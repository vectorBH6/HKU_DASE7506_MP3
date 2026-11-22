"""Masked supervised fine-tuning; only unmerged LoRA A/B matrices can train."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "training/configs/lora_sft_v002.json")
    parser.add_argument("--run-name", default="sft_v002")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    out = ROOT / "runs/training" / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    base = os.environ["MP3_BASE_MODEL"]
    torch.set_num_threads(4)
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU training requested but CUDA unavailable")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    prepared = ROOT / cfg.get("prepared_data", "artifacts/prepared_data/sft_v001")
    records = [json.loads(line) for line in (prepared / "train.jsonl").read_text().splitlines()]
    dataset, skipped = [], []
    for row in records:
        prompt = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(row["target"] + tokenizer.eos_token, add_special_tokens=False)
        if len(prefix) + len(target) > cfg["max_length"]:
            skipped.append({"id": row["id"], "state": row["state"], "tokens": len(prefix) + len(target)})
            continue
        dataset.append({"input_ids": prefix + target, "labels": [-100] * len(prefix) + target})
    def collate(batch):
        length = math.ceil(max(len(r["input_ids"]) for r in batch) / 8) * 8
        return {"input_ids": torch.tensor([r["input_ids"] + [tokenizer.pad_token_id] * (length-len(r["input_ids"])) for r in batch]),
                "labels": torch.tensor([r["labels"] + [-100] * (length-len(r["labels"])) for r in batch]),
                "attention_mask": torch.tensor([[1] * len(r["input_ids"]) + [0] * (length-len(r["input_ids"])) for r in batch])}
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(cfg["seed"]), num_workers=0)
    model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, trust_remote_code=False,
                use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    base_parameters = sum(p.numel() for p in model.parameters())
    if cfg.get("init_adapter"):
        model = PeftModel.from_pretrained(model, ROOT / cfg["init_adapter"], is_trainable=True, local_files_only=True)
    else:
        model = get_peft_model(model, LoraConfig(r=cfg["rank"], lora_alpha=cfg["alpha"], lora_dropout=cfg["dropout"],
                    target_modules=cfg["target_modules"], bias="none", task_type="CAUSAL_LM", modules_to_save=None))
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    trainable = {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(".lora_A." in n or ".lora_B." in n for n in trainable), "Forbidden trainable weights"
    total_parameters = sum(p.numel() for p in model.parameters())
    assert total_parameters <= 1000000000
    if cfg.get("target_only_logits") and cfg["batch_size"] != 1:
        raise ValueError("Target-only projection currently requires micro-batch 1")
    if cfg.get("target_only_logits"):
        # Verify causal indexing against the ordinary masked model loss before
        # using the standard logits_to_keep API to avoid projecting prompt tokens.
        model.eval()
        sample = next(iter(loader))
        sample = {k: v.to("cuda") for k, v in sample.items()}
        positions = (sample["labels"][0, 1:] != -100).nonzero(as_tuple=True)[0]
        with torch.no_grad():
            reference = model(**sample, use_cache=False).loss.float()
            selected = model(input_ids=sample["input_ids"], attention_mask=sample["attention_mask"],
                             logits_to_keep=positions, use_cache=False).logits
            selected_loss = torch.nn.functional.cross_entropy(selected[0].float(), sample["labels"][0, positions + 1])
        difference = abs(reference.item() - selected_loss.item())
        if difference > 0.03:
            raise RuntimeError(f"Target-only loss verification failed: {difference}")
        print(f"target_only_loss_reference={reference.item():.6f} selected={selected_loss.item():.6f} difference={difference:.6f}", flush=True)
        del reference, selected, selected_loss, sample
        torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"], weight_decay=0.01)
    per_epoch = math.ceil(len(loader) / cfg["gradient_accumulation"])
    total_steps = per_epoch * cfg["epochs"]
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=max(1, int(total_steps * .05)), num_training_steps=total_steps)
    metadata = {"training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "config": cfg, "base_parameters": base_parameters, "total_parameters": total_parameters,
                "trainable_parameters": sum(trainable.values()), "trainable_names": trainable,
                "prepared_data_sha256": hashlib.sha256((prepared / "train.jsonl").read_bytes()).hexdigest(),
                "provenance": json.loads((prepared / "provenance.json").read_text()),
                "examples_used": len(dataset), "skipped_long_examples": skipped,
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "loss": "cross entropy on assistant target (SQL or action JSON) + EOS only; prompts/padding masked",
                "optimizer": "AdamW, betas=(0.9,0.999), eps=1e-8, weight_decay=0.01, grad_clip=1.0",
                "init_adapter_sha256": hashlib.sha256((ROOT / cfg["init_adapter"] / "adapter_model.safetensors").read_bytes()).hexdigest() if cfg.get("init_adapter") else None,
                "target_only_loss_check_difference": difference if cfg.get("target_only_logits") else None,
                "status": "running"}
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"examples={len(dataset)} skipped={len(skipped)} LoRA_parameters={sum(trainable.values())} steps={total_steps}", flush=True)
    started = time.perf_counter()
    global_step = 0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, cfg["epochs"] + 1):
        loss_sum, batches = 0., 0
        for i, batch in enumerate(loader):
            batch = {k: v.to("cuda") for k, v in batch.items()}
            group_start = (i // cfg["gradient_accumulation"]) * cfg["gradient_accumulation"]
            group_size = min(cfg["gradient_accumulation"], len(loader) - group_start)
            if cfg.get("target_only_logits"):
                positions = (batch["labels"][0, 1:] != -100).nonzero(as_tuple=True)[0]
                output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               logits_to_keep=positions, use_cache=False)
                raw_loss = torch.nn.functional.cross_entropy(output.logits[0].float(), batch["labels"][0, positions + 1])
            else:
                output = model(**batch, use_cache=False)
                raw_loss = output.loss
            if not torch.isfinite(raw_loss):
                raise RuntimeError("Non-finite loss")
            (raw_loss / group_size).backward()
            loss_sum += raw_loss.item()
            batches += 1
            if (i + 1) % cfg["gradient_accumulation"] == 0 or i + 1 == len(loader):
                norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                record = {"epoch": epoch, "step": global_step, "mean_loss": loss_sum / batches,
                          "learning_rate": scheduler.get_last_lr()[0], "grad_norm": float(norm),
                          "elapsed_seconds": time.perf_counter() - started}
                with (out / "loss.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                if global_step % 5 == 0:
                    print(json.dumps(record), flush=True)
        adapter = ROOT / "artifacts/adapters" / args.run_name / f"epoch_{epoch:02d}"
        adapter.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(adapter, safe_serialization=True)
        # PEFT generates a model card automatically; the prescribed adapter
        # artifact must contain exactly the two allowed weight/config files.
        (adapter / "README.md").unlink(missing_ok=True)
        print(f"epoch={epoch} mean_loss={loss_sum / batches:.6f} adapter={adapter}", flush=True)
    metadata.update({"status": "completed", "seconds": time.perf_counter() - started,
                     "steps": global_step, "max_cuda_memory_allocated": torch.cuda.max_memory_allocated(),
                     "final_epoch_mean_loss": loss_sum / batches})
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

if __name__ == "__main__":
    main()
