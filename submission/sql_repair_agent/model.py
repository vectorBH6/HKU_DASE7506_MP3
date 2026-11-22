"""One offline, unmerged model per process; complete generation audit trail."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
import time

REPOSITORY = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


class Generator:
    def __init__(self, use_adapter):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        base = os.environ["MP3_BASE_MODEL"]
        self.device = os.environ.get("MP3_DEVICE", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            attn_implementation="sdpa").to(self.device)
        self.base_parameters = sum(p.numel() for p in self.model.parameters())
        self.identity = {"repository": REPOSITORY, "revision": REVISION,
                         "base_path": str(Path(base).resolve()), "adapter": None,
                         "base_parameters": self.base_parameters, "device": self.device,
                         "dtype": str(next(self.model.parameters()).dtype)}
        if use_adapter:
            from peft import PeftModel
            adapter = Path(os.environ["MP3_ADAPTER_DIR"])
            self.model = PeftModel.from_pretrained(self.model, adapter, is_trainable=False, local_files_only=True)
            self.identity["adapter"] = {"path": str(adapter.resolve()),
                "sha256": hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest()}
        self.identity["total_parameters"] = sum(p.numel() for p in self.model.parameters())
        if self.identity["total_parameters"] > 1000000000:
            raise ValueError("Actual model + unmerged adapter exceeds assignment limit")
        self.model.eval()
        print("Loaded model: " + json.dumps(self.identity), flush=True)

    def generate(self, messages, task_id, attempt, config):
        import torch
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(self.device)
        input_tokens = inputs["input_ids"].shape[1]
        if input_tokens > 8192:
            raise ValueError("Prompt exceeds explicit 8192-token input budget")
        started = time.perf_counter()
        # Sampling is opt-in for train rollouts and explicitly declared experiments.
        # Historical/selected configurations retain exactly greedy generation.
        sampling = bool(config.get("do_sample", False))
        sampling_args = {}
        if sampling:
            sampling_args = {"temperature": float(config.get("temperature", 1.0)),
                             "top_p": float(config.get("top_p", 1.0)),
                             "top_k": int(config.get("top_k", 0))}
            seed_text = f"{config.get('seed', 42)}:{task_id}:{attempt}:{config.get('sample_index', 0)}"
            seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
            torch.manual_seed(seed)
            if self.device == "cuda":
                torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=sampling, max_new_tokens=config["max_new_tokens"],
                                         pad_token_id=self.tokenizer.eos_token_id, use_cache=True, **sampling_args)
        generated = output[0, input_tokens:]
        result = self.tokenizer.decode(generated, skip_special_tokens=True)
        record = {"task_id": task_id, "attempt": attempt, "messages": messages, "rendered_prompt": rendered,
                  "generated_text": result, "input_tokens": input_tokens, "new_tokens": len(generated),
                  "seconds": time.perf_counter() - started, "identity": self.identity,
                  "generation": {"do_sample": sampling, "enable_thinking": False, "max_new_tokens": config["max_new_tokens"],
                                 **sampling_args, **({"seed": seed} if sampling else {})}}
        log = os.environ.get("MP3_GENERATION_LOG")
        if not log:
            raise ValueError("MP3_GENERATION_LOG is required for live-model auditability")
        path = Path(log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"generation task={task_id} attempt={attempt} tokens={len(generated)} seconds={record['seconds']:.2f}", flush=True)
        return result


@lru_cache(maxsize=1)
def get_generator(use_adapter):
    return Generator(use_adapter)
