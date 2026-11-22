"""On-policy multi-turn GRPO on official train episodes, only LoRA A/B updates.

One policy update group consists of G independent complete episodes of the same
task. Tool and prompt tokens are never in the loss. The unchanged official train
scorer supplies one binary all-three-instance terminal reward per episode.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'submission')]
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from mp3_sql.common import digest, write_json
from mp3_sql.environment import Tools, rollout
from sql_repair_agent import model as model_module
from sql_repair_agent.contract_agent import ContractAgent
from training.v5_utils import TrainReward, append_line, read_lines


def action_logps(model, sequence):
    """Causal likelihood of generated tokens; no logits/loss on tool/prompt tokens."""
    ids = torch.tensor([sequence['input_ids']], device='cuda', dtype=torch.long)
    start = sequence['prompt_length'] - 1
    positions = torch.arange(start, ids.shape[1] - 1, device='cuda')
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                   logits_to_keep=positions).logits[0].float()
    return torch.log_softmax(logits, dim=-1).gather(1, ids[0, start + 1:].unsqueeze(1)).squeeze(1)


class RolloutGenerator:
    def __init__(self, generator, out, seed):
        self.model = generator.model
        self.tokenizer = generator.tokenizer
        self.out = out
        self.seed = seed
        self.group = 0
        self.sample = 0
        self.sequences = []

    def generate(self, messages, task_id, attempt, config):
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
                 add_generation_prompt=True, enable_thinking=False)
        encoded = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
        length = encoded['input_ids'].shape[1]
        if length + config['max_new_tokens'] > 6144:
            raise ValueError('Training sequence exceeds declared 6144-token bound')
        seed = int(hashlib.sha256(f'{self.seed}:{self.group}:{self.sample}:{task_id}:{attempt}'.encode()).hexdigest()[:8], 16)
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        self.model.eval()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**encoded, do_sample=True, temperature=1.0,
                     top_p=1.0, top_k=0, repetition_penalty=1.0,
                     max_new_tokens=config['max_new_tokens'], use_cache=True,
                     pad_token_id=self.tokenizer.eos_token_id)
        sequence = {'input_ids': output[0].tolist(), 'prompt_length': length}
        self.sequences.append(sequence)
        text = self.tokenizer.decode(output[0, length:], skip_special_tokens=True)
        append_line(self.out / 'generations.jsonl', {'id': task_id, 'group': self.group,
                    'sample': self.sample, 'attempt': attempt, 'messages': messages,
                    'generated_text': text, 'input_tokens': length,
                    'new_tokens': len(sequence['input_ids']) - length, 'seed': seed,
                    'seconds': time.perf_counter() - started,
                    'sampling': {'temperature': 1.0, 'top_p': 1.0, 'top_k': 0}})
        return text


def save_adapter(model, run_name, label):
    path = ROOT / 'artifacts/adapters' / run_name / label
    path.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(path, safe_serialization=True)
    (path / 'README.md').unlink(missing_ok=True)
    assert {p.name for p in path.iterdir()} == {'adapter_config.json', 'adapter_model.safetensors'}
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest(path / 'adapter_model.safetensors')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--run-name', default='grpo_v5')
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    out = ROOT / 'runs/training' / args.run_name
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg['seed']); random.seed(cfg['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('This training configuration requires CUDA')
    os.environ['MP3_ADAPTER_DIR'] = str(ROOT / cfg['init_adapter'])
    os.environ['MP3_DEVICE'] = 'cuda'
    generator = model_module.get_generator(True)
    actor = generator.model
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_('.lora_A.' in name or '.lora_B.' in name)
    for module in actor.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    trainable = {n: p.numel() for n, p in actor.named_parameters() if p.requires_grad}
    assert trainable and all('.lora_A.' in n or '.lora_B.' in n for n in trainable)
    reference_base = AutoModelForCausalLM.from_pretrained(os.environ['MP3_BASE_MODEL'],
                     local_files_only=True, trust_remote_code=False, use_safetensors=True,
                     torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
    reference = PeftModel.from_pretrained(reference_base, ROOT / cfg['init_adapter'],
                                         is_trainable=False, local_files_only=True).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    bridge = RolloutGenerator(generator, out, cfg['seed'])
    model_module.get_generator = lambda use_adapter: bridge
    reward = TrainReward()
    tools = Tools(reward.folder, reward.manifest, cache={})
    policy_config = json.loads((ROOT / 'submission/configs/contract_v4_selected.json').read_text())
    policy_config.update({'max_new_tokens': cfg['max_new_tokens'], 'max_completions': cfg['max_completions']})
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad],
                                  lr=cfg['learning_rate'], betas=(0.9, 0.999), weight_decay=0.0)
    # Sampling preference comes exclusively from previously observed TRAIN failures.
    previous = json.loads((ROOT / 'runs/e30_feedback_collection/train/r001/score.json').read_text())
    hard = {r['id'] for r in previous['per_question'] if not r['passed']}
    coverage = read_lines(ROOT / 'runs/v5_train_coverage/episodes.jsonl')
    seen = {}
    for item in coverage:
        seen.setdefault(item['id'], set()).add(bool(item['passed']))
    hard |= {k for k, v in seen.items() if False in v}
    if 'hard_task_ids' in cfg:
        hard = set(cfg['hard_task_ids'])
    if not hard <= set(reward.by_id):
        raise ValueError('Curriculum may contain official train IDs only')
    hard_rows = [r for r in reward.rows if r['id'] in hard]
    easy_rows = [r for r in reward.rows if r['id'] not in hard]
    rng = random.Random(cfg['seed'])
    schedule = []
    while len(schedule) < cfg['max_groups']:
        block = hard_rows[:] * 2 + easy_rows[:]
        rng.shuffle(block); schedule.extend(block)
    metadata = {'status': 'running', 'config': cfg, 'policy_config': policy_config,
                'algorithm': 'on-policy GRPO, group-normalized binary terminal reward, clipped token ratio, sampled k3 KL to frozen initial SFT adapter',
                'trainable_names': trainable, 'trainable_parameters': sum(trainable.values()),
                'total_parameters': sum(p.numel() for p in actor.parameters()),
                'base_identity': generator.identity, 'initial_adapter_sha256': digest(ROOT / cfg['init_adapter'] / 'adapter_model.safetensors'),
                'train_bank_sha256': digest(reward.folder / 'train.jsonl'),
                'train_references_sha256': digest(reward.folder / 'train_references.json'),
                'script_sha256': digest(Path(__file__)),
                'schema_split': 'official train only; validation/test absent from rollouts, labels, schedule and gradients',
                'reward_access': 'train-only grader object; never attached to actor or its observations',
                'loss_mask': 'only sampled model-generated action tokens including EOS; every prompt, history and tool token excluded',
                'dropout': 'disabled during both likelihood evaluation and training for consistent policy ratios',
                'sampling': 'unwarped categorical policy: temperature=1,top_p=1,top_k=0,repetition_penalty=1',
                'gpu': torch.cuda.get_device_name(), 'checkpoints': [],
                'hard_train_tasks_at_start': sorted(hard), 'reward_scale': '1 iff all three train instances pass, otherwise 0; no tool-count reward'}
    assert metadata['total_parameters'] <= 1000000000
    write_json(out / 'training_metadata.json', metadata)
    started = time.perf_counter(); updates = 0; attempted = 0
    try:
        for group, row in enumerate(schedule[:cfg['max_groups']], start=1):
            attempted = group; bridge.group = group
            episodes = []
            for sample in range(cfg['group_size']):
                bridge.sample = sample; bridge.sequences = []
                agent = ContractAgent(policy_config)
                prediction = rollout(row, tools, agent)
                grade = reward.score(prediction)
                episodes.append({'reward': float(grade['passed']), 'prediction': prediction,
                                 'sequences': bridge.sequences[:], 'generations': agent.completions,
                                 'rejections': len(agent.rejections)})
            rewards = torch.tensor([e['reward'] for e in episodes], dtype=torch.float32)
            std = rewards.std(unbiased=False)
            record = {'group': group, 'id': row['id'], 'updates_before': updates,
                      'rewards': rewards.tolist(), 'mixed': bool(std > 0),
                      'elapsed_seconds': time.perf_counter() - started,
                      'tool_calls': [e['prediction']['tool_calls'] for e in episodes]}
            append_line(out / 'episodes.jsonl', {'group': group, 'id': row['id'],
                        'episodes': [{k: v for k, v in e.items() if k != 'sequences'} for e in episodes]})
            if std == 0:
                append_line(out / 'groups.jsonl', record)
                print('GRPO_SKIP', json.dumps(record), flush=True)
                continue
            advantages = ((rewards - rewards.mean()) / (std + 1e-4)).tolist()
            actor.eval()
            with torch.no_grad():
                for episode in episodes:
                    for sequence in episode['sequences']:
                        sequence['old_logps'] = action_logps(actor, sequence).detach()
                        sequence['ref_logps'] = action_logps(reference, sequence).detach()
            loss_records = []
            for iteration in range(cfg['num_iterations']):
                optimizer.zero_grad(set_to_none=True); actor.train()
                total_loss = 0.0; total_kl = 0.0; clip_fraction = 0.0; ratio_error = 0.0
                for episode, advantage in zip(episodes, advantages):
                    tokens = sum(len(s['input_ids']) - s['prompt_length'] for s in episode['sequences'])
                    for sequence in episode['sequences']:
                        current = action_logps(actor, sequence)
                        old = sequence['old_logps']; ref = sequence['ref_logps']
                        log_ratio = current - old
                        ratio = torch.exp(log_ratio)
                        delta = ref - current
                        kl = torch.exp(delta) - delta - 1
                        unclipped = ratio * advantage
                        clipped = torch.clamp(ratio, 1 - cfg['clip_epsilon'], 1 + cfg['clip_epsilon']) * advantage
                        loss = (-torch.minimum(unclipped, clipped) + cfg['beta'] * kl).sum() / (tokens * cfg['group_size'])
                        if not torch.isfinite(loss):
                            raise RuntimeError('Non-finite GRPO loss')
                        loss.backward()
                        total_loss += float(loss.detach())
                        total_kl += float(kl.detach().sum()) / (tokens * cfg['group_size'])
                        clip_fraction += float(((ratio.detach() - 1).abs() > cfg['clip_epsilon']).sum()) / (tokens * cfg['group_size'])
                        ratio_error = max(ratio_error, float(log_ratio.detach().abs().max()))
                        del current, log_ratio, ratio, delta, kl, unclipped, clipped, loss
                if iteration == 0 and ratio_error > 0.05:
                    raise RuntimeError(f'On-policy likelihood consistency failed: {ratio_error}')
                norm = torch.nn.utils.clip_grad_norm_([p for p in actor.parameters() if p.requires_grad], 1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError('Non-finite gradient norm')
                optimizer.step()
                loss_records.append({'iteration': iteration + 1, 'loss': total_loss, 'kl': total_kl,
                                     'clip_fraction': clip_fraction, 'grad_norm': float(norm),
                                     'max_log_ratio_difference': ratio_error})
            updates += 1
            record.update({'updates_after': updates, 'advantages': advantages, 'losses': loss_records,
                           'max_cuda_memory_allocated': torch.cuda.max_memory_allocated()})
            append_line(out / 'groups.jsonl', record)
            print('GRPO_UPDATE', json.dumps(record), flush=True)
            if updates in cfg['save_updates']:
                saved = save_adapter(actor, args.run_name, f'update_{updates:03d}')
                metadata['checkpoints'].append({**saved, 'updates': updates, 'groups': group})
                write_json(out / 'training_metadata.json', metadata)
            if updates >= cfg['max_updates']:
                break
        saved = save_adapter(actor, args.run_name, 'final')
        metadata['checkpoints'].append({**saved, 'updates': updates, 'groups': attempted})
        metadata['status'] = 'completed'
    except BaseException as exc:
        metadata.update({'status': 'failed', 'error': repr(exc)})
        raise
    finally:
        metadata.update({'seconds': time.perf_counter() - started, 'updates': updates, 'groups': attempted,
                         'max_cuda_memory_allocated': torch.cuda.max_memory_allocated()})
        write_json(out / 'training_metadata.json', metadata)


if __name__ == '__main__':
    main()
