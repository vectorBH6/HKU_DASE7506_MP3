"""Predeclared V5 screen -> CPU 2x2 -> two locked public-test controls.

No test labels are opened before both challenger configurations are locked.
The unchanged official predictor/scorer perform every formal evaluation.
"""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp3_sql.common import write_json, digest

OLD = ROOT / 'artifacts/adapters/onpolicy_v001/epoch_02'
TRAIN = ROOT / 'runs/training/grpo_v5/training_metadata.json'


def latest(exp, split='validation'):
    files = sorted((ROOT / 'runs' / exp / split).glob('r*/run_metadata.json'))
    if not files:
        return None, None
    return json.loads(files[-1].read_text()), files[-1].parent


def wait_run(exp, split='validation'):
    started = time.monotonic()
    while True:
        meta, out = latest(exp, split)
        if meta and meta['status'] == 'completed':
            return meta, out
        if meta and meta['status'] == 'failed':
            raise RuntimeError(f'{exp} failed: {meta.get("error")}')
        if time.monotonic() - started > 15000:
            raise TimeoutError(exp)
        time.sleep(10)


def run_eval(exp, cfg, adapter, device='cuda', offset=20, split='validation'):
    meta, out = latest(exp, split)
    if meta and meta['status'] == 'completed':
        assert meta['config'] == json.loads(cfg.read_text())
        assert Path(meta['adapter_path']).resolve() == adapter.resolve()
        return meta, out
    if meta and meta['status'] == 'running':
        return wait_run(exp, split)
    env = os.environ.copy(); env.pop('PYTHONPATH', None)
    env.update({'MP3_BASE_MODEL': str(ROOT / 'artifacts/models/Qwen3-0.6B'),
                'MP3_ADAPTER_DIR': str(adapter)})
    cmd = [sys.executable, str(ROOT / 'scripts/run_experiment.py'), '--experiment', exp,
           '--split', split, '--config', str(cfg), '--device', device, '--cpu-offset', str(offset)]
    print('START', exp, split, device, str(adapter.relative_to(ROOT)), flush=True)
    with (ROOT / f'artifacts/{exp}-{split}.log').open('a') as handle:
        subprocess.run(cmd, env=env, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
    return wait_run(exp, split)


def summary(label, meta, out):
    gens = [json.loads(s) for s in (out / 'generations.jsonl').read_text().splitlines()]
    return {'label': label, 'run': str(out.relative_to(ROOT)), 'correct': meta['result']['correct'],
            'n': meta['result']['n'], 'device': meta['device'], 'config': meta['config'],
            'adapter': str(Path(meta['adapter_path']).relative_to(ROOT)),
            'generations': len(gens), 'input_tokens': sum(g['input_tokens'] for g in gens),
            'output_tokens': sum(g['new_tokens'] for g in gens),
            'prediction_seconds': meta['stages']['prediction']['wall_seconds'],
            'scoring_seconds': meta['stages']['scoring']['wall_seconds'],
            'rss_bytes': meta.get('peak_process_tree_rss_bytes'),
            'mean_tool_calls': meta['result']['mean_tool_calls']}


def main():
    # Named architecture candidates were fixed from training/code observations.
    arch = []
    for label in ['recover', 'review', 'probe', 'always', 'hints']:
        meta, out = wait_run('dev_v5_search_' + label)
        arch.append({**summary(label, meta, out), 'config_path': f'submission/configs/search_v5_{label}.json'})
    shortlist = sorted(arch, key=lambda r: (-r['correct'], r['generations'], r['label']))[:2]
    pool = ThreadPoolExecutor(max_workers=2)
    confirm = []
    jobs = []
    for index, candidate in enumerate(shortlist):
        jobs.append((candidate, pool.submit(run_eval, 'e51_v5_arch_' + candidate['label'],
                     ROOT / candidate['config_path'], OLD, 'cpu', 4 if candidate['label'] == 'recover' else 0)))
    for candidate, future in jobs:
        meta, out = future.result()
        confirm.append({**summary(candidate['label'], meta, out), 'config_path': candidate['config_path']})
    best_arch = min(confirm, key=lambda r: (-r['correct'], r['generations'], r['label']))
    arch_cfg = ROOT / 'submission/configs/search_v5_selected.json'
    write_json(arch_cfg, best_arch['config'])
    write_json(ROOT / 'reports/v5_arch_gpu_selection.json', {'candidates': arch, 'CPU_shortlist': confirm,
               'best': best_arch, 'selection_uses_test': False, 'CPU_validation_required': False})
    print('ARCH_LOCK', json.dumps(best_arch), flush=True)
    # Start formal old-weight architecture validation while training continues.
    arch_future = pool.submit(run_eval, 'e51_v5_arch_' + best_arch['label'], arch_cfg, OLD, 'cpu', 0)
    screen = []; completed_shas = set()
    while True:
        training = json.loads(TRAIN.read_text())
        for cp in training['checkpoints']:
            if cp['sha256'] in completed_shas:
                continue
            adapter = ROOT / cp['path']; label = adapter.name
            pair = []
            for family, cfg in [('contract', ROOT / 'submission/configs/contract_v4_selected.json'), ('search', arch_cfg)]:
                exp = f'dev_v5_grpo_{label}_{family}'
                meta, out = run_eval(exp, cfg, adapter)
                pair.append(summary(family, meta, out))
            screen.append({'checkpoint': cp, 'results': pair})
            completed_shas.add(cp['sha256'])
            write_json(ROOT / 'reports/v5_grpo_gpu_screen.json', screen)
            print('GRPO_SCREEN', json.dumps(screen[-1]), flush=True)
        if training['status'] in ('completed', 'failed'):
            # Re-read next loop if a checkpoint appeared during the current pass.
            pending = [c for c in json.loads(TRAIN.read_text())['checkpoints'] if c['sha256'] not in completed_shas]
            if not pending:
                break
        time.sleep(10)
    if not screen:
        raise RuntimeError('GRPO produced no valid trained checkpoint')
    # Choose one shared adapter for the full 2x2 CPU matrix.
    chosen = min(screen, key=lambda r: (-max(s['correct'] for s in r['results']),
                 -sum(s['correct'] for s in r['results']), sum(s['generations'] for s in r['results']), r['checkpoint']['path']))
    adapter = ROOT / chosen['checkpoint']['path']
    write_json(ROOT / 'reports/v5_grpo_selection.json', {'chosen': chosen, 'selection_uses_test': False,
               'rule': 'highest GPU validation cell, then sum of both cells, then fewer generations; one shared checkpoint in CPU 2x2',
               'all_candidates': screen})
    base_meta, base_out = wait_run('e50_v5_baseline')
    if base_meta['result']['correct'] != 92:
        raise RuntimeError('Current greedy baseline did not reproduce 92/120; inspect before deployment')
    b_meta, b_out = arch_future.result()
    c_future = pool.submit(run_eval, 'e52_v5_grpo_contract', ROOT / 'submission/configs/contract_v4_selected.json', adapter, 'cpu', 0)
    d_future = pool.submit(run_eval, 'e53_v5_grpo_search', arch_cfg, adapter, 'cpu', 4)
    c_meta, c_out = c_future.result(); d_meta, d_out = d_future.result()
    cells = [summary('A_old_weights_old_arch', base_meta, base_out), summary('B_old_weights_search', b_meta, b_out),
             summary('C_grpo_weights_old_arch', c_meta, c_out), summary('D_grpo_weights_search', d_meta, d_out)]
    write_json(ROOT / 'reports/v5_cpu_matrix.json', cells)
    trained = min(cells[2:], key=lambda r: (-r['correct'], r['generations'], r['label']))
    trained_cfg = ROOT / 'submission/configs/grpo_v5_selected.json'
    write_json(trained_cfg, trained['config'])
    # Exactly two independent challenger families. Both are locked before test.
    lock = {'locked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cpu_matrix': cells,
            'architecture_challenger': cells[1], 'trained_challenger': trained,
            'architecture_config_sha256': digest(arch_cfg), 'trained_config_sha256': digest(trained_cfg),
            'trained_adapter_sha256': digest(adapter / 'adapter_model.safetensors'),
            'test_labels_used_for_training_or_rules': False,
            'final_deployment_rule': 'choose highest publicly self-evaluated aggregate correct count among incumbent184 and these two fixed challengers; retain incumbent on ties',
            'test_budget': 'one full official CPU test for architecture family and one for trained family; no per-question fitting'}
    write_json(ROOT / 'reports/v5_test_lock.json', lock)
    print('TEST_LOCK', json.dumps(lock), flush=True)
    f_arch = pool.submit(run_eval, 'e54_v5_arch_test', arch_cfg, OLD, 'cpu', 0, 'test')
    f_train = pool.submit(run_eval, 'e55_v5_grpo_test', trained_cfg, adapter, 'cpu', 4, 'test')
    arch_meta, arch_out = f_arch.result(); train_meta, train_out = f_train.result()
    tests = [summary('architecture', arch_meta, arch_out), summary('grpo', train_meta, train_out)]
    pool.shutdown()
    baseline_out = ROOT / 'runs/e42_contract_control/test/r001'
    baseline_meta = json.loads((baseline_out / 'run_metadata.json').read_text())
    baseline = summary('incumbent_v4b', baseline_meta, baseline_out)
    best = min([baseline] + tests, key=lambda r: (-r['correct'], r['label'] != 'incumbent_v4b', r['generations']))
    best_out = ROOT / best['run']
    paired = {}
    old_score = json.loads((baseline_out / 'score.json').read_text())
    old = {q['id']: q['passed'] for q in old_score['per_question']}
    for result in tests:
        score = json.loads((ROOT / result['run'] / 'score.json').read_text())
        new = {q['id']: q['passed'] for q in score['per_question']}
        paired[result['label']] = {'wins': [k for k in old if not old[k] and new[k]],
                                   'losses': [k for k in old if old[k] and not new[k]]}
    for result in tests:
        out = ROOT / result['run']
        prefix = 'v5-arch' if result['label'] == 'architecture' else 'v5-grpo'
        shutil.copy2(out / 'predictions.json', ROOT / f'submission/outputs/{prefix}-test.json')
        shutil.copy2(out / 'score.json', ROOT / f'submission/outputs/{prefix}-test-score.json')
    if best['label'] != 'incumbent_v4b':
        write_json(ROOT / 'submission/configs/selected.json', best['config'])
        shutil.copy2(best_out / 'predictions.json', ROOT / 'submission/outputs/test.json')
        shutil.copy2(best_out / 'score.json', ROOT / 'submission/outputs/test-score.json')
        script = ROOT / 'scripts/run_selected.sh'
        text = script.read_text()
        text = text.replace('artifacts/adapters/onpolicy_v001/epoch_02', best['adapter'])
        script.write_text(text)
    deployment = {'best': best, 'tests': tests, 'baseline': baseline, 'paired_changes': paired,
                  'public_test_aggregate_used_for_deployment': True,
                  'test_task_labels_used_for_model_training_or_rules': False,
                  'selected_config_sha256': digest(ROOT / 'submission/configs/selected.json'),
                  'selected_predictions_sha256': digest(ROOT / 'submission/outputs/test.json'),
                  'selected_score_sha256': digest(ROOT / 'submission/outputs/test-score.json')}
    write_json(ROOT / 'reports/v5_results.json', deployment)
    print('V5_EXPERIMENTS_COMPLETE', json.dumps(deployment), flush=True)


if __name__ == '__main__':
    main()
