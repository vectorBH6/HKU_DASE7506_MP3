"""Verify frozen v3 model, sources, budgets, replay evidence and default assets."""
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from analyze_recovery_runs import details
from analyze_review_runs import latest

ROOT = Path(__file__).resolve().parents[1]
CRITICAL = ['submission/submission.py'] + [f'submission/sql_repair_agent/{name}.py' for name in
    ['recovery_agent', 'review_agent', 'review_prompts', 'sql_utils', 'model']]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selection = json.loads((ROOT / 'reports/v3_selection.json').read_text())
    artifact = json.loads((ROOT / 'reports/v3_artifact_verification.json').read_text())
    assert not artifact['official_files_modified'] and not artifact['base_files_modified']
    assert artifact['only_lora_A_B_trainable'] and artifact['total_parameters'] == 606142464
    locked = ROOT / selection['locked_config']
    assert sha(locked) == selection['config_sha256']
    configs = [json.loads((ROOT / 'submission/configs' / name).read_text())
               for name in ['recovery_v3_4.json', 'recovery_v3_6.json']]
    differing = [k for k in configs[0].keys() | configs[1].keys() if configs[0].get(k) != configs[1].get(k)]
    assert differing == ['max_completions']
    adapter = ROOT / selection['adapter_path']
    assert {p.name for p in adapter.iterdir()} == {'adapter_config.json', 'adapter_model.safetensors'}
    assert sha(adapter / 'adapter_model.safetensors') == selection['adapter_sha256']
    rows = []
    for experiment, split in [('e20_recovery4', 'validation'), ('e21_recovery6', 'validation'), ('e22_recovery_selected', 'test')]:
        run, meta = latest(experiment, split)
        result, predictions, score = details(experiment, split)
        assert score['n'] == (120 if split == 'validation' else 240)
        assert all(len(q['cases']) == 3 for q in score['per_question'])
        assert all(sha(ROOT / name) == meta['source_sha256'][name] for name in CRITICAL)
        assert meta['offline'] and meta['device'] == 'cpu' and len(meta['cpu_affinity']) == 4
        assert meta['python_executable'] == str(ROOT / '.venv/bin/python')
        assert meta['peak_process_tree_rss_bytes'] <= 16 * 1024 ** 3
        assert result['prediction_seconds'] <= 7200 and result['scoring_seconds'] <= 180
        assert result['max_model_calls'] <= meta['config']['max_completions'] <= 6
        assert result['max_new_tokens'] <= meta['config']['max_new_tokens'] <= 384
        assert result['max_tool_calls'] <= 8
        assert (run / 'prediction.log').read_text().count('Loaded model: ') == 1
        logs = [json.loads(s) for s in (run / 'generations.jsonl').read_text().splitlines()]
        decisions = [json.loads(s) for s in (run / 'decisions.jsonl').read_text().splitlines()]
        by_task = defaultdict(dict)
        for row in logs:
            identity = row['identity']
            assert identity['device'] == 'cpu' and identity['total_parameters'] == 606142464
            assert identity['revision'] == selection['base_revision']
            assert identity['adapter']['sha256'] == selection['adapter_sha256']
            assert row['generation']['do_sample'] is False
            assert row['attempt'] not in by_task[row['task_id']]
            by_task[row['task_id']][row['attempt']] = row
        for attempts in by_task.values():
            assert sorted(attempts) == list(range(1, len(attempts) + 1))
        retries = [d for d in decisions if d['guard_reason'] == 'recovery_retry']
        for retry in retries:
            attempts = by_task[retry['task_id']]
            before = attempts[retry['attempt']]
            after = attempts[retry['attempt'] + 1]
            assert before['messages'] != after['messages']
            assert 'controller_feedback' in after['messages'][1]['content']
        if split == 'test':
            assert meta['config'] == json.loads(locked.read_text())
            assert datetime.fromisoformat(selection['selected_at']) < datetime.fromisoformat(meta['started_at'])
            assert sha(ROOT / 'submission/outputs/v3-test.json') == sha(run / 'predictions.json')
            assert sha(ROOT / 'submission/outputs/v3-test-score.json') == sha(run / 'score.json')
        rows.append({k: result[k] for k in ['experiment', 'split', 'source', 'correct', 'n', 'model_calls', 'max_model_calls', 'max_new_tokens', 'max_tool_calls', 'prediction_seconds', 'scoring_seconds']}
                    | {'peak_process_tree_rss_bytes': meta['peak_process_tree_rss_bytes'], 'verified_recovery_retries': len(retries),
                       'predictions_sha256': sha(run / 'predictions.json'), 'score_sha256': sha(run / 'score.json')})
    default_exp = 'e22_recovery_selected' if selection['promote'] else 'e15_review_selected'
    default_run, _ = latest(default_exp, 'test')
    assert sha(ROOT / 'submission/outputs/test.json') == sha(default_run / 'predictions.json')
    assert sha(ROOT / 'submission/outputs/test-score.json') == sha(default_run / 'score.json')
    active = ROOT / 'submission/configs/selected.json'
    expected = locked if selection['promote'] else ROOT / 'submission/configs/review_v2_selected.json'
    assert active.read_bytes() == expected.read_bytes()
    assert not (ROOT / 'peer_reviews').exists() and not (ROOT / 'dist').exists()
    report = {'all_checks_passed': True, 'validation_configs_differ_only': differing,
              'selection_before_test': True, 'test_metrics_used_for_selection': False,
              'critical_current_sources_match_all_new_run_snapshots': True,
              'model_loaded_once_each_run': True, 'private_recovery_inputs_changed': True,
              'adapter_sha256': selection['adapter_sha256'], 'default_promoted': selection['promote'],
              'default_test_source': str(default_run.relative_to(ROOT)),
              'single_venv': str(ROOT / '.venv'), 'peer_reviews_created': False, 'dist_created': False,
              'artifact_verification': 'reports/v3_artifact_verification.json', 'runs': rows}
    (ROOT / 'reports/v3_runtime_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Recovery audit passed for both validation runs and the locked full test')


if __name__ == '__main__':
    main()
