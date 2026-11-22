"""Compare a completed official run with V8 and print aggregate diagnostics as JSON."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

if __package__:
    from . import summarize_v8_pool as verify
else:
    import summarize_v8_pool as verify


def generation_signatures(run):
    by_task = defaultdict(list)
    with (run / 'generations.jsonl').open() as lines:
        for line in lines:
            row = json.loads(line)
            # Wall time is machine-dependent. The prompt, generated text, token
            # counts and generation arguments describe the actual inference.
            payload = [row[k] for k in ('attempt', 'rendered_prompt', 'generated_text',
                                        'input_tokens', 'new_tokens', 'generation')]
            by_task[row['task_id']].append(hashlib.sha256(json.dumps(
                payload, ensure_ascii=False, sort_keys=True).encode()).digest())
    return by_task


def summarize(run, split, draft_sql, draft_pass, official, adapter):
    score, doc, passed = verify.score_evidence(run, split)
    meta = verify.read(run / 'run_metadata.json')
    verify.require(meta['status'] == 'completed' and meta['split'] == split, f'{run}: incomplete run or split')
    verify.require(all(score[k] == official[k] for k in verify.OFFICIAL), f'{run}: benchmark identity differs')
    verify.require(all(meta['result'][k] == score[k] for k in ('correct', 'n', 'accuracy', 'mean_tool_calls')),
                   f'{run}: runner/scorer mismatch')
    config = meta['config']
    verify.require(config['max_completions'] == 4 and config['max_new_tokens'] == 384,
                   f'{run}: generation limits differ from V8')
    command = meta['stages']['prediction']['command']
    config_path = Path(command[command.index('--checkpoint') + 1])
    snapshot = run / 'source_snapshot' / config_path.resolve().relative_to(verify.ROOT)
    verify.require(verify.read(snapshot) == config, f'{run}: config differs from run snapshot')
    groups, traces = verify.trace_details(doc, score, draft_sql, draft_pass)
    result = {
        'run': str(run.relative_to(verify.ROOT)), 'started_at': meta['started_at'],
        'n': score['n'], 'correct': score['correct'], 'accuracy_percent': score['accuracy'],
        **groups, 'total_charged_tools': sum(groups['charged_tool_counts'].values()),
        'generations': verify.generation_details(run, passed, adapter),
        'resources': verify.resource_details(run, meta),
        'config_sha256': verify.digest(snapshot),
        'run_files_sha256': {name: verify.digest(run / name) for name in
                             ('score.json', 'predictions.json', 'generations.jsonl', 'run_metadata.json')},
    }
    final_sql = {row['id']: row['sql'] for row in doc['predictions']}
    return result, passed, traces, final_sql, generation_signatures(run)


def compare_run(new_run):
    new_run = new_run.resolve()
    metadata = verify.read(new_run / 'run_metadata.json')
    split = metadata['split']
    verify.require(split in verify.SIZES, 'Expected a full validation or test run')
    v8_run = verify.ROOT / 'runs/e82_v8_struct' / split / 'r001'
    old = verify.read(verify.ROOT / 'reports/v8_pool_results.json')[split]
    draft_run = verify.ROOT / old['draft']['run']
    draft_score, draft_doc, draft = verify.score_evidence(draft_run, split)
    verify.require(verify.digest(draft_run / 'score.json') == old['draft']['score_sha256'], 'Draft baseline changed')
    draft_sql = {row['id']: row['sql'] for row in draft_doc['predictions']}
    adapter = old['arms']['struct']['generations']['adapter_sha256']
    v8, before, traces_before, sql_before, gen_before = summarize(v8_run, split, draft_sql, draft, draft_score, adapter)
    saved = old['arms']['struct']['evidence']['run_files_sha256']
    verify.require(all(saved[k] == v for k, v in v8['run_files_sha256'].items()), 'Historical V8 files changed')
    # Historical V8/V9 runs share one adapter. Later configurations can pin a
    # different LoRA while keeping the same frozen base model and scoring bank.
    candidate_adapter = metadata['config'].get('adapter_sha256', adapter)
    adapter_path = Path(metadata['adapter_path']) / 'adapter_model.safetensors'
    verify.require(verify.digest(adapter_path) == candidate_adapter, 'Declared candidate adapter SHA mismatch')
    new, after, traces_after, sql_after, gen_after = summarize(
        new_run, split, draft_sql, draft, draft_score, candidate_adapter)
    identity = ('repository', 'revision')
    verify.require(all(v8['generations'][k] == new['generations'][k] for k in identity), 'Model identity drift')
    prefix = verify.trace_comparison(traces_before, traces_after, prefix=True)
    prefix['definition'] = 'Entire charged V8 trace is a prefix of the new trace, excluding submit'
    delta = verify.compare(before, after)
    same_transcripts = sum(gen_before[i] == gen_after[i] for i in before)
    delta.update({
        'changed_final_sql': sum(sql_before[i] != sql_after[i] for i in before),
        'same_model_inferences': same_transcripts if adapter == candidate_adapter else 0,
        'additional_charged_tools': new['total_charged_tools'] - v8['total_charged_tools'],
        'additional_generations': new['generations']['total'] - v8['generations']['total'],
    })
    if candidate_adapter != adapter:
        delta['same_generation_transcripts_with_distinct_adapters'] = same_transcripts
    return {'split': split, 'official_sha256': {k: draft_score[k] for k in verify.OFFICIAL},
            'v8': v8, 'candidate': new, 'candidate_vs_v8': delta, 'charged_prefix_vs_v8': prefix}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='Completed run directory, e.g. runs/NAME/test/r001')
    args = parser.parse_args()
    print(json.dumps(compare_run(args.run), ensure_ascii=False, indent=2))
