"""Verify frozen runs and summarize end-point differences without exposing individual tasks."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from math import comb, isclose
from pathlib import Path
import re

if __package__:
    from .summarize_v7_official import changes, read, scoremap
else:
    from summarize_v7_official import changes, read, scoremap


ROOT = Path(__file__).resolve().parents[1]
ARMS = {'first': 'e80_v8_first', 'choose': 'e81_v8_choose', 'struct': 'e82_v8_struct'}
BASELINE = {'validation': 'e71_v7_paired', 'test': 'e73_v7_paired_test'}
SIZES = {'validation': 120, 'test': 240}
LOCK = ROOT / 'reports/v8_official_lock.json'
OFFICIAL = ('references_sha256', 'bank_sha256', 'manifest_sha256', 'scorer_sha256', 'executor_sha256')
SUPERVISORS = ('mp3_sql/predict.py', 'mp3_sql/score.py', 'mp3_sql/environment.py',
               'mp3_sql/sqlite_tool.py', 'scripts/run_experiment.py', 'scripts/limited_run.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_lock():
    lock = read(LOCK)
    require(lock.get('files') and instant(lock['locked_utc']).tzinfo is not None, 'Missing dated file lock')
    for relative, sha in lock['files'].items():
        require(digest(ROOT / relative) == sha, f'Frozen file changed: {relative}')
    require(digest(ROOT / 'PACKAGE_MANIFEST.json') == lock['official_manifest_sha256'],
            'Official package manifest changed after freeze')
    return lock


def verify_frozen_assets(lock):
    official = read(ROOT / 'PACKAGE_MANIFEST.json')['files']
    require(len(official) == lock['official_file_count'], 'Official package count mismatch')
    for relative, record in official.items():
        require(digest(ROOT / relative) == record['sha256'], f'Official package file changed: {relative}')
    for relative, sha in lock['base_files'].items():
        require(digest(ROOT / relative) == sha, f'Base model file changed: {relative}')
    return {'official_manifest_sha256':lock['official_manifest_sha256'], 'official_files_verified':len(official),
            'base_files_sha256':lock['base_files']}


def instant(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def compare(before, after):
    counts = changes(before, after)
    wins, losses = counts['repaired'], counts['regressed']
    discordant = wins + losses
    probability = min(1.0, 2 * sum(comb(discordant, i) for i in range(min(wins, losses) + 1))
                      / 2 ** discordant)
    return {'wins': wins, 'losses': losses, 'net': wins - losses,
            'stable_correct': sum(before[k] and after[k] for k in before),
            'stable_incorrect': sum(not before[k] and not after[k] for k in before),
            'accuracy_delta_percentage_points': 100 * (wins - losses) / len(before),
            'mcnemar_exact_two_sided_p': probability}


def choose_run(experiment, split, lock):
    parent = ROOT / 'runs' / experiment / split
    completed, attempts = [], []
    for run in sorted(parent.glob('r[0-9]*')):
        file = run / 'run_metadata.json'
        if not file.is_file():
            continue
        meta = read(file)
        if instant(meta['started_at']) < instant(lock['locked_utc']):
            continue
        attempts.append({'run': run.name, 'status': meta['status']})
        if meta['status'] == 'completed':
            completed.append(run)
    # Fixed run order, independent of score. Failed/incomplete retries stay visible.
    return (completed[0] if completed else None), attempts


def resource_details(run, meta):
    require(meta['offline'] and meta['device'] == 'cpu' and len(set(meta['cpu_affinity'])) == 4,
            f'{run}: expected offline CPU run pinned to four cores')
    stages = {}
    for name, timeout in (('prediction', 7200), ('scoring', 180)):
        stage = meta['stages'][name]
        require(stage['returncode'] == 0 and stage['wall_seconds'] <= timeout, f'{run}: {name} time or exit status')
        evidence = (run / f'{name}_resources.txt').read_text()
        def number(label):
            match = re.search(r'^\s*' + re.escape(label) + r':\s*([0-9.]+)\s*$', evidence, re.M)
            require(match, f'{run}: missing {name} GNU time field {label}')
            return float(match.group(1))
        user, system = number('User time (seconds)'), number('System time (seconds)')
        require(number('Exit status') == 0, f'{run}: {name} GNU time exit status')
        command = stage['command']
        memory_limit = int(command[command.index('--memory-gib') + 1])
        require(memory_limit == (0 if name == 'prediction' else 2), f'{run}: unexpected {name} address-space limit')
        stages[name] = {'wall_seconds': stage['wall_seconds'], 'cpu_user_seconds': user,
                        'cpu_system_seconds': system, 'cpu_seconds': round(user + system, 3),
                        'gnu_time_max_rss_bytes': int(number('Maximum resident set size (kbytes)') * 1024),
                        'address_space_limit_gib': memory_limit}
    rss = meta['peak_process_tree_rss_bytes']
    require(0 < rss <= 16 * 1024 ** 3, f'{run}: live prediction process-tree memory limit')
    return {'cpu_affinity': meta['cpu_affinity'], 'offline': True, 'device': 'cpu',
            'prediction_peak_process_tree_rss_bytes': rss, 'stages': stages,
            'total_cpu_seconds': round(sum(s['cpu_seconds'] for s in stages.values()), 3),
            'total_wall_seconds': sum(s['wall_seconds'] for s in stages.values())}


def generation_details(run, ids, adapter_hash):
    counts, lengths, totals, identities = Counter(), [], Counter(), set()
    with (run / 'generations.jsonl').open() as lines:
        for line in lines:
            event = json.loads(line)
            key, length, identity = event['task_id'], event['new_tokens'], event['identity']
            require(key in ids and type(length) is int and 0 <= length <= 384, f'{run}: unknown task or generation size')
            require(identity['device'] == 'cpu' and identity['adapter']['sha256'] == adapter_hash,
                    f'{run}: frozen adapter/device mismatch')
            require(identity['total_parameters'] <= 1_000_000_000, f'{run}: model size')
            counts[key] += 1
            totals[key] += length
            lengths.append(length)
            identities.add((identity['repository'], identity['revision'], identity['adapter']['sha256'], identity['total_parameters']))
    require(len(identities) == 1 and max(counts.values(), default=0) <= 4, f'{run}: mixed model or excess calls')
    repo, rev, adapter, count = identities.pop()
    return {'total': len(lengths), 'tasks_with_generations': len(counts),
            'max_per_task': max(counts.values()), 'max_new_tokens_per_generation': max(lengths),
            'max_total_new_tokens_per_task': max(totals.values()), 'total_new_tokens': sum(lengths),
            'repository': repo, 'revision': rev, 'adapter_sha256': adapter, 'total_parameters': count}


def score_evidence(run, split):
    score = read(run / 'score.json')
    doc = read(run / 'predictions.json')
    passed = scoremap(score)
    require(score['split'] == doc['split'] == split and score['n'] == SIZES[split], f'{run}: split/count')
    require([p['id'] for p in doc['predictions']] == list(passed), f'{run}: prediction/score ordering')
    require(digest(run / 'predictions.json') == score['predictions_sha256'], f'{run}: predictions SHA')
    require(all(score[k] == doc[k] for k in ('bank_sha256', 'manifest_sha256', 'protocol')), f'{run}: protocol/bank')
    require(doc['budget'] == 8 and isclose(score['accuracy'], sum(passed.values()) * 100 / len(passed)),
            f'{run}: budget or accuracy')
    folder = ROOT / 'mp3_sql/data'
    files = {'references_sha256': folder / f'{split}_references.json', 'bank_sha256': folder / f'{split}.jsonl',
             'manifest_sha256': folder / 'manifest.json', 'scorer_sha256': ROOT / 'mp3_sql/score.py',
             'executor_sha256': ROOT / 'mp3_sql/sqlite_tool.py'}
    require(all(digest(file) == score[name] for name, file in files.items()), f'{run}: official evidence SHA')
    return score, doc, passed


def evidence_files(run, meta, lock, arm):
    files = ['predictions.json', 'score.json', 'run_metadata.json', 'generations.jsonl',
             'scoring_resources.txt', 'prediction_resources.txt']
    if (run / 'decisions.jsonl').is_file():
        files.append('decisions.jsonl')
    supervisors = {k: meta['source_sha256'][k] for k in SUPERVISORS}
    require(all(digest(ROOT / k) == h for k,h in supervisors.items()), f'{run}: supervisor digest drift')
    captured = {}
    if arm != 'v7':
        config_path = f'submission/configs/v8_{arm}.json'
        require(config_path in lock['files'], f'{run}: actual arm config not frozen')
        checkpoint = meta['stages']['prediction']['command']
        require(Path(checkpoint[checkpoint.index('--checkpoint') + 1]).resolve() == (ROOT / config_path).resolve(),
                f'{run}: unexpected prediction checkpoint')
        require(meta['config'] == read(run / 'source_snapshot' / config_path), f'{run}: metadata config mismatch')
        for key, sha in lock['files'].items():
            snapshot = run / 'source_snapshot' / key
            if snapshot.is_file():
                require(digest(snapshot) == sha, f'{run}: locked snapshot mismatch {key}')
                captured[key] = sha
            if key in meta['source_sha256']:
                require(meta['source_sha256'][key] == sha, f'{run}: source digest mismatch {key}')
    return {'run_files_sha256': {name: digest(run / name) for name in files},
            'supervisor_source_sha256': supervisors, 'locked_files_in_run_snapshot': captured}


def action_signature(event, include_observation):
    fields = event if include_observation else {k: v for k,v in event.items() if k != 'observation'}
    blob = json.dumps(fields, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(blob).digest()


def trace_details(doc, score, draft_sql, draft_pass):
    passed = scoremap(score)
    ids = list(passed)
    require(ids == list(draft_pass) == list(draft_sql), 'Draft and arm IDs/order differ')
    raw_counts = {e['id']:e['tool_calls'] for e in score['per_question']}
    groups = Counter()
    called, tools = [], Counter()
    traces = {}
    for row in doc['predictions']:
        key, trace = row['id'], row['trace']
        charged = trace[:-1]
        require(trace and trace[-1] == {'tool':'submit', 'sql':row['sql']} and
                all(e['tool'] in {'schema', 'check', 'execute', 'retrieve', 'repair_hint'} for e in charged),
                f'{key}: missing terminal submit or illegal charged action')
        require(len(charged) == row['tool_calls'] == raw_counts[key] and len(charged) <= 8, f'{key}: tool accounting')
        checks = [e for e in charged if e['tool'] == 'check']
        require(checks and charged[0] == checks[0] and checks[0]['sql'] == draft_sql[key], f'{key}: original not checked first')
        final = [e for e in checks if e['sql'] == row['sql']]
        require(final, f'{key}: final SQL lacks check')
        groups['original_right' if draft_pass[key] else 'original_wrong'] += 1
        if draft_pass[key]:
            groups['preserved'] += passed[key]
            groups['original_right_regressed'] += not passed[key]
        else:
            group = 'ambiguous_draft' if checks[0]['observation']['matches_example'] else 'visibly_wrong_draft'
            groups[group] += 1
            groups[group + '_repaired'] += passed[key]
        final_match = final[-1]['observation']['matches_example']
        groups['public_matching_final'] += final_match
        groups['public_matching_final_still_wrong'] += final_match and not passed[key]
        groups['no_public_matching_candidate'] += not any(e['observation']['matches_example'] for e in checks)
        tools.update(e['tool'] for e in charged)
        called.append(len(charged))
        commands = tuple(action_signature(e, False) for e in charged)
        require(len(set(commands)) == len(commands), f'{key}: duplicate charged action')
        traces[key] = {'actions':commands, 'with_observations':tuple(action_signature(e, True) for e in charged),
                       'checks':tuple(action_signature(e, True) for e in checks)}
    mean_calls = sum(called) / len(called)
    require(isclose(mean_calls, score['mean_tool_calls']), 'mean tool cost differs from official score')
    return {**dict(groups), 'mean_charged_tools':mean_calls, 'max_charged_tools':max(called),
            'charged_tool_counts':dict(tools)}, traces


def trace_comparison(base, new, prefix):
    require(base.keys() == new.keys(), 'Trace ID sets differ')
    result = {'n':len(base), 'definition':'charged V7 prefix, excluding terminal submit' if prefix else 'whole charged history, excluding terminal submit'}
    for field in ('actions', 'with_observations', 'checks'):
        exact = 0
        for key, old in base.items():
            before, after = old[field], new[key][field]
            exact += before == (after[:len(before)] if prefix else after)
        result[field + '_same'] = exact
        result[field + '_different'] = len(base) - exact
    return result


def load_arm(run, split, arm, lock, draft_score, draft_sql, weight):
    meta = read(run / 'run_metadata.json')
    require(meta['status'] == 'completed' and meta['split'] == split, f'{run}: incomplete or wrong split')
    if arm != 'v7':
        require(meta['experiment'] == ARMS[arm] and instant(meta['started_at']) >= instant(lock['locked_utc']),
                f'{run}: label or freeze mismatch')
        config = meta['config']
        require(config.get('controller') == 'review_pool_v8' and config.get('review_pool_policy') == arm,
                f'{run}: variant mismatch')
    require(meta['config']['max_completions'] == 4 and meta['config']['max_new_tokens'] == 384,
            f'{run}: configured generation limits')
    score, doc, passed = score_evidence(run, split)
    require(all(score[k] == draft_score[k] for k in OFFICIAL), f'{run}: draft/arm official hash mismatch')
    require(all(meta['result'][k] == score[k] for k in ('correct','n','accuracy','mean_tool_calls')),
            f'{run}: metadata/scorer mismatch')
    draft = scoremap(draft_score)
    details, traces = trace_details(doc, score, draft_sql, draft)
    generations = generation_details(run, passed, weight)
    resources = resource_details(run, meta)
    resources['cpu_seconds_per_task'] = resources['total_cpu_seconds'] / score['n']
    resources['wall_seconds_per_task'] = resources['total_wall_seconds'] / score['n']
    evidence = evidence_files(run, meta, lock, arm)
    report = {'run':str(run.relative_to(ROOT)), 'n':score['n'], 'correct':score['correct'],
              'accuracy_percent':score['accuracy'], 'compared_to_draft':compare(draft, passed), **details,
              'generations':generations, 'resources':resources, 'evidence':evidence}
    return report, passed, traces


def summarize_split(split, lock, weight, partial=False):
    draft_dir = ROOT / 'runs/e00_draft' / split / 'r001'
    draft_score, draft_doc, draft = score_evidence(draft_dir, split)
    raw = {p['id']:p['sql'] for p in draft_doc['predictions']}
    require(all(p['tool_calls'] == 0 and p['trace'] == [{'tool':'submit','sql':p['sql']}] for p in draft_doc['predictions']),
            f'{split}: expected unchanged zero-call draft run')
    runs = {'v7': ROOT / 'runs' / BASELINE[split] / split / 'r001'}
    attempts = {}
    for arm, experiment in ARMS.items():
        run, attempts[arm] = choose_run(experiment, split, lock)
        if run:
            runs[arm] = run
        elif not partial:
            raise ValueError(f'{split}: no completed post-lock run for {arm}: {attempts[arm]}')
    arms, passed, traces = {}, {}, {}
    for arm, run in runs.items():
        arms[arm], passed[arm], traces[arm] = load_arm(run, split, arm, lock, draft_score, raw, weight)
        if arm != 'v7':
            model_keys = ('repository', 'revision', 'adapter_sha256', 'total_parameters')
            require(all(arms[arm]['generations'][key] == arms['v7']['generations'][key] for key in model_keys),
                    f'{arm}: deployed model differs from V7')
            arms[arm]['compared_to_v7'] = compare(passed['v7'], passed[arm])
            arms[arm]['v7_prefix'] = trace_comparison(traces['v7'], traces[arm], prefix=True)
    paired = {f'{a}_vs_{b}':compare(passed[b], passed[a])
              for a,b in [('choose','first'), ('struct','first'), ('struct','choose')] if a in passed and b in passed}
    consistency = None
    if 'first' in traces and 'choose' in traces:
        consistency = trace_comparison(traces['first'], traces['choose'], prefix=False)
    warnings = []
    if consistency and consistency['with_observations_different']:
        warnings.append('First and choose charged histories differ; their score contrast does not isolate terminal selection.')
    for arm in ARMS:
        if arm in arms and arms[arm]['v7_prefix']['with_observations_different']:
            warnings.append(f'{arm}: some pre-existing V7 charged actions or observations differ; see v7_prefix counts.')
    return {'draft':{'run':str(draft_dir.relative_to(ROOT)), 'n':len(draft), 'correct':sum(draft.values()),
                     'score_sha256':digest(draft_dir / 'score.json'), 'predictions_sha256':draft_score['predictions_sha256']},
            'official_sha256':{k:draft_score[k] for k in OFFICIAL}, 'arms':arms,
            'post_lock_attempts':attempts,'pending_arms':[a for a in ARMS if a not in arms],
            'between_arms':paired,'first_choose_charged_history':consistency,'comparison_warnings':warnings}


def verify_validation_selection(result):
    path = ROOT / 'reports/v8_validation_selection.json'
    record = read(path)
    require(record['official_lock_sha256'] == result['lock_sha256'], 'Validation selection uses another lock')
    arms = result['validation']['arms']
    names = ['V7', *ARMS]
    for name in names:
        actual = arms['v7' if name == 'V7' else name]
        locked = record['outcomes'][name]
        require(locked['correct'] == actual['correct'] and locked['score_sha256'] == actual['evidence']['run_files_sha256']['score.json']
                and locked['path'] == actual['run'] and isclose(locked['mean_tools'], actual['mean_charged_tools']),
                f'{name}: selected validation evidence differs from summarized run')
    best = min(names, key=lambda n: (-record['outcomes'][n]['correct'], record['outcomes'][n]['mean_tools']))
    require(record['selected'] == best, 'Validation selection differs from declared accuracy/cost rule')
    for name in ARMS:
        test_meta = read(ROOT / result['test']['arms'][name]['run'] / 'run_metadata.json')
        require(instant(test_meta['started_at']) >= instant(record['locked_utc']), f'{name}: test began before validation selection')
    return {'selected':best, 'locked_utc':record['locked_utc'], 'selection_file_sha256':digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=tuple(SIZES), help='Print a single split, including pending runs; write no report.')
    args = parser.parse_args()
    lock = checked_lock()
    weight = read(ROOT / 'reports/v7_official_lock.json')['arms']['paired']['adapter_sha256']
    if args.split:
        print(json.dumps(summarize_split(args.split, lock, weight, partial=True), indent=2))
        return
    result = {'locked_utc':lock['locked_utc'], 'lock_sha256':digest(LOCK), 'locked_file_sha256':lock['files'],
              'run_selection_rule':'earliest completed numbered attempt started after the freeze, independent of scores',
              'validation':summarize_split('validation', lock, weight), 'test':summarize_split('test', lock, weight)}
    result['frozen_assets'] = verify_frozen_assets(lock)
    result['validation_selection'] = verify_validation_selection(result)
    output = ROOT / 'reports/v8_pool_results.json'
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output)
    for split in SIZES:
        print(split, {k:v['correct'] for k,v in result[split]['arms'].items()}, result[split]['comparison_warnings'])


if __name__ == '__main__':
    main()
