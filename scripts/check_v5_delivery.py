"""Inventory a reviewable source export without creating dist or an archive."""
from pathlib import Path
import gzip
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp3_sql.common import digest, write_json


def main():
    result = json.loads((ROOT / 'reports/v5_results.json').read_text())
    audit = json.loads((ROOT / 'reports/v5_runtime_audit.json').read_text())
    assert audit['all_checks_passed']
    old = json.loads((ROOT / 'reports/v4_delivery_manifest.json').read_text())
    files = {p for p in old['files'] if (ROOT / p).is_file()}
    for directory in ['submission', 'training', 'scripts', 'tests']:
        for p in (ROOT / directory).rglob('*'):
            relative = p.relative_to(ROOT)
            if p.is_file() and p.suffix in {'.py', '.sh', '.json', '.md', '.txt'} and '__pycache__' not in p.parts:
                if directory == 'submission' and 'outputs' in relative.parts:
                    continue
                files.add(str(relative))
    files.update(['submission/outputs/test.json', 'submission/outputs/test-score.json'])
    for p in (ROOT / 'reports').glob('v5*'):
        if p.is_file() and p.name not in {'v5_delivery_manifest.json', 'v5_final_checks.json'}:
            files.add(str(p.relative_to(ROOT)))
    matrix = json.loads((ROOT / 'reports/v5_cpu_matrix.json').read_text())
    arch = json.loads((ROOT / 'reports/v5_arch_gpu_selection.json').read_text())
    grpo = json.loads((ROOT / 'reports/v5_grpo_gpu_screen.json').read_text())
    formal = {r['run'] for r in matrix + result['tests'] + arch['CPU_shortlist']}
    development = {r['run'] for r in arch['candidates']}
    development.update(r['run'] for c in grpo for r in c['results'])
    compressed = []; folder = ROOT / 'reports/v5_reproduction_logs'; folder.mkdir(exist_ok=True)
    def compress(source):
        path = ROOT / source
        if not path.is_file():
            return
        destination = folder / (source.replace('/', '__') + '.gz')
        content = gzip.compress(path.read_bytes(), compresslevel=9, mtime=0)
        if not destination.exists() or destination.read_bytes() != content:
            destination.write_bytes(content)
        target = str(destination.relative_to(ROOT)); files.add(target)
        compressed.append({'original': source, 'gzip': target,
                           'original_sha256': digest(path), 'gzip_sha256': digest(destination)})
    for run in sorted(formal | development):
        for name in ['run_metadata.json', 'score.json', 'prediction_resources.txt', 'scoring_resources.txt']:
            p = ROOT / run / name
            if p.is_file():
                files.add(str(p.relative_to(ROOT)))
        if run in formal:
            for name in ['predictions.json', 'generations.jsonl', 'decisions.jsonl']:
                compress(run + '/' + name)
    for directory, names in [
        ('runs/training/grpo_v5', ['training_metadata.json', 'groups.jsonl', 'generations.jsonl', 'episodes.jsonl']),
        ('runs/v5_train_coverage', ['collection_config.json', 'generations.jsonl', 'decisions.jsonl', 'episodes.jsonl'])]:
        for name in names:
            compress(directory + '/' + name)
    for p in (ROOT / 'runs/training/grpo_v5/source_snapshot').rglob('*.py'):
        files.add(str(p.relative_to(ROOT)))
    # This train-derived helper is read by the training reproduction script.
    files.add('runs/e30_feedback_collection/train/r001/score.json')
    for alias in ['v5-arch-test.json', 'v5-arch-test-score.json', 'v5-grpo-test.json', 'v5-grpo-test-score.json']:
        compress('submission/outputs/' + alias)
    index = folder / 'index.json'; write_json(index, {'lossless': True, 'raw_files_retained': True, 'files': compressed})
    files.add(str(index.relative_to(ROOT)))
    omitted = []
    def size():
        return sum((ROOT / s).stat().st_size for s in files)
    # Preserve all raw local evidence. Historical GPU prompt dumps are optional
    # in the next export; current formal/test and training evidence stays intact.
    def optional_history(source):
        if source.startswith('reports/v4_reproduction_logs/runs__dev_v4_'):
            return 0
        if source.startswith('reports/v4_reproduction_logs/') and source.endswith('.gz') and 'runs__e42_contract_control__test' not in source:
            return 1
        if source.startswith('reports/recovery_logs/') and source.endswith('.gz'):
            return 2
        if source.startswith('reports/reproduction_logs/') and source.endswith('.gz') and 'e10_onpolicy_collection_train' not in source:
            return 3
        if source.startswith('runs/') and source in old['files']:
            if not any(source.startswith(p) for p in ['runs/training/', 'runs/e10_onpolicy_collection/',
                                                      'runs/e42_contract_control/', 'runs/e30_feedback_collection/train/r001/score.json']):
                return 4
        if source == 'reports/v5_reproduction_logs/runs__v5_train_coverage__generations.jsonl.gz':
            return 5
        return 100
    for source in sorted(files, key=lambda s: (optional_history(s), s)):
        if size() + 256 * 1024 <= 16 * 1024 ** 2:
            break
        if optional_history(source) < 100:
            files.remove(source); omitted.append(source)
    assert size() + 256 * 1024 <= 16 * 1024 ** 2, 'Export exceeds limit: reduce optional history, never hide runtime assets'
    # The historical manifest includes this small provenance record, not weights.
    model_metadata = {'artifacts/models/base_manifest.json'}
    assert all(s in model_metadata or not s.startswith(
        ('artifacts/models/', 'artifacts/adapters/', '.venv/')) for s in files)
    entries = {s: {'bytes': (ROOT / s).stat().st_size, 'sha256': digest(ROOT / s)} for s in sorted(files)}
    manifest = {'source_bytes': sum(v['bytes'] for v in entries.values()),
                'administrative_reserve_bytes': 256 * 1024, 'limit_bytes': 16 * 1024 ** 2,
                'files': entries, 'optional_historical_or_diagnostic_evidence_not_exported': omitted,
                'raw_files_deleted': False, 'archive_created': False, 'dist_created': False,
                'selected_adapter': result['best']['adapter'], 'adapter_is_separate_two_file_asset': True}
    target = ROOT / 'reports/v5_delivery_manifest.json'; write_json(target, manifest)
    links = 0
    for name in ['reports/v5_report.md', 'submission/REPRODUCE_V5.md', 'submission/README.md']:
        path = ROOT / name
        for ref in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            if ref.startswith(('http:', 'https:', '#')):
                continue
            assert (path.parent / ref.split('#', 1)[0]).exists(), (name, ref)
            links += 1
    checks = {'all_checks_passed': True, 'local_document_links_checked': links,
              'payload_files_verified': len(entries), 'payload_bytes': manifest['source_bytes'],
              'source_budget_including_256KiB_administrative_reserve': manifest['source_bytes'] + 256 * 1024,
              'source_limit_bytes': 16 * 1024 ** 2, 'no_archive_or_dist_created': True,
              'single_venv': str(ROOT / '.venv'), 'canonical_test_score_exact_copy': True}
    write_json(ROOT / 'reports/v5_final_checks.json', checks)
    assert target.stat().st_size + (ROOT / 'reports/v5_final_checks.json').stat().st_size < 256 * 1024
    for s, item in entries.items():
        assert digest(ROOT / s) == item['sha256']
    print('V5_DELIVERY_CHECK_COMPLETE', len(entries), manifest['source_bytes'] / 1024 ** 2, 'MiB', flush=True)


if __name__ == '__main__':
    main()
