"""Inventory v3 review assets and lossless logs without creating an archive."""
import gzip
import hashlib
import json
from pathlib import Path
from analyze_review_runs import latest

ROOT = Path(__file__).resolve().parents[1]


def main():
    previous = json.loads((ROOT / 'reports/v2_delivery_manifest.json').read_text())
    candidates = {ROOT / name for name in previous['files']}
    for folder in ['submission', 'scripts', 'training', 'tests', 'reports']:
        for path in (ROOT / folder).rglob('*'):
            if (path.is_file() and '__pycache__' not in path.parts
                    and not path.name.endswith('delivery_manifest.json')
                    and path.suffix in {'.py', '.md', '.txt', '.json', '.csv', '.sh', '.gz'}):
                candidates.add(path)
    candidates = {p for p in candidates if p.exists() and not p.name.endswith('delivery_manifest.json')}
    directory = ROOT / 'reports/recovery_logs'
    directory.mkdir(exist_ok=True)
    index = []
    for experiment, split in [('e20_recovery4', 'validation'), ('e21_recovery6', 'validation'), ('e22_recovery_selected', 'test')]:
        found = latest(experiment, split)
        assert found, f'Run incomplete: {experiment}/{split}'
        run, _ = found
        for name in ['run_metadata.json', 'predictions.json', 'score.json', 'prediction_resources.txt', 'scoring_resources.txt']:
            candidates.add(run / name)
        for name in ['generations.jsonl', 'decisions.jsonl']:
            source = run / name
            raw = source.read_bytes()
            target = directory / f'{experiment}_{split}_{run.name}_{name}.gz'
            target.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
            assert gzip.decompress(target.read_bytes()) == raw
            candidates.add(target)
            index.append({'source': str(source.relative_to(ROOT)), 'compressed': str(target.relative_to(ROOT)),
                          'raw_sha256': hashlib.sha256(raw).hexdigest()})
    index_path = directory / 'index.json'
    index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    candidates.add(index_path)
    files = {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(candidates)}
    total = sum(item['bytes'] for item in files.values())
    assert total <= 16 * 1024 ** 2, (total, 'Source assets exceed 16 MiB')
    result = {'created_archive': False, 'source_bytes': total, 'source_limit_bytes': 16 * 1024 ** 2,
              'note': 'Future export inventory only. All hashes reflect current files. Raw logs remain intact; gzip restores exact bytes. Base, datasets, venv and adapters excluded; adapter exported separately.',
              'lossless_logs': index, 'files': files}
    (ROOT / 'reports/v3_delivery_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Future source assets: {total / 1024 ** 2:.3f} MiB, {len(files)} files; no archive or dist created')


if __name__ == '__main__':
    main()
