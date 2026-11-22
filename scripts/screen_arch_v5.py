"""Small, predeclared validation-only architecture screen; no test selection."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp3_sql.common import write_json


def main():
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update({'MP3_BASE_MODEL': str(ROOT / 'artifacts/models/Qwen3-0.6B'),
                'MP3_ADAPTER_DIR': str(ROOT / 'artifacts/adapters/onpolicy_v001/epoch_02')})
    results = []
    for label in ['recover', 'review', 'probe']:
        exp = 'dev_v5_search_' + label
        parent = ROOT / 'runs' / exp / 'validation'
        while True:
            folders = sorted(parent.glob('r*/run_metadata.json')) if parent.exists() else []
            metadata = json.loads(folders[-1].read_text()) if folders else None
            if metadata and metadata['status'] == 'running':
                time.sleep(10)
                continue
            break
        if not metadata or metadata['status'] != 'completed':
            command = [sys.executable, str(ROOT / 'scripts/run_experiment.py'), '--experiment', exp,
                       '--split', 'validation', '--config', str(ROOT / f'submission/configs/search_v5_{label}.json'),
                       '--device', 'cuda', '--cpu-offset', '12']
            with (ROOT / f'artifacts/v5-arch-{label}.log').open('a') as log:
                subprocess.run(command, env=env, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT)
            folders = sorted(parent.glob('r*/run_metadata.json'))
            metadata = json.loads(folders[-1].read_text())
        out = folders[-1].parent
        generations = [json.loads(s) for s in (out / 'generations.jsonl').read_text().splitlines()]
        item = {'label': label, 'experiment': exp, 'run': str(out.relative_to(ROOT)),
                'config': f'submission/configs/search_v5_{label}.json', 'correct': metadata['result']['correct'],
                'n': metadata['result']['n'], 'generations': len(generations),
                'prediction_seconds': metadata['stages']['prediction']['wall_seconds'],
                'development_gpu_only': True}
        results.append(item)
        write_json(ROOT / 'reports/v5_arch_gpu_screen.json', results)
        print('ARCH_SCREEN', json.dumps(item), flush=True)
    best = min(results, key=lambda r: (-r['correct'], r['generations'], r['label']))
    selected = json.loads((ROOT / best['config']).read_text())
    write_json(ROOT / 'submission/configs/search_v5_selected.json', selected)
    write_json(ROOT / 'reports/v5_arch_gpu_selection.json', {'best': best, 'screen': results,
               'CPU_validation_required': True, 'selection_uses_test': False})


if __name__ == '__main__':
    main()
