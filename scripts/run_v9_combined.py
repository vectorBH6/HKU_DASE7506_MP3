"""Freeze and evaluate the combined V9 terminal policy in the official CPU runner."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

from run_pool_v8 import ADAPTER, BASE, ROOT, read, sha, write_once
from summarize_v8_pool import verify_frozen_assets


EXPERIMENT = "e90_v9_combined"
CONFIG = Path("submission/configs/v9_combined.json")
LOCK = Path("reports/v9_official_lock.json")
BASELINE = Path("runs/e82_v8_struct")


def freeze():
    original = read(Path("reports/v8_official_lock.json"))
    verify_frozen_assets(original)
    runtime = sorted(Path("submission/sql_repair_agent").glob("*.py"))
    runtime += [Path("submission/submission.py"), CONFIG, Path("submission/configs/pool_v8_weights.json")]
    runtime += sorted(p for p in CONFIG.parent.glob("*v9*.json") if p != CONFIG)
    runtime += [Path("scripts/run_v9_combined.py"), Path("scripts/run_experiment.py"),
                Path("scripts/limited_run.py"), Path("scripts/run_pool_v8.py"), Path("scripts/summarize_v8_pool.py")]
    weights = sorted(ADAPTER.glob("*"))
    for p in weights:
        if sha(p) != original["files"][str(p)]:
            raise ValueError(f"Adapter changed: {p}")
    write_once(LOCK, {
        "experiment": EXPERIMENT, "config": str(CONFIG),
        "comparison": "Same V8 baseline and same frozen V9 configuration for both full splits; test is an endpoint.",
        "files": {str(p): sha(p) for p in runtime + weights},
        "baseline_scores": {s: sha(BASELINE / s / "r001/score.json") for s in ("validation", "test")},
        "base_files": original["base_files"], "official_file_count": original["official_file_count"],
        "official_manifest_sha256": original["official_manifest_sha256"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    freeze()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", MP3_BASE_MODEL=str(ROOT / BASE), MP3_ADAPTER_DIR=str(ROOT / ADAPTER))
    subprocess.run([sys.executable, "-B", "scripts/run_experiment.py", "--experiment", EXPERIMENT,
                    "--config", str(CONFIG), "--split", args.split, "--device", "cpu", "--cpu-offset", "0"],
                   env=env, check=True)


if __name__ == "__main__":
    main()
