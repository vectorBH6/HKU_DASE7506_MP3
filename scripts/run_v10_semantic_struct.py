"""Package one student method and run the unmodified four-thread official experiment runner."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from run_pool_v8 import BASE, ROOT, read, sha, write_once
from summarize_v8_pool import verify_frozen_assets

NAME = "e100_v10_semantic_struct"
CONFIG = Path("submission/configs/v10_semantic_struct.json")
METHOD = Path("submission/v10_semantic.py")
ADAPTER = Path("artifacts/adapters/feedback_v4/epoch_01")
STAGE = ROOT / "artifacts/prepared_data/v10_runtime"
LOCK = Path("reports/v10_official_lock.json")


def package():
    old = read(Path("reports/v8_official_lock.json"))
    verify_frozen_assets(old)
    assert sha(Path("PACKAGE_MANIFEST.json")) == old["official_manifest_sha256"]
    cfg = read(CONFIG)
    assert sha(ADAPTER / "adapter_model.safetensors") == cfg["adapter_sha256"]
    src = sorted(Path("submission/sql_repair_agent").glob("*.py"))
    src += [METHOD, CONFIG, Path("submission/configs/pool_v8_weights.json"),
            Path("scripts/run_experiment.py"), Path("scripts/limited_run.py")]
    destinations = {p: Path("submission/submission.py") if p == METHOD else p for p in src}
    deps = [Path("scripts/run_v10_semantic_struct.py"), *sorted(ADAPTER.glob("*"))]
    write_once(LOCK, {
        "experiment": NAME, "configuration": str(CONFIG), "method": str(METHOD),
        "adapter": str(ADAPTER), "base_files": old["base_files"],
        "official_manifest_sha256": old["official_manifest_sha256"],
        "official_file_count": old["official_file_count"],
        "files": {str(p): sha(p) for p in src + deps},
        "selection": "V4 semantic CPU prefix and V8 struct suffix selected on validation replay. Same configuration for both official endpoints.",
        "baseline": {s: sha(Path("runs/e82_v8_struct") / s / "r001/score.json") for s in ("validation", "test")},
    })
    for p, dest in destinations.items():
        target = STAGE / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(p, target)
        if sha(target) != sha(p):
            raise ValueError(f"Packaged file differs from the frozen source: {target}")
    for folder in ("mp3_sql", "training", "runs"):
        link = STAGE / folder
        if not link.exists():
            link.symlink_to(ROOT / folder, target_is_directory=True)
        assert link.resolve() == ROOT / folder


def run(split):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", MP3_BASE_MODEL=str(ROOT / BASE), MP3_ADAPTER_DIR=str(ROOT / ADAPTER))
    # The kernel gives these simultaneous, independent four-thread runs disjoint CPU sets.
    offset = "4" if split == "validation" else "0"
    subprocess.run([sys.executable, "-B", str(STAGE / "scripts/run_experiment.py"), "--split", split,
                    "--device", "cpu", "--cpu-offset", offset, "--config", str(ROOT / CONFIG),
                    "--experiment", NAME], env=env, cwd=STAGE, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test", "both"), required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    package()
    if args.split == "both":
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(run, ["validation", "test"]))
    else:
        run(args.split)
