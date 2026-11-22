"""Run the frozen textual repair on a V10 package, optionally waiting for available cores."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_v10_semantic_struct as v10
from run_pool_v8 import BASE, ROOT, read, sha, write_once


CONFIG = Path("submission/configs/v11_text_fallback.json")
METHOD = Path("submission/v11_text_fallback.py")
STAGE = ROOT / "artifacts/prepared_data/v11_runtime"
NAME = "e101_v11_text_fallback"


def copy_verified(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(source, dest)
    if sha(dest) != sha(source):
        raise ValueError(f"Frozen package mismatch: {dest}")


def package():
    v10.package()
    old = read(v10.LOCK)
    sources = {Path(name):Path(name) for name in old["files"] if name.startswith("submission/")}
    sources.update({METHOD:Path("submission/submission.py"),CONFIG:CONFIG})
    sources.update({Path("scripts")/name:Path("scripts")/name for name in ("run_experiment.py","limited_run.py")})
    write_once(Path("reports/v11_official_lock.json"), {
        "experiment": NAME, "configuration": str(CONFIG), "method": str(METHOD),
        "v10_lock_sha256": sha(v10.LOCK), "base_files": old["base_files"], "adapter": str(v10.ADAPTER),
        "files": {str(p):sha(p) for p in [*sources, Path("scripts/run_v11_text_fallback.py"),*sorted(v10.ADAPTER.glob("*"))]},
        "selection": "Positive V10 validation replay; trial both full endpoints, independent of V10 public test outcome. Enable edits only on the exact publicly failed SQL, with spare calls.",
    })
    for src, dest in sources.items():
        copy_verified(src, STAGE / dest)
    for name in ("mp3_sql", "training", "runs"):
        link = STAGE / name
        if not link.exists():
            link.symlink_to(ROOT / name, target_is_directory=True)
        assert link.resolve() == ROOT / name


def run(split, wait_for_cpu):
    previous, offset = ("validation", "4") if split == "test" else ("test", "0")
    if wait_for_cpu:
        status_file = ROOT / "runs" / v10.NAME / previous / "r001/run_metadata.json"
        while read(status_file)["status"] == "running":
            time.sleep(10)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1",MP3_BASE_MODEL=str(ROOT / BASE),MP3_ADAPTER_DIR=str(ROOT / v10.ADAPTER))
    subprocess.run([sys.executable, "-B", str(STAGE / "scripts/run_experiment.py"), "--split", split,
                    "--device", "cpu", "--cpu-offset", offset, "--config", str(ROOT / CONFIG),
                    "--experiment", NAME], env=env, cwd=STAGE, check=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=("validation","test","both"),required=True)
    parser.add_argument("--wait-for-v10-cores",action="store_true")
    args=parser.parse_args()
    os.chdir(ROOT)
    package()
    splits=["test","validation"] if args.split=="both" else [args.split]
    with ThreadPoolExecutor(max_workers=len(splits)) as pool:
        list(pool.map(lambda s:run(s,args.wait_for_v10_cores),splits))
