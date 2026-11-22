"""Run the replay-locked V12 student factory with new CPU generations."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from run_pool_v8 import ROOT, BASE, read, sha
from run_v11_text_fallback import copy_verified

LOCK = Path("reports/v12/v12_replay_lock.json")
STAGE = ROOT / "artifacts/prepared_data/v12_runtime"


def verify_project_assets(lock):
    manifest = Path("PACKAGE_MANIFEST.json")
    if sha(manifest) != lock["official_manifest_sha256"]:
        raise ValueError("The official package manifest has changed")
    records = read(manifest)["files"]
    if len(records) != lock["official_file_count"]:
        raise ValueError("Unexpected number of official package files")
    for relative, record in records.items():
        path = Path(relative)
        # The project overview occupies README.md. The original specification
        # retains its released bytes under README_en.md in this checkout.
        if relative == "README.md" and Path("README_en.md").is_file():
            path = Path("README_en.md")
        if sha(path) != record["sha256"]:
            raise ValueError(f"Official starter file differs: {path}")
    for relative, expected in lock["base_files"].items():
        if sha(Path(relative)) != expected:
            raise ValueError(f"Base model file differs: {relative}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("validation","test"))
    parser.add_argument("--cpu-offset", default="8")
    args = parser.parse_args()
    os.chdir(ROOT)
    lock = read(LOCK)
    verify_project_assets(lock)
    for name, expected in lock["files"].items():
        p = Path(name)
        if sha(p) != expected:
            raise ValueError(f"File changed after policy lock: {p}")
        if name.startswith("submission/"):
            target = Path("submission/submission.py") if name == lock["entrypoint"] else p
            copy_verified(p,STAGE / target)
    runner = Path("scripts/run_experiment.py")
    for path in (runner, Path("scripts/limited_run.py")):
        assert sha(path)==read(Path("reports/v8/v8_official_lock.json"))["files"][str(path)]
        copy_verified(path,STAGE / path)
    for folder in ("mp3_sql","training","runs"):
        link = STAGE / folder
        if not link.exists():
            link.symlink_to(ROOT / folder, target_is_directory=True)
        assert link.resolve() == ROOT / folder
    env = dict(os.environ)
    env.pop("PYTHONPATH",None)
    env.update(PYTHONDONTWRITEBYTECODE="1",MP3_BASE_MODEL=str(ROOT / BASE),MP3_ADAPTER_DIR=str(ROOT / lock["adapter"]))
    subprocess.run([sys.executable,"-B",str(STAGE / runner),"--experiment","e121_v12_fresh",
                    "--split",args.split,"--config",str(ROOT / lock["config"]),"--device","cpu",
                    "--cpu-offset",args.cpu_offset],env=env,cwd=STAGE,check=True)


if __name__ == "__main__":
    main()
