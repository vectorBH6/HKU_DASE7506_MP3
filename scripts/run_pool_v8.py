"""Freeze three V8 policies, select on validation, and run all prespecified public endpoints."""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARM_RUNS = {"first": ("e80_v8_first", 0), "choose": ("e81_v8_choose", 4),
            "struct": ("e82_v8_struct", 8)}
ADAPTER = Path("artifacts/adapters/v7_paired/epoch_01")
BASE = Path("artifacts/models/Qwen3-0.6B")
LOCK = Path("reports/v8_official_lock.json")
SELECT = Path("reports/v8_validation_selection.json")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def config(arm):
    return Path("submission/configs") / ("v8_" + arm + ".json")


def write_once(path, payload):
    if path.exists():
        existing = read(path)
        assert all(existing[k] == v for k, v in payload.items()), f"Frozen record changed: {path}"
    else:
        with path.open("x") as handle:
            json.dump({**payload, "locked_utc": datetime.now(timezone.utc).isoformat()}, handle, indent=2)
            handle.write("\n")


def freeze():
    baseline = read(Path("reports/v7_official_lock.json"))["arms"]["paired"]
    assert sha(ADAPTER / "adapter_model.safetensors") == baseline["adapter_sha256"]
    assert sha(ADAPTER / "adapter_config.json") == baseline["adapter_config_sha256"]
    basefiles = read(Path("reports/v5_initial_manifest.json"))["base_files"]
    for name, expected in basefiles.items():
        assert sha(Path(name)) == expected, name
    official = read(Path("PACKAGE_MANIFEST.json"))["files"]
    for name, record in official.items():
        assert sha(Path(name)) == record["sha256"], name
    paths = [config(arm) for arm in ARM_RUNS] + sorted(Path("submission/sql_repair_agent").glob("*.py"))
    paths += list(ADAPTER.glob("*")) + [Path(s) for s in [
        "submission/configs/pool_v8_weights.json", "reports/v8_plan_cn.md", "submission/submission.py",
        "scripts/run_pool_v8.py", "scripts/run_experiment.py", "scripts/limited_run.py",
        "runs/v8_pool_train/audit.json", "training/train_pool_v8.py", "artifacts/prepared_data/v6/split.json",
    ]]
    files = {str(p): sha(p) for p in paths}
    write_once(LOCK, {"files": files, "arms": list(ARM_RUNS), "adapter": str(ADAPTER),
                      "base_files": basefiles, "official_file_count": len(official),
                      "official_manifest_sha256": sha(Path("PACKAGE_MANIFEST.json"))})


def run_arm(arm, split):
    exp, offset = ARM_RUNS[arm]
    cfg = config(arm)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(PYTHONDONTWRITEBYTECODE="1", MP3_BASE_MODEL=str(ROOT / BASE),
               MP3_ADAPTER_DIR=str(ROOT / ADAPTER))
    folder = Path("runs") / exp / split
    def completed():
        for p in sorted(folder.glob("r*/run_metadata.json"), reverse=True):
            meta = read(p)
            if (meta["status"] == "completed" and meta["started_at"] > read(LOCK)["locked_utc"]
                    and meta["config"] == read(cfg) and meta["adapter_path"] == env["MP3_ADAPTER_DIR"]):
                return p.parent
        return None
    path = completed()
    if path is None:
        subprocess.run([sys.executable, "-B", "scripts/run_experiment.py", "--cpu-offset", str(offset),
                        "--experiment", exp, "--split", split, "--device", "cpu", "--config", str(cfg)],
                       env=env, cwd=ROOT, check=True)
        path = completed()
    assert path is not None
    return path


def select(validation):
    outcomes = {}
    for name, path in validation.items():
        raw = read(path / "score.json")
        assert raw["split"] == "validation" and raw["n"] == 120
        outcomes[name] = {"correct": raw["correct"], "mean_tools": raw["mean_tool_calls"],
                          "score_sha256": sha(path / "score.json"), "path": str(path)}
    winner = min(outcomes, key=lambda name: (-outcomes[name]["correct"], outcomes[name]["mean_tools"]))
    write_once(SELECT, {"selected": winner, "outcomes": outcomes, "official_lock_sha256": sha(LOCK)})
    print("Validation selected", winner, flush=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    freeze()
    with ThreadPoolExecutor(max_workers=len(ARM_RUNS)) as pool:
        baseline = {"V7": Path("runs/e71_v7_paired/validation/r001")}
        vals = pool.map(lambda a: run_arm(a, "validation"), ARM_RUNS)
        select({**baseline, **dict(zip(ARM_RUNS, vals, strict=True))})
        for path in pool.map(lambda a: run_arm(a, "test"), ARM_RUNS):
            print("Completed", path, flush=True)
