"""Lock the two V7 data-only training arms and evaluate both on official CPU splits."""
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("submission/configs/contract_v4_selected.json")
PLAN = Path("reports/v7_plan_cn.md")
BASE = Path("artifacts/models/Qwen3-0.6B")
ARMS = {"ordinary": (0, "e70_v7_ordinary", "e72_v7_ordinary_test"),
        "paired": (4, "e71_v7_paired", "e73_v7_paired_test")}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lock_arms():
    frozen = read(Path("reports/v6_selection.json"))
    assert sha(CONFIG) == frozen["config_sha256"]
    init = frozen["rows"]["baseline"]
    assert sha(Path(init["adapter"]) / "adapter_model.safetensors") == init["adapter_sha256"]
    arms = {}
    comparable = []
    for name in ARMS:
        run_name = "v7_" + name
        t = read(Path("runs/training") / run_name / "training_metadata.json")
        samples = Path("artifacts/prepared_data/v7") / name / "train.jsonl"
        assert t["status"] == "completed" and t["examples_used"] > 0 and not t["skipped_long_examples"]
        assert t["init_adapter_sha256"] == init["adapter_sha256"] and t["prepared_data_sha256"] == sha(samples)
        cfg = dict(t["config"])
        cfg.pop("prepared_data")
        comparable.append((cfg, t["examples_used"], t["steps"]))
        ad = Path("artifacts/adapters") / run_name / "epoch_01"
        assert {p.name for p in ad.iterdir()} == {"adapter_model.safetensors", "adapter_config.json"}
        arms[name] = {"adapter": str(ad), "adapter_sha256": sha(ad / "adapter_model.safetensors"),
                      "adapter_config_sha256": sha(ad / "adapter_config.json"), "training_sha256": sha(samples),
                      "examples": t["examples_used"], "steps": t["steps"]}
    assert comparable[0] == comparable[1]
    record = {"plan_sha256": sha(PLAN), "inference_config": str(CONFIG), "config_sha256": sha(CONFIG),
              "initial_adapter_sha256": init["adapter_sha256"], "arms": arms,
              "test_arms": list(ARMS), "training_parameters": comparable[0][0]}
    path = Path("reports/v7_official_lock.json")
    if path.exists():
        previous = read(path)
        previous.pop("locked_utc")
        assert previous == record
    else:
        with path.open("x") as f:
            json.dump({**record, "locked_utc": datetime.now(timezone.utc).isoformat()}, f, indent=2)
            f.write("\n")
    return arms


def run_arm(name, arm):
    offset, validation_name, test_name = ARMS[name]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(MP3_BASE_MODEL=str(ROOT / BASE), MP3_ADAPTER_DIR=str(ROOT / arm["adapter"]),
               PYTHONDONTWRITEBYTECODE="1")
    for split, experiment in [("validation", validation_name), ("test", test_name)]:
        folder = Path("runs") / experiment / split
        completed = []
        for path in sorted(folder.glob("r*/run_metadata.json")):
            meta = read(path)
            if meta["status"] == "completed" and meta["config"] == read(CONFIG):
                log = path.parent / "generations.jsonl"
                with log.open() as f:
                    first = json.loads(next(f))
                if first["identity"]["adapter"]["sha256"] == arm["adapter_sha256"]:
                    completed.append(path.parent)
        if completed:
            print("Using completed run", completed[-1], flush=True)
            continue
        subprocess.run([sys.executable, "-B", "scripts/run_experiment.py", "--experiment", experiment,
                        "--split", split, "--config", str(CONFIG), "--device", "cpu", "--cpu-offset", str(offset)],
                       cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    os.chdir(ROOT)
    arms = lock_arms()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_arm, name, arm) for name, arm in arms.items()]
        for job in futures:
            job.result()
