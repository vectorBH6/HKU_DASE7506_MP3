"""Run unchanged official prediction/scoring and preserve reproduction evidence."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", choices=["train", "validation", "test"], required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--cpu-offset", type=int, default=0, help="Choose four available CPUs; useful for independent experiments")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {"mode": "draft"}
    live = config.get("mode") == "qwen"
    parent = ROOT / "runs" / args.experiment / args.split
    parent.mkdir(parents=True, exist_ok=True)
    index = 1
    while (parent / f"r{index:03d}").exists():
        index += 1
    out = parent / f"r{index:03d}"
    out.mkdir()
    for folder in ["submission", "scripts", "training"]:
        shutil.copytree(ROOT / folder, out / "source_snapshot" / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    cpus = sorted(os.sched_getaffinity(0))[args.cpu_offset:args.cpu_offset + 4]
    if len(cpus) != 4:
        raise ValueError("Exactly four CPUs must be available at the requested offset")
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT / "submission") + os.pathsep + str(ROOT),
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                "TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "PYTHONHASHSEED": "42", "MP3_DEVICE": args.device, "MP3_CPU_AFFINITY": ",".join(map(str, cpus)), "MP3_GENERATION_LOG": str(out / "generations.jsonl")})
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    prediction = ["--split", args.split, "--workers", "1" if live else "4", "--output", str(out / "predictions.json")]
    if args.config:
        prediction += ["--method", str(ROOT / "submission/submission.py"), "--checkpoint", str(args.config.resolve())]
    scoring = ["--split", args.split, "--workers", "4", "--predictions", str(out / "predictions.json"), "--output", str(out / "score.json")]
    metadata = {"experiment": args.experiment, "split": args.split, "status": "running",
                "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
                "config": config, "device": args.device, "offline": True, "cpu_affinity": cpus,
                "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                "source_sha256": {str(p.relative_to(ROOT)): sha(p) for folder in ["submission", "scripts", "training", "mp3_sql"] for p in (ROOT / folder).rglob("*.py")},
                "base_model_path": env.get("MP3_BASE_MODEL"), "adapter_path": env.get("MP3_ADAPTER_DIR"),
                "stages": {}}
    def save():
        temporary = out / "run_metadata.json.tmp"
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(out / "run_metadata.json")
    save()
    print(str(out.relative_to(ROOT)), flush=True)
    experiment_started = time.monotonic()
    try:
        for stage, module, rest in [("prediction", "mp3_sql.predict", prediction), ("scoring", "mp3_sql.score", scoring)]:
            cmd = ["/usr/bin/time", "-v", "-o", str(out / f"{stage}_resources.txt"), sys.executable,
                   str(ROOT / "scripts/limited_run.py"), "--memory-gib", "0" if live and stage == "prediction" else "2",
                   "--module", module] + rest
            started = time.monotonic()
            with (out / f"{stage}.log").open("w") as log:
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    timeout = (7200 if stage == "prediction" else 180) if live else max(1, 180 - (time.monotonic() - experiment_started))
                    if live and stage == "prediction":
                        import psutil
                        peak_tree_rss = 0
                        while proc.poll() is None:
                            root = psutil.Process(proc.pid)
                            processes = [root] + root.children(recursive=True)
                            rss = 0
                            for child in processes:
                                try:
                                    rss += child.memory_info().rss
                                except psutil.NoSuchProcess:
                                    pass
                            peak_tree_rss = max(peak_tree_rss, rss)
                            if rss > 16 * 1024 ** 3:
                                raise RuntimeError("Live-model process tree exceeded 16 GiB RSS")
                            if time.monotonic() - started > timeout:
                                raise subprocess.TimeoutExpired(cmd, timeout)
                            time.sleep(0.2)
                        metadata["peak_process_tree_rss_bytes"] = peak_tree_rss
                        rc = proc.returncode
                    else:
                        rc = proc.wait(timeout=timeout)
                except BaseException:
                    if proc.poll() is None:
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                    raise
            metadata["stages"][stage] = {"command": cmd, "returncode": rc, "wall_seconds": time.monotonic() - started}
            save()
            print((out / f"{stage}.log").read_text()[-6000:], flush=True)
            if rc:
                raise RuntimeError(f"{stage} failed with exit code {rc}")
        score = json.loads((out / "score.json").read_text())
        metadata["result"] = {k: score[k] for k in ("n", "correct", "accuracy", "mean_tool_calls", "scoring_seconds")}
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        raise
    finally:
        metadata["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()

if __name__ == "__main__":
    main()
