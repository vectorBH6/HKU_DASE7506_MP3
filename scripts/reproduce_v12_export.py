"""Reproduce V12 with the released MP3 predictor and three-instance scorer on Linux."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil


def run_stage(module, arguments, env, out, name, seconds):
    stdout = out / f"{name}.log"
    resources = out / f"{name}_resources.txt"
    command = [
        "/usr/bin/time",
        "-v",
        "-o",
        str(resources),
        sys.executable,
        "-B",
        "-m",
        module,
        *map(str, arguments),
    ]
    with stdout.open("w") as log:
        child = subprocess.Popen(
            command,
            cwd=out,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        peak = 0
        start = time.monotonic()
        try:
            while child.poll() is None:
                root = psutil.Process(child.pid)
                rss = 0
                for p in [root, *root.children(recursive=True)]:
                    try:
                        rss += p.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss > 16 * 1024**3 or time.monotonic() - start > seconds:
                    raise RuntimeError(f"{name} exceeded its reproduction budget")
                time.sleep(0.2)
            if child.returncode:
                raise RuntimeError(
                    f"{name} exited with code {child.returncode}; see {stdout}"
                )
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return {
        "wall_seconds": time.monotonic() - start,
        "peak_process_tree_rss_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--starter",
        type=Path,
        required=True,
        help="Root of the original course starter",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="A fresh directory for this run"
    )
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--score-only", action="store_true", help="Score the submitted predictions"
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    starter = args.starter.resolve()
    out = args.output_dir.resolve()
    if not (starter / "mp3_sql/predict.py").is_file():
        parser.error("--starter must contain the official mp3_sql package")
    if out.exists() and any(out.iterdir()):
        parser.error("--output-dir must be empty to keep independent run evidence")
    out.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))[:4]
    if len(cpus) != 4:
        parser.error("The approved CPU route needs four available logical CPUs")
    os.sched_setaffinity(0, cpus)
    env = dict(os.environ)
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, (source, starter))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONHASHSEED="42",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="",
        MP3_DEVICE="cpu",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
        MP3_GENERATION_LOG=str(out / "generations.jsonl"),
    )
    prediction = source / "outputs" / (args.split + ".json")
    stages = {}
    if not args.score_only:
        for k, file in (
            ("MP3_BASE_MODEL", "config.json"),
            ("MP3_ADAPTER_DIR", "adapter_model.safetensors"),
        ):
            p = Path(env.get(k, "/nonexistent")).resolve()
            if not (p / file).is_file():
                parser.error(f"Set {k} to the local directory containing {file}")
            env[k] = str(p)
        prediction = out / (args.split + ".json")
        args_predict = [
            "--split",
            args.split,
            "--method",
            source / "submission.py",
            "--checkpoint",
            source / "configs/selected.json",
            "--workers",
            "1",
            "--output",
            prediction,
        ]
        stages["prediction"] = run_stage(
            "mp3_sql.predict", args_predict, env, out, "prediction", 7200
        )
    score = out / (args.split + "-score.json")
    args_score = [
        "--split",
        args.split,
        "--predictions",
        prediction,
        "--workers",
        "4",
        "--output",
        score,
    ]
    stages["scoring"] = run_stage("mp3_sql.score", args_score, env, out, "scoring", 180)
    (out / "resources.json").write_text(
        json.dumps({"cpus": cpus, "stages": stages}, indent=2) + "\n"
    )
    result = json.loads(score.read_text())
    print(
        f"{args.split}: {result['correct']}/{result['n']} ({result['accuracy']:.2f}%)"
    )


if __name__ == "__main__":
    main()
