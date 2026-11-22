"""Run the student entry point on real tools without opening final references."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path
import time

from mp3_sql.common import BUDGET, PROTOCOL, ROOT, digest, load_bank, write_json
from mp3_sql.environment import Tools, rollout


def load_factory(path):
    specification = importlib.util.spec_from_file_location("sql_student_method", path)
    if specification is None or specification.loader is None:
        raise ValueError("Cannot load agent module")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.create_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation", "test", "audit"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--method", type=Path, default=ROOT / "submission.py")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    rows, manifest, folder = load_bank(args.split, args.data_dir)
    tools = Tools(folder, manifest)
    factory = load_factory(args.method)
    def run(row):
        return rollout(row, tools, factory(args.checkpoint))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        predictions = list(pool.map(run, rows))
    document = {"protocol": PROTOCOL, "split": args.split, "budget": BUDGET,
                "bank_sha256": digest(folder / f"{args.split}.jsonl"),
                "manifest_sha256": digest(folder / "manifest.json"),
                "predictions": predictions, "prediction_seconds": time.perf_counter() - started}
    write_json(args.output, document)
    print(json.dumps({"n": len(rows), "prediction_seconds": document["prediction_seconds"]}))


if __name__ == "__main__":
    main()
