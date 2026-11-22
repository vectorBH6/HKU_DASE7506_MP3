"""Replay actual tools independently and execute new final SQL on all grading databases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

from mp3_sql.common import BUDGET, PROTOCOL, ROOT, digest, load_bank, result_fingerprint, write_json
from mp3_sql.environment import Tools, replay
from mp3_sql.sqlite_tool import execute_sql


def load_cases(split, folder, references=None):
    path = Path(references) if references is not None else folder / f"{split}_references.json"
    if split == "audit" and references is None:
        raise FileNotFoundError("No audit split is included in the public benchmark")
    if not path.is_file():
        raise FileNotFoundError(f"Grading references missing: {path}")
    cases = json.loads(path.read_text())
    if cases["bank_sha256"] != digest(folder / f"{split}.jsonl"):
        raise ValueError("References do not match the bank")
    return cases, path.parent


def score_document(document, rows, manifest, folder, cases, case_root, isolated=False, workers=4,
                   cache=None):
    if (document.get("protocol") != PROTOCOL or type(document.get("budget")) is not int
            or document["budget"] != BUDGET
            or document.get("bank_sha256") != cases["bank_sha256"]
            or document.get("manifest_sha256") != digest(folder / "manifest.json")):
        raise ValueError("Prediction protocol or bank identity mismatch")
    predictions = document.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise ValueError("Exactly one prediction per task is required")
    if cases["ids"] != [row["id"] for row in rows]:
        raise ValueError("Reference ID/order mismatch")
    tools = Tools(folder, manifest, isolated=isolated, cache=cache)
    def grade(pair):
        row, prediction = pair
        sql = replay(row, tools, prediction)
        metadata = manifest["databases"][row["db_id"]]
        tables = [table["name"] for table in metadata["schema"]]
        outcomes = []
        for case in cases["tasks"][row["id"]]:
            path = case_root / case["path"]
            if digest(path) != case["database_sha256"]:
                raise ValueError("Grading database hash mismatch")
            identity = (case["database_sha256"], sql, isolated)
            result = cache.get(identity) if cache is not None else None
            if result is None:
                result = execute_sql(path, sql, tables, isolated)
                if cache is not None and len(cache) < 20000:
                    cache[identity] = result
            passed = result["ok"] and result_fingerprint(result, row["ordered"]) == case["expected"]
            outcomes.append({"passed": bool(passed), "error": result["error"]})
        return {"id": row["id"], "passed": all(item["passed"] for item in outcomes),
                "cases": outcomes, "tool_calls": prediction["tool_calls"]}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(grade, zip(rows, predictions)))
    correct = sum(item["passed"] for item in results)
    return {"protocol": PROTOCOL, "n": len(rows), "correct": correct,
            "accuracy": 100 * correct / len(rows),
            "mean_tool_calls": sum(item["tool_calls"] for item in results) / len(rows),
            "per_question": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "validation", "test", "audit"), required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, help="Optional reference bundle; released split references are the default")
    parser.add_argument("--isolated", action="store_true", help="Linux bubblewrap verification")
    parser.add_argument("--workers", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    rows, manifest, folder = load_bank(args.split, args.data_dir)
    cases, case_root = load_cases(args.split, folder, args.references)
    document = json.loads(args.predictions.read_text())
    if document.get("split") != args.split:
        raise ValueError("Prediction split mismatch")
    result = score_document(document, rows, manifest, folder, cases, case_root,
                            args.isolated, args.workers)
    result.update({
        "split": args.split,
        "benchmark": manifest["benchmark"],
        "manifest_sha256": digest(folder / "manifest.json"),
        "bank_sha256": digest(folder / f"{args.split}.jsonl"),
        "references_sha256": digest(args.references if args.references is not None
                                    else folder / f"{args.split}_references.json"),
        "scorer_sha256": digest(Path(__file__)),
        "executor_sha256": digest(ROOT / "sqlite_tool.py"),
        "predictions_sha256": digest(args.predictions),
    })
    result["scoring_seconds"] = time.perf_counter() - started
    if args.output:
        write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "per_question"},
                     indent=2))


if __name__ == "__main__":
    main()
