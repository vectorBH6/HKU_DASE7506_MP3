"""Greedy rollout or independent grading on the incremental train-schema holdout."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import digest, load_bank, read_jsonl, write_json


def predict(rows, manifest, folder, config, out):
    from mp3_sql.environment import Tools, rollout
    from mp3_sql.predict import load_factory

    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    start = time.monotonic()
    os.environ["MP3_GENERATION_LOG"] = str(out / "generations.jsonl")
    public_tools = Tools(folder, manifest)
    factory = load_factory(ROOT / "submission/submission.py")
    with (out / "predictions.jsonl").open("x") as stream:
        for row in rows:
            result = rollout(row, public_tools, factory(config))
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
    write_json(out / "prediction_metadata.json", {
        "rows": len(rows), "config_sha256": digest(config),
        "adapter_sha256": digest(Path(os.environ["MP3_ADAPTER_DIR"]) / "adapter_model.safetensors"),
        "device": os.environ.get("MP3_DEVICE", "cpu"), "prediction_seconds": time.monotonic() - start
    })


def grade(rows, out):
    from training.v5_utils import TrainReward

    predictions = read_jsonl(out / "predictions.jsonl")
    if [p["id"] for p in predictions] != [r["id"] for r in rows]:
        raise ValueError("Missing or unordered holdout predictions")
    reward = TrainReward()
    metrics = []
    for prediction in predictions:
        final = reward.score(prediction)["passed"]
        candidates = reward.candidates(prediction)
        metrics.append({"id": prediction["id"], "passed": final,
                        "covered": any(c["passed"] for c in candidates),
                        "selected_with_coverage": bool(final and any(c["passed"] for c in candidates)),
                        "tool_calls": prediction["tool_calls"]})
    summary = {"n": len(metrics), "correct": sum(r["passed"] for r in metrics),
               "covered": sum(r["covered"] for r in metrics),
               "selected_with_coverage": sum(r["selected_with_coverage"] for r in metrics),
               "per_question": metrics}
    write_json(out / "score.json", summary)
    print({k: v for k, v in summary.items() if k != "per_question"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["predict", "grade"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = json.loads((ROOT / "artifacts/prepared_data/v6/split.json").read_text())
    rows, manifest, folder = load_bank("train")
    wanted = set(split["holdout_ids"])
    chosen = [r for r in rows if r["id"] in wanted]
    if not wanted or {r["db_id"] for r in chosen} != set(split["holdout_db_ids"]) or len(chosen) != len(wanted):
        raise ValueError("Invalid train schema holdout")
    if args.stage == "predict":
        predict(chosen, manifest, folder, ROOT / "submission/configs/contract_v4_selected.json", args.out)
    else:
        grade(chosen, args.out)


if __name__ == "__main__":
    main()
