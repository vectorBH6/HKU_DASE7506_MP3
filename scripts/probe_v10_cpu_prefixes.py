"""Compare released CPU validation prefixes under one unchanged V8 review policy."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]

from mp3_sql.common import digest, load_bank, write_json
from mp3_sql.environment import Tools
from mp3_sql.predict import load_factory
from mp3_sql.score import load_cases, score_document
from replay_validation_v8 import content_digest, continue_episode, paired


def read(path):
    return json.loads(path.read_text())


def sources():
    # Only complete official CPU validation runs predating the V10 candidate lock.
    lock = read(ROOT / "reports/v10_official_lock.json")
    cutoff = lock["locked_utc"]
    for sf in sorted((ROOT / "runs").glob("*/validation/r*/score.json")):
        folder = sf.parent
        meta = read(folder / "run_metadata.json")
        config = meta["config"]
        controller = config.get("controller")
        if meta["status"] != "completed" or meta["finished_at"] >= cutoff:
            continue
        if controller in ("review_pool_v8", "review_terminal_v9") or config["mode"] == "draft":
            continue
        if meta.get("device") != "cpu" and not (meta.get("device") is None and config["mode"] == "feedback"):
            continue
        for stage, module in (("prediction", "mp3_sql.predict"), ("scoring", "mp3_sql.score")):
            assert module in meta["stages"][stage]["command"] and meta["stages"][stage]["returncode"] == 0
        adapter = None
        if config.get("use_adapter"):
            weights = Path(meta["adapter_path"]) / "adapter_model.safetensors"
            adapter = digest(weights)
            with (folder / "generations.jsonl").open() as log:
                generation = json.loads(next(log))
            assert generation["identity"]["adapter"]["sha256"] == adapter
        yield folder, config, adapter


def main():
    rows, manifest, folder = load_bank("validation")
    ids = [row["id"] for row in rows]
    factory = load_factory(ROOT / "submission/submission.py")
    cfg = ROOT / "submission/configs/v8_struct.json"
    tools = Tools(folder, manifest, cache={})
    seen_task_prefix = {}
    doc_v10 = read(ROOT / "runs/v10_prefix_probe/validation/predictions.json")
    results = {}
    unique = {}

    for source, config, adapter in sources():
        prefix = read(source / "predictions.json")
        raw = read(source / "score.json")
        assert prefix["split"] == raw["split"] == "validation" and digest(source / "predictions.json") == raw["predictions_sha256"]
        assert [p["id"] for p in prefix["predictions"]] == ids and [p["id"] for p in raw["per_question"]] == ids
        assert raw["n"] == 120 and len(ids) == 120
        key = content_digest(prefix["predictions"])
        config_key = content_digest([config, adapter])
        if key not in unique:
            predictions = []
            for row, old in zip(rows, prefix["predictions"], strict=True):
                # Exact same task, actions, observations and final SQL give the same deterministic continuation.
                task_key = content_digest(old)
                if task_key not in seen_task_prefix:
                    seen_task_prefix[task_key] = continue_episode(row, tools, old, factory(cfg))
                predictions.append(seen_task_prefix[task_key])
            unique[key] = predictions
        results[str(source.relative_to(ROOT))] = {
            "policy": config.get("controller", config["mode"]), "config": config, "adapter_sha256": adapter,
            "config_and_adapter_digest": config_key, "prefix_sha256": key,
            "source_predictions_sha256": raw["predictions_sha256"], "source_correct": raw["correct"],
        }

    # Controllers have already completed every rollout; only grade below this point.
    cases, case_root = load_cases("validation", folder)
    cache = {}
    def labels(predictions):
        doc = {**doc_v10, "predictions": predictions}
        score = score_document(doc, rows, manifest, folder, cases, case_root, workers=1, cache=cache)
        return [item["passed"] for item in score["per_question"]]

    v10 = labels(doc_v10["predictions"])
    assert sum(v10) == 100
    grades = {}
    for key, predictions in unique.items():
        grades[key] = labels(predictions)
    for name, result in results.items():
        key = result["prefix_sha256"]
        output = grades[key]
        pred = unique[key]
        result.update(correct=sum(output), paired_v10=paired(v10, output), total_checks_and_tools=sum(p["tool_calls"] for p in pred),
                      max_tools=max(p["tool_calls"] for p in pred), continuation_sha256=content_digest(pred))
        print(name, result["source_correct"], result["correct"], result["paired_v10"], flush=True)
    report = {
        "scope": "Validation-only comparison. Official CPU prefixes completed before V10 was frozen; feedback-only CPU arms without a device field were included. Draft, GPU, and previous V8/V9 tails excluded.",
        "method": "A single frozen V8 struct continuation, retaining all actual prefix actions and observations. Exact repeated prefix predictions share computation; entire trajectories and same config/adapter identities are separately identified. Three-case labels opened for grading after rollout. No future test decisions based on per-test outputs.",
        "n": len(rows), "source_runs": len(results), "distinct_full_prefix_predictions": len(unique),
        "distinct_config_adapter_and_predictions": len({(v["config_and_adapter_digest"],v["prefix_sha256"]) for v in results.values()}),
        "unique_task_prefixes": len(seen_task_prefix), "v10_correct": sum(v10), "suffix_sha256": digest(cfg),
        "runs": results,
    }
    write_json(ROOT / "reports/v10_cpu_prefix_sweep.json", report)


if __name__ == "__main__":
    main()
