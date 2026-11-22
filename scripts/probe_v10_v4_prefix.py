"""Evaluate a V4 CPU prefix and a frozen V8 or V9 deterministic suffix on validation."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]

from mp3_sql.common import digest, load_bank, write_json
from mp3_sql.environment import Tools, rollout
from mp3_sql.predict import load_factory
from mp3_sql.score import load_cases, score_document
from replay_validation_v8 import continue_episode, paired


def read(path):
    return json.loads(path.read_text())


def main():
    runs = ROOT / "runs"
    models = {
        "v4": runs / "e34_feedback_variant/validation/r001",
        "v8": runs / "e82_v8_struct/validation/r001",
        "v9": runs / "e90_v9_combined/validation/r001",
        "draft": runs / "e00_draft/validation/r001",
    }
    rows, manifest, folder = load_bank("validation")
    ids = [r["id"] for r in rows]
    labels = {}
    for model, run in models.items():
        score = read(run / "score.json")
        assert score["split"] == "validation" and [r["id"] for r in score["per_question"]] == ids
        labels[model] = [r["passed"] for r in score["per_question"]]
    prefix = read(models["v4"] / "predictions.json")
    assert [p["id"] for p in prefix["predictions"]] == ids
    metadata = read(models["v4"] / "run_metadata.json")
    assert metadata["device"] == "cpu" and metadata["config"]["controller"] == "semantic_v4"

    public_tools = Tools(folder, manifest, cache={})
    factory = load_factory(ROOT / "submission/submission.py")
    results = {}
    predictions = {}
    configs = {}
    for arm in ("v8_struct", "v9_combined"):
        config = ROOT / "submission/configs" / (arm + ".json")
        configs[arm] = digest(config)
        predictions[arm] = [
            continue_episode(r, public_tools, p, factory(config))
            for r, p in zip(rows, prefix["predictions"], strict=True)
        ]

    # Allow a controller-wiring cross-check independent of replayed tool actions.
    # Every prompt must equal the corresponding archived prompt before its response can be reused.
    generations = [json.loads(line) for line in (models["v4"] / "generations.jsonl").read_text().splitlines()]
    indexed = {(g["task_id"], g["attempt"]): g for g in generations}
    used = set()

    class ArchivedGenerator:
        def generate(self, messages, task_id, attempt, config):
            key = (task_id, attempt)
            old = indexed[key]
            assert key not in used and messages == old["messages"]
            assert old["identity"]["adapter"]["sha256"] == config["adapter_sha256"]
            used.add(key)
            return old["generated_text"]

    live_factory = load_factory(ROOT / "submission/v10_semantic.py")
    v10_config = ROOT / "submission/configs/v10_semantic_struct.json"
    with patch("sql_repair_agent.model.get_generator", return_value=ArchivedGenerator()):
        factory_predictions = [rollout(row, public_tools, live_factory(v10_config)) for row in rows]
    assert len(used) == len(generations) and factory_predictions == predictions["v8_struct"]
    assert len(factory_predictions) == len(rows) == 120
    replay_path = ROOT / "runs/v10_prefix_probe/validation/predictions.json"
    write_json(replay_path, {**{k: prefix[k] for k in ("protocol", "split", "budget", "bank_sha256", "manifest_sha256")},
                             "predictions": factory_predictions})

    # The three-instance labels are opened by the independent scorer after both agents finish.
    cases, case_root = load_cases("validation", folder)
    cache = {}
    for arm, preds in predictions.items():
        score = score_document({**prefix, "predictions": preds}, rows, manifest, folder,
                               cases, case_root, workers=1, cache=cache)
        passed = [r["passed"] for r in score["per_question"]]
        draft_public = []
        for row, prediction in zip(rows, preds, strict=True):
            draft_check = next(e for e in prediction["trace"] if e["tool"] == "check")
            assert draft_check["sql"] == row["issue_sql"]
            draft_public.append(draft_check["observation"]["matches_example"])
        correct = [i for i, passed_draft in enumerate(labels["draft"]) if passed_draft]
        hidden = [i for i, passed_draft in enumerate(labels["draft"]) if not passed_draft and draft_public[i]]
        visible = [i for i, passed_draft in enumerate(labels["draft"]) if not passed_draft and not draft_public[i]]

        def count(group):
            return {"correct": sum(passed[i] for i in group), "n": len(group)}
        results[arm] = {
            "correct": score["correct"], "n": score["n"],
            "paired": {m: paired(labels[m], passed) for m in ("v4", "v8", "v9")},
            "originally_correct": count(correct), "wrong_but_publicly_matched": count(hidden),
            "visibly_wrong": count(visible), "charged_tools": sum(p["tool_calls"] for p in preds),
            "max_tools": max(p["tool_calls"] for p in preds), "new_generations": 0,
        }
    report = {
        "scope": "Exploratory validation replay only. No test predictions or labels used.",
        "method": "Replay archived official V4 semantic CPU actions and verify every observation. Continue to submit with unchanged V8 or V9 controller and real tools within the same eight-call episode. Both arms share the previously generated V4 prefix. Grade all 120 final SQL on three official instances.",
        "prefix": str(models["v4"].relative_to(ROOT)), "prefix_controller": metadata["config"],
        "adapter": metadata["adapter_path"], "prefix_device": metadata["device"],
        "prefix_generations": len(generations), "v10_factory_exact_replayed_predictions": len(factory_predictions),
        "replayed_predictions_file": str(replay_path.relative_to(ROOT)),
        "replayed_predictions_sha256": digest(replay_path),
        "v10_factory_prompts_matching_prefix": len(used),
        "config_sha256": configs, "score_sha256": {m: digest(r / "score.json") for m, r in models.items()},
        "prefix_predictions_sha256": digest(models["v4"] / "predictions.json"),
        "arms": results,
    }
    write_json(ROOT / "reports/v10_v4_prefix_probe.json", report)
    print(json.dumps(results))


if __name__ == "__main__":
    main()
