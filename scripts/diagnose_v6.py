"""Offline analysis of already-completed train/validation trajectories using official replay."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mp3_sql.common import digest, load_bank, read_jsonl, write_json
from mp3_sql.score import load_cases, score_document


def load_run(folder):
    document = json.loads((folder / "predictions.json").read_text())
    split = document["split"]
    if split not in {"train", "validation"}:
        raise ValueError("This development diagnostic accepts only train/validation runs")
    metadata = json.loads((folder / "run_metadata.json").read_text())
    score = json.loads((folder / "score.json").read_text())
    if metadata["status"] != "completed" or metadata["split"] != split or score["split"] != split:
        raise ValueError(f"Incomplete or inconsistent run: {folder}")
    if score["predictions_sha256"] != digest(folder / "predictions.json"):
        raise ValueError(f"Saved official score does not identify these predictions: {folder}")
    if score["scorer_sha256"] != digest(ROOT / "mp3_sql/score.py"):
        raise ValueError("Current scorer differs from the one used for this run")
    passed = {p["id"]: p["passed"] for p in score["per_question"]}
    ids = [p["id"] for p in document["predictions"]]
    if ids != list(passed) or len(ids) != score["n"] or sum(passed.values()) != score["correct"]:
        raise ValueError("Prediction and score populations differ")
    return document, metadata, passed


def diagnose(run, baseline, selected_ids):
    document, metadata, passed = load_run(run)
    split = document["split"]
    bank, manifest, folder = load_bank(split)
    cases, case_root = load_cases(split, folder)
    if selected_ids is None:
        selected_ids = set(passed)
    if not selected_ids or not selected_ids <= set(passed):
        raise ValueError("Subset must be nonempty and contained in the run")
    rows = {r["id"]: r for r in bank}
    predictions = {p["id"]: p for p in document["predictions"] if p["id"] in selected_ids}
    checks = {i: [j for j, e in enumerate(p["trace"]) if e["tool"] == "check"]
              for i, p in predictions.items()}
    results = {i: [] for i in selected_ids}
    execution_cache = {}
    for slot in range(max(map(len, checks.values()), default=0)):
        batch_rows = []
        batch_predictions = []
        for task_id, positions in checks.items():
            if slot >= len(positions):
                continue
            position = positions[slot]
            prefix = predictions[task_id]["trace"][:position + 1]
            sql = prefix[-1]["sql"]
            batch_rows.append(rows[task_id])
            batch_predictions.append({"id": task_id, "sql": sql, "tool_calls": len(prefix),
                                      "trace": prefix + [{"tool": "submit", "sql": sql}]})
        ordered_ids = [r["id"] for r in batch_rows]
        replayed = score_document({**document, "predictions": batch_predictions}, batch_rows,
                                  manifest, folder, {**cases, "ids": ordered_ids}, case_root,
                                  workers=1, cache=execution_cache)
        for item in replayed["per_question"]:
            task_id = item["id"]
            event = predictions[task_id]["trace"][checks[task_id][slot]]
            results[task_id].append((event["sql"], event["observation"]["matches_example"],
                                     item["passed"]))

    correct = {i for i in selected_ids if passed[i]}
    covered = {i for i, items in results.items() if any(ok for _, _, ok in items)}
    false_public = {i for i, items in results.items() if any(pub and not ok for _, pub, ok in items)}
    public_wrong_finals = sum(any(sql == predictions[i]["sql"] and pub for sql, pub, _ in results[i])
                              for i in selected_ids - correct)
    tools = Counter(e["tool"] for p in predictions.values() for e in p["trace"])
    generations_path = run / "generations.jsonl"
    generations = read_jsonl(generations_path) if generations_path.exists() else []
    generations = [g for g in generations if g["task_id"] in selected_ids]
    generation_counts = Counter(g["task_id"] for g in generations)
    n = len(selected_ids)
    costs = {"model_generations": len(generations), "mean_generations": len(generations) / n,
             "max_generations": max(generation_counts.values(), default=0),
             "input_tokens": sum(g["input_tokens"] for g in generations),
             "output_tokens": sum(g["new_tokens"] for g in generations),
             "mean_charged_tools": sum(p["tool_calls"] for p in predictions.values()) / n,
             "max_charged_tools": max(p["tool_calls"] for p in predictions.values()),
             "official_actions": dict(tools)}
    if n == len(passed):
        costs.update(prediction_wall_seconds=metadata["stages"]["prediction"]["wall_seconds"],
                     scoring_wall_seconds=metadata["stages"]["scoring"]["wall_seconds"],
                     peak_process_tree_rss_bytes=metadata.get("peak_process_tree_rss_bytes"))
    report = {
        "run": str(run), "split": split, "device": metadata["device"], "n": n,
        "official_completed_run_n": len(passed), "final_correct": len(correct),
        "final_accuracy": len(correct) / n,
        "tasks_with_correct_checked_candidate": len(covered),
        "checked_candidate_coverage": len(covered) / n,
        "correct_final_given_correct_checked_candidate": len(covered & correct),
        "selection_accuracy_given_coverage": len(covered & correct) / len(covered) if covered else None,
        "final_failed_without_correct_checked_candidate": len(selected_ids - (correct | covered)),
        "final_failed_with_correct_checked_candidate": len(covered - correct),
        "correct_final_without_correct_checked_candidate": len(correct - covered),
        "official_checks_replayed": sum(map(len, checks.values())),
        "tasks_with_public_match_incorrect_candidate": len(false_public),
        "public_match_incorrect_candidate_occurrences": sum(
            pub and not ok for items in results.values() for _, pub, ok in items),
        "public_match_final_wrong": public_wrong_finals, "costs": costs,
        "predictions_sha256": digest(run / "predictions.json"),
        "official_score_sha256": digest(run / "score.json"),
        "official_scorer_sha256": digest(ROOT / "mp3_sql/score.py"),
        "method": "Each actual check prefix was resubmitted offline and scored with all official instances. "
                  "A task is covered if at least one such prefix passed. All rates are fractions.",
    }
    if baseline is not None:
        base_doc, _, base_pass = load_run(baseline)
        if base_doc["split"] != split or base_doc["bank_sha256"] != document["bank_sha256"] or not selected_ids <= set(base_pass):
            raise ValueError("Baseline and candidate populations differ")
        base_correct = {i for i in selected_ids if base_pass[i]}
        report["paired"] = {"baseline": str(baseline), "baseline_correct": len(base_correct),
                            "wins": len(correct - base_correct), "losses": len(base_correct - correct),
                            "stable_correct": len(correct & base_correct),
                            "stable_incorrect": n - len(correct | base_correct)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--ids", type=Path, help="Optional JSON array of task IDs to summarize")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected_ids = None
    if args.ids is not None:
        values = json.loads(args.ids.read_text())
        if not isinstance(values, list) or not all(isinstance(i, str) for i in values):
            parser.error("--ids requires a JSON array of task ID strings")
        selected_ids = set(values)
    report = diagnose(args.run, args.baseline, selected_ids)
    if args.ids:
        report["subset_ids_sha256"] = digest(args.ids)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
