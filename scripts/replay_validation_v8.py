"""Validation-only suffix diagnostic. Run real tools from a replayed V7 submit boundary."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from diagnose_v6 import load_run
from mp3_sql.common import BUDGET, digest, load_bank, write_json
from mp3_sql.environment import Episode, Tools
from mp3_sql.predict import load_factory
from mp3_sql.score import load_cases, score_document
from sql_repair_agent.sql_utils import sql_key

BASE = Path("runs/e71_v7_paired/validation/r001")
ARMS = ("first", "choose", "struct")


def content_digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def matched(event):
    return event["tool"] == "check" and event["observation"]["matches_example"]


def continue_episode(row, tools, old, agent):
    if old["trace"][-1] != {"tool": "submit", "sql": old["sql"]}:
        raise ValueError("Missing V7 submission")
    episode = Episode(row, tools)
    for e in old["trace"][:-1]:
        episode.step({k: v for k, v in e.items() if k != "observation"})
        if episode.trace[-1] != e:
            raise ValueError("Recorded V7 tool observation differs from independent replay")
    if len(episode.trace) != old["tool_calls"]:
        raise ValueError("V7 tool count mismatch")
    agent.begin_review(episode.observation(), old["sql"])
    while episode.submitted is None:
        episode.step(agent.act(episode.observation()))
    if agent.completions:
        raise ValueError("Model unexpectedly called in continuation")
    return {"id": row["id"], "sql": episode.submitted, "trace": episode.trace,
            "tool_calls": BUDGET - episode.remaining}


def paired(a, b):
    return {"wins": sum(y and not x for x, y in zip(a, b)),
            "losses": sum(x and not y for x, y in zip(a, b)),
            "net": sum(b) - sum(a)}


def main():
    baseline, _, old_labels = load_run(ROOT / BASE)
    if baseline["split"] != "validation":
        raise ValueError("Only validation can be replayed")
    rows, manifest, folder = load_bank("validation")
    cases, case_root = load_cases("validation", folder)
    ids = [r["id"] for r in rows]
    if ids != list(old_labels) or any(len(cases["tasks"][i]) != 3 for i in ids):
        raise ValueError("Expected three validation grading instances per original prediction")
    old = baseline["predictions"]
    public_tools = Tools(folder, manifest, cache={})
    factory = load_factory(ROOT / "submission/submission.py")
    lock_path = Path("reports/v8_official_lock.json")
    lock = json.loads((ROOT / lock_path).read_text())["files"]
    sources = [f"submission/configs/v8_{arm}.json" for arm in ARMS] + [
        "submission/sql_repair_agent/review_pool_v8.py", "submission/sql_repair_agent/pool_edits_v8.py",
        "submission/sql_repair_agent/pool_rank_v8.py", "submission/configs/pool_v8_weights.json",
        "submission/submission.py"]
    for f in sources:
        if digest(ROOT / f) != lock[f]:
            raise ValueError(f"File differs from pre-validation lock: {f}")

    docs = {}
    for arm in ARMS:
        cfg = ROOT / f"submission/configs/v8_{arm}.json"
        docs[arm] = [continue_episode(r, public_tools, p, factory(cfg)) for r, p in zip(rows, old)]

    # Only the evaluation process sees the reference bundle, after all arms finish their rollouts.
    scoring_cache = {}
    def grade(preds, bank=rows):
        subset_cases = {**cases, "ids": [r["id"] for r in bank]}
        doc = {**baseline, "predictions": preds}
        return score_document(doc, bank, manifest, folder, subset_cases, case_root,
                              workers=1, cache=scoring_cache)["per_question"]
    labels = {a: [r["passed"] for r in grade(preds)] for a, preds in docs.items()}
    old_pass = [r["passed"] for r in grade(old)]
    if old_pass != list(old_labels.values()):
        raise ValueError("Independent V7 rescoring disagrees with archived official grading")
    identical = sum(p["trace"][:-1] == q["trace"][:-1] for p, q in zip(docs["first"], docs["choose"]))
    if identical != len(rows):
        raise ValueError("First/choose charged actions or observations differ")

    # Each SQL checked in the largest realized pool has a legal submit-after-check prefix.
    # Re-score that prefix with the unchanged scorer for diagnostic generation/selection coverage.
    positions = []
    for p, q in zip(docs["choose"], docs["struct"]):
        if p["trace"][:-1] != q["trace"][:p["tool_calls"]]:
            raise ValueError("Structure arm does not preserve choose's charged prefix")
        positions.append([j for j, e in enumerate(q["trace"]) if e["tool"] == "check"])
    checked = [dict() for _ in rows]
    for slot in range(max(map(len, positions))):
        subset = [i for i, ps in enumerate(positions) if slot < len(ps)]
        preds = []
        for i in subset:
            prefix = docs["struct"][i]["trace"][:positions[i][slot] + 1]
            sql = prefix[-1]["sql"]
            preds.append({"id": ids[i], "sql": sql, "tool_calls": len(prefix),
                          "trace": prefix + [{"tool": "submit", "sql": sql}]})
        for i, pred, item in zip(subset, preds, grade(preds, [rows[i] for i in subset])):
            checked[i][sql_key(pred["sql"])] = item["passed"]

    arms = {}
    for a, preds in docs.items():
        counts = Counter()
        for i, (base, pred, passed) in enumerate(zip(old, preds, labels[a])):
            suffix = pred["trace"][base["tool_calls"]:-1]
            checks = [e for e in pred["trace"] if e["tool"] == "check"]
            covered = any(checked[i][sql_key(e["sql"])] for e in checks)
            public = any(matched(e) for e in checks)
            counts.update({"tasks_with_suffix_check": any(e["tool"] == "check" for e in suffix),
                           "tasks_with_new_public_match": any(matched(e) for e in suffix),
                           "tasks_with_correct_checked_candidate": covered,
                           "final_fail_with_correct_checked_candidate": not passed and covered,
                           "final_fail_without_correct_checked_candidate": not passed and not covered,
                           "final_fail_with_any_public_match": not passed and public,
                           "final_fail_without_any_public_match": not passed and not public,
                           "final_changed_vs_v7": sql_key(pred["sql"]) != sql_key(base["sql"]),
                           "suffix_tool_calls": len(suffix), "total_tool_calls": pred["tool_calls"]})
        arms[a] = {"correct": sum(labels[a]), "failed": len(rows) - sum(labels[a]),
                   "paired_vs_v7": paired(old_pass, labels[a]), **counts,
                   "predictions_content_sha256": content_digest(preds)}

    structural = Counter()
    for i, (a, b) in enumerate(zip(docs["choose"], docs["struct"])):
        extra = [e for e in b["trace"][a["tool_calls"]:-1] if e["tool"] == "check"]
        structural.update({"tasks_with_new_structure_checks": bool(extra), "extra_checks": len(extra),
                           "tasks_with_new_structure_public_match": any(matched(e) for e in extra),
                           "tasks_with_new_structure_correct_check": any(
                               checked[i][sql_key(e["sql"])] for e in extra)})
    report = {"split": "validation", "method": "Saved V7 prefix independently replayed to identical tool observations; no new LM invocations. Each continuation and each checked-candidate prefix graded with the unchanged official three-instance scorer. These are replay diagnostics; fresh validation runs are separate.",
              "n": len(rows), "instances_per_final": 3, "replayed_prefixes_per_arm": len(rows),
              "v7_baseline_correct": sum(old_pass), "v7_run": str(BASE),
              "v7_predictions_sha256": digest(ROOT / BASE / "predictions.json"),
              "v7_score_sha256": digest(ROOT / BASE / "score.json"),
              "validation_bank_sha256": baseline["bank_sha256"], "validation_references_sha256": digest(folder / "validation_references.json"),
              "official_environment_sha256": digest(ROOT / "mp3_sql/environment.py"),
              "official_scorer_sha256": digest(ROOT / "mp3_sql/score.py"),
              "v8_lock_sha256": digest(ROOT / lock_path), "replay_script_sha256": digest(__file__),
              "artifacts_sha256": {f: lock[f] for f in sources}, "arms": arms,
              "choose_vs_first": {"same_charged_actions_and_observations": identical, **paired(labels["first"], labels["choose"])},
              "struct_vs_choose": {**structural, **paired(labels["choose"], labels["struct"])}}
    write_json(ROOT / "reports/v8_validation_replay.json", report)
    print(json.dumps({k: report[k] for k in ("arms", "choose_vs_first", "struct_vs_choose")}, indent=2))


if __name__ == "__main__":
    main()
