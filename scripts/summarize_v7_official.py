"""Summarize official endpoint scores and distinguish repair from preservation."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINES = {"validation": "e50_v5_baseline", "test": "e42_contract_control"}
NAMES = {"validation": {"ordinary": "e70_v7_ordinary", "paired": "e71_v7_paired"},
         "test": {"ordinary": "e72_v7_ordinary_test", "paired": "e73_v7_paired_test"}}


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def scoremap(result):
    per_task = result["per_question"]
    scores = {q["id"]: q["passed"] for q in per_task}
    assert len(scores) == result["n"] == len(per_task) and sum(scores.values()) == result["correct"]
    assert all(len(q["cases"]) == 3 and q["passed"] == all(c["passed"] for c in q["cases"]) for q in per_task)
    return scores


def changes(before, after):
    assert before.keys() == after.keys()
    return {"repaired": sum(not value and after[k] for k, value in before.items()),
            "regressed": sum(value and not after[k] for k, value in before.items())}


def load_arm(experiment, split, adapter_hash):
    parent = ROOT / "runs" / experiment / split
    for run in reversed(sorted(parent.glob("r*"))):
        metadata = read(run / "run_metadata.json")
        if metadata["status"] != "completed":
            continue
        lines = (run / "generations.jsonl").read_text().splitlines()
        log = [json.loads(line) for line in lines]
        if not all(event["identity"]["adapter"]["sha256"] == adapter_hash for event in log):
            continue
        score = read(run / "score.json")
        prediction = read(run / "predictions.json")
        assert digest(run / "predictions.json") == score["predictions_sha256"]
        assert score["split"] == prediction["split"] == split and score["n"] == (120 if split == "validation" else 240)
        assert metadata["offline"] and metadata["device"] == "cpu" and len(metadata["cpu_affinity"]) == 4
        assert metadata["config"] == read(ROOT / "submission/configs/contract_v4_selected.json")
        assert score["scorer_sha256"] == digest(ROOT / "mp3_sql/score.py")
        assert score["executor_sha256"] == digest(ROOT / "mp3_sql/sqlite_tool.py")
        assert score["bank_sha256"] == digest(ROOT / f"mp3_sql/data/{split}.jsonl")
        for k in ["bank_sha256", "manifest_sha256"]:
            assert score[k] == prediction[k]
        return run, metadata, log, score, prediction
    raise ValueError(f"Missing completed {experiment}/{split} run for frozen adapter")


def summarize_split(split, lock):
    baseline_sha = lock["initial_adapter_sha256"]
    draft_dir = ROOT / "runs/e00_draft" / split / "r001"
    draft_score = read(draft_dir / "score.json")
    draft = scoremap(draft_score)
    original = {p["id"]: p["sql"] for p in read(draft_dir / "predictions.json")["predictions"]}
    experiments = {"baseline": BASELINES[split], **NAMES[split]}
    arms, score_by, initial_by = {}, {}, {}
    for name, experiment in experiments.items():
        weight = baseline_sha if name == "baseline" else lock["arms"][name]["adapter_sha256"]
        run, metadata, log, score, prediction = load_arm(experiment, split, weight)
        for k in ["references_sha256", "bank_sha256", "manifest_sha256", "scorer_sha256", "executor_sha256"]:
            assert score[k] == draft_score[k]
        passed = scoremap(score)
        score_by[name] = passed
        groups = Counter()
        first_match = {}
        n_gen = Counter(e["task_id"] for e in log)
        for p in prediction["predictions"]:
            i = p["id"]; checks = [e for e in p["trace"] if e["tool"] == "check"]
            assert checks[0]["sql"] == original[i] and p["trace"][0] == checks[0]
            first_match[i] = checks[0]["observation"]["matches_example"]
            groups["original_right" if draft[i] else "original_wrong"] += 1
            if draft[i]:
                groups["preserved"] += passed[i]
            else:
                subset = "ambiguous_draft" if first_match[i] else "visibly_wrong_draft"
                groups[subset] += 1
                groups[subset + "_repaired"] += passed[i]
            final_checks = [e for e in checks if e["sql"] == p["sql"]]
            assert final_checks
            groups["public_matching_final_still_wrong"] += (not passed[i] and final_checks[-1]["observation"]["matches_example"])
        initial_by[name] = first_match
        assert first_match.keys() == draft.keys() == passed.keys()
        tool_max = max(p["tool_calls"] for p in prediction["predictions"])
        token_max = max(e["new_tokens"] for e in log)
        assert tool_max <= 8 and max(n_gen.values()) <= 4 and token_max <= 384
        times = {k: v["wall_seconds"] for k,v in metadata["stages"].items()}
        rss = metadata["peak_process_tree_rss_bytes"]
        assert times["prediction"] <= 7200 and times["scoring"] <= 180 and rss <= 16*1024**3
        arms[name] = {"run": str(run.relative_to(ROOT)), "score_sha256": digest(run / "score.json"),
                      "n": score["n"], "correct": score["correct"], "accuracy_percent": score["accuracy"],
                      **dict(groups), "all_wrong_drafts_repaired": changes(draft, passed)["repaired"],
                      "vs_baseline": changes(score_by["baseline"], passed),
                      "generations": len(log), "mean_charged_tools": score["mean_tool_calls"],
                      "max_generations": max(n_gen.values()), "max_generated_tokens": token_max,
                      "max_charged_tools": tool_max, "wall_seconds": times, "peak_rss_bytes": rss}
    assert initial_by["ordinary"] == initial_by["baseline"] == initial_by["paired"]
    return {"arms": arms, "paired_vs_ordinary": changes(score_by["ordinary"], score_by["paired"])}


if __name__ == "__main__":
    lock = read(ROOT / "reports/v7_official_lock.json")
    result = {split: summarize_split(split, lock) for split in NAMES}
    out = ROOT / "reports/v7_official_results.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({s: {a: r["correct"] for a,r in v["arms"].items()} for s,v in result.items()}))
