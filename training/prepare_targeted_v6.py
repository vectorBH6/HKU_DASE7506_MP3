"""Build equally sized V6 repair and candidate-choice arms from official train traces."""
import argparse
from collections import Counter, defaultdict
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import BUDGET, digest, load_bank, read_jsonl, write_json, write_jsonl
from sql_repair_agent.review_agent import parse_proposal
from sql_repair_agent.review_prompts import messages_for
from sql_repair_agent.sql_utils import sql_key

TRACE = ROOT / "runs/v5_train_coverage"
GOLD = ROOT / "artifacts/prepared_data/onpolicy_v001"
normalize = lru_cache(maxsize=None)(sql_key)


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def unpack(messages):
    return json.loads(messages[1]["content"].split("TASK_DATA\n", 1)[1].split("\nEND_TASK_DATA", 1)[0])


def record(episode, state, messages, target, **metadata):
    return {"id": episode["id"], "db_id": episode["db_id"], "sample": episode["sample"],
            "state": state, "messages": messages, "target": compact(target), **metadata}


def split_schemas(rows):
    ordered = sorted({r["db_id"] for r in rows}, key=lambda s: fingerprint("v6-incremental-split:" + s))
    boundary = (len(ordered) + 4) // 5
    holdout = set(ordered[:boundary])
    return {"rule": "lowest ceil(n/5) SHA256('v6-incremental-split:'+schema) values",
            "scope": "held out from V6 updates and replay; historical initialization used official train",
            "fit_db_ids": sorted(set(ordered) - holdout), "holdout_db_ids": sorted(holdout),
            "fit_ids": [r["id"] for r in rows if r["db_id"] not in holdout],
            "holdout_ids": [r["id"] for r in rows if r["db_id"] in holdout]}


def load_gold(ids):
    predictions = GOLD / "gold_validation.predictions.json"
    score = json.loads((GOLD / "gold_validation.score.json").read_text())
    if (score["split"] != "train" or score["correct"] != len(ids)
            or digest(predictions) != score["predictions_sha256"]
            or {r["id"] for r in score["per_question"] if r["passed"]} != ids):
        raise ValueError("Invalid previous three-instance train gold evaluation")
    return {p["id"]: p["sql"] for p in json.loads(predictions.read_text())["predictions"]}


def add_unique(bucket, row):
    key = fingerprint(compact(row["messages"]))
    previous = bucket.setdefault(key, row)
    if previous["target"] != row["target"]:
        raise ValueError("Conflicting labels on identical prompt")


def chooser_examples(episode, task, labels, bucket):
    candidates = episode["checked_candidates"]
    if {c["passed"] for c in candidates if c["matches_example"]} != {False, True}:
        return
    history = []
    present = set()
    for event in episode["prediction"]["trace"]:
        if event["tool"] == "submit":
            break
        history.append(event)
        if event["tool"] == "check" and event["observation"]["matches_example"]:
            present.add(labels[normalize(event["sql"])])
        if present == {False, True} and any(e["tool"] == "schema" for e in history):
            break
    # The deployed model is called only when at least one tool remains.
    if len(history) >= BUDGET or not any(e["tool"] == "schema" for e in history):
        return
    observation = {"task": {"id": task["id"], "question": task["question"]},
                   "issue_sql": task["issue_sql"], "example": task["example"],
                   "ordered": task["ordered"], "history": history, "remaining": BUDGET - len(history)}
    messages = messages_for(observation, {})
    payload = unpack(messages)
    entries = payload["checked_candidates"]
    good = [i for i, c in enumerate(entries) if c["feedback"]["matches_example"] and labels[normalize(c["sql"])]]
    bad = [i for i, c in enumerate(entries) if c["feedback"]["matches_example"] and not labels[normalize(c["sql"])]]
    positive, negative = good[0], bad[0]
    pair = fingerprint(compact([task["id"], normalize(entries[positive]["sql"]), normalize(entries[negative]["sql"])]))
    # Preserve the factual results. The counterbalanced version changes only display order and local IDs.
    for swapped in (False, True):
        ordered = list(range(len(entries)))
        if swapped:
            ordered[positive], ordered[negative] = ordered[negative], ordered[positive]
        new_entries = [{**entries[i], "id": f"c{j}"} for j, i in enumerate(ordered)]
        pos, neg = f"c{ordered.index(positive)}", f"c{ordered.index(negative)}"
        adjusted = [messages[0], {"role": "user", "content": "TASK_DATA\n" + compact(
            {**payload, "checked_candidates": new_entries}) + "\nEND_TASK_DATA\nNext action JSON:"}]
        add_unique(bucket, record(episode, "contrast_public_match", adjusted,
                   {"tool": "submit", "candidate": pos, "reason_code": "consistent"},
                   pair_id=pair, positive_candidate=pos, negative_candidate=neg,
                   order="swapped" if swapped else "original"))


def collect(episodes, tasks, gold):
    """Join exact deployed model prompts to their sampled episodes via the logged deterministic seed."""
    by_seed = {}
    by_task = defaultdict(list)
    truth = defaultdict(dict)
    for episode in episodes:
        tid = episode["id"]
        if tasks[tid]["db_id"] != episode["db_id"]:
            raise ValueError("Train schema mismatch")
        by_task[tid].append(episode)
        for c in episode["checked_candidates"]:
            key = normalize(c["sql"])
            if truth[tid].setdefault(key, c["passed"]) != c["passed"]:
                raise ValueError("Conflicting three-instance SQL label")
        for attempt in range(1, episode["completions"] + 1):
            seed = int(fingerprint(f"42:{tid}:{attempt}:{episode['sample']}")[:8], 16)
            by_seed[tid, attempt, seed] = episode
    if set(by_task) != set(tasks) or any({r["sample"] for r in g} != {0, 1, 2, 3} or len(g) != 4
                                       for g in by_task.values()):
        raise ValueError("Incomplete independent train episodes")
    for tid, sql in gold.items():
        if truth[tid].setdefault(normalize(sql), True) is not True:
            raise ValueError("Gold score conflicts with train rollout labels")

    chooser, repair, replay = {}, {}, {}
    for episode in episodes:
        chooser_examples(episode, tasks[episode["id"]], truth[episode["id"]], chooser)
    observed = set()
    with (TRACE / "generations.jsonl").open() as handle:
        for line in handle:
            event = json.loads(line)
            tid, attempt = event["task_id"], event["attempt"]
            key = tid, attempt, event["generation"]["seed"]
            if key in observed:
                raise ValueError("Duplicate model event in train coverage")
            observed.add(key)
            episode = by_seed[key]
            messages = event["messages"]
            data = unpack(messages)
            if data["training_examples"] or data["question"] != tasks[tid]["question"]:
                raise ValueError("Unexpected retrieved cross-schema data or task mismatch in trace")
            checked = data["checked_candidates"]
            known = {normalize(c["sql"]) for c in checked}
            if not data["schema"] or not checked or data["remaining_tool_calls"] <= 0:
                continue
            # A repair target is never a repeat of a previous check or recorded rejection.
            rejected = {normalize(p["sql"]) for p in data.get("controller_feedback", {}).get("rejected_proposals", [])
                        if p.get("sql")}
            if not episode["passed"] and not any(truth[tid][q] for q in known):
                alternatives = sorted({c["sql"] for r in by_task[tid] for c in r["checked_candidates"] if c["passed"]},
                                      key=lambda q: (len(q), q))
                verified = [(q, "correct_train_rollout") for q in alternatives] + [(gold[tid], "verified_train_gold")]
                available = next(((q, src) for q, src in verified if normalize(q) not in known | rejected), None)
                if available:
                    target, source = available
                    add_unique(repair, record(episode, "repair_failed_context", messages,
                               {"tool": "check", "sql": target, "reason_code": "semantic_repair"}, label_source=source,
                               rejected_context=bool(data.get("controller_feedback")),
                               all_four_failed=not any(r["passed"] for r in by_task[tid]),
                               attempt=attempt))
            if not episode["passed"]:
                continue
            proposed = parse_proposal(event["generated_text"]) or {}
            tool = proposed.get("tool")
            action = None
            if tool == "check" and isinstance(proposed.get("sql"), str):
                q = normalize(proposed["sql"])
                actually_checked = {normalize(c["sql"]) for c in episode["checked_candidates"] if c["passed"]}
                if q not in known | rejected and q in actually_checked:
                    action = {"tool": "check", "sql": proposed["sql"], "reason_code": "semantic_repair"}
            elif tool == "submit":
                chosen = next((c for c in checked if c["id"] == proposed.get("candidate")), None)
                if chosen and truth[tid][normalize(chosen["sql"])] and normalize(chosen["sql"]) == normalize(episode["prediction"]["sql"]):
                    action = {"tool": "submit", "candidate": chosen["id"], "reason_code": "consistent"}
            elif tool == "repair_hint" and tool in data["allowed_actions"] and not any(truth[tid][q] for q in known):
                trace = episode["prediction"]["trace"]
                k = BUDGET - data["remaining_tool_calls"]
                if k < len(trace) and trace[k]["tool"] == "repair_hint":
                    from sql_repair_agent.sql_utils import extract_sql
                    confirmed = {normalize(c["sql"]) for c in episode["checked_candidates"] if c["passed"]}
                    if any(normalize(q) in confirmed for h in trace[k]["observation"] if (q := extract_sql(h))):
                        action = {"tool": "repair_hint", "reason_code": "need_alternative"}
            if action:
                key = fingerprint(compact([tid, messages, action]))
                replay.setdefault(key, record(episode, "ordinary_" + tool, messages, action))
    if set(by_seed) != observed:
        raise ValueError("Generation log does not cover the sampled trajectories")
    return list(repair.values()), list(chooser.values()), list(replay.values())


def spread(rows, limit):
    """Round robin across tasks, taking at most one row from each before second examples."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"]].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: (-r.get("attempt", 0), fingerprint(compact(r["messages"]))))
    keys = sorted(grouped, key=lambda tid: (not grouped[tid][0].get("all_four_failed", False), fingerprint(tid)))
    ordered = []
    for i in range(max((len(v) for v in grouped.values()), default=0)):
        ordered.extend(grouped[tid][i] for tid in keys if len(grouped[tid]) > i)
    return ordered[:limit]


def summary(rows):
    return {"examples": len(rows), "task_count": len({r["id"] for r in rows}),
            "schema_count": len({r["db_id"] for r in rows}), "states": dict(Counter(r["state"] for r in rows)),
            "label_sources": dict(Counter(r["label_source"] for r in rows if "label_source" in r)),
            "repair_contexts_with_rejection_memory": sum(r.get("rejected_context", False) for r in rows),
            "repair_contexts_from_all_four_failed_tasks": sum(r.get("all_four_failed", False) for r in rows),
            "independent_contrast_pairs": len({r["pair_id"] for r in rows if "pair_id" in r}),
            "contrast_orders": dict(Counter(r["order"] for r in rows if "order" in r))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/prepared_data/v6")
    args = parser.parse_args()
    rows, _, folder = load_bank("train")
    tasks = {r["id"]: r for r in rows}
    split = split_schemas(rows)
    write_json(args.output / "split.json", split)
    gold = load_gold(set(tasks))
    episodes_file = TRACE / "episodes.jsonl"
    previous = json.loads((ROOT / "reports/v5_train_coverage.json").read_text())
    if previous["episodes_sha256"] != digest(episodes_file):
        raise ValueError("Coverage episodes differ from audited collection")
    episodes = read_jsonl(episodes_file)
    repair, chooser, replay = collect(episodes, tasks, gold)
    files = [episodes_file, TRACE / "generations.jsonl", TRACE / "collection_config.json",
             GOLD / "gold_validation.score.json", GOLD / "gold_validation.predictions.json", folder / "train.jsonl"]
    audit = {"official_split": "train", "base_control": "contract_v4, onpolicy_v001/epoch_02",
             "split_sha256": digest(args.output / "split.json"), "source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in files},
             "training_data_contains_official_validation_or_test": False, "episodes": len(episodes),
             "methods": {"generator": "Exact failed train rollout prompts, including recovery memory; distinct SQL target passing previous official three-instance replay.",
                         "chooser": "Verified same-trajectory public-matching hard pairs from real check feedback. Each prompt and its candidate-order swap carry local IDs. Only messages and action enter training.",
                         "replay": "Same successful rollout prompts and verified actions in both arms. Positive check, submit and useful hint requests are sampled across tasks.",
                         "sampling": "Equal targeted example counts across fit arms. Replay length capped at targeted count and shared by both arms."},
             "available": {"generator": summary(repair), "chooser": summary(chooser), "ordinary": summary(replay)},
             "sets": {}}
    for partition, ids in (("train", set(split["fit_ids"])), ("holdout", set(split["holdout_ids"]))):
        gen = [r for r in repair if r["id"] in ids]
        choice = [r for r in chooser if r["id"] in ids]
        ordinary = [r for r in replay if r["id"] in ids]
        # Preserve both orderings in the chooser arm when counts are matched.
        if len(gen) < len(choice):
            raise ValueError("Generator pool too small to match full counterbalanced chooser set")
        gen = spread(gen, len(choice)) if partition == "train" else spread(gen, len(gen))
        shared = []
        for state, fraction in (("ordinary_check", 0.4), ("ordinary_submit", 0.4), ("ordinary_repair_hint", 0.2)):
            shared.extend(spread([r for r in ordinary if r["state"] == state], int(len(choice) * fraction)))
        for arm, treatment in (("generator", gen), ("chooser", choice)):
            records = sorted(treatment + shared, key=lambda r: fingerprint(compact([r["messages"], r["target"]])))
            path = args.output / arm / (partition + ".jsonl")
            write_jsonl(path, records)
            audit["sets"][arm + "/" + partition] = {**summary(records), "sha256": digest(path),
                                                       "shared_replay_sha256": fingerprint(compact(shared))}
    for arm in ("generator", "chooser"):
        write_json(args.output / arm / "provenance.json",
                   {"arm": arm, "split": "train", "validation_or_test_training_material": False,
                    "scope": split["scope"], "sources": audit["source_sha256"], "method": audit["methods"][arm],
                    "fixed_split_sha256": audit["split_sha256"], "fit": audit["sets"][arm + "/train"],
                    "holdout": audit["sets"][arm + "/holdout"]})
    write_json(args.output / "audit.json", audit)
    print(json.dumps(audit["sets"], indent=2))


if __name__ == "__main__":
    main()
