"""Fit question-to-SQL operation cues on train, then replay frozen V8 validation decisions."""
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission"), str(ROOT / "scripts")]
from mp3_sql.common import digest, load_bank, read_jsonl, write_json
from mp3_sql.environment import Tools
from mp3_sql.score import load_cases, score_document
from probe_v9_wrapper import extend
from sql_repair_agent.pool_rank_v8 import value
from sql_repair_agent.sql_utils import sql_key


V8 = ROOT / "runs/e82_v8_struct/validation/r001"
SPLIT = ROOT / "artifacts/prepared_data/v6/split.json"
TRAIN_POOL = ROOT / "runs/v8_pool_train/train_pool.jsonl"
TRAIN_GOLD = ROOT / "mp3_sql/data/train_references.json"
WEIGHTS = ROOT / "submission/configs/pool_v8_weights.json"
OPERATORS = (exp.Count, exp.Avg, exp.Sum, exp.Max, exp.Min, exp.Group, exp.Having,
             exp.Order, exp.Limit, exp.Distinct, exp.Union, exp.Intersect, exp.Except,
             exp.In, exp.Exists, exp.Not, exp.Like, exp.Between, exp.Or, exp.And,
             exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ, exp.Join, exp.Is, exp.Subquery)


def read(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=5000)
def operations(sql):
    try:
        tree = parse_one(sql, read="sqlite")
    except (SqlglotError, RecursionError, ValueError):
        return set()
    nodes = list(tree.walk())
    out = {kind.__name__ for kind in OPERATORS if any(isinstance(n, kind) for n in nodes)}
    if any(order.args.get("desc") for order in tree.find_all(exp.Ordered)):
        out.add("OrderDesc")
    limit = tree.args.get("limit")
    if limit and str(limit.expression) == "1":
        out.add("LimitOne")
    if sum(isinstance(n, exp.Select) for n in nodes) > 1:
        out.add("MultiSelect")
    return out


def phrases(question):
    words = re.findall(r"[a-z0-9]+", question.lower())
    return {tuple(words[i:i + n]) for n in (1, 2, 3) for i in range(len(words) - n + 1)}


def fit_rules(rows, gold_sql, fit_ids):
    """One document per training question; database frequency guards against domain names."""
    support = Counter()
    schemas = defaultdict(set)
    positives = defaultdict(Counter)
    overall = Counter()
    for row in rows:
        if row["id"] not in fit_ids:
            continue
        flags = operations(gold_sql[row["id"]])
        overall.update(flags)
        for phrase in phrases(row["question"]):
            support[phrase] += 1
            schemas[phrase].add(row["db_id"])
            positives[phrase].update(flags)
    rules = {}
    for phrase, n in support.items():
        if n < 8 or len(schemas[phrase]) < 5:
            continue
        required = {k for k, count in positives[phrase].items()
                    if count / n >= .95 and count / n - overall[k] / len(fit_ids) >= .30}
        if required:
            rules[phrase] = required
    return rules


def choose_sql(question, public_matching_sqls, incumbent_sql, rules):
    options = {sql_key(s): s for s in public_matching_sqls}
    if sql_key(incumbent_sql) not in options or len(options) < 2:
        return incumbent_sql
    required = set().union(*(rules.get(p, set()) for p in phrases(question)))
    if required <= operations(incumbent_sql):
        return incumbent_sql
    satisfy = [s for s in options.values() if required <= operations(s)]
    return satisfy[0] if len(satisfy) == 1 else incumbent_sql


def select(row, pred, rules):
    matching = [e["sql"] for e in pred["trace"] if e["tool"] == "check" and
                e["observation"].get("matches_example")]
    sql = choose_sql(row["question"], matching, pred["sql"], rules)
    return {**pred, "sql": sql, "trace": [*pred["trace"][:-1], {"tool": "submit", "sql": sql}]}


def pool_diagnostic(rows, partitions, rules):
    weights = read(WEIGHTS)["choose"]
    candidates = read_jsonl(TRAIN_POOL)
    by_id = {row["id"]: row for row in rows}
    report = {}
    for key in ("fit_ids", "holdout_ids"):
        ids = set(partitions[key])
        counts = Counter()
        for item in candidates:
            if item["id"] not in ids:
                continue
            good = [c for c in item["candidates"] if c["public"]]
            if not good:
                continue
            old = max(good, key=lambda c: value(c["features"], weights))
            chosen = choose_sql(by_id[item["id"]]["question"], [c["sql"] for c in good], old["sql"], rules)
            new = next(c for c in good if chosen == c["sql"])
            counts.update({"correct_before": old["correct"], "correct_after": new["correct"],
                           "changes": chosen != old["sql"],
                           "wins": new["correct"] and not old["correct"],
                           "losses": old["correct"] and not new["correct"]})
        report[key] = {"tasks": len(ids), **dict(counts)}
    return report


def main():
    train, _, _ = load_bank("train")
    partitions = read(SPLIT)
    fit_ids = set(partitions["fit_ids"])
    assert len(fit_ids) == 382 and len(partitions["holdout_ids"]) == 98
    assert not set(partitions["fit_db_ids"]) & set(partitions["holdout_db_ids"])
    rules = fit_rules(train, read(TRAIN_GOLD)["gold_sql"], fit_ids)
    train_diagnostic = pool_diagnostic(train, partitions, rules)

    rows, manifest, folder = load_bank("validation")
    doc = read(V8 / "predictions.json")
    tools = Tools(folder, manifest, cache={})
    originals = doc["predictions"]
    assert [p["id"] for p in originals] == [r["id"] for r in rows]
    wrappers = [extend(r, p, tools)[0] for r, p in zip(rows, originals, strict=True)]
    intents = [select(r, p, rules) for r, p in zip(rows, originals, strict=True)]
    combined = [select(r, p, rules) for r, p in zip(rows, wrappers, strict=True)]
    arms = {"baseline": originals, "intent": intents, "wrapper": wrappers, "combined": combined}

    # All decisions are final before this validation grader opens the extra instances.
    cases, case_root = load_cases("validation", folder)
    scores = {}
    cache = {}
    for key, preds in arms.items():
        scored = score_document({**doc, "predictions": preds}, rows, manifest, folder, cases,
                                case_root, workers=4, cache=cache)
        scores[key] = [x["passed"] for x in scored["per_question"]]
    assert scores["baseline"] == [v["passed"] for v in read(V8 / "score.json")["per_question"]]
    draft = read(ROOT / "runs/e00_draft/validation/r001/score.json")["per_question"]
    draft_right = [i for i, v in enumerate(draft) if v["passed"]]
    hidden = [i for i, p in enumerate(originals) if not draft[i]["passed"] and
              p["trace"][0]["observation"]["matches_example"]]
    stats = {}
    for k, preds in arms.items():
        labels = scores[k]
        stats[k] = {"n": len(rows), "correct": sum(labels),
                    "wins_vs_v8": sum(n and not b for n, b in zip(labels, scores["baseline"])),
                    "losses_vs_v8": sum(b and not n for n, b in zip(labels, scores["baseline"])),
                    "drafts_preserved": sum(labels[i] for i in draft_right), "right_drafts": len(draft_right),
                    "hidden_repaired": sum(labels[i] for i in hidden), "hidden_total": len(hidden),
                    "submissions_changed_vs_v8": sum(p["sql"] != old["sql"] for p, old in zip(preds, originals)),
                    "charged_tools": sum(p["tool_calls"] for p in preds),
                    "max_tools_per_task": max(p["tool_calls"] for p in preds), "extra_lm_generations": 0}
    joint_triggers = sum(p["sql"] != old["sql"] and q["tool_calls"] > old["tool_calls"]
                         for old, p, q in zip(originals, intents, wrappers))
    report = {"method": "Conservative terminal selection among already public-matching SQLs, with optional budgeted projection continuation. All 120 endpoint trajectories replayed and graded on all official instances. No fresh LM inference.",
              "cue_rule": "1-3 contiguous question tokens, support >=8 training tasks spanning >=5 train schemas, operation prevalence >=0.95 and >=0.30 above training marginal. When incumbent violates some requirements, switch only if precisely one checked public match satisfies them all.",
              "scope": "Phrase rule derived only from train; thresholds fixed before the validation probe. Direction motivated by prior aggregate validation results. Wrapper was discovered using validation; the combined score is exploratory. No test tasks used.",
              "cue_count": len(rules), "operators_in_learned_cues": len(set.union(*rules.values())),
              "rules": {" ".join(p): sorted(v) for p, v in sorted(rules.items())},
              "train_pool": train_diagnostic, "validation": stats,
              "intent_changes_overlapping_wrapper_checks": joint_triggers,
              "inputs_sha256": {str(p.relative_to(ROOT)): digest(p) for p in (
                  Path(__file__), ROOT / "scripts/probe_v9_wrapper.py", TRAIN_GOLD, TRAIN_POOL, SPLIT, WEIGHTS,
                  V8 / "predictions.json", V8 / "score.json", folder / "validation_references.json",
                  ROOT / "mp3_sql/score.py")}}
    write_json(ROOT / "reports/v9_intent_probe.json", report)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
