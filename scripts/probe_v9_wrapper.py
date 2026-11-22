"""Validation-discovered outer projection diagnostic on frozen, budget-valid trajectories."""
from pathlib import Path
import json
import sys

import sqlglot
from sqlglot import exp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mp3_sql.common import BUDGET, MAX_SQL_BYTES, PROTOCOL, digest, load_bank, read_jsonl, write_json
from mp3_sql.environment import Episode, Tools
from mp3_sql.score import load_cases, score_document


def wrapped_sql(obs, event):
    result = event["observation"]
    expected = obs["example"]
    columns = result.get("columns") or []
    if not result.get("ok") or result["row_count"] != expected["row_count"]:
        return None
    if not 0 < len(expected["columns"]) < len(columns):
        return None
    names = [name.casefold() for name in columns]
    requested = [name.casefold() for name in expected["columns"]]
    if not all(names.count(name) == 1 for name in requested):
        return None
    sql = event["sql"].strip().removesuffix(";").rstrip()
    if obs["ordered"]:
        # An outer SELECT has no implicit ordering guarantee. Retain only SQL
        # with an explicit root LIMIT 1, so order is immaterial on all instances.
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except (ValueError, RecursionError, sqlglot.errors.SqlglotError):
            return None
        limit = tree.args.get("limit")
        if not isinstance(limit, exp.Limit) or not isinstance(limit.expression, exp.Literal):
            return None
        if limit.expression.is_string or limit.expression.this != "1":
            return None

    # Output names come from the actual executed SQL, not example cell values.
    wanted = [columns[names.index(name)] for name in requested]
    projection = ", ".join('v9_result."' + name.replace('"', '""') + '"' for name in wanted)
    query = "SELECT " + projection + " FROM (\n" + sql + "\n) AS v9_result"
    return query if len(query.encode()) <= MAX_SQL_BYTES else None


def extend(row, old, tools):
    ep = Episode(row, tools)
    for event in old["trace"][:-1]:
        ep.step({k: v for k, v in event.items() if k != "observation"})
        if ep.trace[-1] != event:
            raise ValueError("Prefix observation mismatch")
    assert old["trace"][-1] == {"tool": "submit", "sql": old["sql"]}
    checks = [e for e in ep.trace if e["tool"] == "check"]
    seen = {e["sql"] for e in checks}
    can_try = not any(e["observation"]["matches_example"] for e in checks)
    chosen = old["sql"]
    tries = 0
    obs = ep.observation()
    spare = ep.remaining
    if can_try:
        for event in reversed(checks):
            if not ep.remaining:
                break
            candidate = wrapped_sql(obs, event)
            if candidate is None or candidate in seen:
                continue
            seen.add(candidate)
            ep.step({"tool": "check", "sql": candidate})
            tries += 1
            if ep.trace[-1]["observation"]["matches_example"]:
                chosen = candidate
                break
    ep.step({"tool":"submit","sql":chosen})
    return ({"id":row["id"],"sql":chosen,"trace":ep.trace,"tool_calls":BUDGET-ep.remaining},
            {"no_public_match":can_try, "spare":spare, "tries":tries, "changed":chosen!=old["sql"],
             "empty":not bool(row["example"]["row_count"])})


def run(split, old_predictions, old_scores):
    bank, manifest, folder = load_bank(split)
    if split == "validation":
        priors = json.loads(old_predictions.read_text())["predictions"]
    else:
        priors = read_jsonl(old_predictions)
    by_id = {r["id"]:r for r in bank}
    rows = [by_id[p["id"]] for p in priors]
    tools = Tools(folder, manifest, cache={})
    new, stats = zip(*(extend(r,p,tools) for r,p in zip(rows,priors,strict=True)),strict=True)
    # Only the grader opens labels, after every continuation has been committed.
    cases, root = load_cases(split,folder)
    cases = {**cases, "ids":[r["id"] for r in rows]}
    header = {"protocol":PROTOCOL,"budget":BUDGET,"bank_sha256":digest(folder/f"{split}.jsonl"),
              "manifest_sha256":digest(folder/"manifest.json"),"split":split}
    cache = {}
    def grade(predictions):
        score=score_document({**header,"predictions":list(predictions)},rows,manifest,folder,cases,root,workers=4,cache=cache)
        return [e["passed"] for e in score["per_question"]]
    old = grade(priors)
    labels = grade(new)
    recorded = {a["id"]:a["passed"] for a in json.loads(old_scores.read_text())["per_question"]}
    assert old == [recorded[p["id"]] for p in priors]
    changed=[(s,passed) for s,passed in zip(stats,labels,strict=True) if s["changed"]]
    return {"n":len(rows),"baseline_correct":sum(old),"new_correct":sum(labels),
            "gains":sum(not a and b for a,b in zip(old,labels,strict=True)),
            "losses":sum(a and not b for a,b in zip(old,labels,strict=True)),
            "no_public_match":sum(s["no_public_match"] for s in stats),
            "unmatched_with_remaining_budget":sum(s["no_public_match"] and s["spare"]>0 for s in stats),
            "tasks_with_wrapper_checks":sum(s["tries"]>0 for s in stats),
            "additional_wrapper_checks":sum(s["tries"] for s in stats),
            "new_public_matches_selected":len(changed),
            "selected_wrappers_failing_three_instances":sum(not b for _,b in changed),
            "selected_empty_public":sum(s["empty"] for s,_ in changed),
            "empty_public_match_but_wrong":sum(s["empty"] and not b for s,b in changed),
            "max_tool_calls":max(p["tool_calls"] for p in new),
            "input_predictions_sha256":digest(old_predictions),"input_scores_sha256":digest(old_scores)}


if __name__ == "__main__":
    val=ROOT/"runs/e82_v8_struct/validation/r001"
    train=ROOT/"runs/v6_holdout/baseline"
    report = {
        "method":"Validation-discovered exploratory rule. After a frozen no-public-match prefix, project a strictly narrower list of uniquely named output columns over successful checked queries with matching row counts. Recheck in reverse observation order and adopt the first public match, within the original 8-call budget. Ordered tasks additionally require explicit outermost LIMIT 1. No new model invocations.",
        "validation":run("validation",val/"predictions.json",val/"score.json"),
        "train_holdout_historical_v4b":run("train",train/"predictions.jsonl",train/"score.json"),
        "train_scope":"98 historical V4b trajectories on training schemas held out for V8 comparator fitting. Their prefixes differ from the validation V8 prefixes.",
        "script_sha256":digest(Path(__file__)),"official_scorer_sha256":digest(ROOT/"mp3_sql/score.py")}
    write_json(ROOT/"reports/v9_wrapper_probe.json",report)
    print(json.dumps(report,indent=2))
