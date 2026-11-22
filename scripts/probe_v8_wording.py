"""Blind two-SQL wording probe on the frozen V7 synthetic, train-derived pairs."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
DATA = ROOT / "artifacts/prepared_data/v7"
PUBLIC = DATA / "holdout/public.jsonl"
REPHRASED = ROOT / "runs/v7_rephrases/public.jsonl"
POLICIES = ("keep", "margin", "higher_score")
SEED = 750608


def actor():
    # This separate process reads only unlabeled examples from stdin and frozen coefficients.
    from sql_repair_agent.pool_rank_v8 import rank_scores

    request = json.load(sys.stdin)
    decisions = []
    for item in request["items"]:
        obs, schema, sqls = item["obs"], item["schema"], item["candidates"]
        assert set(item) == {"obs", "schema", "candidates"} and set(obs["task"]) == {"question"}
        candidates = [{"sql": sql, "source": "other", "family": "other"} for sql in sqls]
        forward = rank_scores(obs, schema, candidates, "choose")
        reverse = rank_scores(obs, schema, candidates[::-1], "choose")[::-1]
        assert forward == reverse
        by_sql = dict(zip(sqls, forward, strict=True))
        draft = obs["issue_sql"]
        alternate, = set(sqls) - {draft}
        decisions.append({
            "keep": draft,
            "margin": alternate if by_sql[alternate] > by_sql[draft] + request["margin"] else draft,
            "higher_score": alternate if by_sql[alternate] > by_sql[draft] else draft,
            "exact_score_tie": by_sql[alternate] == by_sql[draft],
            "signed_score_gap": by_sql[alternate] - by_sql[draft],
        })
    json.dump(decisions, sys.stdout, allow_nan=False)


def blind_inputs(primary_rows, rewritten_rows):
    from diagnose_pairs_v7 import PublicTools
    from mp3_sql.common import read_jsonl

    # Keep the two alternatives as an unordered set; never join a row to its target.
    pairs = {r["pair_id"]: sorted(set(r["gold_alternatives"]))
             for r in read_jsonl(DATA / "motifs.jsonl") if r["partition"] == "holdout"}
    assert len(pairs) == 24 and all(len(p) == 2 for p in pairs.values())
    rng = random.Random(SEED)
    for pair in pairs.values():
        rng.shuffle(pair)
    primary = {r["id"]: r for r in primary_rows}
    rewritten = {r["id"]: r for r in rewritten_rows}
    assert set(primary) == set(rewritten) and len(primary) == 48
    assert all({k:v for k,v in a.items() if k != "question"} ==
               {k:v for k,v in rewritten[key].items() if k != "question"}
               and a["question"] != rewritten[key]["question"] for key,a in primary.items())
    tools = PublicTools(primary_rows)
    items = []
    for row in primary_rows:
        choices = pairs[row["pair_id"]]
        results = {sql:tools.query(row, sql, True) for sql in choices}
        assert all(feedback["matches_example"] for feedback in results.values())
        # Only the public draft-check observation is used by the scoring model.
        draft = row["issue_sql"]
        event = {"tool":"check", "sql":draft, "observation":results[draft]}
        for wording, public in (("original", row), ("rephrased", rewritten[row["id"]])):
            obs = {"issue_sql":draft, "ordered":row["ordered"], "example":row["example"],
                   "task":{"question":public["question"]}, "history":[event]}
            items.append((wording,row["id"],{"obs":obs,"schema":row["schema"],"candidates":choices}))
    rng.shuffle(items)
    return pairs,items


def assess(graded):
    by_pair = defaultdict(dict)
    for member in graded:
        sides = by_pair[member["pair_id"]]
        assert member["side"] not in sides
        sides[member["side"]] = member
    assert all(set(sides) == {"keep","repair"} for sides in by_pair.values())
    stats = {
        "pairs":len(by_pair), "members":len(graded),
        "exact_score_tie_members":sum(m["exact_score_tie"] for m in graded),
        "both_members_have_tied_scores":sum(all(m["exact_score_tie"] for m in sides.values())
                                           for sides in by_pair.values()),
    }
    for policy in POLICIES:
        stats[policy] = {
            "correct_keep":sum(s["keep"][policy] for s in by_pair.values()),
            "correct_repair":sum(s["repair"][policy] for s in by_pair.values()),
            "both_correct":sum(s["keep"][policy] and s["repair"][policy] for s in by_pair.values()),
            "strict_both_correct":sum(all(m[policy] and not m["exact_score_tie"] for m in sides.values())
                                      for sides in by_pair.values()),
        }
    return stats


def grade(primary_rows, pairs, refs, decisions):
    from mp3_sql.common import digest, read_jsonl, result_fingerprint
    from mp3_sql.sqlite_tool import execute_sql

    private = {r["id"]:r for r in read_jsonl(DATA / "holdout/private.jsonl")}
    assert set(private) == {r["id"] for r in primary_rows}
    known_hashes = json.loads((DATA/"audit.json").read_text())["fixture_sha256"]
    expected = {}
    for row in primary_rows:
        lab = private[row["id"]]
        p = row["pair_id"]
        assert lab["pair_id"] == p and lab["gold_sql"] in pairs[p]
        assert digest(DATA/row["public_path"]) == known_hashes[row["public_path"]]
        tables = [t["name"] for t in row["schema"]]
        for relative in lab["witness_paths"]:
            assert digest(DATA/relative) == known_hashes[relative]
            answers = {}
            for sql in pairs[p]:
                response = execute_sql(DATA/relative,sql,tables)
                assert response["ok"]
                answers[sql] = result_fingerprint(response,row["ordered"])
            assert len(set(answers.values())) == 2
        assert len(lab["witness_paths"]) == 2
        assert (row["issue_sql"] == lab["gold_sql"]) == (lab["side"] == "keep")
        expected[row["id"]] = lab
    groups = defaultdict(list)
    signature = {}
    for (wording,task_id), outcome in zip(refs,decisions,strict=True):
        lab = expected[task_id]
        graded = {policy:outcome[policy] == lab["gold_sql"] for policy in POLICIES}
        groups[wording].append(dict(graded,category=lab["category"],pair_id=lab["pair_id"],
                                     side=lab["side"],exact_score_tie=outcome["exact_score_tie"]))
        signature[wording,task_id] = outcome
    results = {}
    for wording,rows in groups.items():
        results[wording] = {"overall":assess(rows),
                            "by_category":{c:assess([r for r in rows if r["category"]==c])
                                           for c in sorted({r["category"] for r in rows})}}
    results["changed_choice_on_rephrasing"] = {
        policy:sum(signature["original",i][policy] != signature["rephrased",i][policy] for i in expected)
        for policy in POLICIES}
    return results


def main():
    from diagnose_pairs_v7 import population
    from mp3_sql.common import digest, read_jsonl, write_json

    lock = json.loads((ROOT/"reports/v8_official_lock.json").read_text())["files"]
    for name in ("submission/configs/pool_v8_weights.json","submission/sql_repair_agent/pool_rank_v8.py",
                 "submission/configs/v8_choose.json"):
        assert digest(ROOT/name) == lock[name]
    freeze = json.loads((ROOT/"runs/v7_rephrases/frozen.json").read_text())
    assert digest(PUBLIC) == freeze["original_public_sha256"]
    assert digest(REPHRASED) == freeze["rephrased_public_sha256"]
    margin = json.loads((ROOT/"submission/configs/v8_choose.json").read_text())["replacement_margin"]
    assert margin == 0.12
    rows = population()
    pairs, items = blind_inputs(rows,read_jsonl(REPHRASED))
    request = json.dumps({"items":[i for _,_,i in items],"margin":margin},ensure_ascii=False,sort_keys=True)
    completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--actor"], input=request,
                               text=True,capture_output=True,check=True)
    decisions = json.loads(completed.stdout)
    assert len(decisions) == len(items) == 96
    assert digest(DATA/"holdout/private.jsonl") == freeze["private_sha256"]
    result = grade(rows, pairs, [(w,i) for w,i,_ in items], decisions)
    metadata = {"train_only":True,"pairs":len(pairs),"schemas":len({r["source_db_id"] for r in rows}),
                "candidate_source_and_family":"other","both_orders_verified_members":len(items),
                "random_seed":SEED,"margin":margin,"exact_tie_rule":"retain the fixed draft",
                "source":"Fixed two alternatives from each original train-derived motif, shared across its two intents. One discriminating answer is supplied to the actor among the two candidates. Each SQL was checked on its shared support and the two distinct witness outcomes were verified during grading.",
                "scope":"Conditional two-candidate task. No generation, pool coverage, or online policy measured.",
                "actor_input_sha256":hashlib.sha256(request.encode()).hexdigest(),
                "actor_output_sha256":hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "weights_sha256":lock["submission/configs/pool_v8_weights.json"],
                "feature_code_sha256":lock["submission/sql_repair_agent/pool_rank_v8.py"],
                "motifs_sha256":digest(DATA/"motifs.jsonl"),"public_sha256":digest(PUBLIC),
                "rephrased_sha256":digest(REPHRASED),"private_sha256":freeze["private_sha256"],
                "fixture_manifest_sha256":digest(DATA/"audit.json"),"script_sha256":digest(Path(__file__))}
    write_json(ROOT/"reports/v8_wording_probe.json",{"method":metadata,"results":result})
    for wording in ("original","rephrased"):
        print(wording,result[wording]["overall"])


if __name__=="__main__":
    if sys.argv[1:]==["--actor"]:
        actor()
    elif len(sys.argv)==1:
        main()
    else:
        raise SystemExit("Usage: python scripts/probe_v8_wording.py")
