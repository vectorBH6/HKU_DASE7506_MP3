"""Validation-only teacher assignment from the shared, exact initial public draft check."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {"v8":"e82_v8_struct","v9":"e90_v9_combined","v10":"e100_v10_semantic_struct"}


def read(path):
    return json.loads(path.read_text())


def outputs(path):
    return {p["id"]:p for p in read(path)["predictions"]}


def labels(path):
    d=read(path)
    assert d["split"]=="validation" and d["n"]==120
    return {r["id"]:r["passed"] for r in d["per_question"]}


def main():
    runs=ROOT/"runs"
    bank={r["id"]:r for r in (json.loads(s) for s in (ROOT/"mp3_sql/data/validation.jsonl").read_text().splitlines())}
    passed={}
    preds={}
    for arm,name in EXPERIMENTS.items():
        source=runs/name/"validation/r001"
        preds[arm]=outputs(source/"predictions.json")
        passed[arm]=labels(source/"score.json")
    v11path=runs/"v11_prefix_probe/validation/predictions.json"
    v11=read(ROOT/"reports/v11_text_validation_probe.json")
    assert hashlib.sha256(v11path.read_bytes()).hexdigest()==v11["predictions_sha256"]
    preds["v11"]=outputs(v11path)
    changed=[i for i in bank if preds["v10"][i]["sql"]!=preds["v11"][i]["sql"]]
    assert len(changed)==v11["changed_final_sql"]==v11["wins"]==1 and v11["losses"]==0
    assert not passed["v10"][changed[0]] and sum(passed["v10"].values())==v11["v10"]
    passed["v11"]={**passed["v10"],changed[0]:True}
    assert sum(passed["v11"].values())==v11["v11"]
    passed["draft"]=labels(runs/"e00_draft/validation/r001/score.json")
    match={}
    for i,r in bank.items():
        initial=[preds[arm][i]["trace"][0] for arm in preds]
        assert all(ev==initial[0] and ev["tool"]=="check" and ev["sql"]==r["issue_sql"] for ev in initial)
        match[i]=initial[0]["observation"]["matches_example"]
    for old in ("v8","v9"):
        passed[old+"_pass_v11_fail"]={i:passed[old if match[i] else "v11"][i] for i in bank}
    subsets={"all":list(bank),"draft_public_match":[i for i in bank if match[i]],
             "draft_public_mismatch":[i for i in bank if not match[i]]}
    tables={}
    for g,ids in subsets.items():
        tables[g]={}
        for arm,results in passed.items():
            delta={}
            for base in ("draft","v8","v9","v10","v11"):
                b=passed[base]
                delta[base]={"wins":sum(results[i] and not b[i] for i in ids),
                             "losses":sum(b[i] and not results[i] for i in ids)}
            tables[g][arm]={"n":len(ids),"correct":sum(results[i] for i in ids),"vs":delta}
    report={"scope":"Validation-only simulated assignment of previously evaluated teachers, using only their identical first tool observation. V11 per-question outcomes uniquely determined by its archived independently scored replay: exactly one changed final answer and exactly one win, zero losses vs V10. No new evaluation run, SQL generation, or test reads.",
            "groups":tables}
    p=ROOT/"reports/v11_first_check_teachers.json"
    p.write_text(json.dumps(report,indent=2)+"\n")
    for g,t in tables.items():
        print(g, {x:(r["correct"],r["n"]) for x,r in t.items()})
        for key in ("v8_pass_v11_fail","v9_pass_v11_fail"):
            print(key,t[key]["vs"])


if __name__=="__main__":
    main()
