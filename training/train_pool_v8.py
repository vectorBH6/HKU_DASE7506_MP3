"""Fit two fixed linear ranking heads on independently checked official train candidates."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"submission")]
from mp3_sql.common import BUDGET,digest,write_json
from mp3_sql.environment import Episode,Tools
from mp3_sql.score import score_document
from training.v5_utils import TrainReward
from sql_repair_agent import pool_rank_v8 as rank
from sql_repair_agent.pool_edits_v8 import build_pool
from sql_repair_agent.sql_utils import sql_key

SPLIT=ROOT/"artifacts/prepared_data/v6/split.json"
WEIGHTS=ROOT/"submission/configs/pool_v8_weights.json"
WORK=ROOT/"runs/v8_pool_train"


def setup(row,tools):
    ep=Episode(row,tools)
    ep.step({"tool":"schema"})
    ep.step({"tool":"check","sql":row["issue_sql"]})
    ep.step({"tool":"repair_hint"})
    return ep


def prediction(ep,sql):
    return {"id":ep.row["id"],"sql":sql,"trace":[*ep.trace,{"tool":"submit","sql":sql}],
            "tool_calls":BUDGET-ep.remaining}


def collect(row,tools):
    ep=setup(row,tools)
    obs=ep.observation()
    schema=ep.trace[0]["observation"]
    hints=ep.trace[2]["observation"]
    pool=build_pool(obs,schema,hints)
    orig={"sql":row["issue_sql"],"family":"raw","source":"draft"}
    candidates=[orig]
    seen={sql_key(orig["sql"])}
    for c in pool:
        key=sql_key(c["sql"])
        if key not in seen:
            candidates.append(c)
            seen.add(key)
    features=rank.feature_rows(obs,schema,candidates)
    labels=[None]*len(candidates)
    labels[0]=(ep.trace[1]["observation"]["matches_example"],prediction(ep,orig["sql"]))
    costs=[]
    for start in range(1,len(candidates),BUDGET-3):
        if start!=1:
            # This policy always retrieves the same initial inputs, then chooses a
            # predeclared block. No action depends on another trajectory's result.
            ep=setup(row,tools)
        for i,c in enumerate(candidates[start:start+BUDGET-3],start):
            ep.step({"tool":"check","sql":c["sql"]})
            labels[i]=(ep.trace[-1]["observation"]["matches_example"],prediction(ep,c["sql"]))
        ep.step({"tool":"submit","sql":ep.trace[-1]["sql"]})
        costs.append(BUDGET-ep.remaining)
    if not costs:
        ep.step({"tool":"submit","sql":orig["sql"]})
        costs.append(BUDGET-ep.remaining)
    assert max(costs)<=BUDGET and len(labels)==len(candidates)
    public=[dict(sql=c["sql"],source=c["source"],family=c["family"],features=f,public=bool(lab[0]))
            for c,f,lab in zip(candidates,features,labels)]
    return {"id":row["id"],"db_id":row["db_id"],"candidates":public,"trajectory_costs":costs,
            "pool_before_dedup":len(pool),"raw_count":sum(c["family"]=="raw" for c in candidates)},[x[1] for x in labels]


def label_batch(episodes,reward):
    # Exactly the TrainReward scorer/replay contract, batched with one submitted
    # candidate per distinct training question in each score_document invocation.
    for j in range(max(len(labels) for _,labels in episodes)):
        selected=[(result,labels[j]) for result,labels in episodes if j<len(labels)]
        rows=[reward.by_id[result["id"]] for result,pred in selected]
        preds=[pred for result,pred in selected]
        cases={**reward.cases,"ids":[r["id"] for r in rows]}
        scored=score_document({**reward.header,"predictions":preds},rows,reward.manifest,reward.folder,
                              cases,reward.case_root,workers=4,cache=reward.cache)
        for (result,pred),graded in zip(selected,scored["per_question"]):
            assert result["id"]==graded["id"] and pred["tool_calls"]==graded["tool_calls"]<=BUDGET
            candidate=result["candidates"][j]
            candidate["correct"]=bool(graded["passed"])
            assert not candidate["correct"] or candidate["public"]


def masked(row):
    out = {k:v for k,v in row.items() if not any(k.startswith(p) for p in
           ("src_", "edit_", "source_edit_", "draft_public_source_", "draft_public_edit_", "draft_error_edit_"))}
    out.update({"src_other":1., "edit_other":1., "source_edit_other_other":1.,
                "draft_public_source_other":out["draft_public"], "draft_public_edit_other":out["draft_public"],
                "draft_error_edit_other":out["draft_error"]})
    return out


def fit(tasks,stage):
    import numpy as np
    rows,ys,wp,pairs,wr=[],[],[],[],[]
    for t in tasks:
        group = [c for c in t["candidates"] if stage=="check" or c["public"]]
        if not group:
            continue
        for weight,mask in ((.8,False),(.2,True)):
            start=len(rows)
            xs=[masked(c["features"]) if mask else c["features"] for c in group]
            rows.extend(xs)
            ys.extend(float(c["correct"]) for c in group)
            wp.extend([weight/len(group)]*len(group))
            pos=[i+start for i,c in enumerate(group) if c["correct"]]
            neg=[i+start for i,c in enumerate(group) if not c["correct"]]
            for i in pos:
                for j in neg:
                    pairs.append((i,j))
                    wr.append(weight/(len(pos)*len(neg)))
    keys = sorted(set().union(*(r.keys() for r in rows)))
    X=np.array([[r.get(k,0.) for k in keys] for r in rows],dtype=float)
    # The final column is the unpenalized intercept; pairwise differences cancel it.
    X=np.column_stack((X,np.ones(len(X))))
    y=np.asarray(ys)
    wp=np.asarray(wp)
    wp/=wp.sum()
    if pairs:
        a,b=np.asarray(pairs).T
        D=X[a]-X[b]
        wr=np.asarray(wr)
        wr/=wr.sum()
    theta=np.zeros(X.shape[1])
    m,v=theta.copy(),theta.copy()
    for step in range(1,181):
        p=1/(1+np.exp(-np.clip(X@theta,-40,40)))
        gradient=(wp*(p-y))@X
        if pairs:
            p=1/(1+np.exp(np.clip(D@theta,-40,40)))
            gradient=.5*(gradient-(wr*p)@D)
        gradient[:-1]+=.02*theta[:-1]
        m=.9*m+.1*gradient
        v=.999*v+.001*gradient**2
        theta-=.035*(m/(1-.9**step))/(np.sqrt(v/(1-.999**step))+1e-8)
    return {"bias":float(theta[-1]),"coefficients":dict(zip(keys,theta[:-1].tolist()))}


def evaluate(tasks,weights):
    def first_correct(candidates):
        return int(any(c["correct"] for c in candidates))
    counts=Counter()
    by_k={str(k):Counter() for k in range(1,6)}
    for task in tasks:
        candidates=task["candidates"]
        orig=candidates[0]
        assert orig["source"]=="draft" and orig["family"]=="raw"
        static=candidates[1:]
        ranked=sorted(static,key=lambda c:rank.value(c["features"],weights["check"]),reverse=True)
        counts["tasks"]+=1
        counts["candidate_rows"]+=len(candidates)
        counts["candidate_correct"]+=sum(c["correct"] for c in candidates)
        counts["candidate_public"]+=sum(c["public"] for c in candidates)
        counts["full_pool_oracle"]+=first_correct(candidates)
        counts["draft_only"]+=int(orig["correct"])
        counts["raw_oracle"]+=first_correct([c for c in candidates if c["family"]=="raw"])
        counts["non_raw_only_solves"]+=int(first_correct(candidates) and not first_correct([c for c in candidates if c["family"]=="raw"]))
        for k in range(1,6):
            for name,order in (("static",static),("ranked",ranked),("raw_then_ranked",
                    [c for c in static if c["family"]=="raw"]+[c for c in ranked if c["family"]!="raw"])):
                checked=[orig]+order[:k]
                by_k[str(k)][name+"_oracle"]+=first_correct(checked)
                passed=[c for c in checked if c["public"]]
                if passed:
                    chosen=max(passed,key=lambda c:rank.value(c["features"],weights["choose"]))
                    by_k[str(k)][name+"_first_match_correct"]+=int(passed[0]["correct"])
                    by_k[str(k)][name+"_best_score_correct"]+=int(chosen["correct"])
    return {"counts":dict(counts),"additional_checks_after_schema_draft_hint":{k:dict(v) for k,v in by_k.items()}}


def audit(tasks):
    costs=[n for t in tasks for n in t["trajectory_costs"]]
    return {"schema_count":len({t["db_id"] for t in tasks}),"task_count":len(tasks),"independent_trajectories":len(costs),
            "total_charges_over_offline_trajectories":sum(costs),"max_charges_per_trajectory":max(costs),
            "min_unique_candidates":min(len(t["candidates"]) for t in tasks),
            "max_unique_candidates":max(len(t["candidates"]) for t in tasks),
            "unique_candidates":sum(len(t["candidates"]) for t in tasks),
            "pool_size_histogram":dict(sorted(Counter(len(t["candidates"]) for t in tasks).items())),
            "raw_candidates":sum(t["raw_count"] for t in tasks),
            "candidate_families":dict(Counter(c["family"] for t in tasks for c in t["candidates"])),
            "public_matches":sum(c["public"] for t in tasks for c in t["candidates"]),
            "correct_candidates":sum(c["correct"] for t in tasks for c in t["candidates"]),
            "mixed_public_groups":sum(len({c["correct"] for c in t["candidates"] if c["public"]})==2 for t in tasks),
            "mixed_all_groups":sum(len({c["correct"] for c in t["candidates"]})==2 for t in tasks)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size",type=int,default=32)
    args=parser.parse_args()
    if not 1<=args.batch_size<=480:
        raise ValueError("batch-size must lie in [1,480]")
    for var in ("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","OMP_NUM_THREADS"):
        os.environ[var]="1"
    started=time.perf_counter()
    reward=TrainReward()
    split=json.loads(SPLIT.read_text())
    fit_ids=set(split["fit_ids"])
    holdout_ids=set(split["holdout_ids"])
    assert set(reward.by_id)==fit_ids|holdout_ids and not fit_ids&holdout_ids
    assert not set(split["fit_db_ids"])&set(split["holdout_db_ids"])
    tools=Tools(reward.folder,reward.manifest,cache={})
    records=[]
    WORK.mkdir(parents=True,exist_ok=True)
    for offset in range(0,len(reward.rows),args.batch_size):
        batch=[collect(row,tools) for row in reward.rows[offset:offset+args.batch_size]]
        label_batch(batch,reward)
        records.extend(row for row,_ in batch)
        print(f"Labelled {len(records)}/{len(reward.rows)} training questions in {time.perf_counter()-started:.1f}s",flush=True)
    fit_tasks=[t for t in records if t["id"] in fit_ids]
    holdout_tasks=[t for t in records if t["id"] in holdout_ids]
    assert {t["db_id"] for t in fit_tasks}==set(split["fit_db_ids"])
    assert {t["db_id"] for t in holdout_tasks}==set(split["holdout_db_ids"])
    weights={s:fit(fit_tasks,s) for s in ("check","choose")}
    write_json(WEIGHTS,weights)
    # Input labels and per-question SQL stay outside the exported submission.
    with (WORK/"train_pool.jsonl").open("w") as f:
        for t in records:
            f.write(json.dumps(t,separators=(",",":"),allow_nan=False)+"\n")
    report={"source":"official train only, eligible candidates generated using schema, draft check, repair_hint",
            "bank_sha256":reward.header["bank_sha256"],"split_sha256":digest(SPLIT),
            "weights_sha256":digest(WEIGHTS),"labels_sha256":digest(WORK/"train_pool.jsonl"),
            "generator_sha256":digest(ROOT/"submission/sql_repair_agent/pool_edits_v8.py"),
            "features_sha256":digest(ROOT/"submission/sql_repair_agent/pool_rank_v8.py"),
            "trainer_sha256":digest(Path(__file__)), "official_scorer_sha256":digest(ROOT/"mp3_sql/score.py"),
            "training_references_sha256":digest(reward.folder/"train_references.json"),
            "training_labels":"Every unique candidate independently submitted on a valid trajectory prefix, officially replayed and scored on all three instances",
            "observations_used_for_features":"task question, initial draft, initial draft public check, schema, expected public column names, ordering flag; cleared candidate feedback, IDs never indexed",
            "oracle_scope":"Entire generated bounded pool after normalized deduplication. Single-trajectory @K starts with actual draft check, schema, and hint taking 3 charges, leaving at most 5 checks.",
            "no_gold_injection":True,"validation_test_data_access":False,"fit_only":True,
            "hyperparameters":{"task_weighting":"equal pointwise / equal eligible paired task; native 0.8, source-masked 0.2",
                "pairwise_loss_weight":0.5,"l2":0.02,"adam_lr":0.035,"full_batch_steps":180,"initialization":"zeros"},
            "fit":dict(audit=audit(fit_tasks),independent_queue_simulation=evaluate(fit_tasks,weights)),
            "holdout":dict(audit=audit(holdout_tasks),independent_queue_simulation=evaluate(holdout_tasks,weights)),
            "elapsed_seconds":round(time.perf_counter()-started,2)}
    write_json(WORK/"audit.json",report)
    print(json.dumps({"report":str(WORK/"audit.json"),"holdout":report["holdout"]},indent=2))


if __name__=="__main__":
    main()
