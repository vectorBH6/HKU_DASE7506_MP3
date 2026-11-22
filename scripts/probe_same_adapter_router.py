"""GPU development screen: V11, V12, and first-check routing, one fixed adapter per process."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"submission")]
from mp3_sql.common import BUDGET, PROTOCOL, digest, load_bank, read_jsonl, write_json
from mp3_sql.environment import Tools, rollout
from mp3_sql.predict import load_factory
from mp3_sql.score import load_cases, score_document
from sql_repair_agent.model import get_generator as real_generator, REPOSITORY, REVISION

OUT=ROOT/"runs/same_adapter_router"
BASE=ROOT/"artifacts/models/Qwen3-0.6B"
ADAPTERS={"v7":"artifacts/adapters/v7_paired/epoch_01","v4":"artifacts/adapters/feedback_v4/epoch_01"}
ARMS={"v11":"v11_text_fallback","v12":"v12_v9_fallback","router":"first_check_router"}
OLD_V4=("dev_v4_e1_semantic","dev_v4_e1_contract")


def read(p):
    return json.loads(p.read_text())


def genkey(messages,task,attempt,config):
    assert not config.get("do_sample",False) and config["max_new_tokens"]==384
    return task,attempt,json.dumps(messages,sort_keys=True,ensure_ascii=False)


def worker(adapter):
    from transformers import AutoTokenizer
    out=OUT/adapter
    out.mkdir(parents=True,exist_ok=True)
    weights=ROOT/ADAPTERS[adapter]
    expected=digest(weights/"adapter_model.safetensors")
    os.environ.update(MP3_BASE_MODEL=str(BASE),MP3_ADAPTER_DIR=str(weights),MP3_DEVICE="cuda",
                      MP3_GENERATION_LOG=str(out/"fresh_generations.jsonl"))
    tokenizer=AutoTokenizer.from_pretrained(str(BASE),trust_remote_code=False,local_files_only=True)
    cfg=read(ROOT/"submission/configs/v12_v9_fallback.json")
    cfg["adapter_sha256"]=expected
    assert cfg["max_completions"]==4
    ckpt=out/"config.json"
    write_json(ckpt,cfg)
    cache={}
    if adapter=="v4":
        for run in OLD_V4:
            for g in read_jsonl(ROOT/"runs"/run/"validation/r001/generations.jsonl"):
                ident=g["identity"]
                assert (ident["repository"],ident["revision"],ident["device"],ident["dtype"],ident["adapter"]["sha256"])==(
                    REPOSITORY,REVISION,"cuda","torch.bfloat16",expected)
                assert g["generation"]=={"do_sample":False,"enable_thinking":False,"max_new_tokens":384}
                k=genkey(g["messages"],g["task_id"],g["attempt"],cfg)
                val=(g["generated_text"],g["rendered_prompt"],"archived_gpu")
                if k in cache:
                    assert cache[k][:2] == val[:2]
                cache[k]=val
    counts=Counter()
    per_task=Counter()
    class MemoGenerator:
        def generate(self,messages,task,attempt,config):
            assert config["adapter_sha256"]==expected
            k=genkey(messages,task,attempt,config)
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            per_task[task]+=1
            if k in cache:
                answer,saved,origin=cache[k]
                assert saved==prompt
                counts["reuse_"+origin]+=1
                return answer
            gen=real_generator(True)
            assert gen.identity["device"]=="cuda" and gen.identity["adapter"]["sha256"]==expected
            answer=gen.generate(messages,task,attempt,config)
            cache[k]=(answer,prompt,"same_run")
            counts["fresh_gpu"]+=1
            return answer
    rows,manifest,folder=load_bank("validation")
    tools=Tools(folder,manifest,cache={})
    hdr={"protocol":PROTOCOL,"split":"validation","budget":BUDGET,"manifest_sha256":digest(folder/"manifest.json"),
         "bank_sha256":digest(folder/"validation.jsonl")}
    predictions={};stats={}
    with patch("sql_repair_agent.model.get_generator",return_value=MemoGenerator()):
        for label,method in ARMS.items():
            counts.clear();per_task.clear()
            factory=load_factory(ROOT/"submission"/(method+".py"))
            preds=[rollout(r,tools,factory(ckpt)) for r in rows]
            assert max(per_task.values())<=cfg["max_completions"] and max(p["tool_calls"] for p in preds)<=BUDGET
            doc={**hdr,"predictions":preds};predictions[label]=preds
            write_json(out/(label+"_predictions.json"),doc)
            stats[label]={"model_calls":sum(per_task.values()), "max_calls":max(per_task.values()),
                          "max_tools":max(p["tool_calls"] for p in preds), "source_counts":dict(counts),
                          "charged_tools":sum(p["tool_calls"] for p in preds),
                          "predictions_sha256":digest(out/(label+"_predictions.json"))}
    exact=Counter()
    for i,row in enumerate(rows):
        p=predictions["router"][i]; first=p["trace"][0]
        assert first["tool"]=="check" and first["sql"]==row["issue_sql"]
        assert first==predictions["v11"][i]["trace"][0]==predictions["v12"][i]["trace"][0]
        which="v12" if first["observation"]["matches_example"] else "v11"
        assert p==predictions[which][i]
        exact[which]+=1
    write_json(out/"worker.json",{"adapter_sha256":expected,"model":REPOSITORY,"revision":REVISION,"gpu_dtype":"bfloat16",
                                 "branches":stats,"exact_router_trajectories":dict(exact),"configuration_sha256":digest(ckpt)})
    print("WORKER COMPLETE",adapter,stats,dict(exact),flush=True)


def grade():
    # No validation grading references are accessed by either GPU worker.
    assert all((OUT/a/"worker.json").exists() for a in ADAPTERS)
    rows,manifest,folder=load_bank("validation")
    cases,root=load_cases("validation",folder);cache={}
    result={}
    for a in ADAPTERS:
        w=read(OUT/a/"worker.json")
        report={"worker":w};states={};match=[]
        for arm in ARMS:
            path=OUT/a/(arm+"_predictions.json")
            assert w["branches"][arm]["predictions_sha256"]==digest(path)
            doc=read(path)
            score=score_document(doc,rows,manifest,folder,cases,root,workers=1,cache=cache)
            score.update({"split":"validation","benchmark":manifest["benchmark"],"manifest_sha256":doc["manifest_sha256"],
                          "bank_sha256":doc["bank_sha256"],"references_sha256":digest(folder/"validation_references.json"),
                          "scorer_sha256":digest(ROOT/"mp3_sql/score.py"),"executor_sha256":digest(ROOT/"mp3_sql/sqlite_tool.py"),
                          "predictions_sha256":digest(path)})
            write_json(OUT/a/(arm+"_score.json"),score)
            states[arm]=[q["passed"] for q in score["per_question"]]
            match=[p["trace"][0]["observation"]["matches_example"] for p in doc["predictions"]]
        for g,ix in [("all",list(range(len(rows)))),("public_match",[i for i,m in enumerate(match) if m]),
                     ("public_mismatch",[i for i,m in enumerate(match) if not m])]:
            report[g]={"n":len(ix),"correct":{k:sum(s[i] for i in ix) for k,s in states.items()},
                       "router_vs":{k:{"wins":sum(states["router"][i] and not s[i] for i in ix),
                                       "losses":sum(s[i] and not states["router"][i] for i in ix)} for k,s in states.items() if k!="router"}}
        result[a]=report
    write_json(OUT/"results.json",{"scope":"Validation-only GPU development; exactly one fixed adapter per process and per candidate. Greedy decoding, real tools and independent official three-instance grading after all predictions. V4 historical cache restricted to identical GPU identity, task, attempt, messages, rendered prompt and decoding settings; all other requested prompts generated live or reused from same run. First-check routing verified to reproduce every chosen branch's complete trajectory.","arms":result})
    print("RESULT", {a:{k:v for k,v in r.items() if k!="worker"} for a,r in result.items()},flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("stage",choices=["v4","v7","grade"])
    arg=p.parse_args()
    if arg.stage=="grade":
        grade()
    else:
        worker(arg.stage)
