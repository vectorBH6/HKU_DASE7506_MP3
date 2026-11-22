"""Freeze V12, replay unchanged archived CPU generations, and independently score the full split."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/"submission"),str(ROOT),str(ROOT/"scripts")]
from mp3_sql.common import BUDGET, PROTOCOL, digest, load_bank, read_jsonl, write_json
from mp3_sql.environment import Tools, rollout
from mp3_sql.predict import load_factory
from run_pool_v8 import read, sha, write_once
from summarize_v8_pool import verify_frozen_assets

LOCK=Path("reports/v12_replay_lock.json")
METHOD=Path("submission/v12_v9_fallback.py")
CONFIG=Path("submission/configs/v12_v9_fallback.json")
ADAPTER=Path("artifacts/adapters/v7_paired/epoch_01")
BASE=Path("artifacts/models/Qwen3-0.6B")
OUT=Path("runs/e120_v12_v9_fallback_replay")


def freeze():
    v9=read(Path("reports/v9_official_lock.json"))
    verify_frozen_assets(v9)
    assert sha(Path("PACKAGE_MANIFEST.json"))==v9["official_manifest_sha256"]
    two=read(Path("reports/v11_text_validation_probe.json"))
    assert sha(Path("submission/v11_text_fallback.py"))==two["entrypoint_sha256"]
    eval99=Path("runs/v10_text_grounding/v9_frozen_fallback_validation.json")
    assert read(eval99)["combined"]==99 and read(eval99)["entrypoint_sha256"]==two["entrypoint_sha256"]
    cfg=read(CONFIG)
    old=read(Path("submission/configs/v9_combined.json"))
    assert all(cfg[k]==v for k,v in old.items() if k!="controller")
    assert cfg["text_family_order"]==read(eval99)["families"]
    assert cfg["adapter_sha256"]==sha(ADAPTER/"adapter_model.safetensors")
    src=[METHOD,CONFIG,Path("submission/v11_text_fallback.py"),Path("submission/v10_semantic.py"),
         Path("submission/configs/pool_v8_weights.json"),*sorted(Path("submission/sql_repair_agent").glob("*.py"))]
    for p in src:
        if str(p) in v9["files"]:
            assert sha(p)==v9["files"][str(p)]
    summaries={s:read(Path(f"reports/v9_official_{s}.json"))["candidate"] for s in ("validation","test")}
    archives={s:{**{name:sha(Path(x["run"])/name) for name in ("predictions.json","score.json","generations.jsonl",
                                                               "run_metadata.json")},"run":x["run"]} for s,x in summaries.items()}
    for s,x in summaries.items():
        assert all(archives[s][k]==v for k,v in x["run_files_sha256"].items())
    dependencies=src+[Path(__file__).resolve().relative_to(ROOT),Path("scripts/run_pool_v8.py"),Path("scripts/summarize_v8_pool.py"),
                      *sorted(ADAPTER.glob("*")),eval99]
    snapshot=OUT/"frozen_source"
    for p in src:
        target=snapshot/p
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
        assert sha(target)==sha(p)
    write_once(LOCK,{"mode":"Archived official V9 CPU generations reused only on exact structured prompt, "
                           "serialized prompt, token count, model identity, decoding parameters and attempt match. "
                           "New V12 trajectories and official three-instance scoring on both full splits.",
                     "selection":"V9 plus the already-fixed exact-public-failure fallback, chosen from validation 99/120. "
                                 "Report either test outcome without policy changes.",
                     "files":{str(p):sha(p) for p in dependencies},"official_manifest_sha256":v9["official_manifest_sha256"],
                     "official_file_count":v9["official_file_count"],"base_files":v9["base_files"],"v9_archives":archives,
                     "config":str(CONFIG),"entrypoint":str(METHOD),"adapter":str(ADAPTER),
                     "source_snapshot":str(snapshot)})
    return read(LOCK)


def run(split,lock):
    from transformers import AutoTokenizer
    previous=Path(lock["v9_archives"][split]["run"])
    archived_meta=read(previous/"run_metadata.json")
    assert archived_meta["status"]=="completed" and archived_meta["device"]=="cpu"
    cfg=read(CONFIG)
    tokenizer=AutoTokenizer.from_pretrained(str(BASE),local_files_only=True,trust_remote_code=False)
    rows,manifest,folder=load_bank(split)
    expected_size=120 if split=="validation" else 240
    assert len(rows)==expected_size
    tools=Tools(folder,manifest,cache={})
    events=read_jsonl(previous/"generations.jsonl")
    indexed={(e["task_id"],e["attempt"]):e for e in events}
    assert len(indexed)==len(events)
    used=set()
    class ExactGenerationReplay:
        def generate(self,messages,task_id,attempt,config):
            key=(task_id,attempt)
            event=indexed[key]
            assert key not in used and messages==event["messages"]
            rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            assert rendered==event["rendered_prompt"]
            assert len(tokenizer(rendered,add_special_tokens=False)["input_ids"])==event["input_tokens"]
            ident=event["identity"]
            assert ident["repository"]=="Qwen/Qwen3-0.6B" and ident["revision"]=="c1899de289a04d12100db370d81485cdf75e47ca"
            assert ident["device"]=="cpu" and ident["adapter"]["sha256"]==config["adapter_sha256"]
            assert event["generation"]=={"do_sample":False,"max_new_tokens":config["max_new_tokens"],"enable_thinking":False}
            assert event["new_tokens"]<=config["max_new_tokens"]
            used.add(key)
            return event["generated_text"]

    factory=load_factory(ROOT/METHOD)
    with patch("sql_repair_agent.model.get_generator",return_value=ExactGenerationReplay()):
        preds=[rollout(row,tools,factory(ROOT/CONFIG)) for row in rows]
    assert len(used)==len(events)
    doc={"protocol":PROTOCOL,"split":split,"budget":BUDGET,"bank_sha256":digest(folder/f"{split}.jsonl"),
         "manifest_sha256":digest(folder/"manifest.json"),"predictions":preds}
    out=OUT/split/"r001"
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/"predictions.json",doc)
    # The scorer is a separate process. No per-question information goes to stdout.
    subprocess.run([sys.executable,"-B","-m","mp3_sql.score","--split",split,"--workers","4",
                    "--predictions",str(out/"predictions.json"),"--output",str(out/"score.json")],
                    check=True,stdout=subprocess.PIPE,env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1"))

    old=read(previous/"predictions.json")["predictions"]
    before=read(previous/"score.json")
    after=read(out/"score.json")
    counts=Counter()
    for p,q,a,b in zip(old,preds,before["per_question"],after["per_question"],strict=True):
        assert p["id"]==q["id"]==a["id"]==b["id"] and len(b["cases"])==3
        assert b["passed"]==all(c["passed"] for c in b["cases"])
        assert q["trace"][:p["tool_calls"]]==p["trace"][:-1] and q["tool_calls"]<=8
        extra=q["trace"][p["tool_calls"]:-1]
        assert all(e["tool"]=="check" for e in extra)
        eligible=any(e["tool"]=="check" and e["sql"]==p["sql"] and
                     e["observation"]["matches_example"] is False for e in p["trace"])
        if not eligible:
            assert p==q
        counts.update(eligible=int(eligible),changed_final_sql=int(p["sql"]!=q["sql"]),extra_checks=len(extra),
                      tasks_with_extra_checks=int(bool(extra)),wins=int(b["passed"] and not a["passed"]),
                      losses=int(a["passed"] and not b["passed"]),base_charged=p["tool_calls"],new_charged=q["tool_calls"])
    assert after["correct"]-before["correct"]==counts["wins"]-counts["losses"]
    if split=="validation":
        assert after["correct"]==99
    summary={"split":split,"n":expected_size,"v9":before["correct"],"v12":after["correct"],
             "accuracy":after["accuracy"],**dict(counts),"old_prefixes_equal":len(preds),
             "fresh_lm_calls":0,"exact_reused_cpu_generations":len(events),
             "max_tools":max(p["tool_calls"] for p in preds),"lock_sha256":sha(LOCK),
             "predictions_sha256":sha(out/"predictions.json"),"score_sha256":sha(out/"score.json"),
             "v9_archive":str(previous),"policy":"frozen before either replay; only scoring process opens split references"}
    write_json(out/"replay_summary.json",summary)
    print(json.dumps(summary),flush=True)


if __name__=="__main__":
    os.chdir(ROOT)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=("validation","test","both"))
    parser.add_argument("--freeze-only",action="store_true")
    args=parser.parse_args()
    if not args.freeze_only and not args.split:
        parser.error("--split or --freeze-only required")
    locked=freeze()
    print("Frozen",locked["locked_utc"],str(LOCK),flush=True)
    if not args.freeze_only:
        for name in (("validation","test") if args.split=="both" else (args.split,)):
            run(name,locked)
