"""Verify the complete V11 validation trace against the independent budgeted terminal probe."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/"submission"),str(ROOT/"scripts")]
from mp3_sql.common import BUDGET, digest, read_jsonl, write_json
from mp3_sql.environment import rollout
from mp3_sql.predict import load_factory
from mp3_sql.score import load_cases, score_document
from probe_v10_text_grounding import PublicBank, choose


def read(path):
    return json.loads(path.read_text())


def main():
    bank=PublicBank("validation")
    prefix=read(ROOT/"runs/v10_prefix_probe/validation/predictions.json")
    cfg=ROOT/"submission/configs/v11_text_fallback.json"
    method=ROOT/"submission/v11_text_fallback.py"
    factory=load_factory(method)
    ref=read(ROOT/"runs/v10_text_grounding/proven_failure_fallback.json")
    policy={"families":ref["families"],"checks":8,"gate":"failed"}
    history=read_jsonl(ROOT/"runs/e34_feedback_variant/validation/r001/generations.jsonl")
    logs={(g["task_id"],g["attempt"]):g for g in history}
    used=set()

    class GeneratorReplay:
        def generate(self,messages,task_id,attempt,config):
            key=(task_id,attempt)
            logged=logs[key]
            assert messages==logged["messages"] and key not in used
            assert config["adapter_sha256"]==logged["identity"]["adapter"]["sha256"]
            used.add(key)
            return logged["generated_text"]

    with patch("sql_repair_agent.model.get_generator",return_value=GeneratorReplay()):
        preds=[rollout(row,bank.tools,factory(cfg)) for row in bank.rows]
    assert len(used)==len(history)
    extra=changed=pass_kept=failed_only=falling_back=0
    for row,old,real in zip(bank.rows,prefix["predictions"],preds,strict=True):
        option=bank.options(row,old)
        old_fails=any(e["tool"]=="check" and e["sql"]==old["sql"] and
                      e["observation"]["matches_example"] is False for e in old["trace"])
        option["matched"]=not old_fails
        want,checks=choose(option,policy)
        episode=bank.terminal(row,old)
        for q in checks:
            episode.step({"tool":"check","sql":q})
        episode.step({"tool":"submit","sql":want})
        assert want==real["sql"] and real["trace"]==episode.trace
        assert real["tool_calls"]==BUDGET-episode.remaining
        extra+=len(checks)
        changed+=want!=old["sql"]
        failed_only+=old_fails
        if not old_fails:
            assert old==real
            pass_kept+=1
        if checks and all(not e["observation"]["matches_example"] for e in real["trace"][old["tool_calls"]:-1]):
            assert real["sql"]==old["sql"]
            falling_back+=1
    doc={**prefix,"predictions":preds}
    out=ROOT/"runs/v11_prefix_probe/validation/predictions.json"
    write_json(out,doc)
    cases,case_root=load_cases("validation",bank.folder)
    def grades(data):
        score=score_document(data,bank.rows,bank.manifest,bank.folder,cases,case_root,workers=1,cache={})
        return [q["passed"] for q in score["per_question"]]
    v10,v11=grades(prefix),grades(doc)
    result={"scope":"Validation with exact prompt-by-prompt CPU V4 output reuse and real official public tools, no fresh model inference",
            "n":len(preds),"v10":sum(v10),"v11":sum(v11),"wins":sum(y and not x for x,y in zip(v10,v11)),
            "losses":sum(x and not y for x,y in zip(v10,v11)),"independently_equal_full_episodes":len(preds),"max_tools":max(p["tool_calls"] for p in preds),
            "check_verified_failures":failed_only,"other_episodes_identical":pass_kept,"failed_searches_kept_original":falling_back,
            "extra_checks":extra,"changed_final_sql":changed,"matching_generation_prompts":len(used),
            "entrypoint_sha256":digest(method),"config_sha256":digest(cfg),"predictions_sha256":digest(out)}
    write_json(ROOT/"reports/v11_text_validation_probe.json",result)
    print(result,flush=True)


if __name__=="__main__":
    main()
