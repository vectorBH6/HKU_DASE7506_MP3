"""Freeze a validation-only selection before the new full test run."""
import datetime
import hashlib
import json
from pathlib import Path
from analyze_review_runs import load

ROOT=Path(__file__).resolve().parents[1]
CONFIGS={"e12_review_onpolicy":"review_v2_onpolicy.json","e13_review_risk":"review_v2_risk.json","e14_review_off":"review_v2_off.json"}


def main():
    target=ROOT/"reports/v2_selection.json"
    if target.exists():
        raise RuntimeError("Selection is immutable; already exists")
    choices=[]
    for experiment in CONFIGS:
        item=load(experiment,"validation")
        if not item:
            raise RuntimeError("Complete all planned validation runs first: "+experiment)
        choices.append(item[0])
    assert load("e11_review_old_adapter","validation"), "Complete weight ablation first"
    old=load("e03_qwen3_lora","validation")[0]
    best=sorted(choices,key=lambda r:(-r["correct"],r["model_calls"],r["experiment"]))[0]
    default=best["experiment"] if best["correct"]>old["correct"] else old["experiment"]
    cfg=ROOT/"submission/configs"/CONFIGS[best["experiment"]]
    locked=ROOT/"submission/configs/review_v2_selected.json"
    assert not locked.exists()
    adapter=ROOT/"artifacts/adapters/onpolicy_v001/epoch_02/adapter_model.safetensors"
    record={"selected_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rule":"Max validation correct; new ties use fewer model calls then experiment name. Promote only if greater than v1 validation correct; otherwise retain v1.",
            "new_experiment":best["experiment"],"new_config":str(cfg.relative_to(ROOT)),"default_experiment":default,
            "locked_config":str(locked.relative_to(ROOT)),"config_sha256":hashlib.sha256(cfg.read_bytes()).hexdigest(),
            "adapter_path":str(adapter.parent.relative_to(ROOT)),"adapter_sha256":hashlib.sha256(adapter.read_bytes()).hexdigest(),
            "base_revision":"c1899de289a04d12100db370d81485cdf75e47ca","test_metrics_used":False,
            "validation":[{k:r[k] for k in ("experiment","correct","n","accuracy","model_calls","mean_tool_calls","source")} for r in [old]+choices]}
    locked.write_bytes(cfg.read_bytes())
    with target.open("x") as f:
        json.dump(record,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
