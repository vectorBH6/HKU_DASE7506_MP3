"""Freeze validation-only recovery selection before running the new test."""
import datetime
import hashlib
import json
from pathlib import Path
from analyze_recovery_runs import details
ROOT=Path(__file__).resolve().parents[1]
CONFIGS={"e20_recovery4":"recovery_v3_4.json","e21_recovery6":"recovery_v3_6.json"}


def main():
    target=ROOT/"reports/v3_selection.json"
    assert not target.exists(),"Selection is immutable; already exists"
    results=[]
    for exp in CONFIGS:
        item=details(exp,"validation")
        assert item,"Complete both validation experiments first"
        results.append(item[0])
    baseline=details("e13_review_risk","validation")[0]
    best=sorted(results,key=lambda r:(-r["correct"],r["model_calls"],r["experiment"]))[0]
    promote=best["correct"]>baseline["correct"]
    source=ROOT/"submission/configs"/CONFIGS[best["experiment"]]
    locked=ROOT/"submission/configs/recovery_v3_selected.json";assert not locked.exists()
    locked.write_bytes(source.read_bytes())
    old=json.loads((ROOT/"reports/v2_selection.json").read_text())
    record={"selected_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "rule":"Max validation correct; ties prefer fewer model calls, then experiment name. Promote only if strictly above v2 87/120.",
            "new_experiment":best["experiment"],"new_config":str(source.relative_to(ROOT)),
            "locked_config":str(locked.relative_to(ROOT)),"config_sha256":hashlib.sha256(locked.read_bytes()).hexdigest(),
            "promote":promote,"default_experiment":best["experiment"] if promote else "e15_review_selected",
            "test_metrics_used":False,"adapter_path":old["adapter_path"],"adapter_sha256":old["adapter_sha256"],
            "base_revision":old["base_revision"],
            "validation":[{k:r[k] for k in ("experiment","correct","n","accuracy","model_calls","source")} for r in [baseline]+results]}
    if promote:
        active=ROOT/"submission/configs/selected.json";backup=active.with_name("selected_v2.json")
        assert not backup.exists();backup.write_bytes(active.read_bytes());active.write_bytes(locked.read_bytes())
        record["previous_default_config"]=str(backup.relative_to(ROOT))
    with target.open("x") as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write("\n")
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
