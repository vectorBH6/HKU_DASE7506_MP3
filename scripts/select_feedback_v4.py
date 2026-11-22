"""Immutable CPU-validation selection; never inspect new test scores."""
import datetime
import hashlib
import json
from pathlib import Path
from analyze_recovery_runs import details
ROOT=Path(__file__).resolve().parents[1]


def main():
    target=ROOT/'reports/v4_selection.json';assert not target.exists(),'Selection already frozen'
    matrix=json.loads((ROOT/'reports/v4_cpu_matrix.json').read_text())
    required={'e31_feedback_old_weights','e33_feedback_new_weights',matrix['old_controller_new_weights']}
    results=[]
    for exp in ['e31_feedback_old_weights','e32_recovery_new_weights','e33_feedback_new_weights','e34_feedback_variant','e36_semantic_old_weights','e37_contract_old_weights','e38_recovery_epoch1','e39_constraint_old_weights','e40_feedback_variant2','e41_feedback_variant3']:
        item=details(exp,'validation')
        if item:results.append(item[0])
        elif exp in required or (ROOT/'runs'/exp/'validation').exists():raise RuntimeError('Complete core CPU comparison first: '+exp)
    assert all(r['device']=='cpu' for r in results)
    best=sorted(results,key=lambda r:(-r['correct'],r['model_calls'],r['experiment']))[0]
    promote=best['correct']>89
    config=ROOT/'submission/configs/feedback_v4_selected.json';assert not config.exists()
    config.write_text(json.dumps(best['config'],indent=2)+'\n')
    adapter=Path(best['adapter']['path']);adapter_relative=str(adapter.relative_to(ROOT))
    record={'selected_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'rule':'Highest CPU validation correct, then fewer model calls, then experiment name. Promote only if strictly above v3 89/120.',
            'selected_experiment':best['experiment'],'locked_config':str(config.relative_to(ROOT)),
            'config_sha256':hashlib.sha256(config.read_bytes()).hexdigest(),
            'adapter_path':adapter_relative,'adapter_sha256':best['adapter']['sha256'],
            'base_revision':'c1899de289a04d12100db370d81485cdf75e47ca','promote':promote,'test_metrics_used':False,
            'validation':[{k:r[k] for k in ['experiment','correct','n','accuracy','model_calls','source','config','adapter']} for r in results]}
    if promote:
        active=ROOT/'submission/configs/selected.json';backup=active.with_name('selected_v3.json');assert not backup.exists()
        backup.write_bytes(active.read_bytes());active.write_bytes(config.read_bytes())
        script=ROOT/'scripts/run_selected.sh';backup_script=ROOT/'scripts/run_selected_v3.sh';assert not backup_script.exists()
        backup_script.write_bytes(script.read_bytes())
        script.write_text(script.read_text().replace('artifacts/adapters/onpolicy_v001/epoch_02',adapter_relative))
    with target.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(record,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
