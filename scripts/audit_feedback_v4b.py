"""Verify the unchanged prevalidated control and the final public-score deployment."""
from collections import Counter
from datetime import datetime
from functools import lru_cache
import hashlib,json,sys
from pathlib import Path
from analyze_recovery_runs import details
from analyze_review_runs import latest
ROOT=Path(__file__).resolve().parents[1]
@lru_cache(maxsize=None)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 lock=json.loads((ROOT/'reports/v4b_selection.json').read_text());dep=json.loads((ROOT/'reports/v4_deployment.json').read_text())
 run,m=latest('e42_contract_control','test');val,vm=latest('e37_contract_old_weights','validation');r,p,s=details('e42_contract_control','test')
 assert sha(ROOT/lock['locked_config'])==lock['config_sha256']
 assert m['config']==vm['config']==json.loads((ROOT/lock['locked_config']).read_text())
 assert datetime.fromisoformat(lock['selected_at'])<datetime.fromisoformat(m['started_at'])
 assert m['device']=='cpu' and len(m['cpu_affinity'])==4 and m['offline']
 assert m['python_executable']==str(ROOT/'.venv/bin/python')
 assert s['n']==240 and all(len(x['cases'])==3 for x in s['per_question'])
 assert s['scorer_sha256']==sha(ROOT/'mp3_sql/score.py') and s['executor_sha256']==sha(ROOT/'mp3_sql/sqlite_tool.py')
 assert s['predictions_sha256']==sha(run/'predictions.json')
 assert r['prediction_seconds']<=7200 and r['scoring_seconds']<=180 and m['peak_process_tree_rss_bytes']<=16*1024**3
 assert r['max_model_calls']<=m['config']['max_completions']<=6 and r['max_new_tokens']<=m['config']['max_new_tokens']<=2048 and r['max_tool_calls']<=8
 assert (run/'prediction.log').read_text().count('Loaded model: ')==1
 modules=['model','sql_utils','agent','prompts','review_agent','review_prompts','recovery_agent','feedback_agent','semantic_agent','contract_agent']
 for name in modules:
  path='submission/sql_repair_agent/'+name+'.py'
  assert sha(ROOT/path)==m['source_sha256'][path]==vm['source_sha256'][path],path
 assert sha(ROOT/'submission/submission.py')==m['source_sha256']['submission/submission.py']
 logs=[json.loads(x) for x in (run/'generations.jsonl').read_text().splitlines()]
 for g in logs:
  identity=g['identity'];assert identity['device']=='cpu' and identity['total_parameters']==606142464
  assert identity['revision']=='c1899de289a04d12100db370d81485cdf75e47ca' and not g['generation']['do_sample']
  assert identity['adapter']['sha256']==lock['adapter_sha256']==sha(Path(identity['adapter']['path'])/'adapter_model.safetensors')
 counts=Counter(g['task_id'] for g in logs)
 for key,n in counts.items():assert [g['attempt'] for g in logs if g['task_id']==key]==list(range(1,n+1))
 for source,alias in [('predictions.json','v4b-test.json'),('score.json','v4b-test-score.json')]:assert sha(run/source)==sha(ROOT/'submission/outputs'/alias)
 selected,_=latest(dep['selected_experiment'],'test')
 for source,alias in [('predictions.json','test.json'),('score.json','test-score.json')]:assert sha(selected/source)==sha(ROOT/'submission/outputs'/alias)
 assert (ROOT/'submission/configs/selected.json').read_bytes()==(ROOT/dep['config_path']).read_bytes()
 assert dep['public_test_aggregate_used_for_deployment'] and not lock['test_question_labels_used_for_selection'] and not lock['controller_modified_after_validation']
 artifact=json.loads((ROOT/'reports/v4b_artifact_verification.json').read_text());assert not artifact['official_files_modified'] and not artifact['base_files_modified'] and artifact['only_lora_A_B_trainable']
 train_ids={json.loads(x)['id'] for x in (ROOT/'mp3_sql/data/train.jsonl').read_text().splitlines()}
 for name in ['feedback_v4','synthesis_v4']:
  directory=ROOT/'artifacts/prepared_data'/name;records=[json.loads(x) for x in (directory/'train.jsonl').read_text().splitlines()]
  assert {r['id'] for r in records}==train_ids
  provenance=json.loads((directory/'provenance.json').read_text());assert provenance['prepared_sha256']==sha(directory/'train.jsonl')
 sys.path.insert(0,str(ROOT/'submission'));import submission
 agent=submission.create_agent();other=submission.create_agent();assert agent is not other and agent.completions==other.completions==0
 assert agent.config==json.loads((ROOT/dep['config_path']).read_text())
 assert not (ROOT/'dist').exists() and not (ROOT/'peer_reviews').exists()
 report={'all_checks_passed':True,'selected_experiment':dep['selected_experiment'],'default_factory':type(agent).__name__,'fresh_task_memory':True,'control_identical_to_validation':True,'selection_precedes_control_test':True,'public_test_aggregate_used_for_deployment':True,'test_question_labels_used_for_training_or_rules':False,'original_v4_runtime_audit_is_historical':True,'single_venv':str(ROOT/'.venv'),'canonical_exact_copy_of_selected_raw_result':True,'control':{k:r[k] for k in ['correct','n','model_calls','max_model_calls','max_new_tokens','max_tool_calls','prediction_seconds','scoring_seconds','peak_rss_mib']},'control_predictions_sha256':sha(run/'predictions.json'),'control_score_sha256':sha(run/'score.json')}
 (ROOT/'reports/v4b_runtime_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('All final control/deployment checks passed')

if __name__=='__main__':main()
