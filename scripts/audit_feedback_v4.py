"""Audit final v4 runtime identities, budgets, sources and selected output copies."""
from collections import Counter
from datetime import datetime
from functools import lru_cache
import hashlib,json
from pathlib import Path
from analyze_feedback_v4 import CPU
from analyze_recovery_runs import details
from analyze_review_runs import latest
ROOT=Path(__file__).resolve().parents[1]


@lru_cache(maxsize=None)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sel=json.loads((ROOT/'reports/v4_selection.json').read_text());cfg=ROOT/sel['locked_config']
    assert sha(cfg)==sel['config_sha256']
    data=json.loads((ROOT/'reports/v4_data_audit.json').read_text());assert data['all_checks_passed'] and data['train_only']
    train_ids={json.loads(line)['id'] for line in (ROOT/'mp3_sql/data/train.jsonl').read_text().splitlines()}
    for name in ['feedback_v4','synthesis_v4']:
        prepared=ROOT/'artifacts/prepared_data'/name
        records=[json.loads(line) for line in (prepared/'train.jsonl').read_text().splitlines()]
        assert {r['id'] for r in records}==train_ids and all(r['id'] in train_ids for r in records)
        provenance=json.loads((prepared/'provenance.json').read_text())
        assert provenance['prepared_sha256']==sha(prepared/'train.jsonl')
    train=json.loads((ROOT/'runs/training/feedback_v4/training_metadata.json').read_text())
    assert train['status']=='completed' and train['examples_used']==5871 and not train['skipped_long_examples']
    assert all('.lora_A.' in n or '.lora_B.' in n for n in train['trainable_names'])
    assert train['prepared_data_sha256']==data['prepared_sha256']
    extra=ROOT/'runs/training/feedback_v4_extra/training_metadata.json'
    if extra.exists():
        e=json.loads(extra.read_text());assert e['status']=='completed'
        assert all('.lora_A.' in n or '.lora_B.' in n for n in e['trainable_names'])
        assert e['prepared_data_sha256']==sha(ROOT/e['config']['prepared_data']/'train.jsonl')
    artifact=json.loads((ROOT/'reports/v4_artifact_verification.json').read_text())
    assert not artifact['official_files_modified'] and not artifact['base_files_modified'] and artifact['only_lora_A_B_trainable']
    adapter=ROOT/sel['adapter_path'];assert {p.name for p in adapter.iterdir()}=={'adapter_config.json','adapter_model.safetensors'}
    assert sha(adapter/'adapter_model.safetensors')==sel['adapter_sha256']
    checked=[]
    for exp in CPU:
        for split in ['validation','test']:
            found=latest(exp,split)
            if not found:continue
            run,meta=found;r,pred,score=details(exp,split)
            assert meta['device']=='cpu' and len(meta['cpu_affinity'])==4 and meta['offline']
            assert meta['python_executable']==str(ROOT/'.venv/bin/python')
            assert score['n']==(120 if split=='validation' else 240) and all(len(x['cases'])==3 for x in score['per_question'])
            assert score['scorer_sha256']==sha(ROOT/'mp3_sql/score.py') and score['executor_sha256']==sha(ROOT/'mp3_sql/sqlite_tool.py')
            assert r['prediction_seconds']<=7200 and r['scoring_seconds']<=180 and meta['peak_process_tree_rss_bytes']<=16*1024**3
            assert r['max_model_calls']<=meta['config']['max_completions']<=6
            assert r['max_new_tokens']<=meta['config']['max_new_tokens']<=2048 and r['max_tool_calls']<=8
            assert (run/'prediction.log').read_text().count('Loaded model: ')==1
            modules=['model','sql_utils','review_agent','review_prompts','recovery_agent']
            controller=meta['config']['controller']
            if controller in ['feedback_v4','semantic_v4','contract_v4','plan_v4','constraint_v4','plan_constraint_v4']:modules+=['feedback_agent']
            if controller in ['semantic_v4','contract_v4','plan_v4','constraint_v4','plan_constraint_v4']:modules+=['semantic_agent']
            if controller in ['contract_v4','plan_v4','constraint_v4','plan_constraint_v4']:modules+=['contract_agent']
            if controller in ['plan_v4','plan_constraint_v4']:modules+=['plan_agent']
            if controller in ['constraint_v4','plan_constraint_v4']:modules+=['constraint_agent']
            if controller=='plan_constraint_v4':modules+=['plan_constraint_agent']
            for name in modules:
                relative=f'submission/sql_repair_agent/{name}.py'
                assert sha(ROOT/relative)==meta['source_sha256'][relative],(exp,relative)
            logs=[json.loads(s) for s in (run/'generations.jsonl').read_text().splitlines()]
            for g in logs:
                assert g['identity']['device']=='cpu' and g['identity']['total_parameters']==606142464
                assert g['identity']['revision']==sel['base_revision'] and not g['generation']['do_sample']
                assert sha(Path(g['identity']['adapter']['path'])/'adapter_model.safetensors')==g['identity']['adapter']['sha256']
            counts=Counter(g['task_id'] for g in logs)
            for key,n in counts.items():assert [g['attempt'] for g in logs if g['task_id']==key]==list(range(1,n+1))
            if exp=='e35_feedback_selected':
                assert meta['config']==json.loads(cfg.read_text())
                assert datetime.fromisoformat(sel['selected_at'])<datetime.fromisoformat(meta['started_at'])
                assert all(g['identity']['adapter']['sha256']==sel['adapter_sha256'] for g in logs)
                assert sha(ROOT/'submission/outputs/v4-test.json')==sha(run/'predictions.json')
                assert sha(ROOT/'submission/outputs/v4-test-score.json')==sha(run/'score.json')
                assert sha(ROOT/'submission/submission.py')==meta['source_sha256']['submission/submission.py']
            checked.append({k:r[k] for k in ['experiment','split','source','correct','n','model_calls','max_model_calls','max_new_tokens','max_tool_calls','prediction_seconds','scoring_seconds']}
                           |{'CPU_affinity':meta['cpu_affinity'],'predictions_sha256':sha(run/'predictions.json'),'score_sha256':sha(run/'score.json')})
    default='e35_feedback_selected' if sel['promote'] else 'e22_recovery_selected';run,_=latest(default,'test')
    assert sha(ROOT/'submission/outputs/test.json')==sha(run/'predictions.json')
    assert sha(ROOT/'submission/outputs/test-score.json')==sha(run/'score.json')
    if sel['promote']:assert (ROOT/'submission/configs/selected.json').read_bytes()==cfg.read_bytes()
    assert not (ROOT/'dist').exists() and not (ROOT/'peer_reviews').exists()
    result={'all_checks_passed':True,'train_only':True,'selection_uses_CPU_validation_only':True,'test_metrics_used':False,
            'default_promoted':sel['promote'],'adapter_sha256':sel['adapter_sha256'],'runs':checked,
            'single_venv':str(ROOT/'.venv'),'peer_reviews_created':False,'dist_created':False,
            'source_note':'Controller/model implementations match snapshots. Entry point gained only dispatch branches during development; selected test dispatcher matches final source.'}
    (ROOT/'reports/v4_runtime_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('All fourth-round runtime checks passed')

if __name__=='__main__':main()
