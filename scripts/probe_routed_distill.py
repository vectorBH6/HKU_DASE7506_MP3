"""Collect train-only V7 keep examples and evaluate two weights with identical first-check routing."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.common import BUDGET,digest,load_bank,read_jsonl,write_json
from mp3_sql.environment import Tools,rollout
from mp3_sql.predict import load_factory
from mp3_sql.score import load_cases,score_document
from sql_repair_agent.model import get_generator

WORK=ROOT/'runs/routed_distill'
OLD=ROOT/'artifacts/adapters/v7_paired/epoch_01'
NEW=ROOT/'artifacts/adapters/routed_distill_v1/epoch_01'
METHOD=ROOT/'submission/first_check_router.py'
CONFIG=ROOT/'submission/configs/v12_v9_fallback.json'
SPLIT=ROOT/'artifacts/prepared_data/v6/split.json'
ALLOWED={'teacher_pass','holdout_before','holdout_after','validation_after','compare'}


def read(p):return json.loads(p.read_text())


def selection(stage):
    train=stage!='validation_after'
    name='train' if train else 'validation'
    bank,manifest,folder=load_bank(name)
    split=read(SPLIT)
    if stage=='teacher_pass':
        fitted=set(split['fit_ids'])
        source=read(ROOT/'runs/v10_onpolicy_train/predictions.json')
        initial={p['id']:p['trace'][0] for p in source['predictions']}
        rows=[r for r in bank if r['id'] in fitted and initial[r['id']]['observation']['matches_example']]
        assert all(initial[r['id']]['tool']=='check' and initial[r['id']]['sql']==r['issue_sql'] for r in rows)
    elif train:
        ids=set(split['holdout_ids'])
        rows=[r for r in bank if r['id'] in ids]
        assert len(rows)==98 and set(r['db_id'] for r in rows)==set(split['holdout_db_ids'])
    else:
        rows=bank;assert len(rows)==120
    return name,rows,manifest,folder


def run(stage):
    train,rows,manifest,folder=selection(stage)
    prior=stage in ('teacher_pass','holdout_before')
    adapter=OLD if prior else NEW
    target=WORK/('teacher_v7_pass' if stage=='teacher_pass' else stage)
    target.mkdir(parents=True,exist_ok=True)
    if (target/'predictions.json').exists():raise FileExistsError(f'Completed output exists: {target}')
    log=target/'generations.jsonl'
    if log.exists():raise FileExistsError(f'Unfinished output exists: {target}')
    sha=digest(adapter/'adapter_model.safetensors')
    base_cfg=read(CONFIG)
    cfg={**base_cfg,'adapter_sha256':sha}
    assert set(p.name for p in adapter.iterdir())=={'adapter_model.safetensors','adapter_config.json'}
    if not prior:assert sha!=base_cfg['adapter_sha256']
    config_path=target/'config.json';write_json(config_path,cfg)
    os.environ.update(MP3_DEVICE='cuda',MP3_GENERATION_LOG=str(log),MP3_ADAPTER_DIR=str(adapter),
                      MP3_BASE_MODEL=str(ROOT/'artifacts/models/Qwen3-0.6B'),HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    # Materialize once and verify the exact same model+adapter serves every task and both branches.
    model=get_generator(True)
    assert model.identity['adapter']['sha256']==sha and model.identity['device']=='cuda'
    tools=Tools(folder,manifest,cache={});factory=load_factory(METHOD);preds=[]
    for idx,row in enumerate(rows):
        preds.append(rollout(row,tools,factory(config_path)))
        if idx%20==19:print('tasks',stage,idx+1,flush=True)
    header={'protocol':manifest['protocol'],'split':train,'budget':BUDGET,
            'bank_sha256':digest(folder/(train+'.jsonl')),'manifest_sha256':digest(folder/'manifest.json')}
    doc={**header,'predictions':preds}
    write_json(target/'predictions.json',doc)
    records=read_jsonl(log) if log.exists() else []
    assert all(g['identity']==model.identity for g in records)
    counts=Counter(g['task_id'] for g in records)
    assert max(counts.values(),default=0)<=4 and max(x['tool_calls'] for x in preds)<=8
    result={'full_tasks':len(rows),'source_sha256':digest(METHOD),'config_sha256':digest(config_path),
            'adapter_sha256':sha,'adapter_config_sha256':digest(adapter/'adapter_config.json'),
            'device':model.identity['device'],'dtype':model.identity['dtype'],'base_revision':model.identity['revision'],
            'parameters':model.identity['total_parameters'],'generations':len(records),'max_generations':max(counts.values(),default=0),
            'max_tools':max(x['tool_calls'] for x in preds),'predictions_sha256':digest(target/'predictions.json')}
    if stage!='teacher_pass':
        cases,case_root=load_cases(train,folder)
        scoped={**cases,'ids':[r['id'] for r in rows]}
        score=score_document(doc,rows,manifest,folder,scoped,case_root,workers=1,cache={})
        write_json(target/'score.json',score)
        result['correct']=score['correct']
    write_json(target/'run.json',result);print(json.dumps(result),flush=True)


def compare():
    results={}
    for split,base_folder,new_folder,baseline_pred in [('holdout',WORK/'holdout_before',WORK/'holdout_after','predictions.json'),
              ('validation',ROOT/'runs/same_adapter_router/v7',WORK/'validation_after','router_predictions.json')]:
        name,rows,manifest,folder=selection('holdout_after' if split=='holdout' else 'validation_after')
        bp=read(base_folder/baseline_pred);np=read(new_folder/'predictions.json')
        base=bp['predictions'];new=np['predictions']
        assert [r['id'] for r in rows]==[r['id'] for r in base]==[r['id'] for r in new]
        assert all(b['trace'][0]==n['trace'][0] and b['trace'][0]['sql']==r['issue_sql'] for b,n,r in zip(base,new,rows))
        cases,case_root=load_cases(name,folder);scope={**cases,'ids':[r['id'] for r in rows]}; cache={}
        def grade(d):
            sc=score_document(d,rows,manifest,folder,scope,case_root,workers=1,cache=cache)
            return [r['passed'] for r in sc['per_question']]
        lblb=grade(bp);lbln=grade(np)
        groups={'all':list(range(len(rows))),'initial_public_pass':[],'initial_public_fail':[]}
        for i,p in enumerate(base):groups['initial_public_pass' if p['trace'][0]['observation']['matches_example'] else 'initial_public_fail'].append(i)
        report={}
        for k,ix in groups.items():
            report[k]={'n':len(ix),'original_v7_router':sum(lblb[i] for i in ix),'distilled_v7_router':sum(lbln[i] for i in ix),
                       'wins':sum(lbln[i] and not lblb[i] for i in ix),'losses':sum(lblb[i] and not lbln[i] for i in ix)}
        result={'groups':report,'baseline_predictions_sha256':digest(base_folder/baseline_pred),
                'student_predictions_sha256':digest(new_folder/'predictions.json')}
        logs=[(base_folder/('router_generations.jsonl' if split=='validation' else 'generations.jsonl'),OLD),
              (new_folder/'generations.jsonl',NEW)]
        for file,ad in logs:
            ev=read_jsonl(file); sha=digest(ad/'adapter_model.safetensors')
            assert ev and all(g['identity']['adapter']['sha256']==sha and g['identity']['device']=='cuda' and g['identity']['dtype']=='torch.bfloat16' for g in ev)
        results[split]=result
    out={'method':'Same first-check router and greedy single offline Qwen/V7-family LoRA under CUDA bfloat16; only adapter differs. One predeclared training configuration, one epoch; full three-instance scoring after predictions. Earlier teacher and init training touched official train, including tasks held out from this update. No test used.',
         'splits':results,'router_sha256':digest(METHOD),'base_adapter_sha256':digest(OLD/'adapter_model.safetensors'),
         'distilled_adapter_sha256':digest(NEW/'adapter_model.safetensors'),
         'update_data':read(ROOT/'artifacts/prepared_data/routed_distill/provenance.json'),
         'train_metadata_sha256':digest(ROOT/'runs/training/routed_distill_v1/training_metadata.json')}
    write_json(ROOT/'reports/routed_distill_validation.json',out)
    for s,r in results.items():print(s,r['groups'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=sorted(ALLOWED))
    stage=ap.parse_args().stage
    os.sched_setaffinity(0,{12,13,14,15})
    if stage=='compare':compare()
    else:run(stage)
