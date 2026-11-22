"""Build task-balanced train-only distillation examples from accepted teacher actions."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.common import digest, load_bank, read_jsonl, write_json, write_jsonl
from mp3_sql.score import load_cases,score_document
from sql_repair_agent.review_agent import parse_proposal
from sql_repair_agent.review_prompts import checked_candidates
from sql_repair_agent.sql_utils import sql_key

WORK=ROOT/'runs/routed_distill'
DATA=ROOT/'artifacts/prepared_data/routed_distill'
V4=ROOT/'runs/v10_onpolicy_train'
V7=WORK/'teacher_v7_pass'
BASE=ROOT/'artifacts/adapters/v7_paired/epoch_01'
OLD=ROOT/'artifacts/adapters/feedback_v4/epoch_01'


def read(p): return json.loads(p.read_text())


def stable(*items):
    return hashlib.sha256(json.dumps(items,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def prepare(branch):
    split=read(ROOT/'artifacts/prepared_data/v6/split.json'); fit=set(split['fit_ids'])
    rows,manifest,folder=load_bank('train'); by_id={r['id']:r for r in rows}
    source=V4 if branch=='fail' else V7
    prior=read(source/'predictions.json')
    preds={p['id']:p for p in prior['predictions'] if p['id'] in fit}
    raw=read_jsonl(source/'generations.jsonl')
    gen={(g['task_id'],g['attempt']):g for g in raw}; assert len(raw)==len(gen)
    expected=digest((OLD if branch=='fail' else BASE)/'adapter_model.safetensors')
    assert all(g['identity']['adapter']['sha256']==expected for g in raw)
    eligible={}
    for i,p in preds.items():
        first=p['trace'][0]
        assert first['tool']=='check' and first['sql']==by_id[i]['issue_sql']
        if bool(first['observation']['matches_example']) == (branch=='pass'):
            eligible[i]=p
    # Form targets solely from an actual model completion, compatible accepted action,
    # and the original causal position in a real transcript. Never distill fallbacks.
    candidates=defaultdict(list)
    for dec in read_jsonl(source/'decisions.jsonl'):
        i=dec['task_id']; g=gen.get((i,dec['attempt']));p=eligible.get(i)
        a=dec['accepted_action'];reason=dec['guard_reason']
        if not (p and g and a) or reason not in ('model_query','model_candidate_selection'):continue
        target=parse_proposal(g['generated_text'])
        if not isinstance(target,dict) or target.get('tool') not in ('check','submit') or target['tool']!=a['tool']:continue
        # Raw JSON only; fallback parsing a prose reply is not expert target data.
        try: literal=json.loads(g['generated_text'].strip())
        except ValueError:continue
        if literal!=target:continue
        history=p['trace'];offset=8-int(dec['remaining']);key=sql_key(a['sql'])
        if a['tool']=='check':
            if offset>=len(history) or history[offset]['tool']!='check' or history[offset]['sql']!=a['sql']:continue
            if not isinstance(target.get('sql'),str) or sql_key(target['sql'])!=key:continue
            terminal=offset
        else:
            seen=checked_candidates({'history':history[:offset]})
            hit=next((c for c in seen if c['id']==target.get('candidate') and sql_key(c['sql'])==key),None)
            if not hit:continue
            terminal=next(j for j,e in enumerate(history[:offset]) if e['tool']=='check' and sql_key(e['sql'])==key)
        exact=history[terminal]
        if not exact['observation']['matches_example']:continue
        # Require that the eventual teacher submission was this same verified answer.
        if sql_key(p['sql'])!=key:continue
        if a['tool']=='check' and key==sql_key(by_id[i]['issue_sql']):continue
        candidates[i].append((a['tool'],g,exact['sql'],terminal))
    # Independently grade the original complete task. Only successful fit episodes
    # enter the corpus, each with one accepted check/choice leading to that answer.
    ordered=[i for i in by_id if i in candidates]
    cases,case_root=load_cases('train',folder)
    scoped={**cases,'ids':ordered}
    official=score_document({**prior,'predictions':[eligible[i] for i in ordered]},[by_id[i] for i in ordered],
                            manifest,folder,scoped,case_root,workers=1,cache={})
    good={x['id'] for x in official['per_question'] if x['passed']}
    items=[]; types=Counter(); available=Counter()
    for i in ordered:
        if i not in good:continue
        # Failed drafts prioritize the model's actual successful edit. Passed drafts
        # prioritize the final verified selection that the student should preserve.
        opts=sorted(candidates[i],key=lambda x:((x[0]!='check') if branch=='fail' else (x[0]!='submit'),x[1]['attempt']))
        for kind,*_ in opts:available[kind]+=1
        kind,g,s,position=opts[0]; types[kind]+=1
        items.append({'id':i,'db_id':by_id[i]['db_id'],'state':branch+'_'+kind,'messages':g['messages'],
                      'target':g['generated_text'].strip(),'teacher_adapter_sha256':expected,
                      'teacher_attempt':g['attempt'],'checked_trace_offset':position,'target_sql_sha256':stable(s)})
    artifact={'branch':branch,'fit_tasks_in_route':len(eligible),'tasks_with_candidate_actions':len(ordered),
              'officially_correct_distinct_tasks':len(good),'support':dict(available),
              'chosen':dict(types),'source':str(source.relative_to(ROOT)),
              'sha256':{f:digest(source/f) for f in ('predictions.json','decisions.jsonl','generations.jsonl')}}
    DATA.mkdir(parents=True,exist_ok=True)
    write_jsonl(DATA/(branch+'.jsonl'),items);write_json(DATA/(branch+'_audit.json'),artifact)
    print(json.dumps(artifact),flush=True)


def combine():
    split=read(ROOT/'artifacts/prepared_data/v6/split.json')
    a=read_jsonl(DATA/'fail.jsonl');b=read_jsonl(DATA/'pass.jsonl')
    if not a or not b:raise ValueError('Both teachers must contribute successful real actions')
    # Limit either branch to two examples per example from the smaller branch.
    # Fixed task-hash order avoids confidence or validation-based selection.
    a=sorted(a,key=lambda x:stable('fail',x['id']))[:2*len(b)]
    b=sorted(b,key=lambda x:stable('pass',x['id']))[:2*len(a)]
    items=sorted(a+b,key=lambda x:stable('routed-v1',x['id']))
    assert len({r['id'] for r in items})==len(items)
    assert all(r['id'] in split['fit_ids'] and r['db_id'] in split['fit_db_ids'] for r in items)
    assert not set(r['id'] for r in items)&set(split['holdout_ids'])
    write_jsonl(DATA/'train.jsonl',items)
    prov={'scope':'Only the prior 382-task official train subset in 94 databases. The 98 held-out tasks in 24 other train databases were excluded from this update. Older adapters in the teacher/student initialization used the official training corpus. Validation and test excluded from new fitting. One example per training task; no pure fallbacks; verified accepted model actions lead to an officially correct submitted SQL.',
          'branch_data':{m:read(DATA/(m+'_audit.json')) for m in ('fail','pass')},
          'training_examples':len(items),'states':dict(Counter(r['state'] for r in items)),
          'train_sha256':digest(DATA/'train.jsonl'),'split_sha256':digest(ROOT/'artifacts/prepared_data/v6/split.json'),
          'initial_adapter_sha256':digest(BASE/'adapter_model.safetensors'),'schema_count':len({r['db_id'] for r in items})}
    write_json(DATA/'provenance.json',prov);print(json.dumps({k:v for k,v in prov.items() if k!='branch_data'}),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('stage',choices=('fail','pass','combine'))
    action=ap.parse_args().stage
    if action=='combine':combine()
    else:prepare(action)
