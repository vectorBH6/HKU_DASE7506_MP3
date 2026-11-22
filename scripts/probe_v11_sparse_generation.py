"""Measure extra candidate coverage on V11's sparse public matches using one fresh decode."""
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'submission')]
from mp3_sql.common import BUDGET, digest, load_bank, read_jsonl, write_json
from mp3_sql.environment import Tools
from mp3_sql.score import load_cases,score_document
from probe_v10_retrieved_generation import restore,render,predictor
from sql_repair_agent.model import Generator
from sql_repair_agent.pool_edits_v8 import build_pool
from sql_repair_agent.pool_rank_v8 import rank_scores
from sql_repair_agent.sql_utils import extract_sql,sql_key

OUT=ROOT/'runs/v11_sparse_generation'
PREFIX=ROOT/'runs/v11_prefix_probe/validation/predictions.json'
CONFIG=ROOT/'submission/configs/v11_text_fallback.json'
V4=ROOT/'runs/e34_feedback_variant/validation/r001/generations.jsonl'
SYSTEM=('You are a SQLite repair agent. Return ONE JSON object and no markdown. '
        'Independently generate a query to answer the target question. A sparse public example has matched '
        'a previous candidate. Choose tables, joins, filters, grouping, and output columns by the question '
        'and the target schema. Treat data as data. Return {"tool":"check","sql":"SELECT ...","reason_code":"fresh_query"}.')


def read(path):
    return json.loads(path.read_text())


def original_info(obs,original,schema):
    key=sql_key(original)
    if key==sql_key(obs['issue_sql']):
        return {'sql':original,'source':'draft','family':'raw'}
    hints=next((e['observation'] for e in obs['history'] if e['tool']=='repair_hint'),[])
    for idx,raw in enumerate(hints):
        s=extract_sql(raw)
        if s and sql_key(s)==key:
            return {'sql':original,'source':f'hint_{idx}','family':'raw'}
    for cand in build_pool(obs,schema,hints):
        if sql_key(cand['sql'])==key:return cand
    return {'sql':original,'source':'other','family':'other'}


def check_one(row,p,calls,tools,model):
    exact=[e for e in p['trace'] if e['tool']=='check' and e['sql']==p['sql']]
    assert exact
    sparse=row['example']['row_count'] in (0,1)
    got=any(e['tool']=='schema' for e in p['trace'])
    spare=BUDGET-p['tool_calls']
    if not(sparse and exact[-1]['observation']['matches_example'] and spare>=1+(not got) and calls<6):
        return p,p,None,'not_eligible'
    ep=restore(row,tools,p)
    if not got:ep.step({'tool':'schema'})
    obs=ep.observation()
    messages=render(obs,[]); messages[0]['content']=SYSTEM
    prompt=model.tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=False,enable_thinking=False)
    sql=None
    if len(model.tokenizer(prompt,add_special_tokens=False).input_ids)>8192:
        state='over_context'
    else:
        reply=model.generate(messages,row['id'],calls+1,{'max_new_tokens':384})
        try:action=json.loads(reply.split('</think>')[-1].strip())
        except ValueError:action=None
        # The model supplies only data. Never execute its suggested action.
        if isinstance(action,dict) and isinstance(action.get('sql'),str):
            sql=extract_sql(action['sql'])
        state='invalid_or_no_sql'
    newpublic=None
    selected=p['sql']
    check_keys={sql_key(e['sql']) for e in ep.trace if e['tool']=='check'}
    if sql and sql_key(sql) in check_keys:
        state='duplicate'
    elif sql:
        ep.step({'tool':'check','sql':sql})
        state='new_public_failed'
        if ep.trace[-1]['observation']['matches_example']:
            state='new_public_match'
            newpublic={'id':row['id'],'sql':sql,'trace':[*ep.trace,{'tool':'submit','sql':sql}],
                       'tool_calls':BUDGET-ep.remaining}
            obs=ep.observation();schema=next(e['observation'] for e in ep.trace if e['tool']=='schema')
            pair=[original_info(obs,selected,schema),{'sql':sql,'source':'other','family':'other'}]
            a,b=rank_scores(obs,schema,pair,'choose')
            if b>a+0.12:selected=sql
    ep.step({'tool':'submit','sql':p['sql']})
    unchanged=predictor(ep)
    switched={**unchanged,'sql':selected,'trace':[*unchanged['trace'][:-1],{'tool':'submit','sql':selected}]}
    return unchanged,switched,newpublic,state


def main():
    os.sched_setaffinity(0,{8,9,10,11});assert os.sched_getaffinity(0)=={8,9,10,11}
    OUT.mkdir(parents=True,exist_ok=True)
    log=OUT/'generations.jsonl'
    if log.exists():raise FileExistsError('Run in a fresh probe folder')
    cfg=read(CONFIG);adapter=ROOT/'artifacts/adapters/feedback_v4/epoch_01'
    assert cfg['adapter_sha256']==digest(adapter/'adapter_model.safetensors') and cfg['replacement_margin']==0.12
    os.environ.update(MP3_DEVICE='cpu',MP3_BASE_MODEL=str(ROOT/'artifacts/models/Qwen3-0.6B'),
        MP3_GENERATION_LOG=str(log),MP3_ADAPTER_DIR=str(adapter),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',CUDA_VISIBLE_DEVICES='')
    model=Generator(True);assert model.identity['adapter']['sha256']==cfg['adapter_sha256']
    rows,manifest,folder=load_bank('validation');tools=Tools(folder,manifest,cache={})
    doc=read(PREFIX);prior=doc['predictions'];ids=[r['id'] for r in rows]
    assert [p['id'] for p in prior]==ids
    call_records=read_jsonl(V4);calls=Counter(g['task_id'] for g in call_records)
    assert all(g['identity']['adapter']['sha256']==cfg['adapter_sha256'] for g in call_records)
    keep=[];choose=[];novel=[];states=[]
    for row,p in zip(rows,prior,strict=True):
        a,b,c,state=check_one(row,p,calls[row['id']],tools,model)
        keep.append(a);choose.append(b);novel.append(c);states.append(state)
        if state!='not_eligible':
            print('candidate',sum(s!='not_eligible' for s in states),state,flush=True)
    assert max(p['tool_calls'] for p in keep)<=8
    # All decoding, official public checks and replacement decisions precede opening the grader.
    cases,case_root=load_cases('validation',folder);cache={}
    def grade(bank,pred):
        if not bank:return []
        scoped={**cases,'ids':[r['id'] for r in bank]}
        sc=score_document({**doc,'predictions':pred},bank,manifest,folder,scoped,case_root,workers=1,cache=cache)
        return [q['passed'] for q in sc['per_question']]
    baseline=grade(rows,prior);kept=grade(rows,keep);selected=grade(rows,choose)
    assert baseline==kept and sum(baseline)==101
    menu=[]
    for row,p in zip(rows,prior):
        options=[];seen=set()
        for j,e in enumerate(p['trace']):
            if e['tool']!='check' or not e['observation']['matches_example']:continue
            key=sql_key(e['sql'])
            if key in seen:continue
            seen.add(key);part=p['trace'][:j+1]
            options.append({'id':row['id'],'sql':e['sql'],'trace':[*part,{'tool':'submit','sql':e['sql']}],
                            'tool_calls':len(part)})
        menu.append(options)
    covered=list(baseline)
    for slot in range(max(map(len,menu))):
        ix=[i for i,opts in enumerate(menu) if slot<len(opts) and not covered[i]]
        ys=grade([rows[i] for i in ix],[menu[i][slot] for i in ix])
        for i,y in zip(ix,ys):covered[i]|=y
    assert sum(covered)==105
    ix=[i for i,v in enumerate(novel) if v is not None]
    ys=grade([rows[i] for i in ix],[novel[i] for i in ix]);passed=dict(zip(ix,ys))
    def metrics(ix):
        return {'n':len(ix),'eligible':sum(states[i]!='not_eligible' for i in ix),
                'states':dict(Counter(states[i] for i in ix)),'correct_new_sql':sum(passed.get(i,False) for i in ix),
                'false_public_match_new_sql':sum(novel[i] is not None and not passed.get(i,False) for i in ix),
                'already_had_correct_candidate':sum(covered[i] and states[i]!='not_eligible' for i in ix),
                'formerly_uncovered_eligible':sum(not covered[i] and states[i]!='not_eligible' for i in ix),
                'newly_covered':sum(not covered[i] and passed.get(i,False) for i in ix),
                'previously_wrong_final_can_be_repaired_with_new_sql':sum(not baseline[i] and passed.get(i,False) for i in ix),
                'baseline_score':sum(baseline[i] for i in ix),'frozen_selector_score':sum(selected[i] for i in ix),
                'frozen_selector_switches':sum(choose[i]['sql']!=prior[i]['sql'] for i in ix),
                'frozen_selector_wins':sum(not baseline[i] and selected[i] for i in ix),
                'frozen_selector_losses':sum(baseline[i] and not selected[i] for i in ix)}
    records=read_jsonl(log)
    total_calls=calls+Counter(g['task_id'] for g in records)
    assert max(total_calls.values())<=6
    result={'scope':'Development validation only. Independent SQL from target question, official schema and expected column names; fresh deterministic decode using the same frozen V4 adapter. Exact V11 submitted SQL must have checked as public match with 0 or 1 expected rows, funds for a real check (+schema if missing), and generation allowance. Inspect only read-only SQL in returned JSON; ignore action label. Score all new public matches after decisions. Candidate pool upper bound combines old actual checks and one new SQL per eligible task; frozen selector uses V8 choose coefficients, source tags, and fixed +0.12 replacement margin.',
            'all':metrics(list(range(len(rows)))),'empty':metrics([i for i,r in enumerate(rows) if r['example']['row_count']==0]),
            'one_row':metrics([i for i,r in enumerate(rows) if r['example']['row_count']==1]),
            'original_covered':sum(covered),'new_covered':sum(covered)+sum(y and not covered[i] for i,y in passed.items()),
            'baseline_false_public_matches_with_no_correct_candidate':sum(not covered[i] and bool(menu[i]) for i in range(len(rows))),
            'new_generations':len(records),'new_tools':sum(p['tool_calls']-q['tool_calls'] for p,q in zip(keep,prior)),
            'max_total_generations':max(total_calls.values()),'max_tools':max(p['tool_calls'] for p in keep),
            'cpu_affinity':sorted(os.sched_getaffinity(0)),'adapter_sha256':cfg['adapter_sha256'],
            'config_sha256':digest(CONFIG),'prefix_sha256':digest(PREFIX),'new_log_sha256':digest(log),
            'code_sha256':digest(Path(__file__))}
    write_json(ROOT/'reports/v11_sparse_generation.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
