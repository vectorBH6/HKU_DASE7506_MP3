"""One train-only synthesis-focused adjustment with short clause plans."""
from collections import Counter
import copy,json,random,sys
from pathlib import Path
from sqlglot import exp,parse_one
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.common import load_bank,read_jsonl,write_jsonl,write_json,digest
from sql_repair_agent.feedback_agent import INSTRUCTION
from sql_repair_agent.plan_agent import PLAN_INSTRUCTION


def plan(sql):
    tree=parse_one(sql,read='sqlite');select=tree if isinstance(tree,exp.Select) else tree.find(exp.Select)
    result={'project':[e.sql(dialect='sqlite') for e in select.expressions] if select else [],
            'tables':list(dict.fromkeys(t.sql(dialect='sqlite') for t in tree.find_all(exp.Table)))}
    for key,kind in [('joins',exp.Join),('filters',exp.Where),('group',exp.Group),('having',exp.Having),('order',exp.Order),('limit',exp.Limit)]:
        values=[e.sql(dialect='sqlite') for e in tree.find_all(kind)]
        if values:result[key]=values
    if isinstance(tree,(exp.Union,exp.Except,exp.Intersect)):result['set_operation']=tree.key.upper()
    return result


def main():
    out=ROOT/'artifacts/prepared_data/synthesis_v4';out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'artifacts/prepared_data/feedback_v4';previous=read_jsonl(source/'train.jsonl');p=json.loads((source/'provenance.json').read_text())
    rows,manifest,folder=load_bank('train');memory={r['id']:r['gold_sql'] for r in read_jsonl(folder/'train_memory.jsonl')}
    records=[]
    for row in rows:
        schema=manifest['databases'][row['db_id']]['schema']
        payload={'question':row['question'],'schema':[{'table':t['name'],'columns':[c['name']+' '+c['type'] for c in t['columns']],'foreign_keys':t.get('foreign_keys',[])} for t in schema],
                 'expected_example':row['example'],'ordered':row['ordered']}
        messages=[{'role':'system','content':'Compose a SQLite query for the supplied training question, schema and public example. Return one JSON object with tool="check", reason_code, a short clause plan, then sql. This is a query-synthesis exercise, not an execution trace. '+PLAN_INSTRUCTION},
                  {'role':'user','content':'TASK_DATA\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\nEND_TASK_DATA\nNext action JSON:'}]
        target={'tool':'check','reason_code':'compose_query','plan':plan(memory[row['id']]),'sql':memory[row['id']]}
        records.append({'id':row['id'],'state':'sql_synthesis_auxiliary','source':'train_question_schema_gold_no_fabricated_feedback','messages':messages,'target':json.dumps(target,separators=(',',':'))})
    groups={};submits=[];hints=[]
    for record in previous:
        if record.get('source')!='train_real_feedback_v4':continue
        action=json.loads(record['target'])
        if action['tool']=='check':groups.setdefault(record['id'],[]).append(record)
        elif action['tool']=='submit':submits.append(record)
        elif action['tool']=='repair_hint':hints.append(record)
    rng=random.Random(2026)
    selected=[]
    for key,values in sorted(groups.items()):
        rng.shuffle(values);seen=set()
        for item in values:
            # Prefer different true feedback states instead of duplicated weights.
            signature=item['messages'][1]['content']
            if signature in seen:continue
            selected.append(item);seen.add(signature)
            if len(seen)==3:break
    rng.shuffle(submits);rng.shuffle(hints);selected+=submits[:600]+hints[:150]
    for original in selected:
        record=copy.deepcopy(original);record['source']='train_feedback_synthesis_adjustment'
        record['messages'][0]['content']=record['messages'][0]['content'].replace(INSTRUCTION,'')+PLAN_INSTRUCTION
        payload=json.loads(record['messages'][1]['content'].split('TASK_DATA\n',1)[1].split('\nEND_TASK_DATA',1)[0]);payload.pop('sql_diagnostics',None)
        record['messages'][1]['content']='TASK_DATA\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\nEND_TASK_DATA\nNext action JSON:'
        action=json.loads(record['target'])
        if action['tool']=='check':action={'tool':'check','reason_code':action.get('reason_code','repair_query'),'plan':plan(action['sql']),'sql':action['sql']}
        record['target']=json.dumps(action,separators=(',',':'));records.append(record)
    retention=[r for r in previous if r.get('source')=='previous_sql_sft_retention'];records+=retention
    assert all(r['id'].startswith('train-') for r in records) and len({r['id'] for r in records})==480
    write_jsonl(out/'train.jsonl',records)
    provenance={'split':'train','tasks':480,'examples':len(records),'states':dict(Counter(r['state'] for r in records)),
                'data_method':'auxiliary question/schema-to-SQL plan supervision plus selected real feedback contexts and original SQL retention; supervised, not RL',
                'validation_or_test_training_material':False,'labels':'official train gold and previously three-instance verified train candidate labels',
                'source_prepared_sha256':p['prepared_sha256'],'prepared_sha256':digest(out/'train.jsonl'),
                'gold_training_validation':p['gold_training_validation'],'source_hashes':{str(x.relative_to(ROOT)):digest(x) for x in [folder/'train.jsonl',folder/'train_memory.jsonl',source/'train.jsonl',Path(__file__)]}}
    write_json(out/'provenance.json',provenance);print(json.dumps(provenance,indent=2),flush=True)

if __name__=='__main__':main()
