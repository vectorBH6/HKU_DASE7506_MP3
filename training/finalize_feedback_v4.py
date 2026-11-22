"""Finalize train-only diagnosis-first targets and reproducible provenance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from sqlglot import exp,parse_one
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.common import load_bank,read_jsonl,write_json,write_jsonl,digest
from sql_repair_agent.feedback_agent import sql_diagnostics


def diagnose(bad,gold,feedback):
    if feedback.get('error'):
        text=str(feedback['error']).lower()
        return 'repair_identifier' if 'column' in text or 'table' in text else 'repair_syntax'
    try:
        a=parse_one(bad,read='sqlite');b=parse_one(gold,read='sqlite')
        sections=[('count_entity',lambda t:[e.sql(normalize=True) for e in t.find_all(exp.Count)]),
                  ('grouping',lambda t:[e.sql(normalize=True) for e in t.walk() if isinstance(e,(exp.Group,exp.Having))]),
                  ('filter',lambda t:[e.sql(normalize=True) for e in t.find_all(exp.Where)]),
                  ('join_keys',lambda t:[e.sql(normalize=True) for e in t.find_all(exp.Join)]),
                  ('projection',lambda t:[e.sql(normalize=True) for e in t.expressions]),
                  ('ordering',lambda t:[e.sql(normalize=True) for e in t.walk() if isinstance(e,(exp.Order,exp.Limit))])]
        changed=[name for name,extract in sections if extract(a)!=extract(b)]
        return 'repair_'+'_'.join(changed[:2]) if changed else 'semantic_repair'
    except Exception:return 'semantic_repair'


def main():
    out=ROOT/'artifacts/prepared_data/feedback_v4';source=ROOT/'runs/e30_feedback_collection/train/r001'
    if (out/'provenance.json').exists() and json.loads((out/'provenance.json').read_text()).get('diagnosis_first'):
        print('Dataset already finalized',flush=True);return
    rows,manifest,folder=load_bank('train');bank={r['id']:r for r in rows};gold={r['id']:r['gold_sql'] for r in read_jsonl(folder/'train_memory.jsonl')}
    records=read_jsonl(out/'train.jsonl');(out/'train_before_diagnosis.jsonl').write_bytes((out/'train.jsonl').read_bytes())
    for record in records:
        if record.get('source')!='train_real_feedback_v4':continue
        assert record['id'].startswith('train-')
        text=record['messages'][1]['content'];payload=json.loads(text.split('TASK_DATA\n',1)[1].split('\nEND_TASK_DATA',1)[0]);candidates=payload['checked_candidates']
        action=json.loads(record['target'])
        if action['tool']=='check' and action.get('reason_code') in ['example_not_proof','repair_from_feedback']:
            last=candidates[-1];action['reason_code']=diagnose(last['sql'],gold[record['id']],last['feedback'])
        # Predict a short diagnosis before generating the SQL/action argument.
        ordered={'tool':action['tool'],'reason_code':action.get('reason_code','consistent')}
        ordered.update({k:v for k,v in action.items() if k not in ordered});record['target']=json.dumps(ordered,separators=(',',':'))
        if 'sql_diagnostics' in payload:
            row=bank[record['id']];schema=manifest['databases'][row['db_id']]['schema']
            payload['sql_diagnostics']={c['id']:sql_diagnostics(c['sql'],schema) for c in candidates}
        rng=random.Random(int(hashlib.sha256((record['id']+record['state']).encode()).hexdigest()[:8],16))
        if rng.random()<0.25:
            rng.shuffle(payload['checked_candidates']);rng.shuffle(payload['schema'])
            record['context_order_augmented']=True
        record['messages'][1]['content']='TASK_DATA\n'+json.dumps(payload,ensure_ascii=False,separators=(',',':'))+'\nEND_TASK_DATA\nNext action JSON:'
    write_jsonl(out/'train.jsonl',records)
    gs=json.loads((out/'augmented_gold.score.json').read_text());assert gs['correct']==480
    diagnostic=json.loads((out/'diagnostic.json').read_text())
    provenance={'split':'train','tasks':480,'examples':len(records),'states':dict(Counter(r['state'] for r in records)),
                'actions':dict(Counter(json.loads(r['target'])['tool'] for r in records if r['target'].startswith('{'))),
                'diagnosis_first':True,'diagnosis_labels':'train gold AST differences and actual public SQL errors; diagnosis precedes SQL',
                'context_order_augmentation':'25% of new contexts shuffle summaries only; candidate IDs and official traces stay intact',
                'data_method':'current-policy real states plus gold-derived perturbation and real exploratory curricula; supervised imitation, not RL',
                'labels':'only official train scorer across all three train instances; train_memory expert next actions',
                'validation_or_test_training_material':False,'source_predictions_sha256':digest(source/'predictions.json'),
                'prepared_sha256':digest(out/'train.jsonl'),
                'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [folder/'train.jsonl',folder/'train_memory.jsonl',source/'generations.jsonl',ROOT/'training/prepare_feedback_v4.py',Path(__file__)]},
                'gold_training_validation':{'correct':gs['correct'],'n':gs['n']},'diagnostic':diagnostic}
    write_json(out/'provenance.json',provenance)
    print(json.dumps({k:v for k,v in provenance.items() if k!='diagnostic'},indent=2),flush=True)

if __name__=='__main__':main()
