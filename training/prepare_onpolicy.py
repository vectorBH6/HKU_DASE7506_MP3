"""Train-only action supervision from real tools and official train grading."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.common import load_bank, read_jsonl, write_json, write_jsonl, digest
from mp3_sql.environment import Episode,Tools
from sql_repair_agent.sql_utils import sql_key,extract_sql
from sql_repair_agent.review_prompts import messages_for,checked_candidates

CFG={'use_hints':True,'use_retrieval':True}

def grade(path, output):
    subprocess.run([sys.executable,'-m','mp3_sql.score','--split','train','--workers','4',
                    '--predictions',str(path),'--output',str(output)],cwd=ROOT,check=True,
                   stdout=(output.with_suffix('.log')).open('w'),stderr=subprocess.STDOUT)
    return json.loads(output.read_text())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-run',type=Path,default=ROOT/'runs/e10_onpolicy_collection/train/r001')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'artifacts/prepared_data/onpolicy_v001')
    args=parser.parse_args()
    rows,manifest,folder=load_bank('train')
    source=args.source_run
    document=json.loads((source/'predictions.json').read_text())
    originals={p['id']:p for p in document['predictions']}
    generated={}
    for record in read_jsonl(source/'generations.jsonl'):
        sql=extract_sql(record['generated_text'])
        if sql: generated.setdefault(record['task_id'],set()).add(sql_key(sql))
    out=args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    scores=[]
    for path in sorted((out/'candidate_scores').glob('candidate_*.score.json')):
        scores.append({r['id']:r['passed'] for r in json.loads(path.read_text())['per_question']})
    if len(scores)!=5:
        raise RuntimeError('Complete all five official candidate scoring bundles first')
    memory={r['id']:r for r in read_jsonl(folder/'train_memory.jsonl')}
    tools=Tools(folder,manifest)
    started=time.perf_counter()
    def hint_episode(row):
        ep=Episode(row,tools)
        ep.step({'tool':'check','sql':row['issue_sql']})
        ep.step({'tool':'schema'})
        ep.step({'tool':'repair_hint'})
        hints=ep.trace[-1]['observation']
        sql=extract_sql(hints[0]) if hints else None
        if sql and sql_key(sql)!=sql_key(row['issue_sql']):
            ep.step({'tool':'check','sql':sql})
        final=sql if sql else row['issue_sql']
        ep.step({'tool':'submit','sql':final})
        return {'id':row['id'],'sql':final,'trace':ep.trace,'tool_calls':ep.remaining*-1+8},sql
    with ThreadPoolExecutor(max_workers=4) as pool:
        hint_runs=list(pool.map(hint_episode,rows))
    hint_document={**document,'predictions':[p for p,sql in hint_runs], 'prediction_seconds':time.perf_counter()-started}
    write_json(out/'hint_first.predictions.json',hint_document)
    hint_score=grade(out/'hint_first.predictions.json',out/'hint_first.score.json')
    hint_truth={r['id']:r['passed'] for r in hint_score['per_question']}
    hint_sqls={row['id']:sql for row,(p,sql) in zip(rows,hint_runs)}

    def prepare(row):
        source_pred=originals[row['id']]
        source_checks=[e for e in source_pred['trace'] if e['tool']=='check']
        truth={sql_key(e['sql']):scores[i][row['id']] for i,e in enumerate(source_checks)}
        hint=hint_sqls[row['id']]
        if hint: truth[sql_key(hint)]=hint_truth[row['id']]
        gold=memory[row['id']]['gold_sql']
        truth[sql_key(gold)]=True
        ep=Episode(row,tools)
        ep.step({'tool':'check','sql':row['issue_sql']})
        ep.step({'tool':'schema'})
        records=[]
        def correct():
            return [c for c in checked_candidates(ep.observation()) if truth.get(sql_key(c['sql']),False)]
        def target(reason, hint_request=False):
            good=correct()
            if good:
                return {'tool':'submit','candidate':good[0]['id'],'reason_code':'consistent'}
            if hint_request and hint and hint_truth[row['id']]:
                return {'tool':'repair_hint','reason_code':'need_alternative'}
            return {'tool':'check','sql':gold,'reason_code':reason}
        def add(state, action):
            obs=ep.observation()
            records.append({'id':row['id'],'state':state,'messages':messages_for(obs,CFG),
                            'target':json.dumps(action,ensure_ascii=False,separators=(',',':')),'source':'official_train_feedback'})
            if any(c['feedback'].get('matches_example') for c in checked_candidates(obs)) and not correct():
                records.append({**records[-1],'state':state+'_public_counterexample_weight'})
        orig_good=truth.get(sql_key(row['issue_sql']),False)
        reason='example_not_proof' if ep.trace[0]['observation']['matches_example'] else 'semantic_repair'
        add('initial_review',target(reason,hint_request=not orig_good))
        if not orig_good:
            ep.step({'tool':'repair_hint'})
            if hint and hint_truth[row['id']]:
                action={'tool':'check','sql':hint,'reason_code':'verify_hint'}
            else:
                action=target('ignore_unreliable_hint')
            add('hint_observation',action)
            if hint and sql_key(hint) not in {sql_key(c['sql']) for c in checked_candidates(ep.observation())}:
                ep.step({'tool':'check','sql':hint})
                add('hint_feedback_review',target('repair_failed_hint'))
        own=[e for e in source_checks[1:] if sql_key(e['sql']) in generated.get(row['id'],set()) and sql_key(e['sql']) not in {sql_key(c['sql']) for c in checked_candidates(ep.observation())}]
        # One actual old-policy candidate per task; both failures and successful
        # alternatives are retained, rather than teaching unconditional edits.
        if own:
            chosen=next((e for e in own if not truth.get(sql_key(e['sql']),False)),own[-1])
            ep.step({'tool':'check','sql':chosen['sql']})
            add('self_generated_feedback',target('repair_self_generated_query'))
        existing={sql_key(c['sql']):c for c in checked_candidates(ep.observation())}
        if sql_key(gold) not in existing:
            ep.step({'tool':'check','sql':gold})
            if not ep.trace[-1]['observation']['matches_example']:
                raise RuntimeError('Gold/public feedback mismatch: '+row['id'])
        gold_candidate=next(c for c in checked_candidates(ep.observation()) if sql_key(c['sql'])==sql_key(gold))
        if len(checked_candidates(ep.observation()))>1:
            add('candidate_comparison',{'tool':'submit','candidate':gold_candidate['id'],'reason_code':'question_consistent'})
        ep.step({'tool':'submit','sql':gold_candidate['sql']})
        final={'id':row['id'],'sql':gold_candidate['sql'],'trace':ep.trace,'tool_calls':8-ep.remaining}
        return records,final
    with ThreadPoolExecutor(max_workers=4) as pool:
        prepared=list(pool.map(prepare,rows))
    records=[r for group,pred in prepared for r in group]
    action_records=len(records)
    retained=read_jsonl(ROOT/'artifacts/prepared_data/sft_v001/train.jsonl')
    records += [{**r,'source':'previous_sql_sft_retention'} for r in retained]
    assert all(r['id'].startswith('train-') for r in records)
    write_jsonl(out/'train.jsonl',records)
    gold_document={**document,'predictions':[pred for group,pred in prepared], 'prediction_seconds':time.perf_counter()-started}
    write_json(out/'gold_validation.predictions.json',gold_document)
    gold_score=grade(out/'gold_validation.predictions.json',out/'gold_validation.score.json')
    if gold_score['correct']!=len(rows): raise RuntimeError('Gold targets did not pass all training cases')
    provenance={'split':'train','tasks':len(rows),'examples':len(records),'action_examples':action_records,
                'sql_retention_examples':len(retained),'states':dict(Counter(r['state'] for r in records)),
                'labels':'train_memory gold_sql + official train scorer correctness for checked candidates',
                'candidate_label_source':'Five independently replayed official train score bundles and first-hint score bundle',
                'data_method':'old-policy self-generated candidates with expert-augmented real tool states; supervised action imitation, not RL',
                'validation_or_test_training_material':False,
                'source_predictions_sha256':digest(source/'predictions.json'),
                'prepared_sha256':digest(out/'train.jsonl'),
                'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [folder/'train.jsonl',folder/'train_memory.jsonl',source/'score.json']},
                'gold_training_validation':{'correct':gold_score['correct'],'n':gold_score['n']}}
    write_json(out/'provenance.json',provenance)
    print(json.dumps(provenance,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
