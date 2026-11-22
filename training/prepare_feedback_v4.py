"""Current-policy and perturbation curricula, labelled only by official train replay."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time
from sqlglot import exp, parse_one

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'submission')]
from mp3_sql.common import load_bank, read_jsonl, write_json, write_jsonl, digest
from mp3_sql.environment import Episode,Tools
from sql_repair_agent.review_prompts import checked_candidates,messages_for
from sql_repair_agent.review_agent import parse_proposal
from sql_repair_agent.feedback_agent import enrich_messages
from sql_repair_agent.sql_utils import sql_key,extract_sql

CFG=json.loads((ROOT/'submission/configs/feedback_v4.json').read_text())


def grade(document,path):
    write_json(path.with_suffix('.predictions.json'),document)
    output=path.with_suffix('.score.json')
    with path.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable,'-m','mp3_sql.score','--split','train','--workers','4',
                        '--predictions',str(path.with_suffix('.predictions.json')),'--output',str(output)],
                       cwd=ROOT,check=True,stdout=log,stderr=subprocess.STDOUT)
    return json.loads(output.read_text())


def label_candidates(document,directory):
    maximum=max(sum(e['tool']=='check' for e in p['trace']) for p in document['predictions'])
    truth={p['id']:{} for p in document['predictions']}
    for index in range(maximum):
        predictions=[]
        for original in document['predictions']:
            checks=[i for i,e in enumerate(original['trace']) if e['tool']=='check']
            end=checks[min(index,len(checks)-1)]
            prefix=copy.deepcopy(original['trace'][:end+1]);sql=prefix[-1]['sql']
            prefix.append({'tool':'submit','sql':sql})
            predictions.append({'id':original['id'],'sql':sql,'trace':prefix,'tool_calls':len(prefix)-1})
        result=grade({**document,'predictions':predictions},directory/f'candidate_{index:02d}')
        for p,s in zip(predictions,result['per_question']):truth[p['id']][sql_key(p['sql'])]=s['passed']
    return truth


def mutations(sql):
    tree=parse_one(sql,read='sqlite');results=[]
    def save(name,root):
        candidate=root.sql(dialect='sqlite')
        if sql_key(candidate)!=sql_key(sql) and candidate not in [s for _,s in results]:results.append((name,candidate))
    for index,node in enumerate(tree.walk()):
        if isinstance(node,(exp.GT,exp.GTE,exp.LT,exp.LTE,exp.EQ)):
            other={exp.GT:exp.GTE,exp.GTE:exp.GT,exp.LT:exp.LTE,exp.LTE:exp.LT,exp.EQ:exp.NEQ}[type(node)]
            root=tree.copy();target=list(root.walk())[index]
            target.replace(other(this=target.this.copy(),expression=target.expression.copy()))
            save('predicate_boundary',root)
        if isinstance(node,exp.Count):
            root=tree.copy();target=list(root.walk())[index]
            if isinstance(target.this,exp.Distinct):
                target.set('this',target.this.expressions[0].copy());save('count_multiplicity',root)
            elif isinstance(target.this,exp.Column):
                target.set('this',exp.Distinct(expressions=[target.this.copy()]));save('count_multiplicity',root)
        if isinstance(node,exp.Literal) and not node.is_string:
            try:value=int(node.this)
            except ValueError:continue
            if 0<=value<=100:
                root=tree.copy();list(root.walk())[index].set('this',str(value+1));save('numeric_boundary',root)
        if isinstance(node,exp.Ordered):
            root=tree.copy();target=list(root.walk())[index];target.set('desc',not bool(target.args.get('desc')));save('ordering',root)
        if isinstance(node,exp.Select):
            for field in ['where','having','distinct']:
                if node.args.get(field):
                    root=tree.copy();list(root.walk())[index].set(field,None);save('remove_'+field,root)
            if len(node.expressions)>1:
                root=tree.copy();target=list(root.walk())[index];target.set('expressions',list(reversed(target.expressions)));save('projection_order',root)
            if not node.args.get('distinct'):
                root=tree.copy();list(root.walk())[index].set('distinct',exp.Distinct());save('add_distinct',root)
    return results


def probe_for(gold,schema):
    tree=parse_one(gold,read='sqlite');tables={t.alias_or_name.lower():t.name for t in tree.find_all(exp.Table)}
    actual={t['name'].lower():t for t in schema}
    # Prefer a column involved in a literal predicate: values can disambiguate a filter.
    columns=[]
    for predicate in tree.find_all(exp.EQ):
        if isinstance(predicate.expression,exp.Literal) and predicate.expression.is_string and isinstance(predicate.this,exp.Column):columns.append(predicate.this)
    columns+=list(tree.find_all(exp.Column))
    for column in columns:
        table=tables.get(column.table.lower()) if column.table else None
        choices=[t for t in actual.values() if any(c['name'].lower()==column.name.lower() for c in t['columns']) and (not table or t['name'].lower()==table.lower())]
        if len(choices)==1:
            t=choices[0]['name'];c=next(c['name'] for c in choices[0]['columns'] if c['name'].lower()==column.name.lower())
            return f'SELECT DISTINCT {exp.to_identifier(c,quoted=True).sql()} FROM {exp.to_identifier(t,quoted=True).sql()} LIMIT 8'
    return None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source-run',type=Path,required=True);ap.add_argument('--output-dir',type=Path,required=True);args=ap.parse_args()
    out=args.output_dir;out.mkdir(parents=True,exist_ok=False)
    rows,manifest,folder=load_bank('train');bank={r['id']:r for r in rows}
    source=json.loads((args.source_run/'predictions.json').read_text());assert source['split']=='train' and len(source['predictions'])==480
    source_score=json.loads((args.source_run/'score.json').read_text())
    originals={p['id']:p for p in source['predictions']}
    memory={r['id']:r['gold_sql'] for r in read_jsonl(folder/'train_memory.jsonl')}
    tools=Tools(folder,manifest,cache={})
    current=out/'current_scores';current.mkdir();truth=label_candidates(source,current)
    final={p['id']:p['passed'] for p in source_score['per_question']}
    coverage={k:any(t.values()) for k,t in truth.items()}
    diagnostic={'tasks':480,'final_correct':sum(final.values()),'checked_candidate_oracle_correct':sum(coverage.values()),
                'selection_losses':[k for k in final if coverage[k] and not final[k]],
                'no_correct_checked_candidate':[k for k in final if not coverage[k]],
                'public_match_wrong_candidates':sum(e['observation']['matches_example'] and not truth[p['id']][sql_key(e['sql'])] for p in source['predictions'] for e in p['trace'] if e['tool']=='check')}
    write_json(out/'diagnostic.json',diagnostic);print('TRAIN_DIAGNOSTIC',json.dumps({k:v if not isinstance(v,list) else len(v) for k,v in diagnostic.items()}),flush=True)
    started=time.perf_counter()
    def augment(row):
        ep=Episode(row,tools);ep.step({'tool':'check','sql':row['issue_sql']});ep.step({'tool':'schema'});ep.step({'tool':'repair_hint'})
        rng=random.Random(int(hashlib.sha256(row['id'].encode()).hexdigest()[:8],16))
        options=mutations(memory[row['id']]);rng.shuffle(options)
        selected=[];types=set()
        for kind,sql in options:
            if kind not in types: selected.append(sql);types.add(kind)
            if len(selected)==3:break
        hints=ep.trace[-1]['observation'];hint=extract_sql(hints[0]) if hints else None
        if hint:selected.insert(0,hint)
        selected.insert(rng.randrange(len(selected)+1),memory[row['id']])
        checked={sql_key(row['issue_sql'])}
        for sql in selected:
            if sql_key(sql) not in checked:
                ep.step({'tool':'check','sql':sql});checked.add(sql_key(sql))
        ep.step({'tool':'submit','sql':memory[row['id']]})
        return {'id':row['id'],'sql':memory[row['id']],'trace':ep.trace,'tool_calls':8-ep.remaining}
    with ThreadPoolExecutor(max_workers=4) as pool:augmented=list(pool.map(augment,rows))
    augmented_doc={**source,'predictions':augmented,'prediction_seconds':time.perf_counter()-started}
    goldscore=grade(augmented_doc,out/'augmented_gold');assert goldscore['correct']==480
    adir=out/'augmented_scores';adir.mkdir();aug_truth=label_candidates(augmented_doc,adir)
    for k,v in aug_truth.items():truth[k].update(v)
    records=[];seen=set()
    def add(row,obs,state,target,messages=None,weight=1):
        base=messages if messages is not None else messages_for(obs,CFG)
        rich=enrich_messages(base,obs)
        variants=[rich]
        # Retain compatibility with v3; do not edit the official observation.
        if int(hashlib.sha256((row['id']+state).encode()).hexdigest()[:4],16)%4==0:variants.append(base)
        for context in variants:
            key=json.dumps([row['id'],context,target],sort_keys=True)
            if key in seen:continue
            seen.add(key)
            item={'id':row['id'],'state':state,'messages':context,'target':json.dumps(target,separators=(',',':')),'source':'train_real_feedback_v4'}
            records.extend([item]*weight)
    def target(row,obs,proposal=None):
        candidates=checked_candidates(obs);good=[c for c in candidates if truth[row['id']].get(sql_key(c['sql']),False)]
        if good:
            chosen=next((c for c in good if proposal and proposal.get('candidate')==c['id']),good[0])
            return {'tool':'submit','candidate':chosen['id'],'reason_code':'question_consistent'}
        if proposal and proposal.get('tool') in ['check','execute'] and isinstance(proposal.get('sql'),str):
            sql=extract_sql(proposal['sql'])
            if sql and truth[row['id']].get(sql_key(sql),False) and sql_key(sql) not in {sql_key(c['sql']) for c in candidates}:
                return {'tool':'check','sql':sql,'reason_code':'verified_repair'}
        return {'tool':'check','sql':memory[row['id']],'reason_code':'example_not_proof' if any(c['feedback'].get('matches_example') for c in candidates) else 'repair_from_feedback'}
    for log in read_jsonl(args.source_run/'generations.jsonl'):
        row=bank[log['task_id']];payload=json.loads(log['messages'][1]['content'].split('TASK_DATA\n',1)[1].split('\nEND_TASK_DATA',1)[0]);n=8-payload['remaining_tool_calls']
        prefix=originals[row['id']]['trace'][:n];obs={'task':{'id':row['id'],'question':row['question']},'issue_sql':row['issue_sql'],'example':row['example'],'ordered':row['ordered'],'remaining':8-n,'history':copy.deepcopy(prefix)}
        proposal=parse_proposal(log['generated_text']);action=target(row,obs,proposal)
        # Preserve genuinely useful hint requests as positive tool supervision.
        if proposal and proposal.get('tool')=='repair_hint' and 'repair_hint' in payload['allowed_actions']:
            hints=tools.repair_hint(row);hint=extract_sql(hints[0]) if hints else None
            if hint and truth[row['id']].get(sql_key(hint),False) and action['tool']!='submit':action={'tool':'repair_hint','reason_code':'need_alternative'}
        add(row,obs,'current_policy',action,log['messages'])
    for p in augmented:
        row=bank[p['id']]
        for end,e in enumerate(p['trace']):
            if e['tool'] not in ['schema','repair_hint','check'] or end<1:continue
            n=end+1
            if n>=8:continue
            obs={'task':{'id':row['id'],'question':row['question']},'issue_sql':row['issue_sql'],'example':row['example'],'ordered':row['ordered'],'remaining':8-n,'history':copy.deepcopy(p['trace'][:n])}
            action=target(row,obs)
            hard=any(c['feedback'].get('matches_example') and not truth[row['id']].get(sql_key(c['sql']),False) for c in checked_candidates(obs))
            add(row,obs,'perturbed_feedback',action,weight=2 if hard and action['tool']=='check' else 1)
    # Small genuine exploratory curricula; tools execute real public queries only.
    bad=[r for r in rows if not truth[r['id']].get(sql_key(r['issue_sql']),False)]
    for index,row in enumerate(bad[:128]):
        ep=Episode(row,tools);ep.step({'tool':'check','sql':row['issue_sql']});ep.step({'tool':'schema'})
        probe=probe_for(memory[row['id']],tools.schema(row))
        if probe and index<96:
            add(row,ep.observation(),'probe_request',{'tool':'execute','sql':probe,'reason_code':'inspect_values'})
            ep.step({'tool':'execute','sql':probe});add(row,ep.observation(),'probe_feedback',target(row,ep.observation()))
        elif index>=96:
            query=row['question'][:2000];add(row,ep.observation(),'retrieve_request',{'tool':'retrieve','query':query,'reason_code':'adapt_pattern'})
            ep.step({'tool':'retrieve','query':query});add(row,ep.observation(),'retrieved_feedback',target(row,ep.observation()))
    old=read_jsonl(ROOT/'artifacts/prepared_data/onpolicy_v001/train.jsonl')
    retention=[r for r in old if r['source']=='previous_sql_sft_retention']
    actions=[r for r in old if r['source']!='previous_sql_sft_retention'];random.Random(42).shuffle(actions)
    records+=retention+actions[:400]
    assert all(r['id'].startswith('train-') for r in records)
    write_jsonl(out/'train.jsonl',records)
    provenance={'split':'train','tasks':480,'examples':len(records),'states':dict(Counter(r['state'] for r in records)),
                'actions':dict(Counter(json.loads(r['target'])['tool'] for r in records if r['target'].lstrip().startswith('{'))),
                'data_method':'current-policy real states plus gold-derived perturbation and exploratory curricula; supervised imitation, not RL',
                'labels':'only official train scorer on all three train instances; train_memory gold for expert next actions',
                'validation_or_test_training_material':False,'source_predictions_sha256':digest(args.source_run/'predictions.json'),
                'prepared_sha256':digest(out/'train.jsonl'),'source_hashes':{str(p.resolve().relative_to(ROOT)):digest(p) for p in [folder/'train.jsonl',folder/'train_memory.jsonl',args.source_run/'generations.jsonl']},
                'gold_training_validation':{'correct':goldscore['correct'],'n':goldscore['n']},'diagnostic':diagnostic}
    write_json(out/'provenance.json',provenance)
    print(json.dumps({k:v for k,v in provenance.items() if k!='diagnostic'},indent=2),flush=True)


if __name__=='__main__':main()
