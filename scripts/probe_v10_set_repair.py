"""Development-only set-operation candidates derived from visible SQL, schemas and questions."""
from collections import Counter
import json
from pathlib import Path
import re
import sys

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT / 'submission')]
from mp3_sql.common import BUDGET, digest, load_bank
from mp3_sql.environment import Episode, Tools
from mp3_sql.score import load_cases, score_document
from sql_repair_agent.sql_utils import sql_key

COMPARISONS = ((r'(?:greater than|more than|above|over)',exp.GT),
               (r'(?:less than|fewer than|below|under)',exp.LT),
               (r'(?:at least|no fewer than)',exp.GTE),
               (r'(?:at most|no more than)',exp.LTE))
DIRECTION = (exp.GT,exp.GTE,exp.LT,exp.LTE)
def stem(word):
    return word[:-3]+'y' if len(word)>4 and word.endswith('ies') else word.rstrip('s')

def tokens(text):
    return {stem(w) for w in re.findall(r'[a-z0-9]+',text.casefold())}

def cols_only(tree):
    return all(isinstance(e.unalias(),exp.Column) for e in tree.expressions)

def set_pair(base, a, b, op):
    select_a = base.where(a.copy()) if a is not None else base.copy()
    select_b = base.where(b.copy())
    return op(this=select_a,expression=select_b,distinct=True).sql(dialect='sqlite')

def edit(sql,q):
    try:
        t=parse_one(sql,read='sqlite')
    except (SqlglotError,ValueError,RecursionError):
        return
    if not isinstance(t,exp.Select) or not cols_only(t) or any(t.args.get(k) for k in ('group','having','limit','order','with','with_','offset','qualify','into')):
        return
    if sum(1 for _ in t.find_all(exp.Select)) != 1 or not (where:=t.args.get('where')):
        return
    terms=where.this.flatten() if isinstance(where.this,exp.And) else [where.this]
    terms=list(terms)
    both=bool(re.search(r'\bboth\b',q))
    exclusion=bool(re.search(r'\b(?:without|never|no|none|except|but not)\b',q))
    if not (both or exclusion): return
    for pos,term in enumerate(terms):
        x=term.unnest()
        other=[c.copy() for i,c in enumerate(terms) if i!=pos]
        base=t.copy()
        base.set('where',None)
        if other:base=base.where(*other)
        if both:
            if isinstance(x,exp.In) and isinstance(x.this,exp.Column) and len(x.expressions)==2 and all(isinstance(v,exp.Literal) for v in x.expressions):
                a,b=[exp.EQ(this=x.this.copy(),expression=v.copy()) for v in x.expressions]
                if a!=b:yield 'intersect',set_pair(base,a,b,exp.Intersect)
            elif isinstance(x,exp.Or) and all(isinstance(c,exp.EQ) and isinstance(c.left,exp.Column) and isinstance(c.right,exp.Literal) for c in (x.left,x.right)) and x.left.left==x.right.left and x.left.right!=x.right.right:
                yield 'intersect',set_pair(base,x.left,x.right,exp.Intersect)
        if exclusion and isinstance(x,DIRECTION) and isinstance(x.left,exp.Column) and isinstance(x.right,exp.Literal) and not x.right.is_string:
            for cue,cls in COMPARISONS:
                if re.search(r'\b'+cue+r'\s+'+re.escape(x.right.this)+r'(?![\d.])',q):
                    positive=cls(this=x.left.copy(),expression=x.right.copy())
                    yield 'exclude',set_pair(base,None,positive,exp.Except)
                    break
    if len(terms)==2 and all(isinstance(c, (exp.EQ,exp.NEQ)) and isinstance(c.left,exp.Column) and isinstance(c.right,exp.Literal) for c in terms):
        a,b=terms
        if a.left!=b.left or a.right==b.right: return
        base=t.copy();base.set('where',None)
        if both and isinstance(a,exp.EQ) and isinstance(b,exp.EQ):
            yield 'intersect',set_pair(base,a,b,exp.Intersect)
        if exclusion and {type(a),type(b)}=={exp.EQ,exp.NEQ}:
            keep,drop=(a,b) if isinstance(a,exp.EQ) else (b,a)
            yield 'exclude',set_pair(base,keep,exp.EQ(this=drop.left.copy(),expression=drop.right.copy()),exp.Except)

def union_tables(row,schema):
    q=row['question'].lower()
    if ' and ' not in q or re.search(r"\b(?:where|who|which have|with|without|most|more|less|each|for each|not|ordered|both in|both from)\b|\d",q):return
    labels=row['example']['columns']
    if not labels or not all(re.fullmatch(r'[\w ]+',s) for s in labels):return
    choices=[]
    for table in schema:
        if not tokens(table['name']) or not tokens(table['name']) <= tokens(q):continue
        names={c['name'].lower():c['name'] for c in table['columns']}
        if all(s.lower() in names for s in labels):
            choices.append(exp.select(*(exp.column(names[s.lower()],quoted=True) for s in labels)).from_(exp.Table(this=exp.to_identifier(table['name'],quoted=True))))
    if len(choices)==2:
        yield 'union',exp.Union(this=choices[0],expression=choices[1],distinct=True).sql(dialect='sqlite')

def candidates(row,sqls,schema):
    generated=set(map(sql_key,sqls))
    q=row['question'].lower()
    for method,query in (pair for sql in sqls for pair in edit(sql,q)):
        key=sql_key(query)
        if key not in generated:generated.add(key);yield method,query
    for method,query in union_tables(row,schema):
        key=sql_key(query)
        if key not in generated:yield method,query

def new_ep(row,tools,trace):
    ep=Episode(row,tools)
    for event in trace:
        ep.step({k:v for k,v in event.items() if k!='observation'})
        assert ep.trace[-1]==event
    return ep

def submit(row,ep,sql):
    ep.step({'tool':'submit','sql':sql})
    return {'id':row['id'],'trace':ep.trace,'sql':sql,'tool_calls':BUDGET-ep.remaining}

def grade(rows,pred,meta,manifest,folder,split):
    if not rows:return []
    cases, root=load_cases(split,folder)
    cases={**cases,'ids':[r['id'] for r in rows]}
    return [q['passed'] for q in score_document({**meta,'predictions':pred},rows,manifest,folder,cases,root,workers=1,cache={})['per_question']]

def train():
    rows,man,folder=load_bank('train');tools=Tools(folder,man,cache={})
    spl=json.loads((ROOT/'artifacts/prepared_data/v6/split.json').read_text())
    pools={p['id']:p['candidates'] for p in map(json.loads,(ROOT/'runs/v8_pool_train/train_pool.jsonl').open())}
    meta=json.loads((ROOT/'runs/e00_draft/train/r001/predictions.json').read_text()) if (ROOT/'runs/e00_draft/train/r001/predictions.json').exists() else {}
    from mp3_sql.common import PROTOCOL
    meta.update(protocol=PROTOCOL,split='train',budget=BUDGET,bank_sha256=digest(folder/'train.jsonl'),manifest_sha256=digest(folder/'manifest.json'))
    outs={}
    for part in ('fit_ids','holdout_ids'):
        ids=set(spl[part]); stats=Counter(); candidates_to_grade=[]; target=[]; covered=[]
        for row in rows:
            if row['id'] not in ids:continue
            pool=pools[row['id']]
            orig_any=any(c['correct'] for c in pool)
            stats['total']+=1;stats['prior_covered']+=orig_any
            ep=Episode(row,tools);ep.step({'tool':'check','sql':row['issue_sql']});ep.step({'tool':'schema'})
            schema=ep.trace[-1]['observation']
            sqls=[c['sql'] for c in pool]
            opts=list(candidates(row,sqls,schema))
            stats['with_novel_candidates']+=bool(opts)
            for method,sql in opts[:6]:
                ep.step({'tool':'check','sql':sql})
                if ep.trace[-1]['observation']['matches_example']:
                    candidates_to_grade.append(submit(row,ep,sql));target.append(row);covered.append(orig_any);stats['public_match']+=1
                    break
        correct=grade(target,candidates_to_grade,meta,man,folder,'train')
        stats['first_public_match_correct']=sum(correct)
        stats['new_coverage']=sum(y and not old for y,old in zip(correct,covered))
        outs[part]=dict(stats)
    return outs

def validate():
    src=ROOT/'runs/v10_prefix_probe/validation/predictions.json';base=json.loads(src.read_text()); rows,man,folder=load_bank('validation');tools=Tools(folder,man,cache={})
    new=[];stats=Counter()
    for row,p in zip(rows,base['predictions'],strict=True):
        assert row['id']==p['id']
        checks=[e for e in p['trace'] if e['tool']=='check']
        schema=next((e['observation'] for e in p['trace'] if e['tool']=='schema'),[])
        active=any(e['observation']['matches_example'] for e in checks)
        sqls=[p['sql'],*[e['sql'] for e in checks]]
        opts=list(candidates(row,sqls,schema))
        if active:
            # Overrule a public match only when the incumbent interprets explicit
            # "both" as membership in either value of the same database column.
            allowed={sql_key(s) for kind,s in edit(p['sql'],row['question'].lower()) if kind=='intersect'}
            opts=[(k,s) for k,s in opts if sql_key(s) in allowed]
        if not opts or p['tool_calls']==8:new.append(p);continue
        ep=new_ep(row,tools,p['trace'][:-1]);sql=p['sql'];stats['attempted_tasks']+=1
        for kind,s in opts[:ep.remaining]:
            ep.step({'tool':'check','sql':s});stats['extra_tools']+=1
            if ep.trace[-1]['observation']['matches_example']:
                stats[kind+'_adopted']+=1;sql=s;break
        new.append(submit(row,ep,sql))
    after=grade(rows,new,base,man,folder,'validation')
    before=grade(rows,base['predictions'],base,man,folder,'validation')
    return {**stats,'baseline':sum(before),'correct':sum(after),'n':len(rows),'wins':sum(x and not y for x,y in zip(after,before)),'losses':sum(y and not x for x,y in zip(after,before)),'max_tools':max(p['tool_calls'] for p in new)}

if __name__=='__main__':
    out={'method':'Compile explicit entity-level both, exclusion and union into set SQL. One fixed policy, official train/validation only; no test.', 'train_pool_first_new_public_match':train(), 'validation_v10':validate()}
    (ROOT/'reports/v10_set_probe.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out))
