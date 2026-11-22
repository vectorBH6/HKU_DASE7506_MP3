"""Small, question/example-grounded SQL edits, all checked by official tools."""
from itertools import permutations
import re
from sqlglot import exp,parse_one
from .feedback_agent import FeedbackAgent
from .review_prompts import checked_candidates
from .sql_utils import sql_key


def name_key(text):
    text=str(text).lower()
    text=re.sub(r'\b[a-z_][a-z_0-9]*\.', '', text)
    return re.sub(r'[\s`"\[\]]','',text)


def contract_edits(obs,sql):
    try:tree=parse_one(sql,read='sqlite')
    except Exception:return []
    if not isinstance(tree,exp.Select):return []
    result=[];seen={sql_key(sql)}
    def add(root,label):
        value=root.sql(dialect='sqlite');key=sql_key(value)
        if key not in seen:result.append((value,label));seen.add(key)
    expressions=tree.expressions;expected=obs['example']['columns']
    if 1<len(expressions)==len(expected)<=4:
        keys=[]
        for expression in expressions:
            item=expression.this if isinstance(expression,exp.Alias) else expression
            keys.append({name_key(item.sql(dialect='sqlite')),name_key(expression.alias_or_name)})
        wanted=[name_key(x) for x in expected]
        if not all(w in k for w,k in zip(wanted,keys)):
            for order in permutations(range(len(expressions))):
                if all(wanted[i] in keys[index] for i,index in enumerate(order)):
                    root=tree.copy();root.set('expressions',[root.expressions[i] for i in order]);add(root,'projection_contract')
                    if len(result)>=2:break
    # Explicit unique output rows, not a speculative COUNT DISTINCT rewrite.
    if re.search(r'\b(distinct|unique)\b',obs['task']['question'],re.I) and not tree.args.get('distinct') and not any(tree.find_all(exp.AggFunc)):
        root=tree.copy();root.set('distinct',exp.Distinct());add(root,'explicit_distinct')
    return result


class SemanticAgent(FeedbackAgent):
    def __init__(self,config):
        super().__init__(config);self.contract_checks=0

    def best(self,obs):
        candidates=checked_candidates(obs)
        matching=[c for c in candidates if c['feedback'].get('matches_example')]
        if matching:
            return min(matching,key=lambda c:len(contract_edits(obs,c['sql'])))
        return super().best(obs)

    def guard(self,obs,proposal):
        action,reason=super().guard(obs,proposal)
        if action and action['tool']=='submit':
            best=self.best(obs)
            if best['feedback'].get('matches_example') and len(contract_edits(obs,best['sql']))<len(contract_edits(obs,action['sql'])):
                return {'tool':'submit','sql':best['sql']},'static_contract_selection'
        return action,reason

    def act(self,obs):
        candidates=checked_candidates(obs)
        if candidates and obs['remaining'] and self.contract_checks<self.config.get('max_contract_checks',2):
            best=self.best(obs)
            sources=[best] if best['feedback'].get('matches_example') else [best]+list(reversed(candidates))
            checked={sql_key(c['sql']) for c in candidates}
            for source in sources:
                for sql,label in contract_edits(obs,source['sql']):
                    if sql_key(sql) not in checked:
                        self.contract_checks+=1
                        action={'tool':'check','sql':sql};self.audit(obs,None,action,'static_'+label)
                        return action
        action=super().act(obs)
        # Parent's initial low-risk fast path does not call best().
        if action['tool']=='submit' and candidates:
            best=self.best(obs)
            if best['feedback'].get('matches_example') and len(contract_edits(obs,best['sql']))<len(contract_edits(obs,action['sql'])):
                action={'tool':'submit','sql':best['sql']};self.audit(obs,None,action,'static_contract_selection')
        return action
