"""Conservative explicit numeric/unique-count constraints, grounded in the question."""
import re
from sqlglot import exp,parse_one
from .contract_agent import ContractAgent
from .semantic_agent import contract_edits
from .review_prompts import checked_candidates
from .sql_utils import sql_key

NUMBERS={word:i for i,word in enumerate(['zero','one','two','three','four','five','six','seven','eight','nine','ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen','twenty'])}
PHRASES={'at least':exp.GTE,'no less than':exp.GTE,'not less than':exp.GTE,'at most':exp.LTE,'no more than':exp.LTE,
         'more than':exp.GT,'greater than':exp.GT,'older than':exp.GT,'over':exp.GT,
         'less than':exp.LT,'fewer than':exp.LT,'younger than':exp.LT,'under':exp.LT}
PATTERN=re.compile(r'\b('+'|'.join(sorted(PHRASES,key=len,reverse=True))+r')\s+(\d+|'+'|'.join(NUMBERS)+r')\b',re.I)


def explicit_edits(obs,sql):
    try:tree=parse_one(sql,read='sqlite')
    except Exception:return []
    results=[];question=obs['task']['question'];seen={sql_key(sql)}
    def add(root,kind):
        value=root.sql(dialect='sqlite');key=sql_key(value)
        if len(value.encode())<=16000 and key not in seen:results.append((value,kind));seen.add(key)
    nodes=list(tree.walk())
    for phrase,number in PATTERN.findall(question):
        threshold=int(number) if number.isdigit() else NUMBERS[number.lower()];desired=PHRASES[phrase.lower()]
        # Only change a unique numeric comparison with the exact question threshold.
        compatible=(exp.GT,exp.GTE) if desired in (exp.GT,exp.GTE) else (exp.LT,exp.LTE)
        matches=[]
        for index,node in enumerate(nodes):
            if isinstance(node,compatible) and isinstance(node.expression,exp.Literal) and not node.expression.is_string:
                try:value=float(node.expression.this)
                except ValueError:continue
                if value==threshold:matches.append(index)
        if len(matches)==1 and not isinstance(nodes[matches[0]],desired):
            root=tree.copy();node=list(root.walk())[matches[0]]
            node.replace(desired(this=node.this.copy(),expression=node.expression.copy()));add(root,'explicit_numeric_boundary')
    if re.search(r'\b(distinct|different|unique)\b',question,re.I) and re.search(r'\b(how many|number|count)\b',question,re.I) and not re.search(r'\bdifferent\s+(from|than)\b',question,re.I):
        outer=tree if isinstance(tree,exp.Select) else None
        if outer:
            for index,node in enumerate(nodes):
                if isinstance(node,exp.Count) and isinstance(node.this,exp.Column):
                    root=tree.copy();target=list(root.walk())[index]
                    target.set('this',exp.Distinct(expressions=[target.this.copy()]));add(root,'explicit_unique_count')
    return results


def all_edits(obs,sql):return contract_edits(obs,sql)+explicit_edits(obs,sql)


class ConstraintAgent(ContractAgent):
    def best(self,obs):
        candidates=checked_candidates(obs);matching=[c for c in candidates if c['feedback'].get('matches_example')]
        return min(matching,key=lambda c:len(all_edits(obs,c['sql']))) if matching else super().best(obs)

    def guard(self,obs,proposal):
        action,reason=super().guard(obs,proposal)
        if action and action['tool']=='submit':
            best=self.best(obs)
            if best['feedback'].get('matches_example') and len(all_edits(obs,best['sql']))<len(all_edits(obs,action['sql'])):
                return {'tool':'submit','sql':best['sql']},'static_explicit_selection'
        return action,reason

    def act(self,obs):
        candidates=checked_candidates(obs)
        if candidates and obs['remaining'] and self.contract_checks<self.config.get('max_contract_checks',3):
            best=self.best(obs);sources=[best] if best['feedback'].get('matches_example') else [best]+list(reversed(candidates))
            used={sql_key(c['sql']) for c in candidates}
            for source in sources:
                for sql,label in explicit_edits(obs,source['sql']):
                    if sql_key(sql) not in used:
                        self.contract_checks+=1;action={'tool':'check','sql':sql};self.audit(obs,None,action,'static_'+label);return action
        action=super().act(obs)
        if action['tool']=='submit' and candidates:
            best=self.best(obs)
            if best['feedback'].get('matches_example') and len(all_edits(obs,best['sql']))<len(all_edits(obs,action['sql'])):
                action={'tool':'submit','sql':best['sql']};self.audit(obs,None,action,'static_explicit_selection')
        return action
