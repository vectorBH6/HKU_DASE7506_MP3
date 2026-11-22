"""Bounded search and semantic review using only unchanged official tools."""
import json
from sqlglot import exp, parse_one
from .contract_agent import ContractAgent
from .review_prompts import checked_candidates, risk_flags
from .sql_utils import extract_sql, sql_key


def multiplicity_probe(obs, sql):
    """Inspect a joined COUNT's row/primary-key multiplicity on the public DB."""
    schema = next((e['observation'] for e in obs['history'] if e['tool'] == 'schema'), [])
    try:
        tree = parse_one(sql, read='sqlite')
        if not isinstance(tree, exp.Select) or not tree.args.get('joins'):
            return None
        if not any(isinstance(n, exp.Count) for n in tree.find_all(exp.AggFunc)):
            return None
        if tree.args.get('with') or any(tree.find_all(exp.Subquery)):
            return None
        first = tree.args.get('from')
        if first is None or not isinstance(first.this, exp.Table):
            return None
        table = first.this
        meta = next((t for t in schema if t['name'].lower() == table.name.lower()), None)
        keys = [c['name'] for c in meta['columns'] if c['pk']] if meta else []
        if len(keys) != 1:
            return None
        column = exp.column(keys[0], table=table.alias_or_name, quoted=True)
        probe = tree.copy()
        probe.set('expressions', [exp.alias_(exp.Count(this=exp.Star()), 'joined_rows'),
                  exp.alias_(exp.Count(this=exp.Distinct(expressions=[column])), 'distinct_base_entities')])
        for key in ('group', 'having', 'order', 'limit', 'offset', 'distinct'):
            probe.set(key, None)
        return extract_sql(probe.sql(dialect='sqlite'))
    except Exception:
        return None


class SearchAgent(ContractAgent):
    def __init__(self, config):
        super().__init__(config)
        self.second_pass = None
        self.hint_review_pending = False

    def act(self, obs):
        if self.hint_review_pending:
            self.hint_review_pending = False
            hints = next((e['observation'] for e in obs['history'] if e['tool'] == 'repair_hint'), [])
            checked = {sql_key(c['sql']) for c in checked_candidates(obs)}
            if obs['remaining']:
                for hint in hints:
                    sql = extract_sql(hint)
                    if sql and sql_key(sql) not in checked:
                        action = {'tool': 'check', 'sql': sql}
                        self.audit(obs, None, action, 'controller_hint_candidate_check')
                        return action
        return super().act(obs)

    def model_messages(self, obs):
        messages = super().model_messages(obs)
        if self.second_pass is None:
            return messages
        payload = json.loads(messages[1]['content'].split('TASK_DATA\n', 1)[1].split('\nEND_TASK_DATA', 1)[0])
        payload['independent_review'] = {
            'selected_sql': self.second_pass,
            'request': 'Independently reconstruct the required SQLite query from the question and schema. '
                       'Compare counted entities, joins, filters, grouping, projection and ordering. '
                       'If a concrete mismatch exists, check a different corrected SQL. '
                       'If the selected query is semantically right, submit its checked candidate ID. '
                       'Public example matches and multiplicity probes are evidence, not correctness labels.'}
        messages[1] = {'role': 'user', 'content': 'TASK_DATA\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\nEND_TASK_DATA\nNext action JSON:'}
        return messages

    def guard(self, obs, proposal):
        action, reason = super().guard(obs, proposal)
        if (action and action['tool'] == 'submit' and self.config.get('hint_review', False)
                and self.config.get('use_hints', True)
                and not any(e['tool'] == 'repair_hint' for e in obs['history'])
                and risk_flags(obs) and obs['remaining'] >= 3
                and self.completions < self.config.get('max_completions', 6)):
            self.second_pass = action['sql']
            self.hint_review_pending = True
            return {'tool': 'repair_hint'}, 'controller_hint_review'
        if (action and action['tool'] == 'submit' and self.config.get('second_pass', False)
                and self.second_pass is None and risk_flags(obs) and obs['remaining'] > 0
                and self.completions < self.config.get('max_completions', 6)):
            self.second_pass = action['sql']
            if self.config.get('multiplicity_probe', False) and obs['remaining'] >= 2:
                probe = multiplicity_probe(obs, action['sql'])
                used = {sql_key(e['sql']) for e in obs['history'] if e['tool'] == 'execute'}
                if probe and sql_key(probe) not in used:
                    return {'tool': 'execute', 'sql': probe}, 'controller_multiplicity_probe'
            return None, 'independent_semantic_review'
        return action, reason

    def fallback(self, obs):
        # New evidence is obtained only while another model decision is affordable.
        if (self.config.get('recover_retrieve', True) and self.rejections
                and self.completions < self.config.get('max_completions', 6)
                and obs['remaining'] >= 2 and not any(e['tool'] == 'retrieve' for e in obs['history'])):
            return {'tool': 'retrieve', 'query': obs['task']['question'][:2000]}
        return super().fallback(obs)
