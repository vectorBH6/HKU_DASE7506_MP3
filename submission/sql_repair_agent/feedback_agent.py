"""Evidence-rich candidate review using the same bounded model and official tools."""
import json
from sqlglot import exp, parse_one
from sqlglot.optimizer.qualify import qualify
from .recovery_agent import RecoveryAgent
from .review_prompts import checked_candidates

INSTRUCTION = (
    ' SQL_DIAGNOSTICS are deterministic syntax/schema summaries, not correctness labels. '
    'Compare the counted entity, join multiplicity, filters, grouping and projection order '
    'with the question and expected columns. A correct query can have risk flags. '
    'Do not add DISTINCT, joins or conditions without evidence from the question. '
    'Use execution errors to repair the offending identifier or expression. '
    'For an ambiguous data value or join multiplicity, a useful read-only execute probe is allowed. '
    'Submit a checked candidate only after comparing its meaning with the question; '
    'never prefer a candidate merely because it is first or last.'
)


def schema_from_observation(obs):
    return next((e['observation'] for e in obs['history'] if e['tool'] == 'schema'), [])


def sql_diagnostics(sql, schema):
    try:
        tree = parse_one(sql, read='sqlite')
    except Exception as exc:
        return {'parse_error': str(exc)[:240]}
    selects = list(tree.find_all(exp.Select))
    outer = tree if isinstance(tree, exp.Select) else (selects[0] if selects else None)
    diagnostics = {
        'tables': sorted({t.name for t in tree.find_all(exp.Table)}),
        'projection': [e.sql(dialect='sqlite') for e in outer.expressions] if outer else [],
        'aggregates': [e.sql(dialect='sqlite') for e in tree.find_all(exp.AggFunc)],
        'group_by': [e.sql(dialect='sqlite') for e in tree.find_all(exp.Group)],
        'filters': [e.sql(dialect='sqlite') for e in tree.walk() if isinstance(e, (exp.Where, exp.Having))],
        'joins': [e.sql(dialect='sqlite') for e in tree.find_all(exp.Join)],
        'distinct': bool(outer and outer.args.get('distinct')),
    }
    if schema:
        mapping = {t['name']: {c['name']: c['type'] or 'UNKNOWN' for c in t['columns']} for t in schema}
        try:
            qualify(tree.copy(), dialect='sqlite', schema=mapping, identify=False,
                    quote_identifiers=False, expand_stars=False, validate_qualify_columns=True)
            diagnostics['schema_binding'] = 'resolved'
        except Exception as exc:
            # Unsupported SQLite constructs/ambiguous names are warnings only.
            known = {c['name'].lower() for t in schema for c in t['columns']}
            quoted = [c.name for c in tree.find_all(exp.Column) if not c.table
                      and c.this.args.get('quoted') and c.name.lower() not in known]
            if quoted:
                diagnostics['schema_binding'] = 'SQLite double-quoted identifier/value ambiguity; rely on actual execution'
                diagnostics['quoted_tokens'] = quoted[:4]
            else:
                diagnostics['schema_binding_warning'] = str(exc)[:240]
    diagnostics['flags'] = []
    if diagnostics['joins'] and any(isinstance(e, exp.Count) and isinstance(e.this, exp.Star) for e in tree.find_all(exp.Count)):
        diagnostics['flags'].append('COUNT_STAR_OVER_JOIN: check counted entity and duplicate rows')
    if outer and outer.args.get('limit'):
        diagnostics['limit'] = outer.args['limit'].sql(dialect='sqlite')
    if outer and outer.args.get('order'):
        diagnostics['order'] = outer.args['order'].sql(dialect='sqlite')
    for key,value in list(diagnostics.items()):
        if isinstance(value,list): diagnostics[key]=[str(item)[:220] for item in value[:8]]
    return diagnostics


def enrich_messages(messages, obs):
    schema = schema_from_observation(obs)
    raw = messages[1]['content'].split('TASK_DATA\n', 1)[1].split('\nEND_TASK_DATA', 1)[0]
    payload = json.loads(raw)
    payload['sql_diagnostics'] = {c['id']: sql_diagnostics(c['sql'], schema) for c in checked_candidates(obs)}
    result = [dict(m) for m in messages]
    result[0]['content'] += INSTRUCTION
    result[1]['content'] = 'TASK_DATA\n' + json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\nEND_TASK_DATA\nNext action JSON:'
    return result


class FeedbackAgent(RecoveryAgent):
    def __init__(self, config):
        super().__init__(config)
        self.model_queries = []

    def audit(self, obs, proposal, action, reason):
        if action and action.get('tool') == 'check' and reason == 'model_query':
            self.model_queries.append(action['sql'])
        super().audit(obs, proposal, action, reason)

    def best(self, obs):
        if self.config.get('prefer_model_candidate', False):
            from .sql_utils import sql_key
            candidates = checked_candidates(obs)
            for sql in reversed(self.model_queries):
                choice = next((c for c in candidates if sql_key(c['sql']) == sql_key(sql)
                               and c['feedback'].get('matches_example')), None)
                if choice is not None:
                    return choice
        return super().best(obs)

    def model_messages(self, obs):
        return enrich_messages(super().model_messages(obs), obs)
