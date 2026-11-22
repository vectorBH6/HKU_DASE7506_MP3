"""Mine local AST rewrite examples on train and test novel SQL coverage on held-out train schemas."""
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "submission")]
from mp3_sql.common import BUDGET, MAX_SQL_BYTES, digest, load_bank, read_jsonl
from mp3_sql.environment import Episode, Tools
from mp3_sql.score import load_cases, score_document
from sql_repair_agent.sql_utils import extract_sql, sql_key


def read(path):
    return json.loads((ROOT / path).read_text())


def tree(value):
    if isinstance(value, exp.Expression):
        return ('node', type(value), tuple(sorted((k, tree(v)) for k, v in value.args.items()
            if v is not None and v != [] and k != 'comments')))
    if isinstance(value, list):
        return ('list', tuple(map(tree, value)))
    return ('value', value)


def make(t):
    if t[0] == 'node':
        return t[1](**{k: make(v) for k, v in t[2]})
    if t[0] == 'list':
        return [make(x) for x in t[1]]
    return t[1]


@lru_cache(maxsize=5000)
def parsed(sql, normalize):
    try:
        t = parse_one(sql, read='sqlite')
    except (SqlglotError, ValueError, RecursionError):
        return None
    if not isinstance(t, exp.Query):
        return None
    if normalize:
        # Align alternate formatting and source aliases only when mining examples.
        for query in t.find_all(exp.Select):
            tabs = [n for n in query.find_all(exp.Table) if n.find_ancestor(exp.Select) is query]
            names = Counter(tab.name.casefold() for tab in tabs)
            aliases = {tab.alias_or_name.casefold(): tab.name.casefold() for tab in tabs
                       if names[tab.name.casefold()] == 1}
            for col in query.find_all(exp.Column):
                if col.find_ancestor(exp.Select) is query and col.table.casefold() in aliases:
                    name = aliases[col.table.casefold()]
                    col.set('table', exp.to_identifier(name) if len(tabs) > 1 else None)
            for tab in tabs:
                if tab.alias_or_name.casefold() in aliases:
                    tab.set('alias', None)
            for item in query.expressions:
                if isinstance(item, exp.Alias):
                    item.replace(item.this.copy())
        for ident in t.find_all(exp.Identifier):
            ident.set('this', ident.name.casefold())
            ident.set('quoted', False)
    return t


def child_pairs(a, b):
    if a[0] == 'node' and b[0] == 'node' and a[1] == b[1]:
        da, db = dict(a[2]), dict(b[2])
        if da.keys() == db.keys():
            return [(da[k], db[k], k) for k in da if da[k] != db[k]]
    if a[0] == 'list' and b[0] == 'list' and len(a[1]) == len(b[1]):
        return [(x, y, '') for x, y in zip(a[1], b[1]) if x != y]
    return None


def locus(a, b, parent=None, field=''):
    parts = child_pairs(a, b)
    if parts and len(parts) == 1:
        x, y, key = parts[0]
        if x[0] == y[0] == 'list':
            inner = child_pairs(x, y)
            if inner and len(inner) == 1:
                x, y, _ = inner[0]
        if x[0] == y[0] == 'node':
            # Retain a parent expression for scalar or leaf changes.
            if x[1] not in (exp.Identifier, exp.Literal) and y[1] not in (exp.Identifier, exp.Literal):
                return locus(x, y, a[1], key)
    return a, b, parent, field


def descendants(t):
    yield t
    if t[0] == 'node':
        for _, child in t[2]:
            yield from descendants(child)
    elif t[0] == 'list':
        for child in t[1]:
            yield from descendants(child)


def rewrite_rule(a, b):
    if a == b:
        return None
    a, b, parent, field = locus(a, b)
    common = set(descendants(a)) & set(descendants(b))
    holes = {}

    def pattern(t, source, root=False):
        if t[0] == 'list':
            return ('list', tuple(pattern(c, source) for c in t[1]))
        if t[0] != 'node':
            return t
        if t in holes:
            return ('hole', holes[t], t[1])
        if source and not root and t in common:
            holes[t] = len(holes)
            return ('hole', holes[t], t[1])
        if t[1] in (exp.Identifier, exp.Literal):
            # Only zero and one may be introduced as literal constants.
            if t[1] is exp.Literal and make(t).sql() in ('0', '1'):
                return t
            if not source:
                raise ValueError('Target uses a new identifier or literal')
            holes[t] = len(holes)
            return ('hole', holes[t], t[1])
        return ('node', t[1], tuple((k, pattern(v, source)) for k, v in t[2]))

    try:
        left, right = pattern(a, True, True), pattern(b, False, True)
    except ValueError:
        return None
    # The traversal cap is for general local edits, not entire memorized queries.
    if left[0] != 'node' or left == right or not holes:
        return None
    if sum(n[0] == 'node' for r in (left, right) for n in descendants(r)) > 12:
        return None
    # Never learn a schema or value token via a primitive SQLGlot argument.
    allowed_words = {'left', 'right', 'outer', 'inner', 'cross', 'asc', 'desc', 'first', 'last', '0', '1'}
    if any(t[0] == 'value' and isinstance(t[1], str) and t[1].lower() not in allowed_words
           for r in (left, right) for t in descendants(r)):
        return None
    return parent, field, left, right


def bound(pattern, actual, slots):
    if pattern[0] == 'hole':
        _, i, typ = pattern
        if actual[0] != 'node' or actual[1] is not typ:
            return False
        return slots.setdefault(i, actual) == actual
    if pattern[0] != actual[0]:
        return False
    if pattern[0] == 'node':
        return pattern[1] is actual[1] and len(pattern[2]) == len(actual[2]) and all(
            k == j and bound(v, w, slots) for (k, v), (j, w) in zip(pattern[2], actual[2]))
    if pattern[0] == 'list':
        return len(pattern[1]) == len(actual[1]) and all(bound(v, w, slots) for v, w in zip(pattern[1], actual[1]))
    return pattern == actual


def substitute(t, slots):
    if t[0] == 'hole':
        return slots[t[1]]
    if t[0] == 'node':
        return ('node', t[1], tuple((k, substitute(v, slots)) for k, v in t[2]))
    if t[0] == 'list':
        return ('list', tuple(substitute(v, slots) for v in t[1]))
    return t


def fit(rows, pools, fit_ids, gold):
    votes = defaultdict(set)
    schemas = defaultdict(set)
    count = 0
    for row in rows:
        i = row['id']
        if i not in fit_ids:
            continue
        candidates = pools[i]
        positives = {c['sql'] for c in candidates if c['correct']} | {gold[i]}
        negatives = {c['sql'] for c in candidates if not c['correct']}
        for wrong in sorted(negatives):
            for correct in sorted(positives):
                ws, cs = parsed(wrong, True), parsed(correct, True)
                if ws is None or cs is None:
                    continue
                count += 1
                rule = rewrite_rule(tree(ws), tree(cs))
                if rule:
                    votes[rule].add(i)
                    schemas[rule].add(row['db_id'])
    keep = [r for r in votes if len(votes[r]) >= 2 and len(schemas[r]) >= 2]
    keep.sort(key=lambda r: (-len(schemas[r]), -len(votes[r]), repr(r)))
    return keep, {'fit_pairs':count, 'distinct_single_site_rules':len(votes), 'cross_schema_rules':len(keep),
                  'supports':[(len(votes[r]),len(schemas[r])) for r in keep]}


def proposals(sqls, rules, excluded, limit):
    sources = [(sql, parsed(sql, False)) for sql in sqls]
    seen = set(excluded)
    for parent, field, left, right in rules:
        for sql, original in sources:
            if original is None:
                continue
            for j, node in enumerate(original.walk()):
                if type(node) is not left[1] or type(node.parent) is not (parent or type(None)):
                    continue
                if (node.arg_key or '') != field:
                    continue
                slots = {}
                if not bound(left, tree(node), slots):
                    continue
                new = make(substitute(right, slots))
                if node.parent is not None:
                    changed = original.copy()
                    list(changed.walk())[j].replace(new)
                else:
                    changed = new
                try:
                    result = changed.sql(dialect='sqlite')
                    key = sql_key(result)
                except (SqlglotError, ValueError, RecursionError):
                    continue
                if key not in seen and len(result.encode()) <= MAX_SQL_BYTES:
                    yield result
                    seen.add(key)
                    limit -= 1
                    if not limit:
                        return

def main():
    rows, manifest, folder = load_bank('train')
    split = read('artifacts/prepared_data/v6/split.json')
    pools = {r['id']:r['candidates'] for r in read_jsonl(ROOT/'runs/v8_pool_train/train_pool.jsonl')}
    gold = read('mp3_sql/data/train_references.json')['gold_sql']
    rules, summary = fit(rows, pools, set(split['fit_ids']), gold)
    hold = [r for r in rows if r['id'] in set(split['holdout_ids'])]
    print('Fitted cross-schema rules:', summary, flush=True)
    pub = Tools(folder, manifest, cache={})
    trials = defaultdict(list)
    base_covered = {}
    remaining_slots = {}
    for row in hold:
        task = row['id']
        old = pools[task]
        base_covered[task] = any(c['correct'] for c in old)
        remaining_slots[task] = max(0, BUDGET - 3 - (len(old) - 1))
        ep = Episode(row, pub)
        ep.step({'tool':'schema'})
        ep.step({'tool':'check','sql':row['issue_sql']})
        ep.step({'tool':'repair_hint'})
        hints = ep.trace[-1]['observation']
        raw = [row['issue_sql'], *(s for h in hints if (s := extract_sql(h)))]
        # Reference V8 pool was computed exclusively from the same task's draft, schema
        # and hints; only its normalized SQL strings are used here for deduplication.
        excluded = {sql_key(c['sql']) for c in old}
        for sql in proposals(list(dict.fromkeys(raw)), rules, excluded, BUDGET-3):
            ep.step({'tool':'check','sql':sql})
            matched = ep.trace[-1]['observation']['matches_example']
            pred = {'id':task,'sql':sql,'tool_calls':len(ep.trace),
                    'trace':[*ep.trace,{'tool':'submit','sql':sql}]}
            trials[task].append((matched,pred))
        assert len(ep.trace) <= BUDGET
    # Grade only after all held-out proposals have been fixed. Non-matching
    # public results necessarily fail the course's all-three-instances criterion.
    cases, case_root = load_cases('train', folder)
    header = {'protocol':manifest['protocol'],'budget':BUDGET,'split':'train',
              'manifest_sha256':digest(folder/'manifest.json'),'bank_sha256':digest(folder/'train.jsonl')}
    passes = defaultdict(list)
    cache = {}
    for slot in range(BUDGET-3):
        chosen = [r for r in hold if slot < len(trials[r['id']]) and trials[r['id']][slot][0]]
        if not chosen:
            continue
        preds=[trials[r['id']][slot][1] for r in chosen]
        scoped={**cases,'ids':[r['id'] for r in chosen]}
        scores=score_document({**header,'predictions':preds},chosen,manifest,folder,scoped,case_root,workers=1,cache=cache)
        for record in scores['per_question']:
            if record['passed']:
                passes[record['id']].append(slot+1)
    hits={i for i,v in passes.items() if v}
    added={i for i in hits if not base_covered[i]}
    result={'fit':summary,'heldout_tasks':len(hold),'baseline_oracle':sum(base_covered.values()),
            'executed_unique_new_per_task_total':sum(map(len,trials.values())),
            'tasks_with_proposal':sum(bool(v) for v in trials.values()),
            'public_matching':sum(m for group in trials.values() for m,p in group),
            'correct_new_sql':sum(map(len,passes.values())), 'tasks_with_correct_new_sql':len(hits),
            'tasks_without_old_correct':len(hold)-sum(base_covered.values()),'incremental_covered_tasks':len(added),
            'combined_pool_oracle':sum(base_covered.values())+len(added),
            'incremental_fitting_remaining_old_pool_budget':sum(min(passes[i])<=remaining_slots[i] for i in added),
            'new_check_limit':5,'max_actions':max([p['tool_calls'] for group in trials.values() for m,p in group],default=3),
            'scope':'382 official train fit tasks; positives are training gold and officially correct V8 pool members, negatives are wrong pool members; >=2 source databases and 2 tasks. Novel edits conditioned on parent/argument, bound SQL subtrees and operators; no task-specific output identifiers, strings, or arbitrary numbers. Held out 98 other train tasks. Reference V8 pool deduplicated by SQL only. Fixed edit ordering: training database support, task support, stable rule order. Evaluate first five unseen candidates from draft and four raw hints per task on legal independent draft/schema/hint trajectories. Combined pool oracle describes separate budgeted policies, not achieved end-to-end selection. No validation or test used.'}
    paths=['scripts/probe_v10_learned_edits.py','artifacts/prepared_data/v6/split.json','runs/v8_pool_train/train_pool.jsonl','mp3_sql/data/train_references.json','mp3_sql/score.py']
    result['source_sha256']={str(p):digest(ROOT/p) for p in paths}
    out=ROOT/'reports/v10_learned_edits.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
