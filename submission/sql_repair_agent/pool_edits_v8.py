"""Generate candidate SQL from a task's public observations and schema."""
from collections import Counter, defaultdict, deque
from itertools import permutations
import re

import sqlglot
from sqlglot import exp


AGGREGATES = (exp.Count, exp.Avg, exp.Sum, exp.Min, exp.Max)
FAMILIES = ('projection', 'fk_join', 'grouping', 'relation', 'aggregate', 'quantifier', 'distinct')
MAX_EDITS = 48


def _tokens(text):
    words = re.findall(r'[a-z]+', re.sub(r'([a-z])([A-Z])', r'\1 \2', text).lower())
    return {w[:-3] + 'y' if w.endswith('ies') and len(w) > 4 else
            w[:-1] if w.endswith('s') and not w.endswith('ss') and len(w) > 3 else w for w in words}


def _parse_query(sql):
    try:
        statements = sqlglot.parse(sql, read='sqlite')
    except (ValueError, RecursionError, sqlglot.errors.SqlglotError):
        return None
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
        return None
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command, exp.Into)
    return None if any(isinstance(n, forbidden) for n in statements[0].walk()) else statements[0]


def _text(node):
    return node.sql(dialect='sqlite', normalize=True)


def _column(name, alias):
    return exp.column(name, table=alias, quoted=True)


def _bindings(tree, catalog):
    source = tree.args.get('from') or tree.args.get('from_')
    if not source or tree.args.get('with') or tree.args.get('with_'):
        return {}
    expressions = [source.this] + [j.this for j in tree.args.get('joins', [])]
    if any(not isinstance(t, exp.Table) or t.db or t.catalog for t in expressions):
        return {}
    result = {}
    for table in expressions:
        name, alias = table.name.casefold(), table.alias_or_name
        if name not in catalog or alias.casefold() in result:
            return {}
        result[alias.casefold()] = (alias, catalog[name])
    return result


def _resolve(column, bindings):
    """Return one visible schema column or decline when aliases are ambiguous."""
    matches = []
    for alias, table in bindings.values():
        if column.table and column.table.casefold() != alias.casefold():
            continue
        for field in table.get('columns', []):
            if field['name'].casefold() == column.name.casefold():
                matches.append(_column(field['name'], alias))
    return matches[0] if len(matches) == 1 else None


def _header_expression(label, bindings):
    try:
        projection = sqlglot.parse_one('SELECT ' + label, read='sqlite')
    except (ValueError, RecursionError, sqlglot.errors.SqlglotError):
        return None
    if not isinstance(projection, exp.Select) or len(projection.expressions) != 1:
        return None
    if any(value for name, value in projection.args.items() if name != 'expressions'):
        return None
    item = projection.expressions[0]
    if isinstance(item, exp.Column):
        return _resolve(item, bindings)
    if type(item) not in AGGREGATES:
        return None
    argument = item.this
    if isinstance(argument, exp.Star) and isinstance(item, exp.Count):
        return exp.Count(this=exp.Star())
    distinct = isinstance(argument, exp.Distinct)
    if distinct:
        if len(argument.expressions) != 1:
            return None
        argument = argument.expressions[0]
    if not isinstance(argument, exp.Column):
        return None
    resolved = _resolve(argument, bindings)
    if resolved is None:
        return None
    return type(item)(this=exp.Distinct(expressions=[resolved]) if distinct else resolved)


def _project(tree, labels, bindings):
    if not labels or len(labels) > 12:
        return None
    # Numeric GROUP/ORDER ordinals and alias references must keep their meaning.
    for clause in (tree.args.get('group'), tree.args.get('order')):
        if clause and any(isinstance(node, exp.Literal) and not node.is_string
                          for node in clause.walk()):
            return None
    output = []
    for label in labels:
        k = label.casefold()
        direct = [item for item in tree.expressions
                  if k in {item.alias_or_name.casefold(), _text(item.unalias()).casefold()}]
        if len(direct) == 1:
            output.append(direct[0].copy())
            continue
        bound = _header_expression(label, bindings)
        if bound is None:
            return None
        # Keep existing aliases for the same expression. A bare visible name is
        # resolved using the schema before comparing it to a qualified column.
        def scoped_key(item):
            original = item.unalias().copy()
            for c in list(original.find_all(exp.Column)):
                replacement = _resolve(c, bindings)
                if replacement is not None:
                    if c is original:
                        original = replacement
                    else:
                        c.replace(replacement)
            return _text(original)
        same = [x for x in tree.expressions if scoped_key(x) == _text(bound)]
        output.append(same[0].copy() if len(same) == 1 else bound)
    lost_aliases = {x.alias.casefold() for x in tree.expressions if x.alias}
    lost_aliases -= {x.alias.casefold() for x in output if x.alias}
    for clause in (tree.args.get('order'), tree.args.get('having'), tree.args.get('group')):
        if clause and any(not c.table and c.name.casefold() in lost_aliases for c in clause.find_all(exp.Column)):
            return None
    if [_text(x) for x in output] == [_text(x) for x in tree.expressions]:
        return None
    changed = tree.copy()
    changed.set('expressions', output)
    return changed


def _fk_edges(left, right):
    """Yield directed pairs of columns in left,right order."""
    for origin, target, reverse in ((left, right, False), (right, left, True)):
        for fk in origin.get('foreign_keys', []):
            if str(fk.get('table', '')).casefold() != target['name'].casefold():
                continue
            source, dest = fk.get('from'), fk.get('to')
            if source and not dest:
                pk = [c['name'] for c in target['columns'] if c.get('pk')]
                dest = pk[0] if len(pk) == 1 else None
            if source and dest:
                yield (dest, source) if reverse else (source, dest)


def _join_edits(tree, bindings, question_tokens):
    aliases = list(bindings.values())
    joins = tree.args.get('joins') or []
    for index, join in enumerate(joins):
        on = join.args.get('on')
        if not isinstance(on, exp.EQ) or not all(isinstance(x, exp.Column) for x in (on.left, on.right)):
            continue
        alias, table = aliases[index + 1]
        edges = [(peer, a, b) for peer, parent in aliases[:index + 1] for a, b in _fk_edges(parent, table)]
        if len(edges) != 1:
            continue
        peer, left, right = edges[0]
        good = {(peer.casefold(), left.casefold()), (alias.casefold(), right.casefold())}
        actual = {(c.table.casefold(), c.name.casefold()) for x in (on.left, on.right)
                  if (c := _resolve(x, bindings)) is not None}
        if actual != good:
            current = {(x.table.casefold(), x.name.casefold()) for x in (on.left, on.right)}
            if current != good:
                changed = tree.copy()
                changed.args['joins'][index].set('on', exp.EQ(this=_column(left, peer), expression=_column(right, alias)))
                yield 'fk_join', changed
        # Only prune a terminal join under a global aggregate whose explicit
        # inputs and other clauses refer exclusively to the retained tables.
        if index != len(joins) - 1 or tree.args.get('group') or tree.args.get('having'):
            continue
        if join.args.get('side') or not tree.expressions or not all(type(x.unalias()) in AGGREGATES for x in tree.expressions):
            continue
        if _tokens(table['name']) & question_tokens:
            continue
        stripped = tree.copy()
        stripped.set('joins', stripped.args.get('joins', [])[:-1])
        refs = list(stripped.find_all(exp.Column))
        if (not refs or any(isinstance(x, exp.Star) for x in stripped.walk())
                or any((bound := _resolve(c, bindings)) is None or bound.table.casefold() == alias.casefold() for c in refs)):
            continue
        yield 'aggregate', stripped


def _group_edit(tree, question):
    if not re.search(r'\b(each|per|for every|grouped by|broken down by)\b', question, re.I):
        return None
    group_keys = [x.unalias().copy() for x in tree.expressions if isinstance(x.unalias(), exp.Column)]
    if not group_keys or not any(type(x.unalias()) in AGGREGATES for x in tree.expressions):
        return None
    if len(group_keys) + sum(type(x.unalias()) in AGGREGATES for x in tree.expressions) != len(tree.expressions):
        return None
    changed = tree.copy()
    changed.set('group', exp.Group(expressions=group_keys))
    return changed


def _relation_edits(tree, bindings, catalog, question):
    if len(bindings) != 1 or tree.args.get('group') or tree.args.get('having'):
        return
    trigger = re.search(r'\b(?:without|with no|do(?:es)? not have|have no|has no|'
                        r'at least one|one or more|with any)\b', question, re.I)
    if not trigger:
        return
    absent = trigger.group().casefold() in {'without', 'with no', 'do not have', 'does not have', 'have no', 'has no'}
    alias, base = next(iter(bindings.values()))
    after = _tokens(question[trigger.end():])
    for target in catalog.values():
        topic = _tokens(target['name'])
        if (not topic or not topic.issubset(after) or target is base
                or not (edges := list(_fk_edges(base, target))) or len(edges) != 1):
            continue
        left, right = edges[0]
        inner_alias = 'related' if alias.casefold() != 'related' else 'related_row'
        target_table = exp.Table(this=exp.to_identifier(target['name'], quoted=True))
        target_table.set('alias', exp.TableAlias(this=exp.to_identifier(inner_alias, quoted=True)))
        sub = exp.select(exp.Literal.number(1)).from_(target_table)
        sub = sub.where(exp.EQ(this=_column(right, inner_alias), expression=_column(left, alias)))
        predicate = exp.Exists(this=sub)
        if absent:
            predicate = exp.Not(this=predicate)
        yield 'relation', tree.where(predicate)


def _quantifier_edit(tree, question):
    choices = {'at least': exp.GTE, 'no fewer than': exp.GTE, 'at most': exp.LTE, 'no more than': exp.LTE,
               'more than': exp.GT, 'greater than': exp.GT, 'fewer than': exp.LT, 'less than': exp.LT,
               'exactly': exp.EQ}
    numbers = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
               'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
    pattern = r'\b(' + '|'.join(choices) + r')\s+(\d+(?:\.\d+)?|' + '|'.join(numbers) + r')\b'
    matches = re.findall(pattern, question.casefold())
    if len(matches) != 1:
        return None
    cue, stated = matches[0]
    target, value = choices[cue], numbers.get(stated, stated)
    changed = tree.copy()
    comparisons = []
    for clause in (changed.args.get('where'), changed.args.get('having')):
        if clause:
            comparisons.extend(n for n in clause.walk() if isinstance(n, (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ))
                               and isinstance(n.right, exp.Literal) and not n.right.is_string and n.right.this == value)
    if len(comparisons) != 1 or isinstance(comparisons[0], target):
        return None
    current = comparisons[0]
    current.replace(target(this=current.left.copy(), expression=current.right.copy()))
    return changed


def _distinct_edit(tree, question):
    if not re.search(r'\b(distinct|unique|different)\b', question, re.I) or tree.args.get('group'):
        return None
    if len(tree.expressions) == 1 and isinstance(tree.expressions[0].unalias(), exp.Count):
        changed = tree.copy()
        count = changed.expressions[0].unalias()
        if not isinstance(count.this, exp.Column):
            return None
        count.set('this', exp.Distinct(expressions=[count.this.copy()]))
        return changed
    if tree.args.get('distinct') or any(isinstance(x, exp.AggFunc) for x in tree.walk()):
        return None
    changed = tree.copy()
    changed.set('distinct', exp.Distinct())
    return changed


def _joined_relation(tree, bindings, question):
    joins = tree.args.get('joins') or []
    if len(bindings) != 2 or len(joins) != 1 or joins[0].args.get('side'):
        return None
    cue = re.search(r'\b(without|with no|do(?:es)? not have|have no|has no|at least one|one or more|with any)\b',
                    question, re.I)
    if not cue:
        return None
    (outer, first), (inner, second) = list(bindings.values())
    edges = list(_fk_edges(first, second))
    if len(edges) != 1 or not _tokens(second['name']).issubset(_tokens(question[cue.end():])):
        return None
    left, right = edges[0]
    on = joins[0].args.get('on')
    if not isinstance(on, exp.EQ) or not all(isinstance(c, exp.Column) for c in (on.left, on.right)):
        return None
    def owners(node):
        return {c.table.casefold() if (c := _resolve(ref, bindings)) is not None else None
                for ref in node.find_all(exp.Column)}
    def qualified(node):
        return node.transform(lambda n: _resolve(n, bindings) if isinstance(n, exp.Column) else n)
    if owners(on) != {outer.casefold(), inner.casefold()}:
        return None
    for node in [*tree.expressions, tree.args.get('order'), tree.args.get('group'), tree.args.get('having')]:
        if node is not None and not owners(node).issubset({outer.casefold()}):
            return None
    if any(isinstance(p.unalias(), exp.Star) for p in tree.expressions):
        return None
    inner_clauses, outer_clauses = [], []
    where = tree.args.get('where')
    if where:
        for term in where.this.flatten() if isinstance(where.this, exp.And) else [where.this]:
            references = owners(term)
            if references.issubset({outer.casefold()}):
                outer_clauses.append(term.copy())
            elif references == {inner.casefold()} and not term.find(exp.Null):
                inner_clauses.append(qualified(term))
            else:
                return None
    sub = exp.select(exp.Literal.number(1)).from_(joins[0].this.copy())
    sub = sub.where(exp.EQ(this=_column(left, outer), expression=_column(right, inner)), *inner_clauses)
    predicate = exp.Exists(this=sub)
    if cue.group().casefold() in {'without', 'with no', 'do not have', 'does not have', 'have no', 'has no'}:
        predicate = exp.Not(this=predicate)
    changed = tree.copy()
    changed.set('where', None)
    changed.set('joins', [])
    return changed.where(*outer_clauses, predicate)


def _observed_projection(tree, expected, actual, ordered):
    if (not actual.get('ok') or actual.get('matches_example') or expected.get('truncated') or actual.get('truncated')
            or not 2 <= len(tree.expressions) <= 4 or actual.get('row_count') != expected.get('row_count')):
        return None
    rows = actual.get('rows') or []
    wanted = expected.get('rows') or []
    width = len(tree.expressions)
    if not rows or not wanted or any(len(r) != width for r in [*rows, *wanted]):
        return None
    if any(clause and any(isinstance(x, exp.Literal) and not x.is_string for x in clause.walk())
           for clause in (tree.args.get('group'), tree.args.get('order'))):
        return None
    def signature(values):
        tuples = [tuple(row) for row in values]
        return tuples if ordered else Counter(tuples)
    target = signature(wanted)
    if signature(rows) == target:
        return None
    orders = [order for order in permutations(range(width))
              if signature([[r[i] for i in order] for r in rows]) == target]
    if len(orders) != 1:
        return None
    changed = tree.copy()
    changed.set('expressions', [tree.expressions[i].copy() for i in orders[0]])
    return changed


def _nullable_membership(tree, catalog, question):
    """Guard NOT IN on a nullable column, without changing nested alias scopes."""
    if not re.search(r'\b(no|not|never|without|except|none)\b', question, re.I):
        return None
    tree = tree.copy()
    where = tree.args.get('where')
    if not where:
        return None
    outer = _bindings(tree, catalog)
    if not outer:
        return None
    guards = []
    for neg in where.find_all(exp.Not):
        membership = neg.this
        if neg.find_ancestor(exp.Select) is not tree or not isinstance(membership, exp.In):
            continue
        sub = membership.args.get('query')
        if not isinstance(sub, exp.Subquery) or not isinstance(sub.this, exp.Select):
            continue
        inner = sub.this
        if (len(inner.expressions) != 1 or not isinstance(inner.expressions[0], exp.Column)
                or any(n is not inner and isinstance(n, (exp.Query, exp.Subquery)) for n in inner.walk())):
            continue
        bindings = _bindings(inner, catalog)
        output = _resolve(inner.expressions[0], bindings)
        if output is None or len(bindings) != 1 or inner.args.get('limit') or inner.args.get('group'):
            continue
        table = bindings[output.table.casefold()][1]
        field = next(c for c in table['columns'] if c['name'].casefold() == output.name.casefold())
        if field.get('not_null') or field.get('pk'):
            continue
        guards.append((neg, output))
    if len(guards) != 1:
        return None
    old, col = guards[0]
    corrected = old.copy()
    inner = corrected.this.args['query'].this
    predicate = exp.Not(this=exp.Is(this=col, expression=exp.Null()))
    if any(_text(n) == _text(predicate) for n in inner.walk()):
        return None
    corrected.this.args['query'].set('this', inner.where(predicate))
    old.replace(corrected)
    return tree


def _fix_qualifiers(tree, bindings):
    replacements = {}
    for c in tree.find_all(exp.Column):
        if not c.table or _resolve(c, bindings) is not None or c.db or c.catalog:
            continue
        for_alias = {k: pair for k, pair in bindings.items() if pair[1]['name'].casefold() == c.table.casefold()}
        bare = exp.column(c.name)
        bound = _resolve(bare, for_alias) if for_alias else None
        if bound is None:
            bound = _resolve(bare, bindings)
        if bound is not None:
            replacements[_text(c)] = bound
    if not replacements:
        return None
    return tree.transform(lambda n: replacements[_text(n)].copy()
                          if isinstance(n, exp.Column) and _text(n) in replacements else n)


def _extend_projection(tree, labels, bindings, catalog):
    """Add exactly one FK-connected table when it supplies missing output labels."""
    present = {table['name'].casefold() for _, table in bindings.values()}
    options = []
    raw_names = [str(label).casefold() for label in labels]
    for name, new_table in catalog.items():
        if name in present or name in bindings:
            continue
        fields = {c['name'].casefold() for c in new_table['columns']}
        if not any(label in fields or label.startswith(name + '.') for label in raw_names):
            continue
        edges = [(peer, left, right) for peer, table in bindings.values() for left, right in _fk_edges(table, new_table)]
        if len(edges) != 1:
            continue
        additional = dict(bindings)
        additional[name] = (new_table['name'], new_table)
        projected = _project(tree, labels, additional)
        if projected is None:
            projected = tree
        if not any(c.table.casefold() == name for p in projected.expressions for c in p.find_all(exp.Column)):
            continue
        peer, left, right = edges[0]
        target = exp.Table(this=exp.to_identifier(new_table['name'], quoted=True))
        options.append(projected.join(target, on=exp.EQ(this=_column(left, peer), expression=_column(right, new_table['name']))))
    if len(options) == 1:
        return options[0]
    return None


def _edits(tree, catalog, expected, question, actual, ordered):
    if not isinstance(tree, exp.Select):
        return
    if any(n is not tree and isinstance(n, (exp.Query, exp.Subquery)) for n in tree.walk()):
        guarded = _nullable_membership(tree, catalog, question)
        if guarded is not None:
            yield 'quantifier', guarded
        return
    labels = expected.get('columns') or []
    bindings = _bindings(tree, catalog)
    if not bindings:
        return
    scoped = _fix_qualifiers(tree, bindings)
    if scoped is not None:
        yield 'relation', scoped
    projected = _project(tree, labels, bindings)
    if projected is not None:
        yield 'projection', projected
        grouped = _group_edit(projected, question)
        if grouped is not None:
            yield 'grouping', grouped
    elif labels:
        joined = _extend_projection(tree, labels, bindings, catalog)
        if joined is not None:
            yield 'projection', joined
    if actual:
        aligned = _observed_projection(tree, expected, actual, ordered)
        if aligned is not None:
            yield 'projection', aligned
    yield from _join_edits(tree, bindings, _tokens(question))
    yield from _relation_edits(tree, bindings, catalog, question)
    joined_relation = _joined_relation(tree, bindings, question)
    if joined_relation is not None:
        yield 'relation', joined_relation
    for family, function in (('grouping', _group_edit), ('quantifier', _quantifier_edit), ('distinct', _distinct_edit)):
        changed = function(tree, question)
        if changed is not None:
            yield family, changed


def build_pool(obs, schema, hints):
    """Return raw drafts/hints, then diverse bounded edits. No DB access."""
    catalog = {t['name'].casefold(): t for t in schema or []}
    expected = obs.get('example') or {}
    checked = {e.get('sql'): e.get('observation') for e in obs.get('history', []) if e.get('tool') in ('check', 'execute')}
    question = obs.get('task', {}).get('question', '')
    rows, seeds, seen = [], [], set()
    originals = [('draft', obs['issue_sql'])]
    originals.extend((f'hint_{i}', sql) for i, sql in enumerate(hints or []))
    for source, sql in originals:
        if not isinstance(sql, str) or not sql.strip() or len(sql.encode()) > 16000:
            continue
        tree = _parse_query(sql)
        # The rollout checks the exact original bytes before requesting hints.
        # Keep that draft even if invalid; only officially parseable hints follow.
        if tree is None and source != 'draft':
            continue
        key = _text(tree) if tree is not None else sql.strip()
        if key in seen:
            continue
        seen.add(key)
        rows.append({'sql': sql, 'family': 'raw', 'source': source})
        if tree is not None:
            seeds.append((source, tree, checked.get(sql) or {}))
    queues = defaultdict(deque)
    for source, tree, actual in seeds:
        for family, changed in _edits(tree, catalog, expected, question, actual, obs.get('ordered', False)):
            key = _text(changed)
            if key in seen:
                continue
            candidate = changed.sql(dialect='sqlite')
            if len(candidate.encode()) <= 16000:
                seen.add(key)
                queues[family].append({'sql': candidate, 'family': family, 'source': source})
    target_size = len(rows) + MAX_EDITS
    while len(rows) < target_size:
        active = [queues[family] for family in FAMILIES if queues[family]]
        if not active:
            break
        for queue in active[:target_size - len(rows)]:
            rows.append(queue.popleft())
    return rows
