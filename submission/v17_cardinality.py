"""Repair counting units and unused join leaves using the observed schema and question."""
from functools import lru_cache
import json
from pathlib import Path
import re

from sqlglot import exp

from sql_repair_agent.pool_edits_v8 import _bindings, _column, _fk_edges, _parse_query, _resolve, _tokens
from v15_rescue import normalized

FAMILIES = ("count_child", "count_entity", "unused_leaf")
ASSET = Path(__file__).parent / "configs/v17_cardinality.json"
COUNT_REQUEST = re.compile(r"\b(?:how many|(?:total )?number of|count of)\s+(?:different |distinct |unique )?([a-z_ -]+)", re.I)


def primary_key(table):
    keys = [c["name"] for c in table["columns"] if c.get("pk")]
    return keys[0] if len(keys) == 1 else None


def counted_alias(question, bindings):
    match = COUNT_REQUEST.search(question)
    if not match:
        return None
    words = re.split(r"\b(?:are|is|have|has|by|of|from|in|with|who|whose|which|that|for|on|and|per|at)\b",
                     match[1].casefold(), maxsplit=1)[0].split()[:4]
    terms = _tokens(" ".join(words))
    evidence = [(len(_tokens(t["name"]) & terms), key) for key, (_,t) in bindings.items()]
    best = max(n for n, _ in evidence)
    winners = [k for n, k in evidence if n == best]
    return winners[0] if best and len(winners) == 1 else None


def join_edge(join, bindings):
    on = join.args.get("on")
    if not isinstance(on, exp.EQ) or not all(isinstance(c, exp.Column) for c in (on.this, on.expression)):
        return None
    cols = [_resolve(c, bindings) for c in (on.this, on.expression)]
    if not all(c is not None for c in cols):
        return None
    a, b = (c.table.casefold() for c in cols)
    if a == b:
        return None
    valid = {(cols[0].name.casefold(), cols[1].name.casefold())}
    if not any((x.casefold(), y.casefold()) in valid for x, y in _fk_edges(bindings[a][1], bindings[b][1])):
        return None
    return a, b


def repairs(sql, schema, question):
    tree = _parse_query(sql)
    if not isinstance(tree, exp.Select) or len(list(tree.find_all(exp.Select))) != 1:
        return
    joins = tree.args.get("joins", [])
    binds = _bindings(tree, {t["name"].casefold(): t for t in schema})
    if not joins or not binds or any(j.args.get("using") or not join_edge(j, binds) for j in joins):
        return
    alias = counted_alias(question, binds)
    counts = list(tree.find_all(exp.Count))
    if alias and counts:
        node = counts[0]
        if (len(counts) == 1 and node in [p.unalias() for p in tree.expressions]
                and (isinstance(node.this, exp.Star) or isinstance(node.this, exp.Column))):
            target, table = binds[alias]
            pk = primary_key(table)
            if pk is not None:
                left_child = any(j.this.alias_or_name.casefold() == alias and j.side.upper()=="LEFT" for j in joins)
                if left_child and isinstance(node.this, exp.Star):
                    changed = tree.copy()
                    changed.find(exp.Count).set("this", _column(pk, target))
                    yield "count_child", changed.sql(dialect="sqlite")
                changed = tree.copy()
                changed.find(exp.Count).set("this", exp.Distinct(expressions=[_column(pk, target)]))
                yield "count_entity", changed.sql(dialect="sqlite")
    if any(isinstance(x.unalias(), exp.Star) or
           (isinstance(x.unalias(), exp.Column) and x.unalias().is_star) for x in tree.expressions):
        return
    qwords = _tokens(question)
    sources = list(binds)
    for idx, join in enumerate(joins):
        if join.side and join.side.upper() != "LEFT":
            continue
        keys = join_edge(join, binds)
        leaves = [join.this.alias_or_name.casefold()]
        if idx == 0 and len(joins) == 1 and not join.side:
            leaves.append(sources[0])
        for leaf in leaves:
            peer = next(k for k in keys if k != leaf) if leaf in keys else None
            if not peer:
                continue
            keywords = _tokens(binds[leaf][1]["name"]) - _tokens(binds[peer][1]["name"])
            if not keywords or keywords & qwords:
                continue
            changed = tree.copy()
            cjoins = changed.args["joins"]
            if leaf == sources[0]:
                source = changed.args.get("from") or changed.args.get("from_")
                source.set("this", cjoins[0].this.copy())
            changed.set("joins", [j for i, j in enumerate(cjoins) if i != idx])
            bound = [_resolve(col, binds) for col in changed.find_all(exp.Column)]
            if any(col is None or col.table.casefold() == leaf for col in bound):
                continue
            # Pure COUNT(*) has no field to identify its counting unit.
            if not bound and alias != peer:
                continue
            yield "unused_leaf", changed.sql(dialect="sqlite")


@lru_cache(maxsize=1)
def matched_families():
    return frozenset(json.loads(ASSET.read_text())["matched_families"])


class Tail:
    def __init__(self, families=None, matched=None):
        self.families = FAMILIES if families is None else families
        self.matched = matched_families() if matched is None else matched
        self.queue = None
        self.pending = None
        self.failed = False

    def act(self, obs, incumbent):
        keep = {"tool":"submit","sql":incumbent}
        history = obs["history"]
        if self.pending:
            last = history[-1]
            assert last["tool"] == "check" and last["sql"] == self.pending
            if last["observation"]["matches_example"]:
                return {"tool":"submit","sql":self.pending}
            self.pending = None
        if self.queue is None:
            checked = [e for e in history if e["tool"] == "check"]
            original = next((e for e in checked if e["sql"] == incumbent), None)
            if original is None:
                return keep
            self.failed = not original["observation"]["matches_example"]
            if self.failed and any(e["observation"]["matches_example"] for e in checked):
                return keep
            schema = next((e["observation"] for e in history if e["tool"]=="schema"), [])
            if not schema:
                return keep
            seen = {normalized(incumbent)}
            queue = []
            for family, candidate in repairs(incumbent, schema, obs["task"]["question"]):
                key = normalized(candidate)
                if (key not in seen and len(candidate.encode()) <= 16000 and family in self.families
                        and (self.failed or family in self.matched)):
                    seen.add(key)
                    queue.append((key,candidate))
            self.queue = iter(queue)
        previous = {normalized(e["sql"]): e for e in history if e["tool"]=="check"}
        for key, candidate in self.queue:
            if key in previous:
                if previous[key]["observation"]["matches_example"]:
                    return {"tool":"submit","sql":previous[key]["sql"]}
                continue
            if obs["remaining"]:
                self.pending = candidate
                return {"tool":"check","sql":candidate}
            if self.failed:
                return {"tool":"submit","sql":candidate}
        return keep


def create_agent(checkpoint=None):
    from v15_agent import Agent, create_agent as create_v15
    return Agent(create_v15(), [Tail().act])
