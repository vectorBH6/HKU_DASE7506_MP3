"""Generate relation-scope alternatives for an incumbent with a recorded public failure."""
import re

from sqlglot import exp

from sql_repair_agent.pool_edits_v8 import _bindings, _column, _parse_query, _resolve, build_pool
from sql_repair_agent.pool_rank_v8 import rank_scores
from v14_terminal import create_agent as create_v14


def normalized(sql):
    tree = _parse_query(sql)
    return tree.sql(dialect="sqlite", normalize=True) if tree is not None else sql.strip()


def relation_edits(sql, question, schema, feedback, example):
    tree = _parse_query(sql)
    if not isinstance(tree, exp.Select):
        return
    where = tree.args.get("where")
    if not where:
        return
    terms = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
    bindings = _bindings(tree, {table["name"].casefold(): table for table in schema})
    joins = tree.args.get("joins") or []
    source = tree.args.get("from") or tree.args.get("from_")
    if not bindings or not source:
        return
    anchor = source.this.alias_or_name.casefold()

    def owners(expression):
        resolved = [_resolve(col, bindings) for col in expression.find_all(exp.Column)]
        return {col.table.casefold() if col is not None else None for col in resolved}

    # A required inner join can drop an entity before either arm of the OR is tested.
    if joins and re.search(r"\bor\b", question, re.I) and any(
            anchor in owners(term) and len(owners(term)) > 1 for term in where.find_all(exp.Or)):
        if all((not j.args.get("kind") or j.args["kind"].upper() == "INNER")
               and not j.args.get("side") and j.args.get("on") for j in joins):
            changed = tree.copy()
            for join in changed.args["joins"]:
                join.set("kind", None)
                join.set("side", "LEFT")
            yield "outer_disjunction", changed.sql(dialect="sqlite")
    # Row-level inequality on a many-to-many join does not exclude entities with
    # both the forbidden relation and an allowed relation.
    negative = re.search(r"\b(not|never|without|except|excluding|other than|no)\b|n.t\b", question, re.I)
    simple = not any(tree.args.get(k) for k in ("limit", "order", "having", "group", "offset"))
    if joins and negative and simple and all(isinstance(p.unalias(), exp.Column) and owners(p) == {anchor}
                                              for p in tree.expressions):
        pk = [col["name"] for col in bindings[anchor][1]["columns"] if col.get("pk")]
        if len(pk) == 1 and "v15_entity" not in bindings:
            for term in terms:
                if not isinstance(term, exp.NEQ) or owners(term) in ({anchor}, set()) or None in owners(term):
                    continue
                rest = [t for t in terms if t is not term]
                if not all(owners(t) == {anchor} for t in rest):
                    continue
                def outer_ref(n):
                    return _column(n.name, "v15_entity") if isinstance(n, exp.Column) else n
                sub = tree.copy()
                sub.set("expressions", [exp.Literal.number(1)])
                sub.set("distinct", None)
                sub.set("where", None)
                sub = sub.where(exp.EQ(this=_column(pk[0], bindings[anchor][0]),
                                       expression=_column(pk[0], "v15_entity")),
                                exp.EQ(this=term.this.copy(), expression=term.expression.copy()))
                table = source.this.copy()
                table.set("alias", exp.TableAlias(this=exp.to_identifier("v15_entity")))
                changed = exp.select(*(p.transform(outer_ref) for p in tree.expressions)).from_(table)
                changed = changed.where(*(t.transform(outer_ref) for t in rest), exp.Not(this=exp.Exists(this=sub)))
                if tree.args.get("distinct"):
                    changed.set("distinct", tree.args["distinct"].copy())
                yield "exclude_entity", changed.sql(dialect="sqlite")

    if (len(terms) < 2 or not feedback.get("ok") or feedback["row_count"] >= example["row_count"]
            or tree.find(exp.AggFunc)):
        return
    for index, term in enumerate(terms):
        literals = [str(x.this).casefold() for x in term.find_all(exp.Literal)]
        if not literals or any(v.strip("%") in question.casefold() for v in literals):
            continue
        changed = tree.copy()
        changed.set("where", None)
        changed = changed.where(*(t.copy() for i, t in enumerate(terms) if i != index))
        yield "unsupported_conjunct", changed.sql(dialect="sqlite")


def options(obs, incumbent):
    history = obs["history"]
    checks = [e for e in history if e["tool"] == "check"]
    schema = next((e["observation"] for e in history if e["tool"] == "schema"), [])
    hints = next((e["observation"] for e in history if e["tool"] == "repair_hint"), [])
    checked = {normalized(e["sql"]) for e in checks}
    observed = {e["sql"]: e["observation"] for e in checks}
    proposals = []
    for sql in dict.fromkeys([incumbent, *hints, *(e["sql"] for e in reversed(checks))]):
        for family, new_sql in relation_edits(sql, obs["task"]["question"], schema, observed.get(sql, {}),
                                              obs["example"]):
            proposals.append((family, new_sql))
    pool = build_pool(obs, schema, hints)
    scores = rank_scores(obs, schema, pool, "check")
    proposals.extend(("pool", p["sql"]) for _, p in sorted(zip(scores, pool), key=lambda x: -x[0]))
    for family, query in proposals:
        if len(query.encode()) > 16000 or _parse_query(query) is None:
            continue
        key = normalized(query)
        if key not in checked:
            checked.add(key)
            yield family, query


class RescuePhase:
    def __init__(self, allow_free=True):
        self.allow_free = allow_free
        self.queue = None
        self.pending = None

    def action(self, obs, incumbent):
        keep = {"tool": "submit", "sql": incumbent}
        if self.queue is None:
            checks = [e for e in obs["history"] if e["tool"] == "check"]
            if (any(e["observation"]["matches_example"] for e in checks) or
                    not any(e["sql"] == incumbent and e["observation"]["matches_example"] is False for e in checks)):
                return keep
            self.queue = iter(options(obs, incumbent))
        if self.pending is not None:
            event = obs["history"][-1]
            assert event["tool"] == "check" and event["sql"] == self.pending
            if event["observation"]["matches_example"]:
                return {"tool": "submit", "sql": self.pending}
            self.pending = None
        candidate = next(self.queue, None)
        if candidate is None:
            return keep
        if obs["remaining"]:
            self.pending = candidate[1]
            return {"tool": "check", "sql": self.pending}
        return {"tool": "submit", "sql": candidate[1]} if self.allow_free else keep


class RescueAgent:
    def __init__(self, base, allow_free=True):
        self.base = base
        self.phase = RescuePhase(allow_free)
        self.incumbent = None

    def act(self, obs):
        if self.incumbent is None:
            action = self.base.act(obs)
            if action["tool"] != "submit":
                return action
            self.incumbent = action["sql"]
        return self.phase.action(obs, self.incumbent)


def create_agent(checkpoint=None):
    return RescueAgent(create_v14(checkpoint))
