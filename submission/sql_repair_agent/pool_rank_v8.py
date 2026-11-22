"""Train-fitted, task-local structural features for ranking checked SQL candidates."""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError


FAMILIES = ("raw", "projection", "grouping", "fk_join", "relation", "aggregate", "quantifier", "distinct")
AST_TYPES = {
    "join": exp.Join, "where": exp.Where, "group": exp.Group, "agg": exp.AggFunc,
    "distinct": exp.Distinct, "order": exp.Order, "limit": exp.Limit, "or": exp.Or,
    "and": exp.And, "subquery": exp.Subquery, "select": exp.Select, "count": exp.Count,
    "sum": exp.Sum, "avg": exp.Avg, "min": exp.Min, "max": exp.Max, "not": exp.Not,
    "set": exp.SetOperation, "having": exp.Having, "like": exp.Like,
}
WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def _terms(value):
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in WORDS.findall(str(value).casefold())}


def _coverage(xs, ys):
    return len(xs & ys) / max(len(xs), 1)


def _shape(sql):
    try:
        tree = parse_one(sql, read="sqlite")
    except (SqlglotError, ValueError, RecursionError):
        return None
    if tree is None:
        return None
    nodes = list(tree.walk())
    counts = {key: sum(isinstance(n, kind) for n in nodes) for key, kind in AST_TYPES.items()}
    tables = [t for t in nodes if isinstance(t, exp.Table)]
    columns = [c for c in nodes if isinstance(c, exp.Column)]
    outer = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    projections = outer.expressions if outer is not None else []
    names = [(p.alias_or_name or p.sql(dialect="sqlite")).casefold() for p in projections]
    identifiers = {t.name.casefold() for t in tables} | {c.name.casefold() for c in columns}
    ground = set().union(*(_terms(t) for t in identifiers))
    proj_terms = [_terms(p.sql(dialect="sqlite")) for p in projections]
    table_aliases = {t.alias_or_name.casefold(): t.name.casefold() for t in tables}
    join_keys = []
    for join in (n for n in nodes if isinstance(n, exp.Join)):
        condition = join.args.get("on")
        if condition:
            for eq in condition.find_all(exp.EQ):
                a, b = eq.this, eq.expression
                if isinstance(a, exp.Column) and isinstance(b, exp.Column) and a.table and b.table:
                    join_keys.append(frozenset(((table_aliases.get(a.table.casefold(), a.table.casefold()), a.name.casefold()),
                                                (table_aliases.get(b.table.casefold(), b.table.casefold()), b.name.casefold()))))
    return dict(counts=counts, tables={t.name.casefold() for t in tables}, columns={c.name.casefold() for c in columns},
                literals={str(n.this).casefold() for n in nodes if isinstance(n, exp.Literal)},
                identifiers=identifiers, ground=ground, proj_terms=proj_terms, names=names, join_keys=join_keys,
                size=len(nodes), star=any(isinstance(p,exp.Star) or p.is_star for p in projections))


class _Context:
    def __init__(self, obs, schema):
        self.question = _terms(obs["task"]["question"])
        self.draft = _shape(obs["issue_sql"])
        self.expected = [str(c).casefold() for c in obs["example"].get("columns", [])]
        self.expected_terms = [_terms(c) for c in self.expected]
        self.ordered = bool(obs.get("ordered"))
        self.tables = {t["name"].casefold() for t in schema}
        self.columns = {c["name"].casefold() for t in schema for c in t["columns"]}
        self.fk = {frozenset(((t["name"].casefold(), f["from"].casefold()), (f["table"].casefold(), f["to"].casefold())))
                   for t in schema for f in t["foreign_keys"] if all(f.get(k) for k in ("from","table","to"))}
        self.pk = {c["name"].casefold() for t in schema for c in t["columns"] if c.get("pk")}
        self.draft_feedback = next((e["observation"] for e in obs.get("history", [])
                         if e["tool"] == "check" and e.get("sql") == obs["issue_sql"]), {})
        self.draft_sql = obs["issue_sql"]

    def features(self, candidate):
        family = candidate.get("family", "other")
        if family not in FAMILIES:
            family = "other"
        src = candidate.get("source","other")
        if src not in ("draft", "hint_0", "hint_1"):
            src = "later_hint" if src.startswith("hint_") else "other"
        if src == "other":
            family = "other"
        match = float(self.draft_feedback.get("matches_example", False))
        error = float(bool(self.draft_feedback) and not self.draft_feedback.get("ok"))
        out = {"src_"+src: 1., "edit_"+family: 1., "source_edit_"+src+"_"+family: 1.,
               "draft_public_source_"+src: match, "draft_public_edit_"+family: match,
               "draft_error_edit_"+family: error, "draft_error": error, "draft_public": match}
        shape = self.draft if candidate["sql"] == self.draft_sql else _shape(candidate["sql"])
        if shape is None:
            out["parse_failure"] = 1.
            return out
        base = self.draft
        def fit(q):
            return _coverage(q["ground"], self.question)
        nproj = len(shape["names"])
        literals = set().union(*(_terms(x) for x in shape["literals"]))
        out.update({
            "node_size": min(shape["size"],150)/50, "star": float(shape["star"]),
            "arity_ok": float(nproj==len(self.expected)), "arity_gap": min(abs(nproj-len(self.expected)),5)/5,
            "pos_name_exact": sum(a==b for a,b in zip(shape["names"],self.expected))/max(nproj,len(self.expected),1),
            "pos_name_overlap": sum(_coverage(e,p) for p,e in zip(shape["proj_terms"],self.expected_terms))/max(nproj,len(self.expected),1),
            "any_name_overlap": sum(max((_coverage(e,p) for p in shape["proj_terms"]),default=0) for e in self.expected_terms)/max(len(self.expected),1),
            "question_id_overlap": fit(shape), "question_literal_overlap": _coverage(literals,self.question),
            "unknown_tables": min(len(shape["tables"]-self.tables),4)/4, "unknown_columns": min(len(shape["columns"]-self.columns),4)/4,
            "fk_join_fraction": sum(k in self.fk for k in shape["join_keys"])/max(len(shape["join_keys"]),1),
            "primary_columns": min(len(shape["columns"] & self.pk),4)/4,
            "sort_contract": float(self.ordered == bool(shape["counts"]["order"])),
        })
        for k,v in shape["counts"].items():
            out["ast_"+k]=min(v,4)/2
        if base is not None:
            changes = sum(shape["counts"][key]!=v for key,v in base["counts"].items())
            out.update({"ast_change_count": min(changes,10)/5, "id_keep": _coverage(base["identifiers"],shape["identifiers"]),
                        "id_added_q_ground": _coverage(shape["ground"]-base["ground"],self.question),
                        "id_removed_q_ground": _coverage(base["ground"]-shape["ground"],self.question),
                        "q_ground_delta": fit(shape)-fit(base), "size_delta": max(-1.,min((shape["size"]-base["size"])/50,1.))})
            for k,v in shape["counts"].items():
                out["ast_delta_"+k]=max(-2,min(2,v-base["counts"][k]))/2
        return out


def features(obs, schema, candidate):
    return _Context(obs, schema).features(candidate)


def feature_rows(obs, schema, candidates):
    ctx = _Context(obs, schema)
    return [ctx.features(candidate) for candidate in candidates]


@lru_cache(maxsize=1)
def _weights():
    path = Path(__file__).resolve().parents[1] / "configs" / "pool_v8_weights.json"
    return json.loads(path.read_text())


def value(row, weights):
    total = weights["bias"] + sum(weights["coefficients"].get(k,0.)*v for k,v in row.items())
    return 1/(1+math.exp(-max(-40.,min(total,40.))))


def rank_scores(obs, schema, candidates, stage):
    weights = _weights()[stage]
    return [value(row, weights) for row in feature_rows(obs,schema,candidates)]


def rank_score(obs, schema, candidate, stage):
    return value(features(obs,schema,candidate),_weights()[stage])
