"""Review a full personal name compared against a single component on weak public evidence."""
import re

from sqlglot import exp

from sql_repair_agent.pool_edits_v8 import _bindings, _column, _parse_query, _resolve
from sql_repair_agent.sql_utils import sql_key


# Match an actual pair in one visible relation, including staff_first_name/staff_last_name.
NAME_ROLES = (
    ("firstname", "lastname"), ("fname", "lname"),
    ("forename", "surname"), ("givenname", "familyname"),
)


def name_pair(field, table):
    def compact(name):
        return re.sub(r"[^a-z0-9]", "", name.casefold())

    key = compact(field)
    schema = {compact(col["name"]): col["name"] for col in table["columns"]}
    for first, last in NAME_ROLES:
        for suffix in (first, last):
            if key.endswith(suffix):
                prefix = key[:-len(suffix)]
                if prefix + first in schema and prefix + last in schema:
                    return schema[prefix + first], schema[prefix + last]
    return None


def atom_values(eq, bindings):
    for col, literal in ((eq.this, eq.expression), (eq.expression, eq.this)):
        if not isinstance(col, exp.Column):
            continue
        resolved = _resolve(col, bindings)
        if resolved is None:
            continue
        if isinstance(literal, exp.Literal) and literal.is_string:
            return resolved, literal.this
        # SQLite treats an unbound double-quoted identifier as a string literal.
        if (isinstance(literal, exp.Column) and not literal.table
                and literal.this.args.get("quoted") and _resolve(literal, bindings) is None):
            return resolved, literal.name
    return None


def name_candidate(sql, question, schema):
    tree = _parse_query(sql)
    if tree is None:
        return None
    catalog = {table["name"].casefold(): table for table in schema}
    changes = []
    for where in tree.find_all(exp.Where):
        select = where.parent
        if not isinstance(select, exp.Select):
            continue
        bindings = _bindings(select, catalog)
        terms = list(where.this.flatten()) if isinstance(where.this, exp.And) else [where.this]
        for term in terms:
            if not isinstance(term, exp.EQ):
                continue
            atom = atom_values(term, bindings)
            if atom is None:
                continue
            col, value = atom
            table = bindings[col.table.casefold()][1]
            pair = name_pair(col.name, table)
            if not pair:
                continue
            words = value.split()
            if len(words) != 2 or not all(w[0].isupper() and w.replace("-", "").isalpha() for w in words):
                continue
            if not re.search(r"(?<!\w)" + r"\s+".join(map(re.escape, words)) + r"(?!\w)", question, re.I):
                continue

            sibling_terms = []
            ambiguous = False
            for other in terms:
                if other is term:
                    continue
                # A second plain component constraint can be replaced only when
                # its value is one of the same two tokens from the question.
                relevant = []
                for ref in other.find_all(exp.Column):
                    resolved = _resolve(ref, bindings)
                    if (resolved is not None and resolved.table == col.table
                            and resolved.name in pair):
                        relevant.append(resolved)
                if not relevant:
                    continue
                peer = atom_values(other, bindings) if isinstance(other, exp.EQ) else None
                if peer is None or peer[1] not in words or peer[0].table != col.table or peer[0].name not in pair:
                    ambiguous = True
                    break
                sibling_terms.append(other)
            if not ambiguous:
                fixed = [exp.EQ(this=_column(name, col.table), expression=exp.Literal.string(word))
                         for name, word in zip(pair, words)]
                keep = [other.copy() for other in terms if other is not term and
                        all(other is not sibling for sibling in sibling_terms)]
                changes.append((where, exp.and_(*keep, *fixed)))
    if len(changes) != 1:
        return None
    where, correction = changes[0]
    where.set("this", correction)
    candidate = tree.sql(dialect="sqlite")
    return candidate if len(candidate.encode()) <= 16000 else None


def terminal_action(obs, original_sql):
    submit = {"tool": "submit", "sql": original_sql}
    ex = obs["example"]
    empty = ex["row_count"] == 0
    zero = (ex["row_count"] == 1 and ex["rows"] == [[0]] and len(ex["columns"]) == 1
            and re.match(r"^count\s*\(", ex["columns"][0], re.I))
    if not (empty or zero):
        return submit
    history = obs["history"]
    if not any(e["tool"] == "check" and e["sql"] == original_sql and
               e["observation"].get("matches_example") is True for e in history):
        return submit
    schema = next((e["observation"] for e in history if e["tool"] == "schema"), [])
    candidate = name_candidate(original_sql, obs["task"]["question"], schema)
    if candidate is None:
        return submit
    key = sql_key(candidate)
    for entry in history:
        if entry["tool"] == "check" and sql_key(entry["sql"]) == key:
            if entry["observation"]["matches_example"]:
                return {"tool": "submit", "sql": entry["sql"]}
            return submit
    return {"tool": "check", "sql": candidate} if obs["remaining"] else submit

