"""Try training-ranked text edits only after a verified failing V10 submission."""
import json
from pathlib import Path
import re

from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from sql_repair_agent.sql_utils import sql_key
from v10_semantic import SemanticPoolAgent


def name_columns(schema):
    for tab in schema:
        cols = {re.sub("[^a-z0-9]", "", c["name"].lower()): c["name"] for c in tab["columns"]}
        first = [cols[n] for n in ("fname", "firstname", "givenname", "forename") if n in cols]
        last = [cols[n] for n in ("lname", "lastname", "surname", "familyname") if n in cols]
        if len(first) == len(last) == 1:
            yield tab["name"].casefold(), first[0], last[0]


def text_edits(sql, question, schema):
    try:
        tree = parse_one(sql, read="sqlite")
    except (ValueError, SqlglotError, RecursionError):
        return
    if tree is None:
        return
    table_names = {t.alias_or_name.casefold(): t.name.casefold() for t in tree.find_all(exp.Table)}
    for index, eq in enumerate(tree.find_all(exp.EQ)):
        column, literal = eq.this, eq.expression
        if not isinstance(column, exp.Column) or not isinstance(literal, exp.Literal) or not literal.is_string:
            continue
        v = literal.this.strip()
        if not v or "%" in v or "_" in v:
            continue
        folded = exp.Lower(this=column.copy())
        replacements = [
            ("nocase", exp.EQ(this=folded.copy(), expression=exp.Literal.string(v.lower()))),
            ("contains", exp.Like(this=folded.copy(), expression=exp.Literal.string("%" + v.lower() + "%"))),
        ]
        for m in re.finditer(r"(?<!\w)([\w]+)\s+" + re.escape(v) + r"(?!\w)", question, re.I):
            if m[0][0].isupper():
                replacements.append(("prefix", exp.EQ(this=folded.copy(), expression=exp.Literal.string(m[0].lower()))))
        words = v.split()
        if len(words) == 2 and all(w[:1].isupper() and w.replace("-", "").isalpha() for w in words):
            for tab, first, last in name_columns(schema):
                if column.name.casefold() not in (first.casefold(), last.casefold()) or tab not in table_names.values():
                    continue
                if column.table and table_names.get(column.table.casefold()) != tab:
                    continue
                def part(name, value):
                    c = column.copy()
                    c.set("this", exp.to_identifier(name, quoted=True))
                    return exp.EQ(this=c, expression=exp.Literal.string(value))
                replacements.append(("full_name", exp.and_(part(first, words[0]), part(last, words[1]))))
        for family, edit in replacements:
            changed = tree.copy()
            list(changed.find_all(exp.EQ))[index].replace(edit)
            query = changed.sql(dialect="sqlite")
            if len(query.encode()) <= 16000:
                yield family, query


def candidates(row, trace, previous):
    schema = next((e["observation"] for e in trace if e["tool"] == "schema"), [])
    checked = [e["sql"] for e in trace if e["tool"] == "check"]
    seen = {sql_key(s) for s in checked}
    for sql in dict.fromkeys([previous, *reversed(checked)]):
        for family, query in text_edits(sql, row["question"], schema):
            key = sql_key(query)
            if key not in seen:
                seen.add(key)
                yield family, query


class TextFallbackAgent:
    def __init__(self, config):
        self.previous = SemanticPoolAgent(config)
        self.families = config["text_family_order"]
        self.original_sql = None
        self.pending_sql = None
        self.queue = iter(())

    def act(self, obs):
        if self.original_sql is None:
            action = self.previous.act(obs)
            if action["tool"] != "submit":
                return action
            self.original_sql = action["sql"]
            failed = any(e["tool"] == "check" and e["sql"] == self.original_sql and
                         e["observation"].get("matches_example") is False for e in obs["history"])
            if not failed or not obs["remaining"]:
                return action
            possible = list(candidates(obs["task"], obs["history"], self.original_sql))
            self.queue = (sql for family in self.families for f, sql in possible if f == family)

        if self.pending_sql is not None:
            last = obs["history"][-1]
            if last["tool"] != "check" or last["sql"] != self.pending_sql:
                raise ValueError("Missing feedback for the pending check")
            if last["observation"]["matches_example"]:
                return {"tool": "submit", "sql": self.pending_sql}
            self.pending_sql = None

        if obs["remaining"]:
            self.pending_sql = next(self.queue, None)
            if self.pending_sql is not None:
                return {"tool": "check", "sql": self.pending_sql}
        return {"tool": "submit", "sql": self.original_sql}


def create_agent(checkpoint=None):
    path = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/v11_text_fallback.json"
    return TextFallbackAgent(json.loads(path.read_text()))
