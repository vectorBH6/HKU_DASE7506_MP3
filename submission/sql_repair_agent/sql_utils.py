"""Conservative candidate normalization; validity is decided by official tools."""
import re
import sqlglot
from sqlglot import exp


def extract_sql(text):
    text = text.split("</think>")[-1].strip()
    blocks = re.findall(r"```(?:sql|sqlite)?\s*(.*?)```", text, re.I | re.S)
    if blocks:
        text = blocks[0].strip()
    match = re.search(r"\b(SELECT|WITH)\b", text, re.I)
    if not match:
        return None
    text = text[match.start():].strip()
    try:
        statements = sqlglot.parse(text, read="sqlite")
        if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.Union, exp.Intersect, exp.Except)):
            return None
        sql = statements[0].sql(dialect="sqlite")
        return sql if len(sql.encode()) <= 16000 else None
    except sqlglot.errors.SqlglotError:
        return None


def sql_key(sql):
    try:
        return sqlglot.parse_one(sql, read="sqlite").sql(dialect="sqlite", normalize=True)
    except sqlglot.errors.SqlglotError:
        return sql.strip()


def variants(sql, feedback, expected):
    # A duplicate-removal repair is justified only by a successful result with
    # too many rows. The official checker will validate this hypothesis.
    if not feedback.get("ok") or feedback.get("row_count", 0) <= expected.get("row_count", 0):
        return []
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        if not isinstance(tree, exp.Select) or tree.args.get("distinct") or tree.args.get("group"):
            return []
        if any(isinstance(node, exp.AggFunc) for node in tree.walk()):
            return []
        tree.set("distinct", exp.Distinct())
        return [tree.sql(dialect="sqlite")]
    except sqlglot.errors.SqlglotError:
        return []
