"""Continue the V8 policy with a checked projection or a train-fitted intent cue."""
import re

from mp3_sql.common import MAX_SQL_BYTES
from sqlglot import exp, parse_one
from sqlglot.errors import SqlglotError

from .review_pool_v8 import ReviewPoolAgent
from .sql_utils import sql_key


OPERATION_TYPES = (
    exp.Count, exp.Avg, exp.Sum, exp.Max, exp.Min, exp.Group, exp.Having,
    exp.Order, exp.Limit, exp.Distinct, exp.Union, exp.Intersect, exp.Except,
    exp.In, exp.Exists, exp.Not, exp.Like, exp.Between, exp.Or, exp.And,
    exp.GT, exp.GTE, exp.LT, exp.LTE, exp.NEQ, exp.Join, exp.Is, exp.Subquery,
)


def operations(sql):
    try:
        tree = parse_one(sql, read="sqlite")
    except (SqlglotError, ValueError, RecursionError):
        return set()
    if tree is None:
        return set()
    nodes = list(tree.walk())
    flags = {kind.__name__ for kind in OPERATION_TYPES if any(isinstance(node, kind) for node in nodes)}
    if any(order.args.get("desc") for order in tree.find_all(exp.Ordered)):
        flags.add("OrderDesc")
    limit = tree.args.get("limit")
    if limit and str(limit.expression) == "1":
        flags.add("LimitOne")
    if sum(isinstance(node, exp.Select) for node in nodes) > 1:
        flags.add("MultiSelect")
    return flags


def select_match(question, matches, incumbent, rules):
    options = {sql_key(sql): sql for sql in matches}
    if len(options) < 2 or sql_key(incumbent) not in options:
        return incumbent
    words = re.findall(r"[a-z0-9]+", question.lower())
    required = set()
    for size in (1, 2, 3):
        for i in range(len(words) - size + 1):
            required.update(rules.get(" ".join(words[i:i + size]), ()))
    if required <= operations(incumbent):
        return incumbent
    eligible = [sql for sql in options.values() if required <= operations(sql)]
    return eligible[0] if len(eligible) == 1 else incumbent


def project_columns(obs, event):
    feedback = event["observation"]
    expected = obs["example"]
    if not feedback.get("ok") or feedback["row_count"] != expected["row_count"]:
        return None
    columns = feedback.get("columns") or []
    if not 0 < len(expected["columns"]) < len(columns):
        return None
    names = [name.casefold() for name in columns]
    requested = [name.casefold() for name in expected["columns"]]
    if not all(names.count(name) == 1 for name in requested):
        return None

    sql = event["sql"].strip().removesuffix(";").rstrip()
    if obs["ordered"]:
        # A subquery's ordering is immaterial when it returns at most one row.
        try:
            tree = parse_one(sql, read="sqlite")
        except (SqlglotError, ValueError, RecursionError):
            return None
        limit = tree.args.get("limit") if tree is not None else None
        if not isinstance(limit, exp.Limit) or not isinstance(limit.expression, exp.Literal):
            return None
        if limit.expression.is_string or limit.expression.this != "1":
            return None

    wanted = [columns[names.index(name)] for name in requested]
    projection = ", ".join('v9_result."' + name.replace('"', '""') + '"' for name in wanted)
    query = "SELECT " + projection + " FROM (\n" + sql + "\n) AS v9_result"
    return query if len(query.encode("utf-8")) <= MAX_SQL_BYTES else None


class TerminalReviewAgent(ReviewPoolAgent):
    def __init__(self, config):
        super().__init__(config)
        self.rules = config["intent_rules"]
        self.terminal_sql = None
        self.terminal_boundary = None

    def act(self, obs):
        history = obs["history"]
        if self.terminal_sql is None:
            action = super().act(obs)
            if action["tool"] != "submit":
                return action
            self.terminal_sql = action["sql"]
            self.terminal_boundary = len(history)

        checks = [event for event in history if event["tool"] == "check"]
        matches = [event["sql"] for event in checks if event["observation"]["matches_example"]]
        if matches:
            # Additional checks are made only when the V8 prefix had no match.
            incumbent = matches[-1] if len(history) > self.terminal_boundary else self.terminal_sql
            chosen = select_match(obs["task"]["question"], matches, incumbent, self.rules)
            return {"tool": "submit", "sql": chosen}

        if obs["remaining"]:
            seen = {event["sql"] for event in checks}
            for event in reversed(history[:self.terminal_boundary]):
                if event["tool"] == "check":
                    candidate = project_columns(obs, event)
                    if candidate is not None and candidate not in seen:
                        return {"tool": "check", "sql": candidate}
        return {"tool": "submit", "sql": self.terminal_sql}
