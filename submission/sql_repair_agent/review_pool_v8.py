"""Continue the V7 trajectory at its first submit, using only real tool feedback."""
from .contract_agent import ContractAgent
from .pool_edits_v8 import build_pool
from .pool_rank_v8 import rank_scores
from .sql_utils import extract_sql, sql_key


def _matches(event):
    return event["tool"] == "check" and event["observation"].get("matches_example", False)


def _feedback(history, tool):
    return next((e["observation"] for e in history if e["tool"] == tool), None)


class ReviewPoolAgent(ContractAgent):
    def __init__(self, config):
        super().__init__(config)
        self.policy = config["review_pool_policy"]
        if self.policy not in {"first", "choose", "struct"}:
            raise ValueError("Unknown review pool policy")
        self.margin = config["replacement_margin"]
        self.original_sql = None
        self.boundary = None
        self.struct_queue = None
        self.candidate_info = {}

    def begin_review(self, obs, sql):
        """Freeze the V7 submission and the boundary before additional checks."""
        self.original_sql = sql
        self.boundary = len(obs["history"])
        self.candidate_info[sql_key(obs["issue_sql"])] = ("raw", "draft")

    def act(self, obs):
        if self.original_sql is None:
            action = super().act(obs)
            if action["tool"] != "submit":
                return action
            self.begin_review(obs, action["sql"])
        return self.review_action(obs)

    def review_action(self, obs):
        history = obs["history"]
        hints = _feedback(history, "repair_hint")
        schema = _feedback(history, "schema") or []
        checked = {sql_key(e["sql"]) for e in history if e["tool"] == "check"}
        if hints is None and obs["remaining"] >= 2 and any(
            e["tool"] in ("check", "execute") and e["sql"] == obs["issue_sql"] for e in history
        ):
            return {"tool": "repair_hint"}

        for i, text in enumerate(hints or []):
            candidate = extract_sql(text)
            if candidate is None:
                continue
            key = sql_key(candidate)
            self.candidate_info.setdefault(key, ("raw", f"hint_{i}"))
            if obs["remaining"] and key not in checked:
                return {"tool": "check", "sql": candidate}

        if self.policy == "struct" and obs["remaining"]:
            if self.struct_queue is None and not any(_matches(e) for e in history):
                edits = []
                seen = set(checked)
                for candidate in build_pool(obs, schema, hints or []):
                    key = sql_key(candidate["sql"])
                    if candidate["family"] == "raw" or key in seen:
                        continue
                    seen.add(key)
                    edits.append(candidate)
                scores = rank_scores(obs, schema, edits, "check") if edits else []
                self.struct_queue = [c for _, c in sorted(zip(scores, edits), key=lambda x: -x[0])]
            for candidate in self.struct_queue or []:
                key = sql_key(candidate["sql"])
                if key not in checked:
                    self.candidate_info[key] = (candidate["family"], candidate["source"])
                    return {"tool": "check", "sql": candidate["sql"]}

        return {"tool": "submit", "sql": self._choose(obs, schema)}

    def _choose(self, obs, schema):
        history = obs["history"]
        first_new = next((e for e in history[self.boundary:] if _matches(e)), None)
        if first_new is None:
            return self.original_sql
        if self.policy == "first":
            return first_new["sql"]

        matches = {}
        for e in history:
            if _matches(e):
                key = sql_key(e["sql"])
                family, source = self.candidate_info.get(key, ("other", "other"))
                matches.setdefault(key, {"sql": e["sql"], "source": source, "family": family})

        scores = dict(zip(matches, rank_scores(obs, schema, list(matches.values()), "choose")))
        best = max(scores, key=scores.get)
        incumbent = sql_key(self.original_sql)
        if incumbent in scores and scores[best] <= scores[incumbent] + self.margin:
            return self.original_sql
        return matches[best]["sql"]
