"""A deterministic controller whose branches depend on actual tool observations."""
from .sql_utils import extract_sql, sql_key, variants


class RepairAgent:
    def __init__(self, config):
        self.config = config
        self.completions = 0
        if config.get("mode") == "qwen":
            if not 1 <= config.get("max_completions", 2) <= 6 or not 1 <= config.get("max_new_tokens", 256) <= 2048:
                raise ValueError("Invalid model budget")

    def act(self, observation):
        history = observation["history"]
        checks = [e for e in history if e["tool"] == "check"]
        if not checks:
            return {"tool": "check", "sql": observation["issue_sql"]}
        matched = next((e for e in checks if e["observation"].get("matches_example")), None)
        if matched:
            return {"tool": "submit", "sql": matched["sql"]}
        if observation["remaining"] == 0:
            return self.submit_best(checks, observation)
        used = {e["tool"] for e in history}
        if "schema" not in used:
            return {"tool": "schema"}
        checked = {sql_key(e["sql"]) for e in checks}
        live = self.config.get("mode") == "qwen"
        if live and self.config.get("use_retrieval") and "retrieve" not in used and observation["remaining"] >= 2:
            return {"tool": "retrieve", "query": (observation["task"]["question"] + " " + observation["issue_sql"])[:2000]}
        # First model attempt uses original feedback and schema; later attempts
        # see failures from candidates and cached hints as well.
        if live and self.completions == 0:
            candidate = self.generate(observation)
            if candidate and sql_key(candidate) not in checked:
                return {"tool": "check", "sql": candidate}
        if self.config.get("use_hints") and "repair_hint" not in used and observation["remaining"] >= 2:
            return {"tool": "repair_hint"}
        hint_event = next((e for e in history if e["tool"] == "repair_hint"), None)
        if hint_event:
            for text in hint_event["observation"]:
                candidate = extract_sql(text)
                if candidate and sql_key(candidate) not in checked:
                    return {"tool": "check", "sql": candidate}
        if self.config.get("use_variants"):
            for event in reversed(checks):
                for candidate in variants(event["sql"], event["observation"], observation["example"]):
                    if sql_key(candidate) not in checked:
                        return {"tool": "check", "sql": candidate}
        while live and self.completions < self.config["max_completions"]:
            candidate = self.generate(observation)
            if candidate and sql_key(candidate) not in checked:
                return {"tool": "check", "sql": candidate}
        return self.submit_best(checks, observation)

    def generate(self, observation):
        from .model import get_generator
        from .prompts import messages_for
        self.completions += 1
        generator = get_generator(self.config.get("use_adapter", False))
        return extract_sql(generator.generate(messages_for(observation), observation["task"]["id"], self.completions, self.config))

    @staticmethod
    def submit_best(checks, observation):
        expected = observation["example"]
        def rank(event):
            feedback = event["observation"]
            return (bool(feedback.get("ok")),
                    len(feedback.get("columns", [])) == len(expected["columns"]),
                    -abs(feedback.get("row_count", 0) - expected["row_count"]))
        best = max(checks, key=rank)
        return {"tool": "submit", "sql": best["sql"]}
