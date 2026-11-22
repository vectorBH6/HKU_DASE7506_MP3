"""Model proposes actions; Python enforces the unchanged official protocol."""
import json
import os
from pathlib import Path
from .sql_utils import extract_sql, sql_key, variants
from .review_prompts import checked_candidates, messages_for, risk_flags


def parse_proposal(text):
    text = text.split("</think>")[-1].strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else text
    start = text.find("{")
    if start >= 0:
        try:
            value, end = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            pass
    sql = extract_sql(text)
    return {"tool": "check", "sql": sql, "reason_code": "sql_only_fallback"} if sql else None


class ReviewAgent:
    def __init__(self, config):
        self.config = config
        self.completions = 0
        if not 1 <= config.get("max_completions", 4) <= 6 or not 1 <= config.get("max_new_tokens", 384) <= 2048:
            raise ValueError("Invalid model budget")

    def audit(self, obs, proposal, action, reason):
        path = os.environ.get("MP3_GENERATION_LOG")
        if not path:
            return
        record = {"task_id": obs["task"]["id"], "attempt": self.completions, "remaining": obs["remaining"],
                  "proposal": proposal, "accepted_action": action, "guard_reason": reason,
                  "risk_flags": risk_flags(obs)}
        with Path(path).with_name("decisions.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def generate_proposal(self, obs):
        from .model import get_generator
        self.completions += 1
        text = get_generator(self.config.get("use_adapter", True)).generate(
            messages_for(obs, self.config), obs["task"]["id"], self.completions, self.config)
        return parse_proposal(text)

    @staticmethod
    def best(obs):
        candidates = checked_candidates(obs)
        expected = obs["example"]
        def rank(candidate):
            f = candidate["feedback"]
            return (bool(f.get("matches_example")), bool(f.get("ok")),
                    len(f.get("columns", [])) == len(expected["columns"]),
                    -abs(f.get("row_count", 0) - expected["row_count"]))
        return max(candidates, key=rank)

    def guard(self, obs, proposal):
        if not isinstance(proposal, dict):
            return None, "malformed_proposal"
        tool = proposal.get("tool")
        history = obs["history"]
        used = {e["tool"] for e in history}
        if tool == "submit":
            candidates = checked_candidates(obs)
            choice = next((c for c in candidates if c["id"] == proposal.get("candidate")), None)
            if choice is None and isinstance(proposal.get("sql"), str):
                choice = next((c for c in candidates if sql_key(c["sql"]) == sql_key(proposal["sql"])), None)
            if choice is None:
                return None, "submit_requires_checked_candidate"
            if not choice["feedback"].get("matches_example") and (any(c["feedback"].get("matches_example") for c in candidates) or obs["remaining"] > 0):
                return None, "prefer_matching_or_continue"
            return {"tool": "submit", "sql": choice["sql"]}, "model_candidate_selection"
        if obs["remaining"] <= 0:
            return None, "no_tool_budget"
        if tool in ("check", "execute"):
            sql = extract_sql(proposal.get("sql", "")) if isinstance(proposal.get("sql"), str) else None
            if not sql:
                return None, "invalid_query"
            if any(e["tool"] == tool and sql_key(e["sql"]) == sql_key(sql) for e in history):
                return None, "duplicate_action"
            return {"tool": tool, "sql": sql}, "model_query"
        if tool == "repair_hint" and self.config.get("use_hints", True) and tool not in used and obs["remaining"] >= 2:
            return {"tool": tool}, "model_hint_request"
        if tool == "retrieve" and self.config.get("use_retrieval", True) and tool not in used and obs["remaining"] >= 2:
            query = proposal.get("query")
            if not isinstance(query, str) or not query.strip():
                query = obs["task"]["question"] + " " + obs["issue_sql"]
            return {"tool": tool, "query": query[:2000]}, "model_retrieval_request"
        return None, "unavailable_action"

    def fallback(self, obs):
        best = self.best(obs)
        if best["feedback"].get("matches_example") or not obs["remaining"]:
            return {"tool": "submit", "sql": best["sql"]}
        used = {e["tool"] for e in obs["history"]}
        if self.config.get("use_hints", True) and "repair_hint" not in used and obs["remaining"] >= 2:
            return {"tool": "repair_hint"}
        checked = {sql_key(c["sql"]) for c in checked_candidates(obs)}
        hints = next((e["observation"] for e in obs["history"] if e["tool"] == "repair_hint"), [])
        for text in hints:
            sql = extract_sql(text)
            if sql and sql_key(sql) not in checked:
                return {"tool": "check", "sql": sql}
        if self.config.get("use_variants", True):
            for c in reversed(checked_candidates(obs)):
                for sql in variants(c["sql"], c["feedback"], obs["example"]):
                    if sql_key(sql) not in checked:
                        return {"tool": "check", "sql": sql}
        return {"tool": "submit", "sql": best["sql"]}

    def act(self, observation):
        history = observation["history"]
        if not any(e["tool"] == "check" for e in history):
            return {"tool": "check", "sql": observation["issue_sql"]}
        candidates = checked_candidates(observation)
        matched = next((c for c in candidates if c["feedback"].get("matches_example")), None)
        policy = self.config.get("review_matched", "always")
        # Early acceptance is an explicit controlled ablation, not the default.
        if matched and self.completions == 0 and (policy == "off" or (policy == "risk" and not risk_flags(observation))):
            return {"tool": "submit", "sql": matched["sql"]}
        if not observation["remaining"]:
            return {"tool": "submit", "sql": self.best(observation)["sql"]}
        if not any(e["tool"] == "schema" for e in history):
            return {"tool": "schema"}
        if self.completions < self.config.get("max_completions", 4):
            proposal = self.generate_proposal(observation)
            action, reason = self.guard(observation, proposal)
            self.audit(observation, proposal, action, reason)
            if action:
                return action
        action = self.fallback(observation)
        self.audit(observation, None, action, "completion_cap_fallback")
        return action
