"""Bounded recovery after rejected proposals, with private controller memory."""
import json
import os
from pathlib import Path
from .review_agent import ReviewAgent, parse_proposal
from .review_prompts import checked_candidates, messages_for, risk_flags

RECOVERY_INSTRUCTION = (
    "Controller feedback records rejected proposals, not SQL correctness. "
    "A rejected proposal consumed a generation but did not execute a tool. "
    "Do not repeat a rejected or previously checked query. Use the actual SQLite "
    "error and current schema to change the faulty part, request an available "
    "alternative, or select a checked candidate whose semantics fit the question. "
    "A duplicate rejection is not evidence that the SQL is correct. "
    "Read model_generations_after_this when planning further tool use. "
    "Return the same short action JSON format."
)


class RecoveryAgent(ReviewAgent):
    def __init__(self, config):
        super().__init__(config)
        self.rejections = []
        retries = config.get("max_retries_per_observation", 1)
        if not isinstance(retries, int) or not 0 <= retries <= 1:
            raise ValueError("At most one immediate recovery per observation")

    def rejection_record(self, proposal, reason):
        record = {"generation": self.completions, "reason": reason}
        if isinstance(proposal, dict):
            for key in ("tool", "sql", "candidate", "query"):
                if isinstance(proposal.get(key), str):
                    record[key] = proposal[key][:4000]
        self.rejections.append(record)

    def model_messages(self, obs):
        messages = messages_for(obs, self.config)
        # Preserve the trained v2 prompt exactly until a proposal is rejected.
        if not self.rejections:
            return messages
        raw = messages[1]["content"].split("TASK_DATA\n", 1)[1]
        payload = json.loads(raw.split("\nEND_TASK_DATA", 1)[0])
        latest_error = next(
            ({"sql": e["sql"], "error": e["observation"]["error"]}
             for e in reversed(obs["history"])
             if e["tool"] in ("check", "execute") and e["observation"].get("error")), None)
        payload["controller_feedback"] = {
            "rejected_proposals": self.rejections[-2:],
            "latest_sql_error": latest_error,
            "current_generation": self.completions,
            "model_generations_after_this": self.config.get("max_completions", 4) - self.completions,
        }
        messages[0] = {"role": "system", "content": messages[0]["content"] + " " + RECOVERY_INSTRUCTION}
        messages[1] = {"role": "user", "content": "TASK_DATA\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\nEND_TASK_DATA\nNext action JSON:"}
        return messages

    def generate_proposal(self, obs):
        from .model import get_generator
        self.completions += 1
        text = get_generator(self.config.get("use_adapter", True)).generate(
            self.model_messages(obs), obs["task"]["id"], self.completions, self.config)
        return parse_proposal(text)

    def audit(self, obs, proposal, action, reason):
        path = os.environ.get("MP3_GENERATION_LOG")
        if not path:
            return
        record = {"task_id": obs["task"]["id"], "attempt": self.completions,
                  "remaining": obs["remaining"], "proposal": proposal,
                  "accepted_action": action, "guard_reason": reason,
                  "risk_flags": risk_flags(obs), "controller": "recovery_v3",
                  "model_generations_remaining": self.config.get("max_completions", 4) - self.completions,
                  "rejection_count": len(self.rejections)}
        with Path(path).with_name("decisions.jsonl").open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def act(self, observation):
        history = observation["history"]
        if not any(e["tool"] == "check" for e in history):
            return {"tool": "check", "sql": observation["issue_sql"]}
        matched = next((c for c in checked_candidates(observation)
                        if c["feedback"].get("matches_example")), None)
        policy = self.config.get("review_matched", "risk")
        if matched and self.completions == 0 and (policy == "off" or (policy == "risk" and not risk_flags(observation))):
            return {"tool": "submit", "sql": matched["sql"]}
        if not observation["remaining"]:
            return {"tool": "submit", "sql": self.best(observation)["sql"]}
        if not any(e["tool"] == "schema" for e in history):
            return {"tool": "schema"}
        retries = 0
        limit = self.config.get("max_completions", 4)
        while self.completions < limit:
            proposal = self.generate_proposal(observation)
            action, reason = self.guard(observation, proposal)
            if action:
                self.audit(observation, proposal, action, reason)
                return action
            self.rejection_record(proposal, reason)
            self.audit(observation, proposal, None, reason)
            if retries >= self.config.get("max_retries_per_observation", 1):
                break
            if self.completions < limit:
                retries += 1
                self.audit(observation, None, None, "recovery_retry")
        action = self.fallback(observation)
        reason = "model_budget_fallback" if self.completions >= limit else "recovery_limit_fallback"
        self.audit(observation, None, action, reason)
        return action
