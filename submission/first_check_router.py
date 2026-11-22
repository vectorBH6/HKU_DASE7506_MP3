"""Choose a controller from the public draft check while keeping one fixed language model."""
import json
from pathlib import Path

from sql_repair_agent.review_terminal_v9 import TerminalReviewAgent
from v11_text_fallback import TextFallbackAgent


class FirstCheckRouter:
    def __init__(self, config):
        self.config = config
        self.controller = None

    def act(self, obs):
        if self.controller is None:
            if not obs["history"]:
                return {"tool": "check", "sql": obs["issue_sql"]}
            first = obs["history"][0]
            if first["tool"] != "check" or first["sql"] != obs["issue_sql"]:
                raise ValueError("The first observation must be the public check of the original draft")
            self.controller = TextFallbackAgent(self.config)
            if first["observation"]["matches_example"]:
                self.controller.previous = TerminalReviewAgent(self.config)
        return self.controller.act(obs)


def create_agent(checkpoint=None):
    path = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/v12_v9_fallback.json"
    return FirstCheckRouter(json.loads(path.read_text()))
