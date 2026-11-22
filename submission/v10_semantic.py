"""Official entry point for the feedback-trained V4 semantic agent with V8 review."""
import json
from pathlib import Path

from sql_repair_agent.review_pool_v8 import ReviewPoolAgent
from sql_repair_agent.semantic_agent import SemanticAgent


class SemanticPoolAgent:
    def __init__(self, config):
        self.semantic = SemanticAgent(config)
        self.review = ReviewPoolAgent(config)

    def act(self, observation):
        if self.review.original_sql is None:
            action = self.semantic.act(observation)
            if action["tool"] != "submit":
                return action
            self.review.begin_review(observation, action["sql"])
        return self.review.review_action(observation)


def create_agent(checkpoint=None):
    config_path = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/v10_semantic_struct.json"
    return SemanticPoolAgent(json.loads(config_path.read_text()))
