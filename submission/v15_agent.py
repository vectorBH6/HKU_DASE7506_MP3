"""Compose the selected V15 repair stages through the course agent interface."""
import json
from pathlib import Path


def create_agent(checkpoint=None):
    path = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/v15_selected.json"
    return make_agent(json.loads(path.read_text())["stages"])


def make_agent(stages):
    if "early" in stages:
        from v15_early import create_agent as base_factory
    else:
        from v14_terminal import create_agent as base_factory
    tails = []
    if "witness" in stages:
        from v15_witness import terminal_action
        tails.append(terminal_action)
    if "retrieval" in stages:
        from v15_retrieval import Tail
        tails.append(Tail().act)
    if "rescue" in stages:
        from v15_rescue import RescuePhase
        tails.append(RescuePhase().action)
    return Agent(base_factory(), tails)


class Agent:
    def __init__(self, base, tails):
        self.base = base
        self.tails = tails
        self.incumbent = None
        self.index = 0

    def act(self, obs):
        if self.incumbent is None:
            action = self.base.act(obs)
            if action["tool"] != "submit":
                return action
            self.incumbent = action["sql"]
        while self.index < len(self.tails):
            action = self.tails[self.index](obs, self.incumbent)
            if action["tool"] != "submit" or action["sql"] != self.incumbent:
                return action
            self.index += 1
        return {"tool": "submit", "sql": self.incumbent}
