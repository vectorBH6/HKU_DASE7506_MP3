"""Apply frozen terminal SQL repairs after the V12 controller requests submission."""
import json
from pathlib import Path

from v12_v9_fallback import create_agent as create_v12


def terminal_stages(config):
    stages = []
    for name in config["stages"]:
        if name == "literals":
            from v14_literals import terminal_action
        elif name == "structure":
            from v14_structure import terminal_action
        elif name == "consensus":
            from v14_consensus import terminal_action as choose_sql

            def terminal_action(obs, incumbent):
                return {"tool": "submit", "sql": choose_sql(obs, incumbent, config["consensus_policy"])}
        else:
            raise ValueError(f"Unknown terminal stage {name}")
        stages.append(terminal_action)
    return stages


class TerminalAgent:
    def __init__(self, stages):
        self.base = create_v12()
        self.stages = stages
        self.index = 0
        self.incumbent = None

    def act(self, obs):
        if self.incumbent is None:
            action = self.base.act(obs)
            if action["tool"] != "submit":
                return action
            self.incumbent = action["sql"]
        while self.index < len(self.stages):
            action = self.stages[self.index](obs, self.incumbent)
            if action["tool"] != "submit" or action["sql"] != self.incumbent:
                return action
            self.index += 1
        return {"tool": "submit", "sql": self.incumbent}


def create_agent(checkpoint=None):
    config = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/v14_selected.json"
    return TerminalAgent(terminal_stages(json.loads(config.read_text())))
