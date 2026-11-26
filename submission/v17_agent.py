"""Apply V17 repair stages after the frozen V15 agent has proposed a final SQL."""
import json
from pathlib import Path

from v15_agent import Agent, create_agent as create_v15

ORDER = ("semantics", "cardinality", "relations", "branch")


def make_agent(stages):
    if stages == ["early"]:
        from v17_early import create_agent as create_early
        return create_early()
    if stages != [name for name in ORDER if name in stages]:
        raise ValueError("Unknown or out-of-order repair stages")
    tails = []
    if "semantics" in stages:
        from v17_semantics import Tail
        tails.append(Tail().act)
    if "cardinality" in stages:
        from v17_cardinality import Tail
        tails.append(Tail().act)
    if "relations" in stages:
        from v17_relations import Tail
        tails.append(Tail().act)
    if "branch" in stages:
        from v17_branch import Tail
        tails.append(Tail().act)
    return Agent(create_v15(), tails)


def create_agent(checkpoint=None):
    path = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/v17_selected.json"
    return make_agent(json.loads(path.read_text())["stages"])
