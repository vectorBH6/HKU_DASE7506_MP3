"""Apply the fixed public-failure fallback to the V9 terminal policy."""
import json
from pathlib import Path

from sql_repair_agent.review_terminal_v9 import TerminalReviewAgent
from v11_text_fallback import TextFallbackAgent


def create_agent(checkpoint=None):
    path = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/v12_v9_fallback.json"
    config = json.loads(path.read_text())
    controller = TextFallbackAgent(config)
    controller.previous = TerminalReviewAgent(config)
    return controller
