"""Student entry point; official grading files remain unchanged."""
import json
from pathlib import Path
from sql_repair_agent.agent import RepairAgent

def create_agent(checkpoint=None):
    path = Path(checkpoint) if checkpoint else Path(__file__).parent / "configs/selected.json"
    config = json.loads(path.read_text())
    if config.get("controller") == "review_terminal_v9":
        from sql_repair_agent.review_terminal_v9 import TerminalReviewAgent
        return TerminalReviewAgent(config)
    if config.get("controller") == "review_pool_v8":
        from sql_repair_agent.review_pool_v8 import ReviewPoolAgent
        return ReviewPoolAgent(config)
    if config.get("controller") == "search_v5":
        from sql_repair_agent.search_agent import SearchAgent
        return SearchAgent(config)
    if config.get("controller") == "constraint_v4":
        from sql_repair_agent.constraint_agent import ConstraintAgent
        return ConstraintAgent(config)
    if config.get("controller") == "plan_constraint_v4":
        from sql_repair_agent.plan_constraint_agent import PlanConstraintAgent
        return PlanConstraintAgent(config)
    if config.get("controller") == "plan_v4":
        from sql_repair_agent.plan_agent import PlanAgent
        return PlanAgent(config)
    if config.get("controller") == "contract_v4":
        from sql_repair_agent.contract_agent import ContractAgent
        return ContractAgent(config)
    if config.get("controller") == "semantic_v4":
        from sql_repair_agent.semantic_agent import SemanticAgent
        return SemanticAgent(config)
    if config.get("controller") == "feedback_v4":
        from sql_repair_agent.feedback_agent import FeedbackAgent
        return FeedbackAgent(config)
    if config.get("controller") == "recovery_v3":
        from sql_repair_agent.recovery_agent import RecoveryAgent
        return RecoveryAgent(config)
    if config.get("controller") == "review_v2":
        from sql_repair_agent.review_agent import ReviewAgent
        return ReviewAgent(config)
    return RepairAgent(config)
