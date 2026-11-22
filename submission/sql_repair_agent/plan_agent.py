"""Query-synthesis curriculum interface; plans are proposals, never correctness proof."""
from .contract_agent import ContractAgent
PLAN_INSTRUCTION=(
    ' When proposing check or execute SQL, put a short plan before sql in the same JSON. '
    'The plan may name output expressions, relevant tables/joins, filters, grouping and ordering. '
    'Match output order to expected_example columns. A plan is not a validation result. '
    'Keep it concise. Submit/repair_hint/retrieve need no plan. '
    'All actual verification must still use available official tools.'
)

class PlanAgent(ContractAgent):
    def model_messages(self,obs):
        messages=super().model_messages(obs)
        messages[0]['content']+=PLAN_INSTRUCTION
        return messages
