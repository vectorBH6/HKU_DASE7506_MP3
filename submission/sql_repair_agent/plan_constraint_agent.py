from .constraint_agent import ConstraintAgent
from .plan_agent import PLAN_INSTRUCTION

class PlanConstraintAgent(ConstraintAgent):
    def model_messages(self,obs):
        messages=super().model_messages(obs);messages[0]['content']+=PLAN_INSTRUCTION;return messages
