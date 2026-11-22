"""Isolate public output-contract edits from the richer diagnostic prompt."""
from .semantic_agent import SemanticAgent
from .recovery_agent import RecoveryAgent

class ContractAgent(SemanticAgent):
    def model_messages(self,obs):
        return RecoveryAgent.model_messages(self,obs)
