"""Replace the zero-tool starter with your state-dependent SQL repair agent."""


class DirectAgent:
    def act(self, observation):
        return {"tool": "submit", "sql": observation["issue_sql"]}


def create_agent(checkpoint=None):
    if checkpoint is not None:
        raise ValueError("The starter has no checkpoint; omit it or implement loading")
    return DirectAgent()
