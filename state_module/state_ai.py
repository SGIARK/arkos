import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from state_module.state import State
from state_module.state_registry import register_state


@register_state
class StateAI(State):
    type = "agent"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, _context):
        return True

    async def run(self, context, agent):
        agent_response = await agent.call_llm(context=context)
        return agent_response  # ← return, not print
