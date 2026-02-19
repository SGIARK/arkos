import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from state_module.state import State
from state_module.state_registry import register_state


@register_state
class StateRouter(State):
    """
    Pure routing state - no user-facing output.
    Decides whether the request needs a tool call or a direct agent response.
    choose_transition() runs after run() returns None, with a clean context
    (no StateAI hallucination present), so it can correctly route to use_tool.
    """
    type = "router"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    async def run(self, context, agent=None):
        return None
