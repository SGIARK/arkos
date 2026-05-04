import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from state_module.core.base_state import StateOutput
from state_module.core.state import State
from state_module.core.state_registry import register_state


@register_state
class StateUser(State):
    """Terminal state that waits for user input before transitioning."""

    type = "user"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = True

    def check_transition_ready(self, context):
        return True

    async def run(self, context, agent=None):
        return StateOutput(
            content="",
            completion_signal="needs_input",
        )
