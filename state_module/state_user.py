import os
import sys

from state_module.state import State
from state_module.state_registry import register_state

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@register_state
class StateUser(State):
    type = "user"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = True  # Stop after this state

    def check_transition_ready(self, _context):
        # ALWAYS allow transition after user provides input
        return True

    def run(self, _context):
        return None
