from model_module.ArkModelNew import UserMessage
from state_module.state import State
from state_module.state_registry import register_state


@register_state
class StateUser(State):
    type = "user"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    def run(self, context, agent=None):

        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print("safe_shutdown sequence initialized")
            self.is_terminal = True
            return

        else:
            return UserMessage(content=user_input)
