from model_module.ArkModelNew import SystemMessage
from state_module.state import State
from state_module.state_registry import register_state


@register_state
class StateTool(State):
    type = "tool"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    def run(self, context, agent=None):


        # extract tool name 

        # extract tool parameters


        # call tool 
        

        # return tool msg
        print("TOOL RESULT PLACEHOLDER")
        return SystemMessage(content="Result: 3*6 is 18")

