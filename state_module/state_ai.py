from state_module.state import State
from state_module.state_registry import register_state


@register_state
class StateAI(State):
    type = "agent"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

        self.is_terminal = False

    def check_transition_ready(self, context):
        return True

    def run(self, context, agent):

        agent_response = agent.call_llm(context=context)
        print(agent_response.content)
        return agent_response
