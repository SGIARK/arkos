# state.py (Scaffold)
class State:
    def __init__(self, name, metadata):
        # Name and configuration of the state
        pass

    def check_transition_ready(self, context) -> bool:
        # Determine whether this state is ready to advance
        pass

    def run(self, agent_context):
        # Execute the logic within this state
        pass
