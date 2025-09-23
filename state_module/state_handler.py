from typing import Dict, Any
import yaml
from state_module.state_registry import STATE_REGISTRY, auto_register_states
from state_module.state import State

auto_register_states("state_module")


class StateHandler:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r") as f:
            self.graph = yaml.safe_load(f)

        self.states = {}
        for name, config in self.graph.get("states", {}).items():
            state_type = config.get("type")
            if state_type not in STATE_REGISTRY:
                raise ValueError(f"Unknown state type: {state_type}")
            state_class = STATE_REGISTRY[state_type]
            self.states[name] = state_class(name, config)

        self.initial_state_name = self.graph["initial"]

    def get_initial_state(self) -> State:
        return self.states[self.initial_state_name]

    def get_transitions(self, current_state: str, context: Dict[str, Any]):
        state = current_state
        transition_targets = state.transition.get("next", [])

        transition_descs = []
        for t in transition_targets:
            desc = getattr(self.states[t], "description", None)
            transition_descs.append((t, desc))

        return {"td": transition_descs, "tt": transition_targets}

    def get_state(self, state_name: str) -> State:
        return self.states[state_name]
