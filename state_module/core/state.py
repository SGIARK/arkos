from typing import Any

class State:
    """Base class for all states in the state graph."""

    def __init__(self, name: str, config: dict[str, Any]):
        """Initialize a state with its name and configuration from the YAML graph."""
        self.name = name
        self.is_terminal: bool = False
        self.transition = config.get("transition", {})

    def check_transition_ready(self, context: dict[str, Any]) -> bool:
        """USER DEFINED STATES SHOULD OVERRIDE THIS FUNCTION"""
        raise NotImplementedError

    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """USER DEFINED STATES SHOULD OVERRIDE THIS FUNCTION"""
        raise NotImplementedError
