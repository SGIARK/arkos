from enum import Enum
from typing import Any


class AgentState(Enum):
    """Central registry of all possible agent states."""

    GREET_USER = "greet_user"
    FETCH_PRODUCT = "fetch_product"
    SUMMARIZE_RESULT = "summarize_result"
    DONE = "done"


class State:
    def __init__(self, name: str, config: dict[str, Any]):
        self.name = name
        self.is_terminal: bool = False
        self.transition = config.get("transition", {})

    def check_transition_ready(self, context: dict[str, Any]) -> bool:
        """
        USER DEFINED STATES SHOULD OVERRRIDE THIS FUNCTION
        """
        raise NotImplementedError

    def run(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """
        USER DEFINED STATES SHOULD OVERRRIDE THIS FUNCTION
        """
        raise NotImplementedError
