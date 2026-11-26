"""The declared machines agree with the code that enforces them."""

from harness_module import lifecycle
from state_machine_module.machines import SESSION, pairs


def test_pairs_session_matches_lifecycle_allowed() -> None:
    assert pairs(SESSION) == lifecycle.ALLOWED


def test_session_final_matches_lifecycle_terminal() -> None:
    assert frozenset(SESSION.final) == lifecycle.TERMINAL
