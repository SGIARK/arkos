"""The declared machines agree with the code that enforces them."""

import ast
import pathlib
import sys

from harness_module import lifecycle
from state_machine_module.machines import MACHINES, MODE, SESSION, allows, follows, pairs

PACKAGE = pathlib.Path(__file__).resolve().parent.parent / "state_machine_module"


def test_pairs_session_matches_lifecycle_allowed() -> None:
    assert pairs(SESSION) == lifecycle.ALLOWED


def test_session_final_matches_lifecycle_terminal() -> None:
    assert frozenset(SESSION.final) == lifecycle.TERMINAL


def test_allows_refuses_an_undeclared_move() -> None:
    assert allows(SESSION, "idle", "running")
    assert not allows(SESSION, "completed", "idle")


def test_follows_allows_staying_put_but_not_an_undeclared_move() -> None:
    assert follows(MODE, "attended", "attended")
    assert not follows(SESSION, "completed", "idle")


def test_machines_use_only_their_own_states() -> None:
    for machine in MACHINES:
        named = {start.state for start in machine.starts} | set(machine.final)
        named |= {state for move in machine.moves for state in (move.source, move.target)}
        assert named <= set(machine.states), (machine.name, named - set(machine.states))


def test_machines_each_begin_somewhere() -> None:
    assert all(machine.starts for machine in MACHINES)


def test_package_imports_only_the_standard_library() -> None:
    """The checker has to run anywhere, without arkos's database or model dependencies."""
    for path in PACKAGE.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                root = node.module.split(".")[0]
            elif isinstance(node, ast.Import):
                root = node.names[0].name.split(".")[0]
            else:
                continue
            assert root in sys.stdlib_module_names or root == "state_machine_module", (path.name, root)
