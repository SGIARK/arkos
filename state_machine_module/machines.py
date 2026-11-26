"""The state machines arkos is built from, declared as data.

Code that needs a machine reads it from here. The world model in `world.py` notes
any move a machine does not declare, and a test holds `SESSION` to
`harness_module.lifecycle.ALLOWED`, so the declaration cannot drift from the code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Move:
    """One legal move between two states, and what triggers it."""

    source: str
    target: str
    trigger: str


@dataclass(frozen=True, slots=True)
class Machine:
    """A state machine: its states, the ones it ends in, and its legal moves."""

    name: str
    states: tuple[str, ...]
    final: tuple[str, ...]
    moves: tuple[Move, ...]


def allows(machine: Machine, source: str, target: str) -> bool:
    """Whether `machine` declares a move from `source` to `target`."""
    return any(move.source == source and move.target == target for move in machine.moves)
