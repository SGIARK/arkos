"""Try every sequence of events up to a length, and report each rule that breaks.

The search is breadth-first, so the first sequence that breaks a rule is a shortest
one. Run it with:

    python -m state_machine_module.check [--depth N]
"""

from __future__ import annotations

from dataclasses import dataclass

from state_machine_module.rules import Rule
from state_machine_module.world import Event, World

DEFAULT_DEPTH = 6

Parents = dict[World, tuple[World, Event] | None]


@dataclass(frozen=True, slots=True)
class Finding:
    """A broken rule, and the shortest sequence of events that breaks it."""

    rule: Rule
    root: World
    path: tuple[Event, ...]
