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


@dataclass(frozen=True, slots=True)
class Report:
    """What one search covered, and what it found."""

    states: int
    depth: int
    findings: tuple[Finding, ...]


def path_to(world: World, parents: Parents) -> tuple[World, tuple[Event, ...]]:
    """Follow parent links back to a root: the root, and the events from it in order."""
    path: list[Event] = []
    link = parents[world]
    while link is not None:
        world, event = link
        path.append(event)
        link = parents[world]
    return world, tuple(reversed(path))
