"""Try every sequence of events up to a length, and report each rule that breaks.

The search is breadth-first, so the first sequence that breaks a rule is a shortest
one. Run it with:

    python -m state_machine_module.check [--depth N]
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from state_machine_module.rules import RULES, Rule, broken
from state_machine_module.world import ROOT_NAMES, Event, World, describe, events, step

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


def note(world: World, parents: Parents, findings: dict[str, Finding]) -> None:
    """Record each rule `world` breaks that no shorter sequence has broken yet."""
    for rule in broken(world):
        if rule.name not in findings:
            root, path = path_to(world, parents)
            findings[rule.name] = Finding(rule, root, path)


def explore(roots: Iterable[World], depth: int) -> Report:
    """Visit every world reachable from `roots` in at most `depth` events."""
    parents: Parents = {}
    findings: dict[str, Finding] = {}
    frontier: deque[tuple[World, int]] = deque()
    for root in roots:
        parents[root] = None
        note(root, parents, findings)
        frontier.append((root, 0))
    while frontier:
        world, level = frontier.popleft()
        if level == depth:
            continue
        for event in events(world):
            after = step(world, event)
            if after not in parents:
                parents[after] = (world, event)
                note(after, parents, findings)
                frontier.append((after, level + 1))
    return Report(states=len(parents), depth=depth, findings=tuple(findings.values()))


def format_report(report: Report, seconds: float) -> str:
    """The report as text: every rule, and for each broken one, how to break it."""
    found = {finding.rule.name: finding for finding in report.findings}
    lines = [f"Explored {report.states:,} states, every sequence of up to {report.depth} events, in {seconds:.1f} s."]
    for rule in RULES:
        finding = found.get(rule.name)
        if finding is None:
            lines.append(f"holds   {rule.name}: {rule.says}")
            continue
        lines.append(f"BROKEN  {rule.name}: {rule.says}")
        lines.append(f"        from {ROOT_NAMES[finding.root]}:")
        lines += [f"        {number}. {describe(event)}" for number, event in enumerate(finding.path, 1)]
    return "\n".join(lines)
