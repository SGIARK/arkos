"""Draw the declared machines as pages you click through, and serve them to a browser.

Graphviz lays each machine out as an SVG in which every box is a plain link, so the
pages need no JavaScript. The first page shows every machine at once, and clicking
one opens it on its own, with what triggers each move. Run it with:

    python -m state_machine_module.view [--port N] [--no-open]

It needs Graphviz's `dot` on the PATH: `sudo apt install graphviz`.
"""

from __future__ import annotations

import textwrap

from state_machine_module.machines import Machine, Move

LABEL_WIDTH = 26  # characters on each line of a move's label
START_STYLE = 'shape=point, width=0.12, style=filled, fillcolor="#3b3b3b"'


def quoted(text: str) -> str:
    """`text` as a DOT string, its line breaks kept."""
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped.replace("\n", "\\n") + '"'


def wrapped(text: str) -> str:
    """A trigger broken into short lines, so its label stays narrow."""
    return "\n".join(textwrap.wrap(text, LABEL_WIDTH))


def page_path(machine: Machine) -> str:
    """Where the page for `machine` is served."""
    return f"/{machine.name}"


def start_lines(machine: Machine, prefix: str, labelled: bool) -> list[str]:
    """A dot for each way the machine begins, with an arrow into the state it begins in."""
    lines = []
    for number, start in enumerate(machine.starts):
        dot = quoted(f"{prefix}start {number}")
        label = f"label={quoted(wrapped(start.trigger))}" if labelled else f"tooltip={quoted(start.trigger)}"
        lines.append(f"{dot} [{START_STYLE}, tooltip={quoted(start.trigger)}];")
        lines.append(f"{dot} -> {quoted(prefix + start.state)} [{label}];")
    return lines


def frames(machine: Machine) -> dict[str, tuple[str, ...]]:
    """The states drawn framed together: those the machine ends in, and the rest, if two or more each."""
    if not machine.final:
        return {}
    live = tuple(state for state in machine.states if state not in machine.final)
    return {name: states for name, states in (("live", live), ("ended", machine.final)) if len(states) > 1}


def shared_moves(machine: Machine, states: tuple[str, ...]) -> list[Move]:
    """The moves every one of `states` makes alike, to a state outside them: one Move each, from the first."""
    return [
        move
        for move in machine.moves
        if move.source == states[0]
        and move.target not in states
        and all(Move(state, move.target, move.trigger) in machine.moves for state in states)
    ]
