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
FRAME_STYLE = 'style="rounded"; color="#cfcfcf"; fontname="Helvetica-Oblique"; fontsize=11; fontcolor="#6b6b6b";'
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


def drawn_moves(machine: Machine) -> list[tuple[Move, str]]:
    """Each arrow to draw, with the frame it leaves from: a move a whole frame shares is drawn once."""
    arrows: list[tuple[Move, str]] = []
    covered: set[Move] = set()
    for name, states in frames(machine).items():
        for move in shared_moves(machine, states):
            arrows.append((move, name))
            covered |= {Move(state, move.target, move.trigger) for state in states}
    return arrows + [(move, "") for move in machine.moves if move not in covered]


def frame_id(prefix: str, name: str) -> str:
    """The DOT name of a frame; Graphviz draws a subgraph as a frame only if its name starts with cluster."""
    return quoted(f"cluster_{prefix}{name}")


def state_line(machine: Machine, state: str, prefix: str, link: str) -> str:
    """A box for one state, with a double border if the machine can end there."""
    border = ", peripheries=2" if state in machine.final else ""
    return f"{quoted(prefix + state)} [label={quoted(state)}{border}{link}];"


def state_lines(machine: Machine, prefix: str, link: str) -> list[str]:
    """A box for each state, the framed ones inside their frames."""
    framed = frames(machine)
    lines = []
    for name, states in framed.items():
        lines.append(f"subgraph {frame_id(prefix, name)} {{ label={quoted(name)}; {FRAME_STYLE}")
        lines += [state_line(machine, state, prefix, link) for state in states]
        lines.append("}")
    loose = [state for state in machine.states if not any(state in states for states in framed.values())]
    return lines + [state_line(machine, state, prefix, link) for state in loose]
