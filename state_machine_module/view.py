"""Draw the declared machines as pages you click through, and serve them to a browser.

Graphviz lays each machine out as an SVG in which every box is a plain link, so the
pages need no JavaScript. The first page shows every machine at once, and clicking
one opens it on its own, with what triggers each move. Run it with:

    python -m state_machine_module.view [--port N] [--no-open]

It needs Graphviz's `dot` on the PATH: `sudo apt install graphviz`.
"""

from __future__ import annotations

import functools
import html
import subprocess
import textwrap

from state_machine_module.machines import MACHINES, Machine, Move

LABEL_WIDTH = 26  # characters on each line of a move's label
DOT_TIMEOUT_S = 10

GRAPH_STYLE = (
    'graph [fontname="Helvetica", fontsize=12, bgcolor="transparent", nodesep=0.45, ranksep=0.5, compound=true];',
    'node [fontname="Helvetica", fontsize=12, shape=box, style="rounded,filled", fillcolor="white",'
    ' color="#3b3b3b", margin="0.18,0.08"];',
    'edge [fontname="Helvetica", fontsize=10, color="#5c5c5c", fontcolor="#3b3b3b", arrowsize=0.7];',
)
CLUSTER_STYLE = 'style="rounded,dashed"; color="#8c8c8c"; fontname="Helvetica-Bold"; fontsize=13; margin=14;'
FRAME_STYLE = 'style="rounded"; color="#cfcfcf"; fontname="Helvetica-Oblique"; fontsize=11; fontcolor="#6b6b6b";'
START_STYLE = 'shape=point, width=0.12, style=filled, fillcolor="#3b3b3b"'

PAGE_STYLE = """
body { font: 15px/1.5 system-ui, sans-serif; color: #1f1f1f; background: #f7f7f7; margin: 0; }
main { max-width: 80rem; margin: 0 auto; padding: 1.25rem 2rem 3rem; }
a { color: #1a5fb4; }
nav { font-size: 0.9rem; color: #6b6b6b; }
h1 { font-size: 1.5rem; margin: 0.6rem 0 0.3rem; }
.about { max-width: 46rem; margin: 0 0 1.25rem; color: #3b3b3b; }
figure { margin: 0 0 2rem; padding: 1rem; background: #fff; border: 1px solid #e2e2e2; border-radius: 10px; }
figure svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
figcaption { font-size: 0.85rem; color: #6b6b6b; margin-top: 0.5rem; text-align: center; }
svg a:hover path, svg a:hover polygon, svg a:hover ellipse { stroke: #1a5fb4; stroke-width: 2px; }
svg a:hover text { fill: #1a5fb4; }
table { border-collapse: collapse; font-size: 0.93rem; }
th, td { text-align: left; vertical-align: top; padding: 0.4rem 1.5rem 0.4rem 0; border-bottom: 1px solid #e2e2e2; }
td:nth-child(-n+2) { white-space: nowrap; font-family: ui-monospace, monospace; font-size: 0.87rem; }
ul { max-width: 46rem; padding-left: 1.1rem; }
li { margin: 0.35rem 0; }
"""


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


def move_lines(machine: Machine, prefix: str, labelled: bool) -> list[str]:
    """An arrow for each move drawn: labelled with its trigger, or showing it on hover."""
    lines = []
    for move, frame in drawn_moves(machine):
        label = f"label={quoted(wrapped(move.trigger))}" if labelled else f"tooltip={quoted(move.trigger)}"
        tail = f", ltail={frame_id(prefix, frame)}" if frame else ""
        lines.append(f"{quoted(prefix + move.source)} -> {quoted(prefix + move.target)} [{label}{tail}];")
    return lines


def digraph(name: str, body: list[str]) -> str:
    """A whole DOT graph, in the view's style."""
    return "\n".join([f"digraph {quoted(name)} {{", *GRAPH_STYLE, *body, "}"])


def machine_dot(machine: Machine) -> str:
    """One machine on its own, every move labelled with what triggers it."""
    body = start_lines(machine, "", labelled=True) + state_lines(machine, "", link="")
    return digraph(machine.name, body + move_lines(machine, "", labelled=True))


def cluster_lines(machine: Machine) -> list[str]:
    """One machine as a framed group on the overview, its frame and every state linking to its page."""
    prefix = f"{machine.name}."
    href, tooltip = quoted(page_path(machine)), quoted(machine.about)
    return [
        f"subgraph {quoted('cluster_' + machine.name)} {{",
        f"label={quoted(machine.name)}; href={href}; tooltip={tooltip}; {CLUSTER_STYLE}",
        *start_lines(machine, prefix, labelled=False),
        *state_lines(machine, prefix, link=f", href={href}, tooltip={tooltip}"),
        *move_lines(machine, prefix, labelled=False),
        "}",
    ]


def overview_dot() -> str:
    """Every machine side by side, each a framed group that links to its own page."""
    return digraph("arkos", [line for machine in MACHINES for line in cluster_lines(machine)])


@functools.cache
def drawn(source: str) -> str:
    """`source` laid out by Graphviz, as an SVG element to place in a page."""
    result = subprocess.run(
        ["dot", "-Tsvg"], input=source, capture_output=True, text=True, check=True, timeout=DOT_TIMEOUT_S
    )
    return result.stdout[result.stdout.index("<svg") :]


def menu(current: str) -> str:
    """Links to the overview and to every machine, the page you are on in bold."""
    entries = [("arkos", "/"), *((machine.name, page_path(machine)) for machine in MACHINES)]
    links = [
        f"<b>{html.escape(name)}</b>" if name == current else f'<a href="{path}">{html.escape(name)}</a>'
        for name, path in entries
    ]
    return " · ".join(links)


def page(title: str, about: str, body: str) -> str:
    """A whole HTML page. `title` and `about` are text; `body` is already HTML."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · arkos state machines</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
<main>
<nav>{menu(title)}</nav>
<h1>{html.escape(title)}</h1>
<p class="about">{html.escape(about)}</p>
{body}
</main>
</body>
</html>
"""


def figure(svg: str, caption: str) -> str:
    """A drawing with a line under it saying how to read it."""
    return f"<figure>{svg}<figcaption>{html.escape(caption)}</figcaption></figure>"
