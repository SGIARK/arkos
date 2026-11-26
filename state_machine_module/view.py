"""Draw the declared machines as pages you click through, and serve them to a browser.

Graphviz lays each machine out as an SVG in which every box is a plain link, so the
pages need no JavaScript. The first page shows every machine at once, and clicking
one opens it on its own, with what triggers each move. Run it with:

    python -m state_machine_module.view [--port N] [--no-open]

It needs Graphviz's `dot` on the PATH: `sudo apt install graphviz`.
"""

from __future__ import annotations

import textwrap

from state_machine_module.machines import Machine

LABEL_WIDTH = 26  # characters on each line of a move's label


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
