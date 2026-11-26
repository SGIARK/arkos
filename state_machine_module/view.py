"""Serve arkos's docs and state machines to a browser, drawn in plain Python.

The first page shows every state machine at once and lists the docs. Clicking a
machine opens it with what triggers each move, and each doc opens as a page.
From a clone, with nothing installed, run:

    python3 -m state_machine_module.view [--port N] [--no-open]

`layout.py` places each machine, `svg.py` draws it, and `docs.py` renders the
markdown, so the pages need no JavaScript and the server only the standard
library.
"""

from __future__ import annotations

import argparse
import contextlib
import html
import posixpath
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlsplit

from state_machine_module.docs import ROOT, title_of, tracked_docs
from state_machine_module.layout import drawn_moves, layout
from state_machine_module.machines import MACHINES, Machine
from state_machine_module.svg import drawing_svg, overview_svg

HOST = "127.0.0.1"
DOCS = "/docs"
WEB_LINKS = ("http://", "https://", "mailto:")

BY_NAME = {machine.name: machine for machine in MACHINES}

PAGE_STYLE = """
body { font: 15px/1.55 system-ui, sans-serif; color: #1f1f1f; background: #f7f7f7; margin: 0; }
main { max-width: 80rem; margin: 0 auto; padding: 1.25rem 2rem 3rem; }
a { color: #1a5fb4; }
nav { font-size: 0.9rem; color: #6b6b6b; }
h1 { font-size: 1.5rem; margin: 0.6rem 0 0.3rem; }
h2 { font-size: 1.2rem; margin: 2rem 0 0.5rem; }
.about { max-width: 46rem; margin: 0 0 1.25rem; color: #3b3b3b; }
.path { font: 0.85rem ui-monospace, monospace; color: #6b6b6b; }
figure { margin: 0 0 2rem; padding: 1rem; background: #fff; border: 1px solid #e2e2e2; border-radius: 10px;
  overflow-x: auto; }
figure svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
figcaption { font-size: 0.85rem; color: #6b6b6b; margin-top: 0.5rem; text-align: center; }
svg text { font-family: system-ui, sans-serif; }
svg .state rect { fill: #fff; stroke: #3b3b3b; stroke-width: 1.2; }
svg .state text { font-size: 13px; fill: #1f1f1f; text-anchor: middle; }
svg a:hover .state rect { stroke: #1a5fb4; stroke-width: 2; }
svg a:hover text { fill: #1a5fb4; }
svg .frame rect { fill: #fafafa; stroke: #d0d0d0; }
svg .frame text { font-size: 11px; font-style: italic; fill: #6b6b6b; }
svg .machine rect { fill: none; stroke: #9a9a9a; stroke-dasharray: 5 4; }
svg .machine text { font-size: 14px; font-weight: 600; fill: #1f1f1f; }
svg .line { fill: none; stroke: #5c5c5c; stroke-width: 1.2; }
svg .hit { fill: none; stroke: transparent; stroke-width: 12; }
svg .head { fill: #5c5c5c; }
svg .move:hover .line { stroke: #1a5fb4; stroke-width: 2; }
svg .entry circle { fill: #3b3b3b; }
svg .label rect { fill: #fff; fill-opacity: 0.94; }
svg .label text { font-size: 11px; fill: #3b3b3b; text-anchor: middle; }
table { border-collapse: collapse; font-size: 0.93rem; margin: 0 0 1rem; }
th, td { text-align: left; vertical-align: top; padding: 0.4rem 1.5rem 0.4rem 0; border-bottom: 1px solid #e2e2e2; }
.moves td:nth-child(-n+2) { white-space: nowrap; font-family: ui-monospace, monospace; font-size: 0.87rem; }
ul, ol { max-width: 46rem; padding-left: 1.4rem; }
li { margin: 0.35rem 0; }
.doc { max-width: 50rem; }
.doc h1 { font-size: 1.6rem; } .doc h2 { font-size: 1.25rem; } .doc h3 { font-size: 1.05rem; }
.doc pre { background: #fff; border: 1px solid #e2e2e2; border-radius: 8px; padding: 0.75rem 1rem; overflow-x: auto; }
.doc code { font: 0.88em ui-monospace, monospace; }
.doc :not(pre) > code { background: #ececec; padding: 0.1em 0.3em; border-radius: 4px; }
.doc blockquote { margin: 0 0 1rem; padding: 0 1rem; border-left: 3px solid #d0d0d0; color: #4a4a4a; }
.doc hr { border: 0; border-top: 1px solid #e2e2e2; margin: 1.5rem 0; }
"""

OVERVIEW_ABOUT = (
    "A session, drawn as the machines it is made of. They run side by side, and one event can move several"
    " at once: a restart fails the session, kills its box and orphans its lease. Click a machine to see what"
    " moves it."
)
DOCS_ABOUT = "The markdown files this repo keeps, rendered here much as they read on GitHub."


def page_path(machine: Machine) -> str:
    """Where the page for `machine` is served."""
    return f"/{machine.name}"


def doc_path(doc: str) -> str:
    """Where the page for the doc at `doc`, a path in the repo, is served."""
    return f"{DOCS}/{doc}"


def menu(current: str) -> str:
    """Links to the first page, the docs and every machine, the page you are on in bold."""
    entries = [("arkos", "/"), ("docs", DOCS), *((machine.name, page_path(machine)) for machine in MACHINES)]
    links = [
        f"<b>{html.escape(name)}</b>" if name == current else f'<a href="{path}">{html.escape(name)}</a>'
        for name, path in entries
    ]
    return " · ".join(links)


def page(title: str, current: str, body: str) -> str:
    """A whole HTML page: the menu with `current` in bold, then `body`, which is already HTML."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} · arkos</title>
<style>{PAGE_STYLE}</style>
</head>
<body>
<main>
<nav>{menu(current)}</nav>
{body}
</main>
</body>
</html>
"""


def heading(title: str, about: str) -> str:
    """A page's title, and the paragraph under it saying what the page is about."""
    return f'<h1>{html.escape(title)}</h1><p class="about">{html.escape(about)}</p>'


def figure(svg: str, caption: str) -> str:
    """A drawing with a line under it saying how to read it."""
    return f"<figure>{svg}<figcaption>{html.escape(caption)}</figcaption></figure>"


def titled_docs() -> list[tuple[str, str]]:
    """Every doc the repo tracks, as (path, title)."""
    return [(doc, title_of((ROOT / doc).read_text(encoding="utf-8"), doc)) for doc in tracked_docs()]


def docs_list(docs: list[tuple[str, str]]) -> str:
    """Each doc's title, linked to its page, beside where it lives in the repo."""
    items = "".join(
        f'<li><a href="{html.escape(doc_path(doc))}">{html.escape(title)}</a>'
        f' <span class="path">{html.escape(doc)}</span></li>'
        for doc, title in docs
    )
    return f"<ul>{items}</ul>"


def overview_page(svg: str) -> str:
    """The first page: every machine in one drawing, then a line on what each one tracks."""
    machines = "".join(
        f'<li><a href="{page_path(machine)}">{html.escape(machine.name)}</a>: {html.escape(machine.about)}</li>'
        for machine in MACHINES
    )
    caption = "Click a state or a machine's name to open it. Hover over an arrow to see what triggers it."
    body = heading("arkos", OVERVIEW_ABOUT) + figure(svg, caption) + f"<ul>{machines}</ul>"
    return page("arkos", "arkos", body)


def docs_page(docs: list[tuple[str, str]]) -> str:
    """Every doc, listed."""
    return page("docs", "docs", heading("docs", DOCS_ABOUT) + docs_list(docs))


def doc_link(doc: str, target: str, docs: set[str]) -> str | None:
    """Where a link in `doc` should lead: another doc's page, a heading, or a web address. Anything
    else stays plain text, so nothing a doc links to can run in the page."""
    if target.startswith("#") or target.startswith(WEB_LINKS):
        return target
    path, _, heading_id = target.partition("#")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(doc), path)) if path else ""
    if resolved not in docs:
        return None
    return doc_path(resolved) + (f"#{heading_id}" if heading_id else "")


def moves_table(machine: Machine) -> str:
    """Every way into and between the machine's states, as a table."""
    rows = [("(start)", start.state, start.trigger) for start in machine.starts]
    rows += [(move.source, move.target, move.trigger) for move in machine.moves]
    cells = "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<table class="moves"><tr><th>From</th><th>To</th><th>When</th></tr>{cells}</table>'


def legend(machine: Machine) -> str:
    """How to read one machine's drawing, naming only what it uses."""
    parts = ["A dot marks where it begins."]
    if machine.final:
        parts.append("A double border marks a state it can end in.")
    if any(frame for _, frame in drawn_moves(machine)):
        parts.append("An arrow leaving a frame is the same move from every state in it; the table lists each.")
    return " ".join(parts)


def machine_page(machine: Machine, svg: str) -> str:
    """One machine's page: its drawing with every trigger, then its moves as a table."""
    body = heading(machine.name, machine.about) + figure(svg, legend(machine)) + moves_table(machine)
    return page(machine.name, machine.name, body)


def not_found(path: str) -> tuple[HTTPStatus, str]:
    """The page for a path nothing is served at."""
    return HTTPStatus.NOT_FOUND, page("Not found", "", heading("Not found", f"Nothing is served at {path}."))


def route(path: str) -> tuple[HTTPStatus, str]:
    """The status and page for a requested path."""
    path = unquote(urlsplit(path).path)
    if path == "/":
        drawings = [layout(machine, labelled=False) for machine in MACHINES]
        return HTTPStatus.OK, overview_page(overview_svg(drawings, [page_path(machine) for machine in MACHINES]))
    machine = BY_NAME.get(path.removeprefix("/"))
    if machine is None:
        return not_found(path)
    return HTTPStatus.OK, machine_page(machine, drawing_svg(layout(machine, labelled=True)))


class Handler(BaseHTTPRequestHandler):
    """Answers each GET with its page, and keeps quiet about requests."""

    def do_GET(self) -> None:
        status, body = route(self.path)
        data = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        """A line per request would bury the address the server printed."""


def serve(port: int) -> ThreadingHTTPServer:
    """A server for the pages, reachable from this computer only. Port 0 picks a free one."""
    return ThreadingHTTPServer((HOST, port), Handler)


def main(argv: list[str] | None = None) -> int:
    """Serve the pages and open them in your browser, until Ctrl-C."""
    parser = argparse.ArgumentParser(description="Read arkos's docs and click through its state machines.")
    parser.add_argument("--port", type=int, default=0, help="the port to serve on; a free one if left out")
    parser.add_argument("--no-open", action="store_true", help="print the address without opening a browser")
    args = parser.parse_args(argv)
    server = serve(args.port)
    url = f"http://{HOST}:{server.server_port}/"
    print(f"arkos's docs and state machines are at {url} (Ctrl-C stops the server)", flush=True)
    if not args.no_open:
        webbrowser.open(url)
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()
    server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
