"""The viewer serves the first page, every machine and every tracked doc, and nothing else."""

import threading
import urllib.request
from http import HTTPStatus

from state_machine_module.machines import LEASE, MACHINES, SESSION
from state_machine_module.view import doc_link, legend, machine_page, route, serve

TIMEOUT_S = 10
DOCS = {"README.md", ".github/CLAUDE.md", "docs/specs/self-hosted-e2b.md"}


def test_legend_leaves_out_what_a_drawing_does_not_use() -> None:
    assert "frame" in legend(SESSION)
    assert "frame" not in legend(LEASE) and "double border" not in legend(LEASE)


def test_machine_page_lists_every_start_and_move() -> None:
    page = machine_page(SESSION, svg="<svg></svg>")
    assert page.count("<tr>") == 1 + len(SESSION.starts) + len(SESSION.moves)


def test_doc_link_leads_a_relative_link_to_the_doc_it_names() -> None:
    link = doc_link("docs/specs/self-hosted-e2b.md", "../../README.md#running-it", DOCS)
    assert link == "/docs/README.md#running-it"


def test_doc_link_keeps_web_addresses_and_headings() -> None:
    assert doc_link("README.md", "https://example.com/a", DOCS) == "https://example.com/a"
    assert doc_link("README.md", "#tests-and-ci", DOCS) == "#tests-and-ci"


def test_doc_link_leaves_unknown_and_unsafe_targets_unlinked() -> None:
    assert doc_link("README.md", "harness_module/api.py", DOCS) is None
    assert doc_link("README.md", "javascript:alert(1)", DOCS) is None


def test_route_an_unknown_path_is_not_found() -> None:
    assert route("/no-such-machine")[0] == HTTPStatus.NOT_FOUND


def test_route_links_the_first_page_to_every_machine() -> None:
    status, page = route("/")
    assert status == HTTPStatus.OK
    for machine in MACHINES:
        assert f'href="/{machine.name}"' in page, machine.name


def test_route_draws_a_page_for_every_machine() -> None:
    for machine in MACHINES:
        status, page = route(f"/{machine.name}?from=a-link")
        assert status == HTTPStatus.OK and "<svg" in page, machine.name


def test_serve_answers_a_browser() -> None:
    server = serve(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/lease", timeout=TIMEOUT_S) as response:
            assert response.status == HTTPStatus.OK
            assert "orphaned" in response.read().decode()
    finally:
        server.shutdown()
        server.server_close()
