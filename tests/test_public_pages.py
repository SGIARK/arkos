"""The landing page and the cookie notice: what a visitor sees before signing in.

Files only, no server and no database. The `/` route itself is in test_api.py.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from config_module.loader import cfg

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
LANDING = FRONTEND / "landing" / "index.html"


class _Refs(HTMLParser):
    """Every src and href on a page, split into what it loads and where it links."""

    def __init__(self) -> None:
        super().__init__()
        self.loads: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "script" and a.get("src"):
            self.loads.append(a["src"] or "")
        elif tag == "link" and a.get("href"):
            self.loads.append(a["href"] or "")
        elif tag in ("img", "source", "iframe") and a.get("src"):
            self.loads.append(a["src"] or "")
        elif tag == "a" and a.get("href"):
            self.links.append(a["href"] or "")


def _refs(page: Path) -> _Refs:
    parser = _Refs()
    parser.feed(page.read_text())
    return parser


def _offsite(url: str) -> bool:
    return bool(re.match(r"^(https?:)?//", url.strip()))


def _public_pages() -> list[Path]:
    return [LANDING, *sorted((FRONTEND / "legal").glob("*.html")), *sorted((FRONTEND / "help").glob("*.html"))]


def test_landing_loads_nothing_offsite():
    """No script, stylesheet, image or font on the landing page comes from another host."""
    loads = _refs(LANDING).loads
    assert loads, "the landing page should load its own stylesheet"
    assert [u for u in loads if _offsite(u)] == []

    for css in (FRONTEND / "landing" / "landing.css", FRONTEND / "legal" / "legal.css"):
        text = css.read_text()
        for url in re.findall(r"url\(\s*['\"]?([^'\")]+)", text) + re.findall(r"@import\s+['\"]([^'\"]+)", text):
            assert not _offsite(url), f"{css.name} loads {url}"


def test_landing_paths_exist():
    """Every /app/ path the landing page loads or links maps to a file, and Source is the configured repo."""
    refs = _refs(LANDING)
    for url in refs.loads + refs.links:
        if not url.startswith("/app/"):
            continue
        rel = url.split("?", 1)[0].removeprefix("/app/")
        target = FRONTEND / rel
        if rel == "" or rel.endswith("/"):
            target = target / "index.html"
        assert target.is_file(), f"{url} has no file behind it"

    sources = [u for u in refs.links if "github.com" in u]
    assert sources == [str(cfg("legal.source_url"))]


def test_cookie_notice_on_public_pages():
    """The landing, legal and help pages all load the notice, so any first visit sees it."""
    for page in _public_pages():
        loads = [u.split("?", 1)[0] for u in _refs(page).loads]
        assert "/app/legal/cookie-notice.js" in loads, f"{page.relative_to(FRONTEND)} has no cookie notice"
