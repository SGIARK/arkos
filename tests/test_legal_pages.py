"""The legal pages: every one exists, versions match config, and the Privacy Policy
and cookies page say what the code does.

Files only, read with the standard-library HTML parser.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from config_module.loader import cfg

ROOT = Path(__file__).resolve().parent.parent
LEGAL = ROOT / "frontend" / "legal"
# The only stand-ins the public pages may carry until launch (launch plan §0.2), and the
# open decisions they may be tagged with. A launch check fails while any remain.
PLACEHOLDERS = {
    "{{OPERATOR_LEGAL_NAME}}",
    "{{OPERATOR_POSTAL_ADDRESS}}",
    "{{EFFECTIVE_DATE}}",
    "{{SUPPORT_EMAIL}}",
    "{{PRIVACY_EMAIL}}",
    "{{SECURITY_EMAIL}}",
    "{{LEGAL_EMAIL}}",
    "{{GOVERNING_LAW}}",
    "{{MIN_AGE}}",
    "{{LLM_PROVIDER}}",
    "{{PUBLIC_URL}}",
    "{{SOURCE_URL}}",
    "{{MIT_RELATIONSHIP_STATEMENT}}",
    "{{HOSTING_PROVIDER}}",
}
DECISIONS = {f"D{n}" for n in range(1, 20)}

# The pages that ship now; the other legal documents come with their text.
PAGES = ["index", "privacy", "subprocessors", "cookies"]

PRIVACY_SECTIONS = ["collect", "use", "ai", "who-sees", "sharing", "retention", "rights", "contact"]


class _Page(HTMLParser):
    """The version meta, the section ids in order, and the text inside <main>."""

    def __init__(self) -> None:
        super().__init__()
        self.version: str | None = None
        self.ids: list[str] = []
        self.main_words = 0
        self._in_main = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "meta" and a.get("name") == "arkos-doc-version":
            self.version = a.get("content")
        if tag == "main":
            self._in_main = True
        if a.get("id"):
            self.ids.append(a["id"] or "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "main":
            self._in_main = False

    def handle_data(self, data: str) -> None:
        if self._in_main:
            self.main_words += len(data.split())


def _parse(name: str) -> _Page:
    page = _Page()
    page.feed((LEGAL / f"{name}.html").read_text())  # comments never reach handle_data
    return page


def test_every_legal_page_exists():
    """Every shipped page is there, so no footer or index link is dead."""
    missing = [p for p in PAGES if not (LEGAL / f"{p}.html").is_file()]
    assert missing == []


def test_legal_page_versions_match_config():
    """The Privacy Policy's version is the one users accept."""
    assert _parse("privacy").version == cfg("legal.privacy_version")


def test_placeholders_and_decisions_are_known():
    """Every {{placeholder}} is a known stand-in, and every decision tag is a known decision."""
    known = PLACEHOLDERS
    decisions = DECISIONS
    public = [*LEGAL.glob("*.html"), ROOT / "frontend" / "landing" / "index.html"]
    for page in public:
        text = page.read_text()
        unknown = set(re.findall(r"\{\{[A-Z_]+\}\}", text)) - known
        assert unknown == set(), f"{page.name}: {unknown}"
        tags = set(re.findall(r'data-decision="(D\d+)"', text)) | set(re.findall(r"<!-- (D\d+) alternative", text))
        assert tags <= decisions, f"{page.name}: {tags - decisions}"


def test_privacy_policy_is_short_and_complete():
    """Eight anchored sections in order, under 1,500 words of visible text."""
    page = _parse("privacy")
    order = [i for i in page.ids if i in PRIVACY_SECTIONS]
    assert order == PRIVACY_SECTIONS
    assert page.main_words < 1500, f"{page.main_words} words"
    assert "Limited Use" in (LEGAL / "privacy.html").read_text()


def test_cookies_page_lists_every_storage_key():
    """Every arkos-… key the frontend uses is disclosed, and so is the sign-in cookie."""
    keys: set[str] = set()
    for src in (ROOT / "frontend").rglob("*"):
        if src.suffix not in (".js", ".jsx") or "vendor" in src.parts:  # storage is only ever set from scripts
            continue
        keys |= set(re.findall(r"[\"'`](arkos-[a-z0-9-]+)[\"'`]", src.read_text()))
    cookies = (LEGAL / "cookies.html").read_text()
    assert keys, "expected at least the theme key"
    assert sorted(k for k in keys if k not in cookies) == []
    assert str(cfg("auth.cookie_name")) in cookies
