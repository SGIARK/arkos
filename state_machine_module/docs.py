"""The repo's markdown docs: which ones there are, and each one as HTML.

The docs are the markdown files git tracks, so notes kept beside a clone stay
out; without git, every markdown file under the repo counts. The renderer covers
the markdown these docs use (headings, paragraphs, lists holding code, tables,
quotes, rules, links and emphasis) with nothing but the standard library. HTML
comments are dropped, as GitHub drops them, and any other HTML shows as text,
never as markup.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIT_TIMEOUT_S = 10
SKIPPED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def tracked_docs(root: Path = ROOT) -> list[str]:
    """The markdown files git tracks under `root`, as sorted relative paths; when git cannot say,
    every markdown file under it."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z", "--", "*.md"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_S,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return walked_docs(root)
    return sorted(path for path in listed.split("\0") if path and (root / path).is_file())


def walked_docs(root: Path) -> list[str]:
    """Every markdown file under `root`, leaving out virtualenvs, caches and git's own files."""
    found = (path.relative_to(root) for path in root.rglob("*.md") if path.is_file())
    return sorted(path.as_posix() for path in found if not SKIPPED_DIRS & set(path.parts))


def title_of(text: str, fallback: str) -> str:
    """A doc's first heading, or `fallback` if it has none."""
    for line in COMMENT.sub("", text).splitlines():
        match = HEADING.match(line)
        if match:
            return match.group(2).replace("`", "")
    return fallback


def slug(text: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces turned to hyphens."""
    return re.sub(r"[^\w\- ]", "", text.lower()).strip().replace(" ", "-")
