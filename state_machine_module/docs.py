"""The repo's markdown docs: which ones there are, and each one as HTML.

The docs are the markdown files git tracks, so notes kept beside a clone stay
out; without git, every markdown file under the repo counts. The renderer covers
the markdown these docs use (headings, paragraphs, lists holding code, tables,
quotes, rules, links and emphasis) with nothing but the standard library. HTML
comments are dropped, as GitHub drops them, and any other HTML shows as text,
never as markup.
"""

from __future__ import annotations

from pathlib import Path

SKIPPED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})


def walked_docs(root: Path) -> list[str]:
    """Every markdown file under `root`, leaving out virtualenvs, caches and git's own files."""
    found = (path.relative_to(root) for path in root.rglob("*.md") if path.is_file())
    return sorted(path.as_posix() for path in found if not SKIPPED_DIRS & set(path.parts))
