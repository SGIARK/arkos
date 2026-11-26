"""The repo's markdown docs: which ones there are, and each one as HTML.

The docs are the markdown files git tracks, so notes kept beside a clone stay
out; without git, every markdown file under the repo counts. The renderer covers
the markdown these docs use (headings, paragraphs, lists holding code, tables,
quotes, rules, links and emphasis) with nothing but the standard library. HTML
comments are dropped, as GitHub drops them, and any other HTML shows as text,
never as markup.
"""

from __future__ import annotations

import html
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIT_TIMEOUT_S = 10
SKIPPED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})

FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+.-]*)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
QUOTE = re.compile(r"^\s*>\s?(.*)$")
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


def starts_block(line: str) -> bool:
    """Whether `line` opens a block of its own, and so ends a paragraph above it."""
    return any(pattern.match(line) for pattern in (FENCE, HEADING, RULE, QUOTE, ITEM))


def slug(text: str) -> str:
    """The anchor GitHub gives a heading: lower case, punctuation dropped, spaces turned to hyphens."""
    return re.sub(r"[^\w\- ]", "", text.lower()).strip().replace(" ", "-")


def code_block(lines: list[str], index: int, out: list[str]) -> int:
    """A fenced block, kept exactly as written; a language named after the fence becomes its class."""
    fence, language = FENCE.match(lines[index]).groups()
    end = index + 1
    while end < len(lines) and not lines[end].strip().startswith(fence):
        end += 1
    kind = f' class="language-{html.escape(language)}"' if language else ""
    out.append(f"<pre><code{kind}>{html.escape(chr(10).join(lines[index + 1 : end]))}</code></pre>")
    return end + 1


def cells(line: str) -> list[str]:
    """A table row split into cells, its outer pipes dropped; an escaped pipe stays in its cell."""
    row = line.strip().removeprefix("|").removesuffix("|")
    return [cell.strip().replace("\\|", "|") for cell in re.split(r"(?<!\\)\|", row)]


def indent_of(line: str) -> int:
    """How many spaces start `line`."""
    return len(line) - len(line.lstrip(" "))


def continues(lines: list[str], index: int) -> bool:
    """Whether a list goes on past the blank line at `index`: the next line with text is an item,
    or is indented under one."""
    for line in lines[index + 1 :]:
        if line.strip():
            return bool(ITEM.match(line)) or indent_of(line) > 0
    return False


def belongs(lines: list[str], index: int, fenced: bool) -> bool:
    """Whether line `index` belongs to the list item above it: it lies in a code block the item
    opened, is indented under the item, is a blank line the list goes on past, or carries the item's
    text on with no blank line between."""
    line = lines[index]
    if fenced or (line.strip() and indent_of(line) > 0):
        return True
    if not line.strip():
        return continues(lines, index)
    return bool(lines[index - 1].strip()) and not starts_block(line)


def list_items(lines: list[str], index: int) -> tuple[list[tuple[int, str, list[str]]], int]:
    """A list's items from line `index`, each with its indent, marker and lines, and the line after
    the list."""
    items: list[tuple[int, str, list[str]]] = []
    end, fenced = index, False
    while end < len(lines):
        line = lines[end]
        match = None if fenced else ITEM.match(line)
        if match:
            items.append((len(match.group(1)), match.group(2), [match.group(3)]))
        elif belongs(lines, end, fenced):
            items[-1][2].append(line)
            fenced = fenced != bool(FENCE.match(line))
        else:
            break
        end += 1
    return items, end
