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
import textwrap
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GIT_TIMEOUT_S = 10
SKIPPED_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__"})

FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*([\w+.-]*)")
HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
ITEM = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
DIVIDER = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
QUOTE = re.compile(r"^\s*>\s?(.*)$")
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
CODE_SPAN = re.compile(r"(`+)(.+?)\1")
IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
LINK = re.compile(r'\[([^\]]+)\]\(\s*([^)\s]*)(?:\s+"[^"]*")?\s*\)')
AUTOLINK = re.compile(r"&lt;((?:https?|mailto):(?:[^\s&]|&amp;)+)&gt;")
STRONG = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
EMPHASIS = re.compile(r"(?<![*\w])\*(?=\S)(.+?)(?<=\S)\*(?![*\w])")
KEPT = re.compile(r"\x00(\d+)\x00")

Linker = Callable[[str], str | None]


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


def render(text: str, link: Linker) -> str:
    """A markdown doc as HTML. `link` turns a link's target into an href, or None to leave it as text."""
    lines = COMMENT.sub("", text).splitlines()
    out: list[str] = []
    index = 0
    while index < len(lines):
        index = block(lines, index, out, link)
    return "\n".join(out)


def block(lines: list[str], index: int, out: list[str], link: Linker) -> int:
    """Render the block that starts at line `index` into `out`, and return the line after it."""
    line = lines[index]
    if not line.strip():
        return index + 1
    if FENCE.match(line):
        return code_block(lines, index, out)
    heading = HEADING.match(line)
    if heading:
        level, text = len(heading.group(1)), heading.group(2)
        out.append(f'<h{level} id="{slug(text)}">{inline(text, link)}</h{level}>')
        return index + 1
    if RULE.match(line):
        out.append("<hr>")
        return index + 1
    if "|" in line and index + 1 < len(lines) and DIVIDER.match(lines[index + 1]):
        return table(lines, index, out, link)
    if QUOTE.match(line):
        return quote(lines, index, out, link)
    if ITEM.match(line):
        items, end = list_items(lines, index)
        out.append(nested(items, link))
        return end
    return paragraph(lines, index, out, link)


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


def table(lines: list[str], index: int, out: list[str], link: Linker) -> int:
    """A pipe table: the row above the divider is its header, and each row below it a row."""
    end = index + 2
    while end < len(lines) and "|" in lines[end] and lines[end].strip():
        end += 1
    header = "".join(f"<th>{inline(cell, link)}</th>" for cell in cells(lines[index]))
    body = "".join(
        "<tr>" + "".join(f"<td>{inline(cell, link)}</td>" for cell in cells(row)) + "</tr>"
        for row in lines[index + 2 : end]
    )
    out.append(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>")
    return end


def quote(lines: list[str], index: int, out: list[str], link: Linker) -> int:
    """A block quote: its lines with their > taken off, rendered as markdown of their own."""
    end = index
    while end < len(lines) and QUOTE.match(lines[end]):
        end += 1
    inner = "\n".join(QUOTE.match(line).group(1) for line in lines[index:end])
    out.append(f"<blockquote>{render(inner, link)}</blockquote>")
    return end


def paragraph(lines: list[str], index: int, out: list[str], link: Linker) -> int:
    """Lines up to a blank one, or to one that opens another block, as a single paragraph."""
    end = index + 1
    while end < len(lines) and lines[end].strip() and not starts_block(lines[end]):
        end += 1
    out.append(f"<p>{inline(' '.join(line.strip() for line in lines[index:end]), link)}</p>")
    return end


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


def item_html(body: list[str], link: Linker) -> str:
    """An item's text, inline when it is one paragraph, otherwise as blocks of its own."""
    rest = textwrap.dedent("\n".join(body[1:])).splitlines()
    if all(line.strip() and not starts_block(line) for line in rest):
        return inline(" ".join([body[0], *(line.strip() for line in rest)]), link)
    return render("\n".join([body[0], *rest]), link)


def nested(items: list[tuple[int, str, list[str]]], link: Linker) -> str:
    """Items as lists: an item indented past the one above it opens a list inside that one."""
    parts: list[str] = []
    open_lists: list[tuple[int, str]] = []
    for indent, marker, body in items:
        tag = "ol" if marker[0].isdigit() else "ul"
        while open_lists and indent < open_lists[-1][0]:
            parts.append(f"</li></{open_lists.pop()[1]}>")
        if open_lists and indent == open_lists[-1][0]:
            parts.append("</li>")
        else:
            first = int(marker[:-1]) if tag == "ol" else 1
            parts.append(f'<{tag} start="{first}">' if first != 1 else f"<{tag}>")
            open_lists.append((indent, tag))
        parts.append(f"<li>{item_html(body, link)}")
    parts += [f"</li></{tag}>" for _, tag in reversed(open_lists)]
    return "".join(parts)


def anchor(text: str, target: str, link: Linker) -> str:
    """A link, if `link` resolves its target; just the text if not, so no doc can link to anything unsafe."""
    href = link(html.unescape(target))
    return f'<a href="{html.escape(href)}">{text}</a>' if href else text


def inline(text: str, link: Linker) -> str:
    """One run of text's inline markdown as HTML. Code spans come out first, so nothing in them changes."""
    kept: list[str] = []

    def keep(match: re.Match[str]) -> str:
        kept.append(f"<code>{html.escape(match.group(2).strip())}</code>")
        return f"\x00{len(kept) - 1}\x00"

    text = html.escape(CODE_SPAN.sub(keep, text), quote=False)
    text = IMAGE.sub(lambda match: f'<span class="image">{match.group(1)}</span>', text)
    text = LINK.sub(lambda match: anchor(match.group(1), match.group(2), link), text)
    text = AUTOLINK.sub(lambda match: anchor(match.group(1), match.group(1), link), text)
    text = EMPHASIS.sub(r"<em>\1</em>", STRONG.sub(r"<strong>\1</strong>", text))
    return KEPT.sub(lambda match: kept[int(match.group(1))], text)
