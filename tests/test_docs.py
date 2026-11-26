"""The viewer finds the docs git tracks, and renders their markdown without trusting it."""

from pathlib import Path

from state_machine_module.docs import render, title_of, tracked_docs, walked_docs


def no_links(target: str) -> str | None:
    return None


def test_render_gives_headings_anchors() -> None:
    assert render("## Where things are", no_links) == '<h2 id="where-things-are">Where things are</h2>'


def test_render_keeps_code_exactly_as_written() -> None:
    html = render("```bash\necho <b>*not emphasis*</b>\n```", no_links)
    assert html == '<pre><code class="language-bash">echo &lt;b&gt;*not emphasis*&lt;/b&gt;</code></pre>'


def test_render_drops_comments_and_shows_other_html_as_text() -> None:
    assert render("<!-- a note -->\n<script>x</script>", no_links) == "<p>&lt;script&gt;x&lt;/script&gt;</p>"


def test_render_keeps_code_blocks_inside_numbered_steps() -> None:
    html = render("1. Install:\n\n   ```bash\n   pip install x\n   ```\n\n2. Run it.", no_links)
    code = '<pre><code class="language-bash">pip install x</code></pre>'
    assert html == f"<ol><li><p>Install:</p>\n{code}</li><li>Run it.</li></ol>"


def test_render_draws_pipe_tables() -> None:
    html = render("| Path | What |\n| --- | --- |\n| `db/` | the pool |", no_links)
    row = "<tr><td><code>db/</code></td><td>the pool</td></tr>"
    assert html == f"<table><thead><tr><th>Path</th><th>What</th></tr></thead><tbody>{row}</tbody></table>"


def test_render_links_only_what_the_linker_allows() -> None:
    def docs_only(target: str) -> str | None:
        return "/docs/x.md" if target == "x.md" else None

    html = render("[ok](x.md) and [bad](javascript:alert)", docs_only)
    assert html == '<p><a href="/docs/x.md">ok</a> and bad</p>'


def test_render_marks_strong_and_emphasis_but_leaves_snake_case() -> None:
    html = render("**bold** and *this* but not snake_case_name", no_links)
    assert html == "<p><strong>bold</strong> and <em>this</em> but not snake_case_name</p>"


def test_title_of_takes_the_first_heading() -> None:
    assert title_of("<!-- # a note -->\nintro\n# The `arkos` title\n## Next", "fallback") == "The arkos title"


def test_tracked_docs_lists_the_readme() -> None:
    docs = tracked_docs()
    assert "README.md" in docs
    assert all(doc.endswith(".md") for doc in docs)


def test_tracked_docs_walks_the_disk_without_git(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("# Notes")
    assert tracked_docs(tmp_path) == ["notes.md"]


def test_walked_docs_skips_virtualenvs(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Guide")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "vendored.md").write_text("# Not ours")
    assert walked_docs(tmp_path) == ["docs/guide.md"]
