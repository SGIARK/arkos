"""Memory: appended by sessions, searched by them, carried between them.

The store half (append, curate, search) and the four tools over it, plus the
case the whole thing exists for: something learned in one session is there in
the next. Runs against a real Postgres; the box is a fake.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
import pytest_asyncio

from db import pool
from harness_module import runner, store, workspace
from tests.dbgate import require_db
from tests.test_workspace import FakeSandbox, _sweeping
from tool_module import registry
from tool_module.envelope import ToolContext
from tool_module.sandbox import manager as sandbox_manager

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    store.use_blobs(None)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _user() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(user_id)


async def _session(user_id: str, project_id: str | None = None) -> str:
    return str(
        await pool.fetchval(
            "INSERT INTO sessions (user_id, project_id, mode, status) "
            "VALUES ($1, $2, 'attended', 'idle') RETURNING id",
            uuid.UUID(user_id),
            uuid.UUID(project_id) if project_id else None,
        )
    )


def _ctx(user_id: str, session_id: str | None = None) -> ToolContext:
    return ToolContext(user_id=user_id, session_id=session_id)


# --- the append gate ----------------------------------------------------------------


async def test_a_note_is_a_file_of_its_own():
    user_id = await _user()

    path = await store.append_note(user_id, "the human prefers short replies")

    assert path.startswith(f"{store.NOTES_DIR}/")
    notes = await store.read_notes(user_id)
    assert [(n.path, n.text) for n in notes] == [(path, "the human prefers short replies")]


async def test_concurrent_appends_land_as_separate_files():
    """No read-modify-write anywhere on this path, so a race cannot lose a note."""
    user_id = await _user()

    paths = await asyncio.gather(*(store.append_note(user_id, f"note {i}") for i in range(10)))

    assert len(set(paths)) == 10
    assert {n.text for n in await store.read_notes(user_id)} == {f"note {i}" for i in range(10)}


async def test_notes_read_back_oldest_first():
    user_id = await _user()
    first = await store.append_note(user_id, "one")
    second = await store.append_note(user_id, "two")

    assert [n.path for n in await store.read_notes(user_id)] == [first, second]


async def test_one_users_memory_is_not_anothers():
    mine, theirs = await _user(), await _user()
    await store.append_note(theirs, "not yours")
    await store.update_memory(theirs, "# Theirs\n")

    await store.append_note(mine, "mine")

    assert [n.text for n in await store.read_notes(mine)] == ["mine"]
    assert await store.read_memory(mine) == ""
    assert await store.search_memory(mine, "yours") == []


# --- the curated core ---------------------------------------------------------------


async def test_the_core_reads_empty_until_it_is_curated():
    user_id = await _user()
    await store.append_note(user_id, "a note is not the core")

    assert await store.read_memory(user_id) == ""


async def test_curating_replaces_the_core_whole():
    user_id = await _user()

    await store.update_memory(user_id, "# Memory\n\nShort replies.\n")
    await store.update_memory(user_id, "# Memory\n\nShort replies. Ships on Fridays.\n")

    assert await store.read_memory(user_id) == "# Memory\n\nShort replies. Ships on Fridays.\n"
    assert await store.read_notes(user_id) == [], "the core came back as a note"
    rows = await pool.fetchval(
        "SELECT count(*) FROM memory_files WHERE user_id = $1 AND path = $2",
        uuid.UUID(user_id),
        store.MEMORY_CORE,
    )
    assert rows == 1, "a rewrite left a second core behind"


async def test_two_curations_at_once_serialize_on_the_gate():
    """One of them wins whole. Neither writes half a document, and neither errors."""
    user_id = await _user()

    await asyncio.gather(
        store.update_memory(user_id, "# A\n" + "a" * 500),
        store.update_memory(user_id, "# B\n" + "b" * 500),
    )

    core = await store.read_memory(user_id)
    assert core in ("# A\n" + "a" * 500, "# B\n" + "b" * 500)


# --- search -------------------------------------------------------------------------


async def test_search_finds_a_saved_note():
    user_id = await _user()
    await store.append_note(user_id, "The user's accountant is Dana Okafor, reachable at the Tuesday standup.")
    await store.append_note(user_id, "Deployments go out on Fridays, never before the invoices are filed.")

    hits = await store.search_memory(user_id, "accountant")

    assert [h.text for h in hits] == [
        "The user's accountant is Dana Okafor, reachable at the Tuesday standup."
    ]
    assert hits[0].is_core is False


async def test_search_covers_the_core_as_well_as_the_notes():
    user_id = await _user()
    await store.update_memory(user_id, "# Memory\n\nInvoices are filed before any deployment.\n")

    hits = await store.search_memory(user_id, "invoices")

    assert [h.is_core for h in hits] == [True]


async def test_a_query_that_matches_nothing_returns_nothing():
    user_id = await _user()
    await store.append_note(user_id, "something else entirely")

    assert await store.search_memory(user_id, "kangaroo") == []
    assert await store.search_memory(user_id, "   ") == []


# --- the tools ----------------------------------------------------------------------


async def test_save_memory_then_search_memory_finds_it():
    user_id = await _user()
    ctx = _ctx(user_id)

    saved = await registry.dispatch("save_memory", {"text": "The user bills in euros."}, ctx)
    found = await registry.dispatch("search_memory", {"query": "bills"}, ctx)

    assert saved.ok
    assert found.ok
    assert "euros" in found.content


async def test_save_memory_refuses_a_transcript():
    user_id = await _user()

    result = await registry.dispatch("save_memory", {"text": "x" * 3000}, _ctx(user_id))

    assert result.ok is False
    assert "at most" in result.content
    assert await store.read_notes(user_id) == []


async def test_update_memory_requires_reading_the_document_first():
    """The prompt's copy is capped, so a rewrite from it would drop the tail."""
    user_id = await _user()
    await store.update_memory(user_id, "# Memory\n\nOne. Two. Three.\n")
    ctx = _ctx(user_id)

    blind = await registry.dispatch("update_memory", {"content": "# Memory\n\nOne.\n"}, ctx)

    assert blind.ok is False
    assert "read_memory" in blind.content
    assert await store.read_memory(user_id) == "# Memory\n\nOne. Two. Three.\n"

    read = await registry.dispatch("read_memory", {}, ctx)
    rewritten = await registry.dispatch("update_memory", {"content": "# Memory\n\nOne. Two.\n"}, ctx)

    assert "Three" in read.content
    assert rewritten.ok
    assert await store.read_memory(user_id) == "# Memory\n\nOne. Two.\n"


async def test_read_memory_says_so_when_there_is_nothing_yet():
    result = await registry.dispatch("read_memory", {}, _ctx(await _user()))

    assert result.ok
    assert "empty" in result.content


async def test_the_memory_tools_ship_in_the_manifest():
    specs = {s.name: s for s in await registry.manifest(await _user())}

    for name in ("save_memory", "search_memory", "read_memory", "update_memory"):
        assert name in specs, f"{name} is missing from the manifest"
    assert {n for n in specs if n.endswith("_memory") and specs[n].readonly} == {
        "search_memory",
        "read_memory",
    }


# --- across sessions ----------------------------------------------------------------


async def test_what_one_session_learns_the_next_one_knows():
    """The case memory exists for: session A curates, session B is told without asking."""
    user_id = await _user()
    first = await _session(user_id)
    second = await _session(user_id)
    ctx_a = _ctx(user_id, first)

    await registry.dispatch(
        "save_memory", {"text": "The user's accountant is Dana Okafor."}, ctx_a
    )
    await registry.dispatch("read_memory", {}, ctx_a)
    await registry.dispatch(
        "update_memory", {"content": "# Memory\n\nThe user's accountant is Dana Okafor.\n"}, ctx_a
    )

    # A different session, sharing nothing with the first but the user.
    found = await registry.dispatch("search_memory", {"query": "accountant"}, _ctx(user_id, second))
    folded = await runner.fold(await runner.load(second))
    system = folded.messages[0]["content"]

    assert "Dana Okafor" in found.content, "the note did not survive into the next session"
    assert "Dana Okafor" in system, "the curated core did not reach the next session's prompt"
    assert folded.messages[0]["role"] == "system"


async def test_the_prompt_carries_a_capped_core_and_says_where_the_rest_is(monkeypatch):
    user_id = await _user()
    session_id = await _session(user_id)
    await store.update_memory(user_id, "# Memory\n\n" + "long. " * 500)
    monkeypatch.setattr(runner, "_cfg", lambda key, default: 200 if key == "memory.prompt_max_chars" else default)

    system = (await runner.fold(await runner.load(session_id))).messages[0]["content"]

    assert "read_memory" in system
    assert 0 < system.count("long.") < 40, "the cap did not hold"
    assert (await registry.dispatch("read_memory", {}, _ctx(user_id))).content.count("long.") == 500


async def test_a_session_with_no_memory_gets_a_prompt_without_the_section():
    session_id = await _session(await _user())

    system = (await runner.fold(await runner.load(session_id))).messages[0]["content"]

    assert "# MEMORY.md" not in system
    assert "save_memory" in system, "the guidance is not conditional on there being memory"


# --- and it still does not mount ------------------------------------------------------


async def test_a_session_claiming_everything_still_has_no_memory_in_its_box():
    """D30 is open; until it is settled the default posture is that memory stays out."""
    user_id = await _user()
    project_id = str(
        await pool.fetchval(
            "INSERT INTO projects (user_id, title) VALUES ($1, 'Taxes') RETURNING id", uuid.UUID(user_id)
        )
    )
    session_id = await _session(user_id, project_id)
    assert await sandbox_manager.claim_slot(session_id)
    await store.append_note(user_id, "the most sensitive distillate in the system")
    await store.update_memory(user_id, "# Memory\n")
    await store.commit_tree(project_id, [store.FileContent(path="a.txt", content=b"1")])
    sandbox = _sweeping(FakeSandbox())

    claims = [
        workspace.Claim(project_id=project_id, slug="taxes"),
        workspace.Claim(project_id=project_id, slug="taxes-ro", mode="read"),
    ]
    await workspace.materialize(sandbox, session_id, claims)

    landed = set(sandbox.files)
    assert f"{workspace.MOUNT_ROOT}/taxes/a.txt" in landed
    assert not [p for p in landed if "memory" in p.lower()], "memory reached the box"
    assert not [p for p in landed if p.endswith(store.MEMORY_CORE)]
