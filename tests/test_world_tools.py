"""The five world tools: they read, and they read only their own user's rows."""

from __future__ import annotations

import json
import uuid

import pytest
import pytest_asyncio

from db import pool
from tests.dbgate import require_db
from tool_module import registry
from tool_module.envelope import ToolContext

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db():
    await require_db()
    yield
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


async def _project(user_id: str, title: str = "A project") -> str:
    return str(
        await pool.fetchval(
            "INSERT INTO projects (user_id, title) VALUES ($1, $2) RETURNING id",
            uuid.UUID(user_id),
            title,
        )
    )


async def _session(user_id: str, project_id: str | None = None, **cols) -> str:
    return str(
        await pool.fetchval(
            """
            INSERT INTO sessions (user_id, project_id, mode, status, title, goal)
            VALUES ($1, $2, $3, $4, $5, $6) RETURNING id
            """,
            uuid.UUID(user_id),
            uuid.UUID(project_id) if project_id else None,
            cols.get("mode", "attended"),
            cols.get("status", "idle"),
            cols.get("title", "A session"),
            cols.get("goal", "do the thing"),
        )
    )


async def _run(name: str, args: dict, user_id: str):
    return await registry.dispatch(name, args, ToolContext(user_id=user_id))


# --- they are in the manifest -------------------------------------------------


async def test_the_five_world_tools_ship_in_the_manifest():
    specs = {s.name: s for s in (await registry.manifest(await _user())).specs}

    for name in ("list_projects", "get_project", "list_sessions", "get_session", "list_files"):
        assert name in specs, f"{name} is missing from the manifest"
        assert specs[name].readonly, f"{name} is not marked readonly"
        assert not specs[name].requires_approval


# --- they read ----------------------------------------------------------------


async def test_list_projects_returns_the_users_projects_with_session_counts():
    user_id = await _user()
    project_id = await _project(user_id, "Taxes")
    await _session(user_id, project_id)

    result = await _run("list_projects", {}, user_id)
    rows = json.loads(result.content)

    assert result.ok
    assert [r["title"] for r in rows] == ["Taxes"]
    assert rows[0]["sessions"] == 1


async def test_get_project_carries_its_sessions():
    user_id = await _user()
    project_id = await _project(user_id, "Taxes")
    await _session(user_id, project_id, title="File the return")

    result = await _run("get_project", {"project_id": project_id}, user_id)
    body = json.loads(result.content)

    assert body["title"] == "Taxes"
    assert [s["title"] for s in body["sessions"]] == ["File the return"]


async def test_list_sessions_filters_by_project_and_status():
    user_id = await _user()
    project_id = await _project(user_id)
    await _session(user_id, project_id, status="completed", title="done one")
    await _session(user_id, project_id, status="idle", title="open one")
    await _session(user_id, None, status="idle", title="unfiled")

    in_project = json.loads((await _run("list_sessions", {"project_id": project_id}, user_id)).content)
    completed = json.loads((await _run("list_sessions", {"status": "completed"}, user_id)).content)

    assert {s["title"] for s in in_project} == {"done one", "open one"}
    assert [s["title"] for s in completed] == ["done one"]


async def test_get_session_carries_the_tail_of_the_transcript():
    from agent_module.events import BudgetEvent, ContentEvent, UserEvent
    from harness_module import session_log as slog

    user_id = await _user()
    session_id = await _session(user_id, goal="find the receipt")
    await slog.append(session_id, UserEvent(text="where is it"))
    await slog.append(session_id, BudgetEvent(hops_used=1, hops_max=6))
    await slog.append(session_id, ContentEvent(text="in the drawer"))

    body = json.loads((await _run("get_session", {"session_id": session_id}, user_id)).content)

    assert body["goal"] == "find the receipt"
    # recent_events carries user text and replies; budget meters are left out.
    assert [e["kind"] for e in body["recent_events"]] == ["user", "content"]


async def test_list_files_reports_an_empty_project_plainly():
    user_id = await _user()
    project_id = await _project(user_id)

    result = await _run("list_files", {"project_id": project_id}, user_id)

    assert result.ok
    assert "No files" in result.content


# --- and they read nothing else ------------------------------------------------


async def test_another_users_project_is_not_found_not_forbidden():
    """Another user's row and an id that exists nowhere both report `not_found`."""
    mine, theirs = await _user(), await _user()
    their_project = await _project(theirs, "Secret")
    their_session = await _session(theirs, their_project)

    project = await _run("get_project", {"project_id": their_project}, mine)
    session = await _run("get_session", {"session_id": their_session}, mine)
    missing = await _run("get_project", {"project_id": str(uuid.uuid4())}, mine)

    assert (project.ok, project.error_kind) == (False, "not_found")
    assert (session.ok, session.error_kind) == (False, "not_found")
    assert project.error_kind == missing.error_kind


async def test_listing_never_crosses_users():
    mine, theirs = await _user(), await _user()
    await _project(theirs, "Secret")
    await _session(theirs, None, title="Secret work")

    projects = await _run("list_projects", {}, mine)
    sessions = await _run("list_sessions", {}, mine)

    assert "Secret" not in projects.content
    assert "Secret work" not in sessions.content


async def test_another_users_files_are_invisible():
    mine, theirs = await _user(), await _user()
    their_project = await _project(theirs)
    await pool.execute(
        "INSERT INTO project_files (project_id, path, content_hash, size) VALUES ($1, $2, $3, $4)",
        uuid.UUID(their_project),
        "payroll.csv",
        "0" * 64,
        10,
    )

    result = await _run("list_files", {"project_id": their_project}, mine)

    assert "payroll.csv" not in result.content


async def test_a_malformed_id_is_a_miss_not_a_crash():
    user_id = await _user()

    result = await _run("get_project", {"project_id": "not-a-uuid"}, user_id)

    assert (result.ok, result.error_kind) == (False, "not_found")
