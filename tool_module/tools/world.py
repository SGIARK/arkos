"""
The world tools: what the model can see of its own installation.

Reads only, all of them, and every query is scoped to `ctx.user_id` in SQL
rather than filtered afterwards. A row belonging to someone else comes back as
`not_found`, which is also what a row that does not exist comes back as — the
model must not be able to probe for the existence of another user's project.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from db import pool
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok

# A model that asks for "everything" gets a page, not the table.
_DEFAULT_LIMIT = 50
_MAX_LIMIT = 200


def _render(rows: Any) -> str:
    """Serialize rows for the model: JSON, because a tool result is read, not displayed."""
    return json.dumps(rows, indent=2, default=str)


def _limit(args: dict[str, Any]) -> int:
    try:
        wanted = int(args.get("limit") or _DEFAULT_LIMIT)
    except (TypeError, ValueError):
        return _DEFAULT_LIMIT
    return max(1, min(wanted, _MAX_LIMIT))


def _as_uuid(value: Any) -> uuid.UUID | None:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


class ListProjects:
    spec = ToolSpec(
        name="list_projects",
        description=(
            "List the user's projects, most recently updated first. A project is the "
            "container a session belongs to; use it to find work you or the user did before."
        ),
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": f"Default {_DEFAULT_LIMIT}."}},
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        rows = await pool.fetch(
            """
            SELECT p.id, p.title, p.created_at, p.updated_at,
                   count(s.id) AS sessions
              FROM projects p LEFT JOIN sessions s ON s.project_id = p.id
             WHERE p.user_id = $1
             GROUP BY p.id
             ORDER BY p.updated_at DESC
             LIMIT $2
            """,
            _as_uuid(ctx.user_id),
            _limit(args),
        )
        if not rows:
            return ok("No projects yet.")
        return ok(_render([dict(r) for r in rows]))


class GetProject:
    spec = ToolSpec(
        name="get_project",
        description="Read one project and the sessions in it, by id.",
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        project_id = _as_uuid(args["project_id"])
        project = await pool.fetchrow(
            "SELECT id, title, created_at, updated_at FROM projects WHERE id = $1 AND user_id = $2",
            project_id,
            _as_uuid(ctx.user_id),
        )
        if project is None:
            return fail("not_found", f"No project {args['project_id']!r}.")
        sessions = await pool.fetch(
            """
            SELECT id, title, status, mode, terminal_reason, created_at, ended_at
              FROM sessions WHERE project_id = $1 ORDER BY created_at DESC LIMIT $2
            """,
            project_id,
            _MAX_LIMIT,
        )
        return ok(_render({**dict(project), "sessions": [dict(s) for s in sessions]}))


class ListSessions:
    spec = ToolSpec(
        name="list_sessions",
        description=(
            "List the user's sessions, newest first. Optionally filter to one project or "
            "one status (pending, idle, running, awaiting_approval, completed, failed, cancelled)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "project_id": {"type": "string"},
                "status": {"type": "string"},
                "limit": {"type": "integer", "description": f"Default {_DEFAULT_LIMIT}."},
            },
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        # NULL means "no filter", so one query serves all four combinations.
        rows = await pool.fetch(
            """
            SELECT id, project_id, title, goal, status, mode, terminal_reason,
                   hops_used, created_at, ended_at
              FROM sessions
             WHERE user_id = $1
               AND ($2::uuid IS NULL OR project_id = $2)
               AND ($3::text IS NULL OR status = $3)
             ORDER BY created_at DESC
             LIMIT $4
            """,
            _as_uuid(ctx.user_id),
            _as_uuid(args.get("project_id")) if args.get("project_id") else None,
            args.get("status"),
            _limit(args),
        )
        if not rows:
            return ok("No sessions match.")
        return ok(_render([dict(r) for r in rows]))


class GetSession:
    spec = ToolSpec(
        name="get_session",
        description=(
            "Read one session by id: its goal, status, and the tail of its transcript. "
            "Use it to pick up what another session found without re-doing the work."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "events": {"type": "integer", "description": "How many trailing events to include. Default 20."},
            },
            "required": ["session_id"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        session = await pool.fetchrow(
            """
            SELECT id, project_id, title, goal, status, mode, terminal_reason,
                   hops_used, created_at, ended_at
              FROM sessions WHERE id = $1 AND user_id = $2
            """,
            _as_uuid(args["session_id"]),
            _as_uuid(ctx.user_id),
        )
        if session is None:
            return fail("not_found", f"No session {args['session_id']!r}.")

        try:
            wanted = max(0, min(int(args.get("events") or 20), _MAX_LIMIT))
        except (TypeError, ValueError):
            wanted = 20
        # Only the kinds that carry meaning to a reader: a wall of content chunks
        # and budget meters is what the summary is for.
        tail = await pool.fetch(
            """
            SELECT kind, payload, ts FROM (
                SELECT seq, kind, payload, ts FROM session_events
                 WHERE session_id = $1 AND kind IN ('user', 'content', 'tool_call', 'done')
                 ORDER BY seq DESC LIMIT $2
            ) t ORDER BY seq
            """,
            session["id"],
            wanted,
        )
        return ok(_render({**dict(session), "recent_events": [dict(e) for e in tail]}))


class ListFiles:
    spec = ToolSpec(
        name="list_files",
        description=(
            "List the files the user uploaded to a project. These are copied into the "
            "sandbox when it starts, so read them there by name rather than through this tool."
        ),
        input_schema={
            "type": "object",
            "properties": {"project_id": {"type": "string"}},
            "required": ["project_id"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        # Joined to projects rather than trusting the id: project_files has no
        # user column, so the ownership check has to happen in the join.
        rows = await pool.fetch(
            """
            SELECT f.id, f.name, f.size_bytes, f.created_at
              FROM project_files f JOIN projects p ON p.id = f.project_id
             WHERE f.project_id = $1 AND p.user_id = $2
             ORDER BY f.created_at
             LIMIT $3
            """,
            _as_uuid(args["project_id"]),
            _as_uuid(ctx.user_id),
            _MAX_LIMIT,
        )
        if not rows:
            # Ambiguous on purpose: an empty project and someone else's project
            # must read the same, or this becomes an existence oracle.
            return ok("No files in that project.")
        return ok(_render([dict(r) for r in rows]))


TOOLS = [ListProjects(), GetProject(), ListSessions(), GetSession(), ListFiles()]
