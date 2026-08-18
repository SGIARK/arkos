"""The HTTP surface: session snapshots, the event stream, and commands.

Every error response carries `{code, message, retryable}`. The caller is
identified by the session cookie; no endpoint reads a user id from a header,
body or query string, the OAuth callback included.

Auth, chat, files, attention and the MCP connections surface are served here.
Browser frames have no route until Task 9 rebuilds the browser.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import posixpath
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import jwt
from fastapi import Body, Depends, FastAPI, File, Form, Header, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from agent_module.events import TodoEvent, UserEvent
from config_module.loader import config
from db import pool
from harness_module import approvals, hands, jwt_utils, lifecycle, runner, store, system_log, workspace
from harness_module import session_log as slog
from harness_module.stream import LAGGED, stream
from tool_module.browser.stream import broker as frames
from tool_module.sandbox import manager as sandbox_manager
from tool_module.smithery import AuthRequiredError, SmitheryError

logger = logging.getLogger(__name__)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


# --- error shape ---------------------------------------------------------------


class ApiError(Exception):
    """An error rendered to the client as `{code, message, retryable}`."""

    def __init__(self, status: int, code: str, message: str, retryable: bool = False):
        self.status = status
        self.code = code
        self.message = message
        self.retryable = retryable
        super().__init__(message)


def _error(status: int, code: str, message: str, retryable: bool = False) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"code": code, "message": message, "retryable": retryable},
    )


# --- lifespan ------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the process-wide resources."""
    jwt_utils.assert_secure_secrets()
    config.assert_coherent()
    if not _cfg("app.public_url", ""):
        logger.warning("app.public_url is unset: mutations are not origin-checked and OAuth has no return url")

    # Runs before any session can start: `start` refuses a session whose row
    # still says running.
    with contextlib.suppress(Exception):
        await lifecycle.sweep_interrupted()
    # Slots held by a process that died: reclaimed here rather than at whatever
    # hour their expiry happens to pass.
    with contextlib.suppress(Exception):
        await sandbox_manager.sweep_slots()
    await hands.start()
    await system_log.start()
    try:
        yield
    finally:
        await system_log.stop()
        await hands.stop()
        await pool.close()


app = FastAPI(title="ARKOS", lifespan=lifespan)

_origin = str(_cfg("app.public_url", "")).rstrip("/")
app.add_middleware(
    CORSMiddleware,
    # /app and the API share one origin, so the session cookie is same-site.
    allow_origins=[_origin] if _origin else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ApiError)
async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
    return _error(exc.status, exc.code, exc.message, exc.retryable)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return _error(500, "internal", "Something failed on our side.", retryable=True)


# --- who is calling ------------------------------------------------------------


async def current_user(request: Request) -> str:
    """Resolve the caller from the session cookie, and origin-check mutations."""
    cookie = request.cookies.get(str(_cfg("auth.cookie_name", "ark_session")))
    if not cookie:
        raise ApiError(401, "unauthenticated", "No session. Sign in first.")
    try:
        claims = jwt_utils.read_session(cookie)
    except jwt.PyJWTError as e:
        raise ApiError(401, "unauthenticated", f"Session rejected: {e}") from e

    if request.method not in ("GET", "HEAD", "OPTIONS"):
        _check_origin(request)
    return str(claims["sub"])


def _check_origin(request: Request) -> None:
    """Reject a mutation whose Origin header names another origin.

    The message names both sides. `http://127.0.0.1:1121` and
    `http://localhost:1121` are the same server and different origins, and a
    bare "not allowed" leaves someone comparing two strings they cannot see.
    `app.public_url` is public by definition, so saying it costs nothing.
    """
    origin = request.headers.get("origin")
    if origin is None:
        # Same-origin fetches and non-browser clients send no Origin header.
        return
    # With app.public_url unset, `_origin` is empty and every browser mutation
    # is refused.
    if origin.rstrip("/") != _origin:
        logger.warning("refused a mutation from origin %r; app.public_url is %r", origin, _origin)
        expected = _origin or "unset — set app.public_url in config.yaml"
        raise ApiError(403, "bad_origin", f"This request came from {origin}, but this app is served from {expected}.")


CurrentUser = Depends(current_user)

# Bytes per read while an upload is checked against the quota.
_UPLOAD_CHUNK = 1024 * 1024

# Events per read while a stream catches up on the log.
_BACKLOG_PAGE = 1000
JsonBody = Body(...)
UploadedFile = File(...)
UploadedPath = Form(default=None)


# --- auth ----------------------------------------------------------------------


@app.post("/auth/session", status_code=204)
async def create_auth_session(authorization: str | None = Header(default=None)) -> Response:
    """Verify a Supabase token and set the session cookie. The only endpoint that reads a bearer token."""
    token = jwt_utils.extract_bearer(authorization)
    if not token:
        raise ApiError(401, "unauthenticated", "Send the Supabase access token as Authorization: Bearer.")
    try:
        claims = jwt_utils.verify_supabase(token)
    except jwt.PyJWTError as e:
        raise ApiError(401, "unauthenticated", f"Token rejected: {e}") from e

    user_id, email = str(claims["sub"]), claims.get("email")
    await pool.execute(
        """
        INSERT INTO users (id, email) VALUES ($1, $2)
        ON CONFLICT (id) DO UPDATE SET email = COALESCE(EXCLUDED.email, users.email)
        """,
        _uuid(user_id, "user"),
        email,
    )
    await _ensure_home_session(user_id)

    out = Response(status_code=204)
    out.set_cookie(
        key=str(_cfg("auth.cookie_name", "ark_session")),
        value=jwt_utils.mint_session(user_id, email),
        max_age=int(_cfg("auth.session_ttl_s", 604800)),
        httponly=True,
        secure=bool(_cfg("auth.cookie_secure", True)),
        samesite=str(_cfg("auth.cookie_samesite", "lax")),
        path="/",
    )
    return out


async def _ensure_home_session(user_id: str) -> str:
    """Give a user their standing chat, once.

    The app opens this session by default, which is the whole of what makes it
    home: the row is an ordinary attended session in an ordinary project, free
    to sit idle forever or run like any other. Created here because first login
    is the only moment that knows a user is new, and guarded by
    `home_session_id IS NULL` so a second login never makes a second one.
    """
    existing = await pool.fetchval(
        "SELECT home_session_id FROM users WHERE id = $1", _uuid(user_id, "user")
    )
    if existing is not None:
        return str(existing)

    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, $2) RETURNING id",
        _uuid(user_id, "user"),
        "Chat",
    )
    session_id = await pool.fetchval(
        """
        INSERT INTO sessions (user_id, project_id, mode, status, title)
        VALUES ($1, $2, 'attended', 'idle', 'Chat')
        RETURNING id
        """,
        _uuid(user_id, "user"),
        project_id,
    )
    # Conditional, so two first logins racing leave one session as home and the
    # other as an ordinary empty one rather than overwriting each other.
    claimed = await pool.fetchval(
        """
        UPDATE users SET home_session_id = $2
         WHERE id = $1 AND home_session_id IS NULL
        RETURNING home_session_id
        """,
        _uuid(user_id, "user"),
        session_id,
    )
    if claimed is None:
        return str(await pool.fetchval("SELECT home_session_id FROM users WHERE id = $1", _uuid(user_id, "user")))
    return str(claimed)


@app.get("/auth/config")
async def auth_config() -> dict[str, Any]:
    """What the sign-in view needs to talk to Supabase. Public, and necessarily so.

    The anon key is public by design — it identifies the project, authorizes
    nothing on its own, and every row it can reach is behind RLS or behind this
    API. It is served rather than baked into the page because it differs per
    deployment and the page is a checked-in file.
    """
    return {"supabase_url": store.project_url() or "", "anon_key": _anon_key()}


@app.delete("/auth/session", status_code=204)
async def delete_auth_session() -> Response:
    out = Response(status_code=204)
    out.delete_cookie(key=str(_cfg("auth.cookie_name", "ark_session")), path="/")
    return out


@app.get("/auth/me")
async def auth_me(user_id: str = CurrentUser) -> dict[str, Any]:
    """Who is calling, and which session the app opens for them.

    `home_session_id` rides along because the page needs it on the first render
    and there is no other request that would carry it. It is null only for a
    user whose home session was deleted; the next sign-in makes another.
    """
    row = await pool.fetchrow(
        "SELECT id, email, home_session_id FROM users WHERE id = $1", _uuid(user_id, "user")
    )
    if row is None:
        raise ApiError(401, "unauthenticated", "That user no longer exists.")
    return {
        "user_id": str(row["id"]),
        "email": row["email"],
        "home_session_id": str(row["home_session_id"]) if row["home_session_id"] else None,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    """Report process and database health. Unauthenticated, for the uptime check."""
    try:
        await pool.fetchval("SELECT 1")
        database = "ok"
    except Exception as e:  # noqa: BLE001 - the failure is reported in the response body
        database = f"unreachable: {type(e).__name__}"
    return {"status": "ok" if database == "ok" else "degraded", "database": database}


# --- sessions ------------------------------------------------------------------


@app.post("/sessions", status_code=201)
async def create_session(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Open a session on a goal and start its first turn.

    A session is created attended, so only the new-session rate quota applies
    here; the unattended concurrency quota is checked on approve.

    `claims` names what the session may see, as `[{project_id, subpath?,
    mode?}]`. Absent, it gets a write claim on its own project. The set is fixed
    here for the session's life: deciding it up front is what lets the leases be
    taken in one go rather than acquired into a deadlock halfway through.
    """
    goal = str(body.get("goal") or "").strip()
    if not goal:
        raise ApiError(400, "invalid_request", "A session needs a goal.")
    await _check_rate_quota(user_id)

    project_id = body.get("project_id")
    if project_id:
        owned = await pool.fetchval(
            "SELECT id FROM projects WHERE id = $1 AND user_id = $2",
            _uuid(project_id, "project"),
            _uuid(user_id, "user"),
        )
        if owned is None:
            raise ApiError(404, "not_found", "No such project.")
    else:
        project_id = await pool.fetchval(
            "INSERT INTO projects (user_id, title) VALUES ($1, $2) RETURNING id",
            _uuid(user_id, "user"),
            _title(goal),
        )

    session_id = await pool.fetchval(
        """
        INSERT INTO sessions (user_id, project_id, mode, status, title, goal)
        VALUES ($1, $2, 'attended', 'pending', $3, $4)
        RETURNING id
        """,
        _uuid(user_id, "user"),
        _uuid(project_id, "project"),
        _title(goal),
        goal,
    )
    session_id = str(session_id)
    await _record_claims(session_id, str(project_id), body.get("claims"), user_id)
    async with (await pool.pool()).acquire() as conn:
        await lifecycle.touch_project(conn, session_id)

    await _append(session_id, UserEvent(text=goal, source="human"))
    steps = body.get("steps")
    if isinstance(steps, list) and steps:
        items = [{"text": str(s), "status": "pending"} for s in steps]
        await _append(session_id, TodoEvent(items=items))

    await runner.start(session_id)
    return {"session_id": session_id, "project_id": str(project_id)}


@app.get("/sessions")
async def list_sessions(status: str | None = None, user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """The user's sessions across every project, newest activity first.

    `status` narrows it — `?status=running` is what the rail asks for. The
    per-project list does not compose into this: a rail spanning both tabs is a
    cross-project view, and asking it project by project is N requests for a
    sidebar.
    """
    if status is not None and status not in lifecycle.ALL_STATUSES:
        raise ApiError(400, "invalid_request", f"{status!r} is not a session status.")

    rows = await pool.fetch(
        """
        SELECT s.id, s.title, s.status, s.mode, s.terminal_reason, s.hops_used,
               s.project_id, p.title AS project_title,
               COALESCE(max(e.ts), s.created_at) AS last_event_at
          FROM sessions s
          LEFT JOIN projects p ON p.id = s.project_id
          LEFT JOIN session_events e ON e.session_id = s.id
         WHERE s.user_id = $1 AND ($2::text IS NULL OR s.status = $2)
         GROUP BY s.id, p.title
         ORDER BY last_event_at DESC
        """,
        _uuid(user_id, "user"),
        status,
    )
    return [
        {
            "session_id": str(r["id"]),
            "title": r["title"],
            "status": r["status"],
            "mode": r["mode"],
            "terminal_reason": r["terminal_reason"],
            "hops_used": r["hops_used"],
            "hops_max": _budgets_for(r["mode"]),
            "project_id": str(r["project_id"]) if r["project_id"] else None,
            "project_title": r["project_title"],
            "last_event_at": r["last_event_at"].isoformat(),
        }
        for r in rows
    ]


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Return the session and the tail of its transcript, for a just-opened view."""
    row = await _owned_session(session_id, user_id)
    budgets = _budgets_for(row["mode"])
    events = await slog.recent_events(session_id, limit=int(_cfg("harness.snapshot_events", 200)))
    return {
        "session_id": str(row["id"]),
        "title": row["title"],
        "project_id": str(row["project_id"]) if row["project_id"] else None,
        "status": row["status"],
        # The UI reads mode to decide whether to offer the approve control.
        "mode": row["mode"],
        "terminal_reason": row["terminal_reason"],
        "hops_used": row["hops_used"],
        "hops_max": budgets,
        # What this session may see and write, fixed at creation.
        "claims": await _claims_of(session_id),
        "recent_events": [_wire(e) for e in events],
    }


@app.get("/projects")
async def list_projects(user_id: str = CurrentUser) -> list[dict[str, Any]]:
    rows = await pool.fetch(
        """
        SELECT p.id, p.title, p.updated_at,
               count(s.id) FILTER (WHERE s.status = 'running')           AS running,
               count(s.id) FILTER (WHERE s.status = 'awaiting_approval') AS awaiting,
               count(s.id) FILTER (WHERE s.status = 'failed')            AS failed,
               count(s.id)                                               AS sessions
          FROM projects p LEFT JOIN sessions s ON s.project_id = p.id
         WHERE p.user_id = $1
         GROUP BY p.id
         ORDER BY p.updated_at DESC
        """,
        _uuid(user_id, "user"),
    )
    return [
        {
            "id": str(r["id"]),
            "title": r["title"],
            "updated_at": r["updated_at"].isoformat(),
            # Rolled up in the query above, so the project grid costs one round trip.
            "status_rollup": _rollup(r),
            "sessions": r["sessions"],
        }
        for r in rows
    ]


@app.get("/projects/{project_id}/sessions")
async def list_project_sessions(project_id: str, user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """The project's sessions, most recently active first.

    How the grid gets from a bubble to a window: `GET /projects` rolls a project
    up to one dot and one count, which is the right shape for the grid and no
    use at all for opening what it counted.
    """
    await _owned_project(project_id, user_id)
    rows = await pool.fetch(
        """
        SELECT s.id, s.title, s.status, s.mode, s.terminal_reason, s.hops_used,
               s.created_at, s.ended_at,
               COALESCE(max(e.ts), s.created_at) AS last_event_at,
               count(a.id) FILTER (WHERE a.answered_at IS NULL) AS open_questions
          FROM sessions s
          LEFT JOIN session_events e ON e.session_id = s.id
          LEFT JOIN approvals a ON a.session_id = s.id
         WHERE s.project_id = $1
         GROUP BY s.id
         ORDER BY last_event_at DESC
        """,
        _uuid(project_id, "project"),
    )
    return [
        {
            "session_id": str(r["id"]),
            "title": r["title"],
            "status": r["status"],
            "mode": r["mode"],
            "terminal_reason": r["terminal_reason"],
            "hops_used": r["hops_used"],
            "hops_max": _budgets_for(r["mode"]),
            "open_questions": r["open_questions"],
            "created_at": r["created_at"].isoformat(),
            "ended_at": r["ended_at"].isoformat() if r["ended_at"] else None,
            "last_event_at": r["last_event_at"].isoformat(),
        }
        for r in rows
    ]


@app.get("/attention")
async def attention(
    project_id: str | None = None,
    session_id: str | None = None,
    user_id: str = CurrentUser,
) -> list[dict[str, Any]]:
    """Every question waiting on this human, oldest first.

    One query at three scopes: no filter is the whole account (the Command
    Center), `project_id` is that project's list, `session_id` is the one
    window. The same row appears in all three — an approval is a state of the
    session, not something a surface owns — and answering it anywhere writes
    the same response and wakes the session at its cursor.

    Approvals and asks are the same row and the same wait; what differs is the
    answer, so the caller is told `kind` and nothing else branches here.
    """
    if project_id is not None:
        await _owned_project(project_id, user_id)
    if session_id is not None:
        await _owned_session(session_id, user_id)

    rows = await pool.fetch(
        """
        SELECT a.id, a.session_id, a.kind, a.prompt, a.created_at, a.tool_call_id,
               s.title AS session_title, s.project_id, p.title AS project_title
          FROM approvals a
          JOIN sessions s ON s.id = a.session_id
          LEFT JOIN projects p ON p.id = s.project_id
         WHERE s.user_id = $1
           AND a.answered_at IS NULL
           AND ($2::uuid IS NULL OR s.project_id = $2)
           AND ($3::uuid IS NULL OR s.id = $3)
         ORDER BY a.created_at
        """,
        _uuid(user_id, "user"),
        _uuid(project_id, "project") if project_id is not None else None,
        _uuid(session_id, "session") if session_id is not None else None,
    )
    return [
        {
            "approval_id": str(r["id"]),
            "session_id": str(r["session_id"]),
            "session_title": r["session_title"],
            "project_id": str(r["project_id"]) if r["project_id"] else None,
            "project_title": r["project_title"],
            "kind": r["kind"],
            "prompt": r["prompt"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


@app.get("/projects/{project_id}/files")
async def list_project_files(project_id: str, user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """List a project's files from the tree. No sandbox is woken to answer this."""
    await _owned_project(project_id, user_id)
    rows = await pool.fetch(
        "SELECT id, path, size, mtime FROM project_files WHERE project_id = $1 ORDER BY path",
        _uuid(project_id, "project"),
    )
    return [
        {
            "file_id": str(r["id"]),
            "path": r["path"],
            "name": posixpath.basename(r["path"]),
            "size": r["size"],
            "mtime": r["mtime"].isoformat(),
        }
        for r in rows
    ]


@app.get("/projects/{project_id}/files/{file_id}")
async def read_project_file(project_id: str, file_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """One file's contents, read from the store without waking anything.

    The computer view is a filesystem you can read, and listing rows is only
    half of that. Text is returned decoded; anything that is not UTF-8 says so
    rather than arriving as mojibake, because a reader pane that renders a PNG
    as characters is worse than one that admits it cannot.
    """
    await _owned_project(project_id, user_id)
    row = await pool.fetchrow(
        "SELECT path, content_hash, size, mtime FROM project_files WHERE id = $1 AND project_id = $2",
        _uuid(file_id, "file"),
        _uuid(project_id, "project"),
    )
    if row is None:
        raise ApiError(404, "not_found", "No such file.")

    blob = await store.get_blob(row["content_hash"])
    if blob is None:
        raise ApiError(410, "blob_missing", "The store no longer holds this file's contents.")

    try:
        text = blob.decode()
    except UnicodeDecodeError:
        return {
            "path": row["path"],
            "size": row["size"],
            "mtime": row["mtime"].isoformat(),
            "text": None,
            "binary": True,
        }
    return {
        "path": row["path"],
        "size": row["size"],
        "mtime": row["mtime"].isoformat(),
        "text": text,
        "binary": False,
    }


@app.post("/projects/{project_id}/files", status_code=201)
async def upload_project_file(
    project_id: str,
    file: UploadFile = UploadedFile,
    path: str | None = UploadedPath,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Put a file in the project's store, and in any box already holding it.

    The store is where the file lands. A running session with this project
    materialized is written through, so it reads the upload the same turn; every
    other session gets it at its next materialize.
    """
    await _owned_project(project_id, user_id)
    try:
        stored_path = store.safe_path(path or file.filename or "")
    except ValueError as e:
        raise ApiError(400, "invalid_request", str(e)) from e

    content = await _read_within_quota(file)
    stored = await store.put_file(project_id, stored_path, content)
    await pool.execute("UPDATE projects SET updated_at = now() WHERE id = $1", _uuid(project_id, "project"))
    await workspace.write_through(sandbox_manager.manager(), project_id, stored_path, content)

    return {
        "file_id": stored.id,
        "name": posixpath.basename(stored_path),
        "path": stored_path,
        "size": stored.entry.size,
    }


@app.post("/sessions/{session_id}/messages", status_code=202)
async def post_message(
    session_id: str,
    body: dict[str, Any] = JsonBody,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Append a message to a session, starting a turn if one is not running.

    A running session picks the event up at its next hop, so the response is
    202 and carries no reply.
    """
    text = str(body.get("text") or "").strip()
    if not text:
        raise ApiError(400, "invalid_request", "An empty message says nothing.")
    row = await _owned_session(session_id, user_id)
    if row["status"] == "awaiting_approval":
        return await _answer_by_message(session_id, text)

    if not runner.is_running(session_id):
        # Calls left open by a dead run are closed before the message is
        # appended, so no user event lands between a call and its result.
        for closed in await slog.close_dangling(session_id):
            stream.publish(session_id, closed)

    await _append(session_id, UserEvent(text=text, source="human"))
    started = await runner.start(session_id)
    return {"accepted": True, "started": started}


@app.post("/approvals/{approval_id}/respond", status_code=202)
async def respond_to_approval(
    approval_id: str,
    body: dict[str, Any] = JsonBody,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Answer an open question and wake the session that raised it.

    The answer is appended as a `user` event. The tool call that raised the
    question was closed before the session parked, so nothing is back-filled.
    """
    text = str(body.get("answer") or "").strip()
    if not text:
        raise ApiError(400, "invalid_request", "An answer is required.")

    approval = await approvals.get(approval_id, user_id)
    if approval is None:
        raise ApiError(404, "not_found", "No such approval.")

    # The UPDATE matches on answered_at IS NULL, so two people answering at once
    # produce one wake.
    answered = await approvals.answer(approval_id, text)
    if answered is None:
        raise ApiError(409, "already_answered", "That question has already been answered.")

    await _append(answered.session_id, UserEvent(text=text, source="human"))
    started = await runner.start(answered.session_id, reason="answered")
    return {"accepted": True, "session_id": answered.session_id, "started": started}


@app.post("/sessions/{session_id}/approve", status_code=202)
async def approve_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Hand a session over to run unattended, and start it.

    The session keeps its id and its transcript. From here the unattended
    budgets apply and `finish_task` is its only terminal step.
    """
    row = await _owned_session(session_id, user_id)
    if row["mode"] == "unattended":
        raise ApiError(409, "already_unattended", "This session is already running unattended.")
    if row["status"] not in ("idle", "pending"):
        raise ApiError(409, "not_idle", f"A session in {row['status']!r} cannot be handed over.")

    # Sessions are created attended, so this is the only point at which a
    # user's unattended load grows.
    await _check_unattended_quota(user_id)

    started = await runner.start(session_id, mode="unattended", reason="approved")
    if not started:
        raise ApiError(409, "not_idle", "The session moved before it could be started.")
    return {"accepted": True, "mode": "unattended"}


@app.post("/sessions/{session_id}/cancel", status_code=202)
async def cancel_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    await _owned_session(session_id, user_id)
    return {"cancelled": await runner.cancel(session_id)}


@app.get("/results/{ref}")
async def read_result(ref: str, offset: int = 0, limit: int = 2000, user_id: str = CurrentUser) -> dict[str, Any]:
    """Return a slice of a stored oversized result, scoped to its owner."""
    start = max(0, offset)
    text = await slog.read_blob(ref, offset=start, limit=max(1, min(limit, 100_000)), user_id=user_id)
    if text is None:
        raise ApiError(404, "not_found", "No such result.")
    return {"ref": ref, "offset": start, "content": text}


# --- the stream ----------------------------------------------------------------


@app.get("/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    user_id: str = CurrentUser,
) -> StreamingResponse:
    """Stream the session's events, replaying anything after `Last-Event-ID` first.

    Every frame carries `id: <seq>`; `EventSource` returns the last id it saw
    as the `Last-Event-ID` header when it reconnects.
    """
    await _owned_session(session_id, user_id)
    after = _int_or(last_event_id or request.query_params.get("last_event_id"), 0)
    return StreamingResponse(
        _event_stream(session_id, after),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _event_stream(session_id: str, after_seq: int) -> AsyncIterator[str]:
    """Yield SSE frames until the client disconnects.

    The subscription is opened before the backlog is read, so an event appended
    during the read still arrives on the queue; `sent` holds the highest seq
    yielded and drops the overlap.
    """
    keepalive = float(_cfg("harness.sse_keepalive_s", 15))
    try:
        async with stream.subscribe(session_id) as queue:
            sent = after_seq
            async for stored in _backlog(session_id, sent):
                sent = stored.seq
                yield _frame(stored)

            while True:
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=keepalive)
                except TimeoutError:
                    # Proxies and EventSource drop a stream that stays silent.
                    yield ": keepalive\n\n"
                    continue

                if item is LAGGED:
                    # This consumer fell behind its queue; it rejoins from the log.
                    async for stored in _backlog(session_id, sent):
                        sent = stored.seq
                        yield _frame(stored)
                    continue

                if item.seq > sent:
                    sent = item.seq
                    yield _frame(item)
    except asyncio.CancelledError:
        # Starlette cancels this generator when the client disconnects.
        raise
    except Exception as e:  # noqa: BLE001 - the failure is sent as a final frame
        logger.exception("session %s: the event stream failed", session_id)
        # EventSource cannot tell a truncated stream from a finished one, so
        # the failure is delivered as an error event.
        yield "event: error\ndata: " + json.dumps(
            {"code": "stream_failed", "message": f"{type(e).__name__}: {e}", "retryable": True}
        ) + "\n\n"


async def _backlog(session_id: str, after_seq: int) -> AsyncIterator[slog.StoredEvent]:
    """Yield every event after `after_seq`, a page at a time until the log runs out.

    One page is not the backlog. A reader rejoining a long session with
    `Last-Event-ID` is exactly the case that exceeds it, and `sent` only moves
    forward, so anything skipped here cannot be recovered on this connection —
    the transcript would be whole in the log and full of holes on screen.
    """
    cursor = after_seq
    while True:
        page = await slog.get_events(session_id, after_seq=cursor, limit=_BACKLOG_PAGE)
        for stored in page:
            cursor = stored.seq
            yield stored
        if len(page) < _BACKLOG_PAGE:
            return


def _frame(stored: slog.StoredEvent) -> str:
    return f"id: {stored.seq}\nevent: {stored.event.kind}\ndata: {json.dumps(_wire(stored), default=str)}\n\n"


def _wire(stored: slog.StoredEvent) -> dict[str, Any]:
    """Render one stored event for the wire, adding the seq and ts columns."""
    row = stored.event.to_row()
    return {"seq": stored.seq, "ts": stored.ts.isoformat(), **row}


# --- MCP connections ------------------------------------------------------------


@app.get("/sessions/{session_id}/browser/frames")
async def browser_frames(session_id: str, user_id: str = CurrentUser) -> StreamingResponse:
    """Watch what the browser is looking at, while it looks.

    A side-channel, not the event stream: frames are never appended, never
    replayed and carry no seq. Keyed by (user, session) and ownership-checked
    like everything else — the implementation this replaces trusted a user id
    from the query string, which is the violation the deletion closed.
    """
    await _owned_session(session_id, user_id)
    return StreamingResponse(
        _frame_stream(user_id, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _frame_stream(user_id: str, session_id: str) -> AsyncIterator[str]:
    """Yield frames until the viewer goes away. Keepalives, because a browser
    run can think for a while between pictures."""
    keepalive = float(_cfg("harness.sse_keepalive_s", 15))
    async with frames.subscribe(user_id, session_id) as queue:
        while True:
            try:
                frame = await asyncio.wait_for(queue.get(), timeout=keepalive)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            yield f"event: frame\ndata: {json.dumps({'jpeg': frame})}\n\n"


@app.get("/connections")
async def list_connections(user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """List every configured server and this user's connection status.

    A server that is not connected is re-verified before the list is returned;
    `connect()` is idempotent, so a row left behind by an interrupted OAuth
    flow is repaired on read.

    The repair does not hold the response. Waiting on it cost ten seconds with
    seven servers configured — one Smithery round trip per unconnected row,
    serially — and the list was already correct before any of them ran: a row
    that needs repairing reads "not connected", which is exactly what it says
    afterwards too, until the human finishes the OAuth they abandoned. So the
    rows go back now and the repairs run behind them, together and on a budget.
    """
    client = hands.smithery()
    if client is None:
        return []
    rows = await client.connections(user_id)
    broken = [r["server"] for r in rows if client.needs_repair(r)]
    if broken:
        task = asyncio.create_task(_repair_connections(user_id, broken))
        # Held, or the loop may collect a task nobody is awaiting.
        _repairs.add(task)
        task.add_done_callback(_repairs.discard)
    return rows


@app.post("/connections/{server}/connect")
async def connect_server(server: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Start a connection: mint the id, write the pending row, PUT to Smithery. Idempotent."""
    client = _require_smithery()
    if server not in client.servers:
        raise ApiError(404, "not_found", f"No server {server!r} is configured.")
    try:
        await client.connect(user_id, server, return_url=_callback_url(server))
    except AuthRequiredError as e:
        # The user has not authorized the server yet; they are sent to setup_url.
        return {"server": server, "status": e.state, "setup_url": e.setup_url}
    except SmitheryError as e:
        raise ApiError(502, "upstream_error", f"Smithery refused: {e}", retryable=True) from e
    return {"server": server, "status": "connected", "setup_url": None}


@app.delete("/connections/{server}", status_code=204)
async def disconnect_server(server: str, user_id: str = CurrentUser) -> Response:
    client = _require_smithery()
    if server not in client.servers:
        raise ApiError(404, "not_found", f"No server {server!r} is configured.")
    try:
        await client.disconnect(user_id, server)
    except SmitheryError as e:
        raise ApiError(502, "upstream_error", f"Smithery refused: {e}", retryable=True) from e
    return Response(status_code=204)


_POPUP_CLOSE = """<!doctype html><meta charset="utf-8"><title>Connected</title>
<body style="font:14px system-ui;padding:2rem">Connected. You can close this window.
<script>
  try { window.opener && window.opener.postMessage({type:"arkos:connection", server:%s}, "*"); } catch (e) {}
  window.close();
</script></body>"""


@app.get("/oauth/callback/{server}")
async def oauth_callback(server: str, request: Request) -> HTMLResponse:
    """Land Smithery's redirect once the user has authorized a server.

    The caller is read from the session cookie; this is a top-level GET, so
    SameSite=Lax sends it. Verification runs once in the background after the
    response, since dispatch does not re-verify and revalidation skips
    unconnected rows.
    """
    try:
        user_id = await current_user(request)
    except ApiError:
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><body style='font:14px system-ui;padding:2rem'>"
            "Sign in to this app first, then connect again.</body>",
            status_code=401,
        )
    return HTMLResponse(
        _POPUP_CLOSE % json.dumps(server),
        background=BackgroundTask(_verify_once, user_id, server),
    )


# Repairs in flight behind a `GET /connections`, held so they are not collected.
_repairs: set[asyncio.Task[None]] = set()


async def _repair_connections(user_id: str, servers: list[str]) -> None:
    """Re-assert interrupted connections, all at once and bounded.

    Nobody is waiting on this, which is the point — but it still gets a deadline,
    because a vendor that hangs should not leave tasks alive for the life of the
    process.
    """
    budget = float(_cfg("smithery.repair_budget_s", 15))
    try:
        async with asyncio.timeout(budget):
            await asyncio.gather(*(_verify_once(user_id, server) for server in servers))
    except TimeoutError:
        logger.warning("repairing %d connection(s) for user %s ran past %.0fs", len(servers), user_id, budget)


async def _verify_once(user_id: str, server: str) -> None:
    """Re-assert one connection. Idempotent, and not retried."""
    client = hands.smithery()
    if client is None:
        return
    try:
        await client.connect(user_id, server)
    except AuthRequiredError:
        # OAuth did not finish; GET /connections repairs the row on the next read.
        logger.info("oauth callback for %s: still unauthorized", server)
    except Exception:
        logger.exception("oauth callback verification failed for %s", server)


def _require_smithery() -> Any:
    client = hands.smithery()
    if client is None:
        raise ApiError(503, "unavailable", "MCP is not configured on this server.")
    return client


def _callback_url(server: str) -> str | None:
    return f"{_origin}/oauth/callback/{server}" if _origin else None


# --- helpers -------------------------------------------------------------------


async def _owned_session(session_id: str, user_id: str) -> Any:
    """Load a session the caller owns. Another user's session reads as `not_found`."""
    row = await pool.fetchrow(
        """
        SELECT id, user_id, project_id, title, status, mode, terminal_reason, hops_used
          FROM sessions WHERE id = $1 AND user_id = $2
        """,
        _uuid(session_id, "session"),
        _uuid(user_id, "user"),
    )
    if row is None:
        raise ApiError(404, "not_found", "No such session.")
    return row


async def _append(session_id: str, event: Any) -> None:
    """Append an event and publish it to live subscribers."""
    stored = await slog.append(session_id, event)
    stream.publish(session_id, stored)


async def _owned_project(project_id: str, user_id: str) -> None:
    """Raise 404 unless the project is this user's. Someone else's reads as absent."""
    owned = await pool.fetchval(
        "SELECT id FROM projects WHERE id = $1 AND user_id = $2",
        _uuid(project_id, "project"),
        _uuid(user_id, "user"),
    )
    if owned is None:
        raise ApiError(404, "not_found", "No such project.")


async def _read_within_quota(file: UploadFile) -> bytes:
    """Read an upload, refusing it as soon as it passes `quotas.upload_max_mb`.

    Chunked and checked as it goes, so an oversized upload is refused rather
    than held in memory first. Zero bytes is a file like any other — a
    `.gitkeep`, a placeholder — and is stored as one.
    """
    limit = int(_cfg("quotas.upload_max_mb", 25)) * 1024 * 1024
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(_UPLOAD_CHUNK):
        total += len(chunk)
        if total > limit:
            raise ApiError(413, "file_too_large", f"{limit // (1024 * 1024)} MB is the limit for one file.")
        chunks.append(chunk)
    return b"".join(chunks)


async def _check_rate_quota(user_id: str) -> None:
    """Enforce the sliding window on new sessions, before anything is written.

    The home session is excluded: it is created by first login rather than
    asked for, and a quota on what a person does should not be spent by the
    server greeting them.
    """
    limit = int(_cfg("quotas.new_sessions_per_hour", 20))
    recent = await pool.fetchval(
        """
        SELECT count(*) FROM sessions s
          JOIN users u ON u.id = s.user_id
         WHERE s.user_id = $1
           AND s.created_at > now() - interval '1 hour'
           AND (u.home_session_id IS NULL OR s.id <> u.home_session_id)
        """,
        _uuid(user_id, "user"),
    )
    if recent >= limit:
        raise ApiError(429, "quota_exceeded", f"{limit} new sessions an hour is the limit.", retryable=True)


async def _answer_by_message(session_id: str, text: str) -> dict[str, Any]:
    """Route a composer message sent to a parked session.

    A question raised by `ask` is answered by the message, and the session
    wakes. A request raised by `request_approval` is answered only through
    `/approvals/{id}/respond`, where the action being agreed to is on screen.
    """
    open_questions = await approvals.open_for(session_id)
    if open_questions and open_questions[0].kind == "approval":
        raise ApiError(
            409,
            "awaiting_approval",
            "This session is waiting on an approval. Answer it from the approval itself.",
        )

    if open_questions:
        await approvals.answer(open_questions[0].id, text)
    await _append(session_id, UserEvent(text=text, source="human"))
    started = await runner.start(session_id, reason="answered")
    return {"accepted": True, "started": started}


async def _record_claims(session_id: str, project_id: str, declared: Any, user_id: str) -> None:
    """Record what this session may touch, fixed for its life.

    Absent, it gets a write claim on its own project, so a caller that knows
    nothing about claims behaves as it did before they existed.
    """
    claims = declared if isinstance(declared, list) and declared else [{"project_id": project_id}]
    rows = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ApiError(400, "invalid_request", "Each claim is an object with a project_id.")
        target = str(claim.get("project_id") or project_id)
        mode = str(claim.get("mode") or "write")
        if mode not in ("read", "write"):
            raise ApiError(400, "invalid_request", f"A claim is read or write, not {mode!r}.")
        owned = await pool.fetchval(
            "SELECT id FROM projects WHERE id = $1 AND user_id = $2",
            _uuid(target, "project"),
            _uuid(user_id, "user"),
        )
        if owned is None:
            raise ApiError(404, "not_found", "No such project.")
        rows.append((_uuid(target, "project"), str(claim.get("subpath") or "/"), mode))

    for project, subpath, mode in rows:
        await pool.execute(
            """
            INSERT INTO session_claims (session_id, project_id, subpath, mode)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (session_id, project_id, subpath) DO UPDATE SET mode = EXCLUDED.mode
            """,
            _uuid(session_id, "session"),
            project,
            subpath,
            mode,
        )


async def _claims_of(session_id: str) -> list[dict[str, Any]]:
    """The session's claims, for the window to render."""
    rows = await pool.fetch(
        """
        SELECT c.project_id, c.subpath, c.mode, p.title
          FROM session_claims c JOIN projects p ON p.id = c.project_id
         WHERE c.session_id = $1
         ORDER BY c.project_id, c.subpath
        """,
        _uuid(session_id, "session"),
    )
    return [
        {"project_id": str(r["project_id"]), "title": r["title"], "subpath": r["subpath"], "mode": r["mode"]}
        for r in rows
    ]


async def _check_unattended_quota(user_id: str) -> None:
    """Enforce the per-user cap on sessions occupying a worker."""
    limit = int(_cfg("quotas.max_unattended_sessions", 5))
    # An awaiting_approval session still holds its worker slot and will resume.
    busy = await pool.fetchval(
        """
        SELECT count(*) FROM sessions
         WHERE user_id = $1 AND mode = 'unattended' AND status IN ('running', 'awaiting_approval')
        """,
        _uuid(user_id, "user"),
    )
    if busy >= limit:
        raise ApiError(429, "quota_exceeded", f"{limit} unattended sessions at once is the limit.", retryable=True)


def _budgets_for(mode: str) -> int:
    return int(_cfg(f"budgets.{mode}.max_hops", 0))


def _anon_key() -> str:
    """The publishable Supabase key, under either of its two names.

    `sb_publishable_...` is the current format; `SUPABASE_ANON_KEY` is the legacy
    JWT-shaped one. Neither authorizes anything on its own.
    """
    return os.environ.get("SUPABASE_PUBLISHABLE_KEY") or os.environ.get("SUPABASE_ANON_KEY") or ""


def _rollup(row: Any) -> str:
    """Return the most urgent session status in a project, for the grid's dot."""
    if row["awaiting"]:
        return "awaiting_approval"
    if row["running"]:
        return "running"
    if row["failed"]:
        return "failed"
    return "idle"


def _title(goal: str) -> str:
    line = goal.strip().splitlines()[0]
    return line[:77] + "..." if len(line) > 80 else line


def _int_or(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _uuid(value: Any, what: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ApiError(404, "not_found", f"No such {what}.") from e


# --- the app itself --------------------------------------------------------------

# Mounted last, so no API route can be shadowed by a file with the same name.
#
# Same origin as the API, and that is a requirement rather than a convenience:
# the session cookie is SameSite=Lax, which a cross-site `EventSource` does not
# carry, so a frontend served from anywhere else gets 401 on every stream connect
# while the tests — which call `_event_stream` directly — stay green.
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

if _FRONTEND.is_dir():
    # `html=True` serves index.html at /app and for any path the build does not
    # have a file for, which is what a client-routed page needs.
    app.mount("/app", StaticFiles(directory=_FRONTEND, html=True), name="app")
else:  # pragma: no cover - only a broken checkout or a partial image
    logger.error("no frontend/ directory at %s; /app will 404", _FRONTEND)

