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

from agent_module import prompts
from agent_module.events import TodoEvent, UserEvent
from config_module.loader import cfg as _cfg
from config_module.loader import config
from db import pool
from db.ids import as_uuid
from harness_module import approvals, hands, jwt_utils, lifecycle, runner, store, system_log, workspace
from harness_module import session_log as slog
from harness_module.stream import LAGGED, stream
from tool_module import registry, session_tools
from tool_module.browser.stream import broker as frames
from tool_module.sandbox import manager as sandbox_manager
from tool_module.sandbox import tools as sandbox_tools
from tool_module.smithery import AuthRequiredError, SmitheryError

logger = logging.getLogger(__name__)




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
    home: the row is an ordinary attended session, free to sit idle forever or
    run like any other. Created here because first login is the only moment that
    knows a user is new, and guarded by `home_session_id IS NULL` so a second
    login never makes a second one.

    It has NO PROJECT (11.9). It used to mint one called "Chat", because a
    project was the only way to hold a directory and a session needed a
    directory. No project holds a directory now — folders are the store's and
    projects link them — so the shadow project is not cleaned up, it is unmade.
    A chat that has not been given work has nothing durable to write into, which
    is the truth about it: asking it for work makes a project, and that is when
    a folder appears.
    """
    existing = await pool.fetchval(
        "SELECT home_session_id FROM users WHERE id = $1", _uuid(user_id, "user")
    )
    if existing is not None:
        return str(existing)

    session_id = await pool.fetchval(
        """
        INSERT INTO sessions (user_id, project_id, mode, status, title)
        VALUES ($1, NULL, 'attended', 'idle', 'Chat')
        RETURNING id
        """,
        _uuid(user_id, "user"),
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

    `claims` names what the session may see, as `[{folder, subpath?, mode?}]`.
    Absent, it gets a write claim on every folder its project links. The set is
    fixed here for the session's life: deciding it up front is what lets the
    leases be taken in one go rather than acquired into a deadlock halfway
    through, and it is why a folder linked later reaches the agent at the next
    session rather than under this one.
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
        # A session asked for with no project is a new piece of work, so it gets
        # a project and the project gets a folder to keep the work in. The home
        # chat is the other case and gets neither: nobody asked it for anything.
        project_id = await _new_project(user_id, _title(goal))
        await _link_folder(project_id, await _make_folder(user_id, store.slug(_title(goal), "project")))

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
            **_session_core(r),
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
    events = await slog.recent_events(session_id, limit=int(_cfg("harness.snapshot_events", 200)))
    # Read ONCE. `folders` is a projection of the claims, and asking for them
    # twice is two queries that can disagree with each other.
    claims = await _claims_of(session_id)
    return {
        **_session_core(row),
        "project_id": str(row["project_id"]) if row["project_id"] else None,
        # The project's LABEL, so the window's header reads the same however it
        # was opened. It used to come from the grid's navigation state, which is
        # absent when a session is opened from the desk — and the header then
        # fell back to the session's own title.
        "project_title": await pool.fetchval(
            "SELECT title FROM projects WHERE id = $1", _uuid(row["project_id"], "project")
        )
        if row["project_id"]
        else None,
        # The FOLDERS this session writes, in link order — where the work
        # actually lands. Not `project_slug`: a project links folders rather than
        # owning one, so there is no single directory to name, and the session
        # header stopped drawing directory chips when there could be several
        # (11.9). The pane and the plan card say where work goes.
        "folders": [claim["folder"] for claim in claims],
        # The session's newest plan, so the collapsed card an approved run pins
        # is exact. Counting `propose_plan` calls in `recent_events` was the
        # alternative and it drifts: that window is capped, so a long session
        # renders a version the server does not agree with, or none at all.
        "plan": await _latest_plan(session_id),
        # What this session may see and write, fixed at creation.
        "claims": claims,
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


@app.post("/projects", status_code=201)
async def create_project(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Make a project deliberately, rather than as a side effect of starting a session.

    `folders` is what it LINKS: any number of folders that already exist in the
    caller's store, by name. A project owns no folder and never did anything to
    the files under one — linking is a fact about which work reads and writes
    where, and unlinking would leave every file exactly where it is.

    Picking none is the none-case, and it is not "no files": a folder named
    after the project is made for it (uniquified, because folder names are
    unique per user) and linked, so it appears in the Files tab as an ordinary
    folder like any other. It is kept alive by a sentinel, which is how any
    named-but-unfilled folder exists.

    This replaced `seed_from`, which COPIED tree rows from one project's
    directory into another's. There is one store now, so the copy has nothing to
    do: pointing two projects at the same folder is linking it twice, and the
    file is one file rather than two rows with one blob under them.
    """
    title = str(body.get("title") or "").strip()
    if not title:
        raise ApiError(400, "invalid_request", "A project needs a name.")

    asked = body.get("folders")
    if asked is not None and not isinstance(asked, list):
        raise ApiError(400, "invalid_request", "folders is a list of folder names.")
    wanted = [str(name).strip().strip("/") for name in (asked or []) if str(name).strip().strip("/")]

    project_id = await _new_project(user_id, title)
    if wanted:
        existing = {f.name for f in await store.folders(user_id)}
        unknown = [name for name in wanted if name not in existing]
        if unknown:
            raise ApiError(404, "not_found", f"No such folder: {', '.join(sorted(unknown))}.")
        linked = wanted
    else:
        linked = [await _make_folder(user_id, store.slug(title, "project"))]

    for folder in linked:
        await _link_folder(project_id, folder)

    # Sentinels are excluded for the same reason `store.folders` excludes them:
    # a folder that has been named and not filled holds nothing, and reporting
    # the file that keeps it alive as content is reporting a file nobody put
    # there.
    files = await pool.fetchval(
        """
        SELECT count(*) FROM files
         WHERE user_id = $1
           AND split_part(path, '/', 1) = ANY($2::text[])
           AND path NOT LIKE '%/' || $3
        """,
        _uuid(user_id, "user"),
        linked,
        store.DIR_SENTINEL,
    )
    return {"id": str(project_id), "title": title, "folders": linked, "files": int(files)}


@app.post("/projects/{project_id}/folders", status_code=201)
async def link_project_folder(
    project_id: str,
    body: dict[str, Any] = JsonBody,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Link one more store folder to this project.

    The UI shows it at once and the AGENT sees it from the NEXT session, because
    claims are fixed for a session's life (`workspace.claims_for`). That is not a
    lag to be papered over: a folder appearing under a run mid-hop is a mount and
    a lease the model was never told about, and a fact recorded at session start
    is one it can be told. Linking twice is the same link.
    """
    await _owned_project(project_id, user_id)
    folder = str(body.get("folder") or "").strip().strip("/")
    if not folder:
        raise ApiError(400, "invalid_request", "A link names a folder.")
    if folder not in {f.name for f in await store.folders(user_id)}:
        raise ApiError(404, "not_found", f"No such folder: {folder}.")
    await _link_folder(project_id, folder)
    await _touch_project(project_id)
    return {"id": project_id, "folders": await _folders_of(project_id)}


@app.patch("/projects/{project_id}")
async def rename_project(
    project_id: str,
    body: dict[str, Any] = JsonBody,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Rename a project. The title is a LABEL and nothing durable is keyed by it.

    No folder moves, and now there is nothing a rename could even reach for: a
    project LINKS folders and the folders are the store's, derived from the
    paths of files that exist. The Files tab's headers are those segments, so
    renaming a project cannot change one — which is the bug this replaced, where
    the headers were project titles and a rename renamed the filesystem.
    """
    await _owned_project(project_id, user_id)
    title = str(body.get("title") or "").strip()
    if not title:
        raise ApiError(400, "invalid_request", "A project needs a name.")
    row = await pool.fetchrow(
        "UPDATE projects SET title = $2, updated_at = now() WHERE id = $1 "
        "RETURNING id, title, updated_at",
        _uuid(project_id, "project"),
        title,
    )
    return {
        "id": str(row["id"]),
        "title": row["title"],
        # The links, so a surface can show where the work actually lands. The
        # rename did not touch them and cannot.
        "folders": await _folders_of(project_id),
        "updated_at": row["updated_at"].isoformat(),
    }


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
            **_session_core(r),
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

    A `plan` row carries one field the others do not: `version`, which is a fact
    about the session's plan HISTORY rather than about this row. It is read in
    one extra query, and only when a plan is actually waiting.
    """
    if project_id is not None:
        await _owned_project(project_id, user_id)
    if session_id is not None:
        await _owned_session(session_id, user_id)

    rows = await pool.fetch(
        """
        SELECT a.id, a.session_id, a.kind, a.prompt, a.created_at, a.tool_call_id,
               a.tool_name, a.tool_args,
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
            # Only a `call` carries these: the tool that runs if it is approved,
            # so the human decides on the call rather than a description of it.
            "tool_name": r["tool_name"],
            "tool_args": json.loads(r["tool_args"]) if isinstance(r["tool_args"], str) else r["tool_args"],
            "created_at": r["created_at"].isoformat(),
            **(await _plan_context(str(r["session_id"])) if r["kind"] == "plan" else {}),
        }
        for r in rows
    ]


async def _plan_context(session_id: str) -> dict[str, Any]:
    """Which version of this session's plan the open row is.

    The version of a plan IS its position in the session's history — there is no
    counter column, because a counter that disagreed with the rows would be a
    second source of truth for the same fact.

    The previous version's args and the reply that produced this one were sent
    too, for a "changed since v{n-1}" list on the card. They are not any more:
    edits stack, so by v3 the list was longer than the plan and said less. A
    reply is answered by a whole new plan, and the plan is what the human reads.
    """
    history = await approvals.plan_history(session_id)
    return {"version": len(history)} if history else {}


async def _latest_plan(session_id: str) -> dict[str, Any] | None:
    """The session's newest plan and what became of it, or None if it has none.

    `answer` is the row's verbatim decision — `approve`, `decline`, `superseded`,
    or the feedback that was sent — so a surface can tell an approved plan from
    a dismissed one without a second request.
    """
    history = await approvals.plan_history(session_id)
    if not history:
        return None
    newest = history[-1]
    return {
        "approval_id": newest.id,
        "version": len(history),
        "goal": (newest.tool_args or {}).get("goal"),
        "answer": newest.answer,
    }


# --- the store ------------------------------------------------------------------
#
# ONE flat namespace per user, and a folder is a top-level segment of it. These
# routes are the store itself; the project-scoped one below is a VIEW of it,
# narrowed to what one project links.


@app.get("/folders")
async def list_folders(user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """Every folder in the caller's store, with how many files are under it.

    What the create modal's checklist and the `+ link` picker read. There is no
    folders table to query: this is the first segment of every path the user
    has, grouped — which is why a folder cannot be stale, and why one appears
    the moment a file lands under a new first segment.
    """
    return [{"name": f.name, "files": f.files} for f in await store.folders(user_id)]


@app.post("/folders", status_code=201)
async def create_folder(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Make a folder durable the moment it is named.

    A folder is not a row — it is a path segment — so what lands is a zero-byte
    sentinel inside it. Holding the folder in the browser until a file arrived
    would put the truth in two places, and the first reload would disagree with
    one of them.
    """
    try:
        folder = store.safe_path(str(body.get("path") or ""))
    except ValueError as e:
        raise ApiError(400, "invalid_request", str(e)) from e
    if posixpath.basename(folder) == store.DIR_SENTINEL:
        raise ApiError(
            400, "invalid_request", f"{store.DIR_SENTINEL} is how an empty folder is kept, not a folder to make."
        )

    taken = await pool.fetchval(
        "SELECT 1 FROM files WHERE user_id = $1 AND (path = $2 OR path LIKE $3) LIMIT 1",
        _uuid(user_id, "user"),
        folder,
        f"{folder}/%",
    )
    if taken:
        raise ApiError(409, "already_exists", f"{folder} is already in the store.")

    sentinel = store.dir_sentinel(folder)
    await store.put_file(user_id, sentinel, b"")
    await workspace.write_through(sandbox_manager.manager(), user_id, sentinel, b"")
    return {"path": folder, "sentinel": sentinel}


@app.get("/files")
async def list_files(user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """The caller's whole store, as tree rows. No sandbox is woken to answer this."""
    rows = await pool.fetch(
        "SELECT id, path, size, mtime FROM files WHERE user_id = $1 ORDER BY path",
        _uuid(user_id, "user"),
    )
    return [_file_row(r) for r in rows]


@app.get("/files/{file_id}")
async def read_file(file_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """One file's contents, read from the store without waking anything.

    Text is returned decoded; anything that is not UTF-8 says so rather than
    arriving as mojibake, because a reader pane that renders a PNG as characters
    is worse than one that admits it cannot.
    """
    row = await pool.fetchrow(
        "SELECT path, content_hash, size, mtime FROM files WHERE id = $1 AND user_id = $2",
        _uuid(file_id, "file"),
        _uuid(user_id, "user"),
    )
    if row is None:
        raise ApiError(404, "not_found", "No such file.")

    blob = await store.get_blob(row["content_hash"])
    if blob is None:
        raise ApiError(410, "blob_missing", "The store no longer holds this file's contents.")

    common = {"path": row["path"], "size": row["size"], "mtime": row["mtime"].isoformat()}
    try:
        return {**common, "text": blob.decode(), "binary": False}
    except UnicodeDecodeError:
        return {**common, "text": None, "binary": True}


@app.post("/files", status_code=201)
async def upload_file(
    file: UploadFile = UploadedFile,
    path: str | None = UploadedPath,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Put a file in the store, and in any box already holding its folder.

    The store is where the file lands, so it exists whether or not anything is
    awake. A running session whose claim covers the path is written through and
    reads it the same turn; every other session gets it at its next materialize.

    A path is REQUIRED to name a folder, because every file in the store is in
    one: the folder is the first segment, so a bare filename would be its own
    folder holding nothing.
    """
    try:
        stored_path = store.in_folder(store.safe_path(path or file.filename or ""))
    except ValueError as e:
        raise ApiError(400, "invalid_request", str(e)) from e

    content = await _read_within_quota(file)
    stored = await store.put_file(user_id, stored_path, content)
    await _touch_linked_projects(user_id, store.folder_of(stored_path))
    await workspace.write_through(sandbox_manager.manager(), user_id, stored_path, content)

    return {
        "file_id": stored.id,
        "name": posixpath.basename(stored_path),
        "path": stored_path,
        "folder": store.folder_of(stored_path),
        "size": stored.entry.size,
    }


@app.post("/files/move")
async def move_file(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Move a file or a whole subtree inside the store: store first, boxes after.

    The store is the record and moves in one transaction. A live box is a cache
    and is corrected immediately after, because flush commits what is on disk —
    a box left holding the old path would put it back and delete the new one
    when the turn ends, undoing the move without saying so. Sessions whose box
    refused come back as `stale_sessions` rather than being swallowed.

    Moving BETWEEN folders is an ordinary move now: one namespace, a row edit,
    no copy. Moving a FOLDER is refused — that is a rename of the thing claims
    and mounts are keyed by, and it is its own card.
    """
    try:
        src = store.safe_path(str(body.get("from") or ""))
        dst = store.safe_path(str(body.get("to") or ""))
    except ValueError as e:
        raise ApiError(400, "invalid_request", str(e)) from e

    try:
        moves = await store.move_path(user_id, src, dst)
    except store.MissingPath as e:
        raise ApiError(404, "not_found", str(e)) from e
    except store.StoreError as e:
        raise ApiError(409, "move_refused", str(e)) from e

    if not moves:
        return {"from": src, "to": dst, "moved": [], "stale_sessions": []}

    for folder in {store.folder_of(src), store.folder_of(dst)}:
        await _touch_linked_projects(user_id, folder)
    _, stale = await workspace.move_through(sandbox_manager.manager(), user_id, moves)
    if stale:
        system_log.record(
            "workspace.move_stale",
            level="error",
            user_id=user_id,
            sessions=stale,
            moved=len(moves),
        )
    return {
        "from": src,
        "to": dst,
        "moved": [{"from": was, "to": now} for was, now in moves],
        "stale_sessions": stale,
    }


@app.post("/files/rename")
async def rename_file(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Rename anything in the store: a file, a directory, or a top-level folder.

    A rename changes what a thing is CALLED and not where it is, so `name` is a
    name — a `/` in it is refused rather than quietly making this a move, which
    has its own route and its own rules.

    Renaming a top-level FOLDER carries the projects that link it and the claims
    that mount it along with the paths, in one transaction. It is refused
    (`409 folder_busy`) while a live box has that folder materialized: the
    session's claims and its manifest live in the runner's memory as well as in
    the database, so a box left at `~/store/<old>/` would flush its work back
    under the old name and resurrect the folder — losing everything written
    since it was materialized. Stopping the run first is the answer, and saying
    so is better than a rename that half-happens.

    Nothing else is disturbed: blobs are content-addressed and immutable, so not
    one byte is re-uploaded, and every open reader keeps its `file_id`.
    """
    try:
        path = store.safe_path(str(body.get("path") or ""))
    except ValueError as e:
        raise ApiError(400, "invalid_request", str(e)) from e
    name = str(body.get("name") or "")

    try:
        destination = store.renamed_to(path, name)
    except ValueError as e:
        raise ApiError(400, "invalid_request", f"{str(e)} — a rename takes a name, not a path.") from e

    if "/" not in path:
        await _folder_is_free(user_id, path)

    try:
        moves = await store.rename_path(user_id, path, name)
    except store.MissingPath as e:
        raise ApiError(404, "not_found", str(e)) from e
    except store.StoreError as e:
        raise ApiError(409, "already_exists", str(e)) from e

    if not moves:
        return {"from": path, "to": destination, "moved": [], "stale_sessions": []}

    for folder in {store.folder_of(path), store.folder_of(destination)}:
        await _touch_linked_projects(user_id, folder)

    # A nested rename is a move as far as a live box is concerned, and the boxes
    # holding it are corrected in this same request for the reason every move is:
    # flush commits what is on disk, so a box left on the old path would put it
    # back. A top-level rename never reaches here with a live box — it was
    # refused above.
    _, stale = await workspace.move_through(sandbox_manager.manager(), user_id, moves)
    if stale:
        system_log.record(
            "workspace.rename_stale", level="error", user_id=user_id, sessions=stale, moved=len(moves)
        )
    return {
        "from": path,
        "to": destination,
        "moved": [{"from": was, "to": now} for was, now in moves],
        "stale_sessions": stale,
    }


async def _folder_is_free(user_id: str, folder: str) -> None:
    """Refuse to restructure a folder a run currently has on its disk.

    The same rule delete, undo and a folder rename all need, for the same
    reason: the runner holds that session's claims and its manifest in MEMORY,
    so nothing here can correct the box. A box that still has the files would
    put them back at its next flush, and a box that lost them would commit the
    loss. Stopping the run first is the answer, and saying so beats a change
    that half-happens.
    """
    busy = await workspace.boxes_holding(user_id, folder)
    if busy:
        raise ApiError(
            409,
            "folder_busy",
            f"{folder}/ is mounted in {len(busy)} running session(s). Stop them and try again after.",
        )


@app.delete("/files")
async def delete_file(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Delete a file or a whole subtree, and hand back the way to take it back.

    The rows go; the BLOBS do not, because they are content-addressed,
    immutable and never collected. Undo is therefore exact — the same content
    under the same id — rather than a best effort.

    A folder exists exactly as long as a file exists under it, so a delete that
    empties one takes the folder with it, and the project links that named it go
    into the same batch so they come back together. `folders` in the response is
    what ceased to exist, which is what a surface has to stop drawing.
    """
    try:
        path = store.safe_path(str(body.get("path") or ""))
    except ValueError as e:
        raise ApiError(400, "invalid_request", str(e)) from e

    await _folder_is_free(user_id, store.folder_of(path))
    try:
        gone = await store.delete_path(user_id, path)
    except store.MissingPath as e:
        raise ApiError(404, "not_found", str(e)) from e

    await _touch_linked_projects(user_id, store.folder_of(path))
    return {
        "path": gone.path,
        "batch": gone.batch,
        "files": gone.files,
        "unlinked": gone.unlinked,
        "folders": list(gone.folders),
    }


@app.post("/files/undo")
async def undo_delete(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Put back exactly what one delete removed, links included.

    `batch` names the gesture, so undo restores what that click took and not
    whatever happened to be deleted most recently. `409` when something has
    since been put at one of those paths: it was put there afterwards and is not
    this batch's to overwrite.
    """
    batch = str(body.get("batch") or "").strip()
    try:
        as_uuid(batch)
    except ValueError as e:
        raise ApiError(404, "not_found", "There is nothing to undo.") from e

    restored = await pool.fetchval(
        "SELECT path FROM deleted_files WHERE user_id = $1 AND batch = $2 LIMIT 1",
        _uuid(user_id, "user"),
        _uuid(batch, "batch"),
    )
    if restored is None:
        raise ApiError(404, "not_found", "There is nothing to undo.")
    await _folder_is_free(user_id, store.folder_of(restored))

    try:
        back = await store.undo_delete(user_id, batch)
    except store.MissingPath as e:
        raise ApiError(404, "not_found", str(e)) from e
    except store.StoreError as e:
        raise ApiError(409, "already_exists", str(e)) from e

    for folder in back.folders:
        await _touch_linked_projects(user_id, folder)
    return {
        "path": back.path,
        "files": back.files,
        "relinked": back.unlinked,
        "folders": list(back.folders),
    }


@app.get("/projects/{project_id}/files")
async def list_project_files(project_id: str, user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """The files under this project's linked folders — the working-files pane.

    A VIEW of the store, not a tree of its own: the rows are `files` rows and
    the paths are store paths, so clicking one in the pane and finding it in the
    Files tab is the same file at the same path rather than two listings that
    have to agree.
    """
    await _owned_project(project_id, user_id)
    rows = await pool.fetch(
        """
        SELECT f.id, f.path, f.size, f.mtime
          FROM files f
         WHERE f.user_id = $1
           AND split_part(f.path, '/', 1) IN (SELECT folder FROM project_folders WHERE project_id = $2)
         ORDER BY f.path
        """,
        _uuid(user_id, "user"),
        _uuid(project_id, "project"),
    )
    return [_file_row(r) for r in rows]


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
        stream.publish_all(session_id, await slog.close_dangling(session_id))

    await _append(session_id, UserEvent(text=text, source="human"))
    started = await runner.start(session_id)
    return {"accepted": True, "started": started}


@app.post("/approvals/{approval_id}/respond", status_code=202)
async def respond_to_approval(
    approval_id: str,
    body: dict[str, Any] = JsonBody,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Answer an open question, decide a gated call, or answer a plan.

    The kinds are answered differently because they mean different things.
    A question (`ask`, `approval`) takes prose, which is appended as a `user`
    event for the model to read. A `call` takes exactly `approve` or `decline`:
    it is a decision about a tool call that is still open in the transcript, and
    the resumed run executes or closes that call itself. Nothing is appended as
    a user message for a call — the model was not asked a question, and telling
    it "approve" as though the human had spoken would be a second, false record
    of what happened.

    A `plan` takes THREE answers, because a plan is replied to and a call is
    not. The approve word saves the approved args as `plan.md` and starts the
    unattended run; the decline word closes the park and leaves an attended
    chat; anything else is a reply, which closes this plan and asks for the next
    one. This is the only place a session's mode flips to unattended.
    """
    text = str(body.get("answer") or "").strip()
    if not text:
        raise ApiError(400, "invalid_request", "An answer is required.")

    approval = await approvals.get(approval_id, user_id)
    if approval is None:
        raise ApiError(404, "not_found", "No such approval.")

    if approval.gated_call:
        text = text.lower()
        if text not in (approvals.APPROVE, approvals.DECLINE):
            raise ApiError(
                400,
                "invalid_request",
                f"A tool call is decided, not discussed: send "
                f'{{"answer": "{approvals.APPROVE}"}} or {{"answer": "{approvals.DECLINE}"}}.',
            )

    if approval.is_plan:
        return await _answer_plan(approval, text, user_id)
    if approval.is_resume:
        return await _answer_resume(approval, text, user_id)

    # The UPDATE matches on answered_at IS NULL, so two people answering at once
    # produce one wake.
    answered = await approvals.answer(approval_id, text)
    if answered is None:
        raise ApiError(409, "already_answered", "That question has already been answered.")

    if not answered.gated_call:
        await _append(answered.session_id, UserEvent(text=text, source="human"))
    started = await runner.start(answered.session_id, reason="answered")
    return {"accepted": True, "session_id": answered.session_id, "started": started}


async def _answer_resume(approval: approvals.Approval, text: str, user_id: str) -> dict[str, Any]:
    """Resume a stopped run, steer it, or cancel it for real.

    The same three-answer shape as a plan, and the same reason: what a human
    does to a held run is not one bit. The approve word picks the run up where
    it held — same plan, same mode, same hop budget, because the hold wrote no
    terminal. Prose resumes it too, WITH the message appended: "skip that step,
    do X instead" is the resume, and the model reads it beside the call it just
    saw closed. The decline word is the hard cancel the button used to be, and
    it is the only thing here that spends the plan.
    """
    verdict = text.strip().lower()

    if verdict == approvals.DECLINE:
        answered = await approvals.answer(approval.id, approvals.DECLINE)
        if answered is None:
            raise ApiError(409, "already_answered", "That run has already been answered.")
        cancelled = await runner.cancel(answered.session_id)
        return {"accepted": True, "session_id": answered.session_id, "cancelled": cancelled}

    # `approve` is bare consent to carry on and appends nothing. Anything else is
    # steering, and the human said it, so it lands as their turn.
    steer = None if verdict == approvals.APPROVE else text
    answered = await approvals.answer(approval.id, approvals.APPROVE if steer is None else text)
    if answered is None:
        raise ApiError(409, "already_answered", "That run has already been answered.")
    if steer is not None:
        await _append(answered.session_id, UserEvent(text=steer, source="human"))
    # No `mode`: the hold never moved it, so the plan's approval still stands.
    started = await runner.start(answered.session_id, reason="resumed")
    return {"accepted": True, "session_id": answered.session_id, "started": started}


async def _answer_plan(approval: approvals.Approval, text: str, user_id: str) -> dict[str, Any]:
    """Approve, decline or workshop a proposed plan.

    The quota is checked BEFORE the row is answered: a user at their unattended
    limit should get a 429 and keep their plan, not lose it to an approval that
    then cannot start anything.
    """
    verdict = text.strip().lower()
    args = approval.tool_args or {}
    row = await _owned_session(approval.session_id, user_id)
    decision = verdict in (approvals.APPROVE, approvals.DECLINE)
    # The two words are stored NORMALIZED, so the row says what was decided
    # rather than how it was typed: "Approve" starting a run while the row read
    # unapproved would make the consent table disagree with what happened, on
    # the one table whose entire job is binding consent. Feedback is prose and
    # is stored exactly as written.
    recorded = verdict if decision else text

    if verdict == approvals.APPROVE:
        if await runner.plan_folder(approval.session_id) is None:
            # The prompt tells an unattended run its plan is `plan.md` at the
            # root of its first linked folder. A session that claims no folder
            # has nowhere to write it, and starting anyway would make that
            # promise a lie. Checked BEFORE the row is answered, like the quota.
            raise ApiError(
                409, "no_folder", "A plan is saved in a folder, and this session was given none."
            )
        if row["mode"] != "unattended":
            # Sessions are created attended and the play button no longer flips
            # the mode, so this is the only point at which a user's unattended
            # load grows — and it does not grow at all when the session
            # proposing is already unattended, which is why that case is exempt
            # rather than counting itself and refusing at a limit of one.
            await _check_unattended_quota(user_id)

    answered = await approvals.answer(approval.id, recorded)
    if answered is None:
        raise ApiError(409, "already_answered", "That plan has already been answered.")

    if verdict == approvals.APPROVE:
        version = len(await approvals.plan_history(answered.session_id))
        # What was approved is what is saved. Written before the run starts, so
        # the first materialize copies it into the box with everything else.
        await runner.save_plan(answered.session_id, args, version)
        # Mode and status move in ONE conditional UPDATE, inside `start`.
        started = await runner.start(answered.session_id, mode="unattended", reason="plan_approved")
        if not started:
            # The session moved under us. Put the card back rather than leaving a
            # plan stamped approved that nothing ran and nothing can approve
            # again — the human's only other recourse being to get the whole plan
            # proposed afresh.
            logger.warning("session %s: an approved plan could not start it", answered.session_id)
            await approvals.reopen(answered.id)
            raise ApiError(409, "not_idle", "The session moved before the plan could start it.")
        return {"accepted": True, "session_id": answered.session_id, "started": True, "mode": "unattended"}

    if verdict == approvals.DECLINE:
        # Nothing ran and nothing is owed to the model: the park simply closes
        # and the session goes back to being a chat.
        await lifecycle.transition(answered.session_id, "awaiting_approval", "idle", "plan_declined")
        return {"accepted": True, "session_id": answered.session_id, "started": False, "mode": "attended"}

    # A reply. It lands as the human's own turn — they typed it — followed by the
    # instruction that makes it produce a PLAN rather than a paragraph. Without
    # that second event the model reads the reply, answers it inline, and the
    # session goes idle with the card gone and nothing to approve: the run the
    # human was setting up quietly stops existing.
    await _append(answered.session_id, UserEvent(text=text, source="human"))
    await _append(answered.session_id, UserEvent(text=prompts.plan_reply(), source="system"))
    started = await runner.start(answered.session_id, reason="plan_reply")
    return {"accepted": True, "session_id": answered.session_id, "started": started, "mode": "attended"}


@app.post("/sessions/{session_id}/approve", status_code=202)
async def approve_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Ask for a plan for this session. It does NOT start an unattended run.

    The button used to flip the mode here and hand the model a transcript, which
    is not a task: the 2026-08-20 Marketplace run went unattended with the
    model's own unanswered question as the last event and burned its budget
    greeting nobody. Now pressing it appends a `user{source: system}` handoff and
    starts an ORDINARY ATTENDED TURN, whose job is to call `propose_plan`.

    So the mode flips in exactly one place — approving that plan — and both
    entries to an unattended run, this button and the model proposing off its own
    judgement, are the same tool and the same card. Pressed on a session with no
    conversation at all, this still yields a plan card: the handoff copy forbids
    asking in prose, so the gaps arrive as `missing` and the card opens as the
    intake form.
    """
    row = await _owned_session(session_id, user_id)
    if row["mode"] == "unattended":
        raise ApiError(409, "already_unattended", "This session is already running unattended.")
    # A TERMINAL session is a legal starting point, and it is the important one:
    # pressing this on a cancelled run is how a continuation gets drafted. The
    # handoff copy tells the model to read plan.md and the transcript and resume
    # from what is verifiably done rather than planning the work twice. The
    # `terminal -> running` reopen already exists for exactly this.
    if row["status"] not in ("idle", "pending") and row["status"] not in lifecycle.TERMINAL:
        raise ApiError(409, "not_idle", f"A session in {row['status']!r} cannot be handed over.")

    await _append(session_id, UserEvent(text=prompts.plan_handoff(), source="system"))
    started = await runner.start(session_id, reason="plan_requested")
    if not started:
        # A 202 with started:false would leave the surface waiting on a plan that
        # nothing is drafting, and the handoff instruction sitting in the
        # transcript with no hop to read it — the exact shape this card exists to
        # remove. The event stays: the next turn this session runs will read it
        # and propose, which is what was asked for.
        raise ApiError(409, "not_idle", "The session moved before it could be started.")
    return {"accepted": True, "started": True, "mode": "attended"}


@app.post("/sessions/{session_id}/stop", status_code=202)
async def stop_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Hold a running turn, without ending it.

    The run control has two faces and this is the first: **Stop while running,
    Cancel while stopped.** Stop closes the calls in flight as
    `cancelled_by_user`, refuses the rest of the hop, and parks on a `resume`
    row at the hop boundary — no `done`, no terminal, and NO MODE FLIP, so the
    plan the run was approved from is still approved and resuming costs nothing.

    Cancel was the only control before this, and it is `task.cancel()` on the
    whole turn: one slow step stopped, one approved plan spent. That is now the
    second face, and the fallback here when there is no live turn to hold.

    `stopped: false` means the turn is not running in this process, so the
    caller should cancel instead.
    """
    row = await _owned_session(session_id, user_id)
    if row["status"] != "running":
        raise ApiError(409, "not_running", f"A session in {row['status']!r} is not running.")
    return {"accepted": True, "stopped": await runner.stop(session_id)}


@app.post("/sessions/{session_id}/cancel", status_code=202)
async def cancel_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """End a run for good. The second face of the button, and the backstop.

    A stopped session is cancelled from here too, and its `resume` row closes
    with it: leaving the row open would put a card in front of a human offering
    to resume a session that is already terminal.
    """
    await _owned_session(session_id, user_id)
    for open_row in await approvals.open_for(session_id):
        if open_row.is_resume:
            await approvals.answer(open_row.id, approvals.DECLINE)
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


# --- what this session may reach ------------------------------------------------


@app.get("/sessions/{session_id}/tools")
async def session_tools_state(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """The tool budget for one session: the meter's numbers and a row per connected server.

    `budget` is `llm.max_tools - ours`, so the meter moves on its own when we add
    a local tool — ours are always loaded and never spend the human's allowance.
    `used` is the tool count of the servers this session has been given.

    What this does NOT say is what the model will actually be handed. Recording a
    toggle and obeying it are two cards: until Task 11.5 the loop builds its
    manifest without reading any of this, so the panel is honest about state
    before it is honest about effect.
    """
    await _owned_session(session_id, user_id)
    return await _tools_document(session_id, user_id)


@app.put("/sessions/{session_id}/tools/{server}")
async def set_session_tool(
    session_id: str,
    server: str,
    body: dict[str, Any] = JsonBody,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """Give this session a server, or take it away. Returns the whole document.

    The refusals are the point of the endpoint. A toggle that would put the
    manifest over `llm.max_tools` is refused HERE, with the numbers in the
    message, rather than being recorded and discovered later as a `bad_request`
    from the provider with nothing near the connection that caused it. The panel
    refuses it too, before the request; this is the half that holds when the
    panel is stale, and it is what makes the recorded state a state the system
    could actually ship.
    """
    await _owned_session(session_id, user_id)
    if "enabled" not in body:
        raise ApiError(400, "invalid_request", 'Send {"enabled": true|false}.')
    wanted = bool(body["enabled"])

    document = await _tools_document(session_id, user_id)
    row = next((r for r in document["servers"] if r["server"] == server), None)
    if row is None:
        raise ApiError(404, "not_found", f"No server {server!r} is configured.")

    if wanted and not row["enabled"]:
        if row["status"] != "connected":
            raise ApiError(
                409,
                "not_connected",
                f"{row['name']} is not connected. Authorize it in settings before adding it to a session.",
            )
        left = document["budget"] - document["used"]
        if row["tool_count"] > left:
            raise ApiError(
                409,
                "tool_budget",
                f"{row['name']} needs {row['tool_count']} of the {left} tool slot(s) left "
                f"({document['used']}/{document['budget']} in use). Turn something off first.",
            )

    await session_tools.set_enabled(session_id, row["mcp_url"], wanted)
    return await _tools_document(session_id, user_id)


async def _tools_document(session_id: str, user_id: str) -> dict[str, Any]:
    """Build the meter and the server rows from config, the connections and the toggles."""
    client = hands.smithery()
    rows = await client.connections(user_id) if client is not None else []
    on = set(await session_tools.enabled_urls(session_id))

    ours = len(registry.local_tools())
    max_tools = int(_cfg("llm.max_tools", 128))
    budget = max(0, max_tools - ours)

    servers = [{**row, "enabled": row["mcp_url"] in on} for row in rows]
    return {
        "max_tools": max_tools,
        "ours": ours,
        "budget": budget,
        "used": sum(s["tool_count"] for s in servers if s["enabled"]),
        "servers": servers,
    }


# --- the session's disk ---------------------------------------------------------


@app.get("/sessions/{session_id}/fs")
async def list_sandbox_dir(
    session_id: str,
    path: str = sandbox_tools.HOME,
    user_id: str = CurrentUser,
) -> dict[str, Any]:
    """List a directory on the session's live sandbox disk.

    Ownership-checked the way the frame stream is, and it never boots anything:
    a box that has parked or been reaped reads as 404, because starting compute
    so that somebody can look at a folder is not a thing a browse should do.
    """
    await _owned_session(session_id, user_id)
    try:
        entries = await sandbox_manager.manager().browse(session_id, path)
    except sandbox_manager.BoxNotAwake as e:
        raise _no_box(session_id) from e
    except Exception as e:  # noqa: BLE001 - e2b raises its own types
        raise ApiError(404, "not_found", f"Could not list {path!r} in this session's box.") from e
    return {"path": path, "entries": entries}


@app.get("/sessions/{session_id}/fs/file")
async def read_sandbox_file(session_id: str, path: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """One file from the session's live sandbox disk, on the same terms as the listing.

    Text is returned decoded and anything that is not UTF-8 says so rather than
    arriving as mojibake — the same bargain `GET /projects/{id}/files/{file_id}`
    makes about the store. A file past `sandbox.browse_max_bytes` comes back cut
    short and SAYS it was, because half a file rendered as a whole one is the
    kind of quiet lie a reader pane should never tell.
    """
    await _owned_session(session_id, user_id)
    cap = int(_cfg("sandbox.browse_max_bytes", 1048576))
    try:
        blob, truncated = await sandbox_manager.manager().peek(session_id, path, max_bytes=cap)
    except sandbox_manager.BoxNotAwake as e:
        raise _no_box(session_id) from e
    except Exception as e:  # noqa: BLE001 - e2b raises its own types
        raise ApiError(404, "not_found", f"Could not read {path!r} in this session's box.") from e

    try:
        text = blob.decode()
    except UnicodeDecodeError:
        return {"path": path, "size": len(blob), "text": None, "binary": True, "truncated": truncated}
    return {"path": path, "size": len(blob), "text": text, "binary": False, "truncated": truncated}


def _no_box(session_id: str) -> ApiError:
    """The one answer for a session whose disk is not there to read."""
    return ApiError(
        404,
        "not_found",
        "This session has no computer running. Its disk exists only while it is awake.",
    )


# --- helpers -------------------------------------------------------------------


async def _touch_project(project_id: str) -> None:
    """Mark a project as changed. Written inline at three call sites before.

    Separate from `lifecycle.touch_project`, which takes a connection and a
    SESSION id because it runs inside the transaction that moves a session. This
    one is for the routes that have a project id and no transaction to join.
    """
    await pool.execute("UPDATE projects SET updated_at = now() WHERE id = $1", _uuid(project_id, "project"))


async def _touch_linked_projects(user_id: str, folder: str) -> None:
    """Mark every project that links this folder as changed.

    A file route no longer knows which project it is acting for, because it is
    not acting for one: it writes to the store, and a project is whatever links
    the folder the write landed in. Nought, one or several — the grid's "when"
    is about work, and work is what a linked folder holds.
    """
    await pool.execute(
        """
        UPDATE projects SET updated_at = now()
         WHERE user_id = $1
           AND id IN (SELECT project_id FROM project_folders WHERE folder = $2)
        """,
        _uuid(user_id, "user"),
        folder,
    )


def _file_row(row: Any) -> dict[str, Any]:
    """One tree row on the wire. `path` is the full store path, folder included."""
    return {
        "file_id": str(row["id"]),
        "path": row["path"],
        "name": posixpath.basename(row["path"]),
        "folder": store.folder_of(row["path"]),
        "size": row["size"],
        "mtime": row["mtime"].isoformat(),
    }


async def _folders_of(project_id: str) -> list[str]:
    """The folders a project links, in the order they were linked."""
    rows = await pool.fetch(
        "SELECT folder FROM project_folders WHERE project_id = $1 ORDER BY created_at, folder",
        _uuid(project_id, "project"),
    )
    return [r["folder"] for r in rows]


async def _link_folder(project_id: str, folder: str) -> None:
    """Record one link. Linking twice is the same link."""
    await pool.execute(
        "INSERT INTO project_folders (project_id, folder) VALUES ($1, $2) ON CONFLICT DO NOTHING",
        _uuid(project_id, "project"),
        folder,
    )


async def _make_folder(user_id: str, base: str) -> str:
    """Reserve a new, empty folder named after `base` and return the name it got.

    The none-case of creating a project. A folder exists exactly as long as a
    file exists under it, so reserving one IS writing the sentinel — the same
    zero-byte file that keeps any named-but-unfilled folder alive, riding
    materialize and flush like everything else.
    """
    name = await store.unique_folder(user_id, base)
    await store.put_file(user_id, store.dir_sentinel(name), b"")
    return name


def _session_core(row: Any) -> dict[str, Any]:
    """The fields every session projection carries, shaped once.

    Three endpoints return a session and each built these seven by hand, so a
    field added to one drifted from the others silently. What differs BETWEEN
    the projections — a project title here, an open-question count there, the
    transcript tail in the snapshot — stays at the call site, because that is
    the part that is genuinely different.
    """
    return {
        "session_id": str(row["id"]),
        "title": row["title"],
        "status": row["status"],
        # The UI reads mode to decide whether to offer the approve control.
        "mode": row["mode"],
        "terminal_reason": row["terminal_reason"],
        "hops_used": row["hops_used"],
        "hops_max": _budgets_for(row["mode"]),
    }


async def _new_project(user_id: str, title: str) -> Any:
    """Create a project row. It links no folder yet; the caller does that.

    `slug` survives as nothing but the default NAME for the folder the none-case
    makes, and it is no longer uniquified here: folder names are unique per
    user because they are segments of unique paths, and `store.unique_folder`
    resolves a collision against the folders that actually exist rather than
    against a column that stopped describing them.
    """
    return await pool.fetchval(
        "INSERT INTO projects (user_id, title, slug) VALUES ($1, $2, $3) RETURNING id",
        _uuid(user_id, "user"),
        title,
        store.slug(title, "project"),
    )


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


# What a composer message is refused with, per park kind. `ask` and `resume` are
# absent on purpose: they are the two parks a typed message legitimately
# answers — one because it asked a question, the other because the consent it
# holds on was given already.
_WAITING_ON = {"call": "a tool call", "approval": "an approval", "plan": "a plan"}


async def _answer_by_message(session_id: str, text: str) -> dict[str, Any]:
    """Route a composer message sent to a parked session.

    A question raised by `ask` is answered by the message, and the session
    wakes. Consent is NOT: an `approval` and a gated `call` are both answered
    only through `/approvals/{id}/respond`, where the thing being agreed to is
    on screen.

    A `call` falling through here was the sharp edge. `approvals.approved` is an
    allow-list, so any prose that is not the approve word reads as a decline —
    "sounds good, go ahead" typed in the composer would have silently declined a
    call the human never saw, which undoes the whole point of binding consent to
    the call.

    A `plan` is refused for the same reason and one more: the card's own input is
    where a reply becomes the next version, and "yes do that" typed in the
    composer would read as a reply rather than as the approval it meant.

    A `resume` is the one park a typed message DOES answer, and it is not an
    exception to the rule — it is the rule. The consent that park waits on is
    the plan, and the plan is already approved; prose here approves nothing new,
    it steers a run the human already said yes to. So typing "skip the browser,
    do it another way" into a stopped run resumes it with that message, which is
    the whole point of stopping rather than cancelling. The decline word cannot
    reach this path at all — cancelling is a card action — so a message that
    happens to contain it still resumes.
    """
    open_questions = await approvals.open_for(session_id)
    if open_questions and open_questions[0].kind in _WAITING_ON:
        waiting = _WAITING_ON[open_questions[0].kind]
        raise ApiError(
            409,
            "awaiting_approval",
            f"This session is waiting on {waiting}. Answer it there, where you can see what it does.",
        )

    if open_questions:
        await approvals.answer(open_questions[0].id, text)
    await _append(session_id, UserEvent(text=text, source="human"))
    started = await runner.start(session_id, reason="answered")
    return {"accepted": True, "started": started}


async def _record_claims(session_id: str, project_id: Any, declared: Any, user_id: str) -> None:
    """Record which FOLDERS this session may touch, fixed for its life.

    Absent, it claims every folder its project links, all write — which is what
    "every session spawned in the project receives them" means, and why a link
    added later reaches the agent at the NEXT session: this runs once, at
    creation, and nothing rewrites it.

    A session with no project claims nothing and mounts nothing. That is the
    home chat, and it is honest: there is no folder it was given.
    """
    rows: list[tuple[str, str, str]] = []
    if isinstance(declared, list) and declared:
        known = {f.name for f in await store.folders(user_id)}
        for claim in declared:
            if not isinstance(claim, dict):
                raise ApiError(400, "invalid_request", "Each claim is an object with a folder.")
            folder = str(claim.get("folder") or "").strip().strip("/")
            if not folder:
                raise ApiError(400, "invalid_request", "Each claim names a folder.")
            if folder not in known:
                raise ApiError(404, "not_found", f"No such folder: {folder}.")
            mode = str(claim.get("mode") or "write")
            if mode not in ("read", "write"):
                raise ApiError(400, "invalid_request", f"A claim is read or write, not {mode!r}.")
            rows.append((folder, str(claim.get("subpath") or "/"), mode))
    elif project_id is not None:
        rows = [(folder, "/", "write") for folder in await _folders_of(str(project_id))]

    # `ord` is the order they were GIVEN, which for the default case is the order
    # the project linked them. It is an explicit column rather than a timestamp
    # because the answer has to be stable: `plan.md` goes to the FIRST folder,
    # and "first" cannot depend on how close together two inserts landed.
    for position, (folder, subpath, mode) in enumerate(rows):
        await pool.execute(
            """
            INSERT INTO session_claims (session_id, folder, subpath, mode, ord)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (session_id, folder, subpath)
            DO UPDATE SET mode = EXCLUDED.mode, ord = EXCLUDED.ord
            """,
            _uuid(session_id, "session"),
            folder,
            subpath,
            mode,
            position,
        )


async def _claims_of(session_id: str) -> list[dict[str, Any]]:
    """The session's claims, for the window to render. Folders, in claim order."""
    rows = await pool.fetch(
        "SELECT folder, subpath, mode FROM session_claims WHERE session_id = $1 "
        "ORDER BY ord, folder, subpath",
        _uuid(session_id, "session"),
    )
    return [{"folder": r["folder"], "subpath": r["subpath"], "mode": r["mode"]} for r in rows]


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
    """Coerce an id from the wire, or 404 with the noun in it.

    The coercion is `db.ids.as_uuid`; what is local is the ANSWER. A malformed
    id over HTTP is indistinguishable from one that simply does not exist, and
    saying "no such project" leaks nothing while "invalid UUID" tells a prober
    it guessed the shape right.
    """
    try:
        return as_uuid(value)
    except ValueError as e:
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

