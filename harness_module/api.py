"""The HTTP surface: session snapshots, the event stream, and commands.

Every error response carries `{code, message, retryable}`. The caller is
identified by the session cookie; no endpoint reads a user id from a header,
body or query string, the OAuth callback included.

Auth, chat and the MCP connections surface are served here.
`/approvals/{id}/respond`, `/attention`, file upload and browser frames have no
route and return 404.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import jwt
from fastapi import Body, Depends, FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.background import BackgroundTask

from agent_module.events import TodoEvent, UserEvent
from config_module.loader import config
from db import pool
from harness_module import approvals, hands, jwt_utils, lifecycle, runner, system_log
from harness_module import session_log as slog
from harness_module.stream import LAGGED, stream
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
    if not _cfg("app.public_url", ""):
        logger.warning("app.public_url is unset: mutations are not origin-checked and OAuth has no return url")

    # Runs before any session can start: `start` refuses a session whose row
    # still says running.
    with contextlib.suppress(Exception):
        await lifecycle.sweep_interrupted()
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
    """Reject a mutation whose Origin header names another origin."""
    origin = request.headers.get("origin")
    if origin is None:
        # Same-origin fetches and non-browser clients send no Origin header.
        return
    # With app.public_url unset, `_origin` is empty and every browser mutation
    # is refused.
    if origin.rstrip("/") != _origin:
        raise ApiError(403, "bad_origin", "Origin not allowed.")


CurrentUser = Depends(current_user)
JsonBody = Body(...)


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


@app.delete("/auth/session", status_code=204)
async def delete_auth_session() -> Response:
    out = Response(status_code=204)
    out.delete_cookie(key=str(_cfg("auth.cookie_name", "ark_session")), path="/")
    return out


@app.get("/auth/me")
async def auth_me(user_id: str = CurrentUser) -> dict[str, Any]:
    row = await pool.fetchrow("SELECT id, email FROM users WHERE id = $1", _uuid(user_id, "user"))
    if row is None:
        raise ApiError(401, "unauthenticated", "That user no longer exists.")
    return {"user_id": str(row["id"]), "email": row["email"]}


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
    async with (await pool.pool()).acquire() as conn:
        await lifecycle.touch_project(conn, session_id)

    await _append(session_id, UserEvent(text=goal, source="human"))
    steps = body.get("steps")
    if isinstance(steps, list) and steps:
        items = [{"text": str(s), "status": "pending"} for s in steps]
        await _append(session_id, TodoEvent(items=items))

    await runner.start(session_id)
    return {"session_id": session_id, "project_id": str(project_id)}


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
            for stored in await slog.get_events(session_id, after_seq=sent, limit=1000):
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
                    for stored in await slog.get_events(session_id, after_seq=sent, limit=1000):
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


def _frame(stored: slog.StoredEvent) -> str:
    return f"id: {stored.seq}\nevent: {stored.event.kind}\ndata: {json.dumps(_wire(stored), default=str)}\n\n"


def _wire(stored: slog.StoredEvent) -> dict[str, Any]:
    """Render one stored event for the wire, adding the seq and ts columns."""
    row = stored.event.to_row()
    return {"seq": stored.seq, "ts": stored.ts.isoformat(), **row}


# --- MCP connections ------------------------------------------------------------


@app.get("/connections")
async def list_connections(user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """List every configured server and this user's connection status.

    A server that is not connected is re-verified before the list is returned;
    `connect()` is idempotent, so a row left behind by an interrupted OAuth
    flow is repaired on read.
    """
    client = hands.smithery()
    if client is None:
        return []
    rows = await client.connections(user_id)
    if not any(client.needs_repair(r) for r in rows):
        return rows

    for row in rows:
        if client.needs_repair(row):
            await _verify_once(user_id, row["server"])
    return await client.connections(user_id)


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


async def _check_rate_quota(user_id: str) -> None:
    """Enforce the sliding window on new sessions, before anything is written."""
    limit = int(_cfg("quotas.new_sessions_per_hour", 20))
    recent = await pool.fetchval(
        "SELECT count(*) FROM sessions WHERE user_id = $1 AND created_at > now() - interval '1 hour'",
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
