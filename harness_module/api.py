"""
The HTTP surface: snapshot, subscribe, commands.

One error shape everywhere — `{code, message, retryable}` — and one way in:
the session cookie. No endpoint takes a user id from a header, a body or a
query string, including the OAuth callback, which is the one request that
arrives as somebody else's redirect.

**Scope.** Task 4: auth, chat on the new loop, the connections surface. The
rows this table is still missing (`/approve`, `/approvals/{id}/respond`,
`/attention`, file upload, browser frames) belong to Tasks 5, 6, 9 and Looking
Glass, and each is absent rather than stubbed — a stub that 202s and does
nothing is worse than a 404.
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
from harness_module import hands, jwt_utils, lifecycle, runner
from harness_module import session_log as slog
from harness_module.stream import LAGGED, stream
from tool_module.smithery import AuthRequiredError, SmitheryError

logger = logging.getLogger(__name__)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


# --- the one error shape -------------------------------------------------------


class ApiError(Exception):
    """Everything the API refuses, in the shape the frontend switches on."""

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
    """Bring the process up in the order the invariants need."""
    jwt_utils.assert_secure_secrets()
    if not _cfg("app.public_url", ""):
        logger.warning("app.public_url is unset: mutations are not origin-checked and OAuth has no return url")

    # Before anything can be woken: a session this process is not running must
    # not still claim to be running, or `start` refuses to touch it forever.
    with contextlib.suppress(Exception):
        await lifecycle.sweep_interrupted()
    await hands.start()
    try:
        yield
    finally:
        await hands.stop()
        await pool.close()


app = FastAPI(title="ARKOS", lifespan=lifespan)

_origin = str(_cfg("app.public_url", "")).rstrip("/")
app.add_middleware(
    CORSMiddleware,
    # Defence in depth, not the thing making this work: /app and the API are
    # served from one origin, so the cookie arrives because it is same-site.
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
    """The CSRF cost of cookies, paid on every mutation."""
    if not _origin:
        return
    origin = request.headers.get("origin")
    # A same-origin fetch from some clients sends no Origin at all; a
    # cross-site form post always does, which is the case this closes.
    if origin is not None and origin.rstrip("/") != _origin:
        raise ApiError(403, "bad_origin", "Origin not allowed.")


CurrentUser = Depends(current_user)
JsonBody = Body(...)


# --- auth ----------------------------------------------------------------------


@app.post("/auth/session", status_code=204)
async def create_auth_session(authorization: str | None = Header(default=None)) -> Response:
    """The ONLY endpoint that reads a bearer token, and the only path to a cookie."""
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
    """Unauthenticated on purpose: it is what the uptime check pings."""
    try:
        await pool.fetchval("SELECT 1")
        database = "ok"
    except Exception as e:  # noqa: BLE001 - the point is to report it, not raise
        database = f"unreachable: {type(e).__name__}"
    return {"status": "ok" if database == "ok" else "degraded", "database": database}


# --- sessions ------------------------------------------------------------------


@app.post("/sessions", status_code=201)
async def create_session(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """
    Open a session on a goal and start its first turn.

    Every session is created attended (D5), which is why the concurrency quota
    is checked at `/approve` and only the rate quota is checked here — at create
    time the unattended load is always zero and that check could never fire.
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

    await _append(session_id, UserEvent(text=goal, source="human"))
    steps = body.get("steps")
    if isinstance(steps, list) and steps:
        items = [{"text": str(s), "status": "pending"} for s in steps]
        await _append(session_id, TodoEvent(items=items))

    await runner.start(session_id)
    return {"session_id": session_id, "project_id": str(project_id)}


@app.get("/sessions/{session_id}")
async def get_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """The just-opened snapshot: the same events the stream carries, read once."""
    row = await _owned_session(session_id, user_id)
    budgets = _budgets_for(row["mode"])
    events = await slog.recent_events(session_id, limit=int(_cfg("harness.snapshot_events", 200)))
    return {
        "session_id": str(row["id"]),
        "title": row["title"],
        "project_id": str(row["project_id"]) if row["project_id"] else None,
        "status": row["status"],
        # The UI needs mode to know whether to offer the approve affordance.
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
            # The grid's status dot, rolled up here so ten projects cost one query.
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
    """
    Say something to a session.

    Idle or finished, this starts a turn. Already running, it is steering: the
    event is in the log and the loop reads it at the next hop, which is why
    this returns 202 rather than waiting for anything.
    """
    text = str(body.get("text") or "").strip()
    if not text:
        raise ApiError(400, "invalid_request", "An empty message says nothing.")
    row = await _owned_session(session_id, user_id)
    if row["status"] == "awaiting_approval":
        # A composer message must not answer a park: an ambiguous "sure" typed
        # into a chat box is not consent. The respond endpoint is Task 6.
        raise ApiError(409, "awaiting_approval", "This session is waiting on an approval. Answer that first.")

    await _append(session_id, UserEvent(text=text, source="human"))
    started = await runner.start(session_id)
    return {"accepted": True, "started": started}


@app.post("/sessions/{session_id}/cancel", status_code=202)
async def cancel_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    await _owned_session(session_id, user_id)
    return {"cancelled": await runner.cancel(session_id)}


@app.get("/results/{ref}")
async def read_result(ref: str, offset: int = 0, limit: int = 2000, user_id: str = CurrentUser) -> dict[str, Any]:
    """Page a stored oversized result. Ownership-checked: unguessable is not access control."""
    text = await slog.read_blob(ref, offset=max(0, offset), limit=max(1, min(limit, 100_000)), user_id=user_id)
    if text is None:
        raise ApiError(404, "not_found", "No such result.")
    return {"ref": ref, "offset": offset, "content": text}


# --- the stream ----------------------------------------------------------------


@app.get("/sessions/{session_id}/events")
async def session_events(
    session_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    user_id: str = CurrentUser,
) -> StreamingResponse:
    """
    Live events, with replay.

    Three UI moments, one mechanism: just-opened reads the snapshot above,
    watching gets these, and a reconnect replays after `Last-Event-ID` and then
    goes live. `id:<seq>` on every frame is what makes the third work with no
    code in the browser beyond `EventSource`.
    """
    await _owned_session(session_id, user_id)
    after = _int_or(last_event_id or request.query_params.get("last_event_id"), 0)
    return StreamingResponse(
        _event_stream(session_id, after),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _event_stream(session_id: str, after_seq: int) -> AsyncIterator[str]:
    """
    Yield SSE frames until the client goes away.

    Subscribing BEFORE reading the backlog is what closes the gap where an event
    appended between the two would reach nobody; `sent` then drops the overlap.
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
                    # Proxies and EventSource drop a stream that sends nothing.
                    yield ": keepalive\n\n"
                    continue

                if item is LAGGED:
                    # This consumer fell behind. The log is the truth, so it
                    # rejoins from there rather than losing what it missed.
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
    except Exception as e:  # noqa: BLE001 - a dead stream must say so, not truncate
        logger.exception("session %s: the event stream failed", session_id)
        # Truncation and completion look identical to EventSource, so a failure
        # that says nothing reads as "the agent stopped talking".
        yield "event: error\ndata: " + json.dumps(
            {"code": "stream_failed", "message": f"{type(e).__name__}: {e}", "retryable": True}
        ) + "\n\n"


def _frame(stored: slog.StoredEvent) -> str:
    return f"id: {stored.seq}\nevent: {stored.event.kind}\ndata: {json.dumps(_wire(stored), default=str)}\n\n"


def _wire(stored: slog.StoredEvent) -> dict[str, Any]:
    """Store-shape equals wire-shape; seq and ts are the two columns the payload has no room for."""
    row = stored.event.to_row()
    return {"seq": stored.seq, "ts": stored.ts.isoformat(), **row}


# --- MCP connections ------------------------------------------------------------


@app.get("/connections")
async def list_connections(user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """
    Every configured server and where the user stands with it.

    Reading this re-verifies anything not yet connected: `connect()` is
    idempotent, so a popup closed before its opener could re-fetch is repaired
    the next time somebody looks at the panel.
    """
    client = hands.smithery()
    if client is None:
        return []
    rows = await client.connections(user_id)
    if not any(client.needs_repair(r) for r in rows):
        return rows

    # Read-repair: one attempt per unconnected server, then re-read. This is the
    # second half of what makes the OAuth callback reliable — the callback fires
    # the verification, and this catches whatever the callback missed.
    for row in rows:
        if client.needs_repair(row):
            await _verify_once(user_id, row["server"])
    return await client.connections(user_id)


@app.post("/connections/{server}/connect")
async def connect_server(server: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Mint the id, write the pending row, PUT to Smithery. Idempotent: a reconnect reuses the id."""
    client = _require_smithery()
    if server not in client.servers:
        raise ApiError(404, "not_found", f"No server {server!r} is configured.")
    try:
        await client.connect(user_id, server, return_url=_callback_url(server))
    except AuthRequiredError as e:
        # Not an error: it is the whole point. The user has to authorize it.
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
    """
    Smithery's redirect lands here when the user finishes authorizing.

    Identity comes from the cookie, never a query parameter: this is a top-level
    GET, so SameSite=Lax sends it, and taking a user id from the URL would let
    anyone finish a flow as anyone.

    The verification fires AFTER the response, as one attempt that blocks
    nothing. It has to fire: dispatch never re-verifies (a tool call must not
    open an OAuth flow) and revalidation skips unconnected rows, so a popup
    closed before its opener re-reads would leave the row saying `auth_required`
    forever after a successful authorization.
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
    """One idempotent re-assert. Never retried on a timer: a poll on the request path is what this replaces."""
    client = hands.smithery()
    if client is None:
        return
    try:
        await client.connect(user_id, server)
    except AuthRequiredError:
        # OAuth did not finish after all. Read-repair on GET /connections covers it.
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
    """Load a session or refuse. Someone else's session is `not_found`, not `forbidden`."""
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
    """Append and publish, so a watcher sees a typed message as soon as it lands."""
    stored = await slog.append(session_id, event)
    stream.publish(session_id, stored)


async def _check_rate_quota(user_id: str) -> None:
    """The sliding window on new sessions. Checked before any state changes."""
    limit = int(_cfg("quotas.new_sessions_per_hour", 20))
    recent = await pool.fetchval(
        "SELECT count(*) FROM sessions WHERE user_id = $1 AND created_at > now() - interval '1 hour'",
        _uuid(user_id, "user"),
    )
    if recent >= limit:
        raise ApiError(429, "quota_exceeded", f"{limit} new sessions an hour is the limit.", retryable=True)


def _budgets_for(mode: str) -> int:
    return int(_cfg(f"budgets.{mode}.max_hops", 0))


def _rollup(row: Any) -> str:
    """The grid's dot: the most urgent thing happening in the project."""
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
