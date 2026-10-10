"""The HTTP surface: session snapshots, the event stream, and commands.

Every error response is `{code, message, retryable}`, and the caller is
identified by the session cookie only — never by a user id in a header, body or
query string, the OAuth callback included.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import posixpath
import signal
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import jwt
from fastapi import Body, Depends, FastAPI, File, Form, Header, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from agent_module import prompts
from agent_module.events import ContentEvent, TodoEvent, UserEvent, todo_item
from config_module.loader import cfg as _cfg
from config_module.loader import config
from db import pool
from db.ids import as_uuid
from harness_module import approvals, blobs, hands, jwt_utils, leases, lifecycle, runner, store, system_log, workspace
from harness_module import session_log as slog
from harness_module.stream import CLOSED, LAGGED, shutdown_streams, stream
from harness_module.stream import attention as attention_channel
from model_module import client as model_client
from tool_module import registry, session_tools
from tool_module.browser.stream import broker as frames
from tool_module.composio_mcp import ComposioError
from tool_module.sandbox import manager as sandbox_manager
from tool_module.sandbox import tools as sandbox_tools

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

    # Must run before any session can start: `start` refuses a session whose row
    # still says running.
    with contextlib.suppress(Exception):
        await lifecycle.sweep_interrupted()
    with contextlib.suppress(Exception):
        await sandbox_manager.sweep_slots()
    await hands.start()
    await system_log.start()
    keeper = asyncio.create_task(_keys_and_sweep(), name="jwks_and_session_sweep")
    # NOT in the `finally` below: uvicorn drains open connections BEFORE it runs
    # the lifespan's shutdown, so a stream waiting to be told to stop blocks the
    # drain that would deliver the message. The signal is the earliest moment.
    _end_streams_on_signal()
    try:
        yield
    finally:
        # Idempotent second call, for a shutdown that arrives without a signal.
        shutdown_streams()
        keeper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await keeper
        await system_log.stop()
        await hands.stop()
        # The store's HTTP client is scoped to this loop, so it closes here.
        await blobs.close_clients()
        model_client.reset_client()
        await pool.close()


def _end_streams_on_signal() -> None:
    """Tell every open stream to end the moment a shutdown signal arrives.

    Chained rather than installed — one handler per signal, so replacing
    uvicorn's would mean the server never learns to stop. Best-effort.
    """
    for signame in ("SIGTERM", "SIGINT"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        previous = signal.getsignal(sig)

        def chained(signum, frame, _previous=previous):
            shutdown_streams()
            if callable(_previous):
                _previous(signum, frame)

        try:
            signal.signal(sig, chained)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            logger.debug("could not chain %s; streams will end at lifespan shutdown", signame)


app = FastAPI(title="Arkos", lifespan=lifespan)

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


async def _keys_and_sweep() -> None:
    """The periodic tick: refresh the signing keys, prune dead sessions.

    ONE loop for both because they share a schedule and neither belongs in a
    request. The JWKS half is what makes verification a pure cache read — the
    host is normally 200ms and occasionally 30s, and a sign-in must never be the
    thing that finds out. The sweep half is bookkeeping: an expired cookie is
    already refused on its own `exp`, so a row past `expires_at` is dead weight.

    COLD START IS THE CASE THIS IS SHAPED AROUND. An empty cache refuses every
    token, so a restart landing in the endpoint's flaky window would be an
    outage for a whole interval. Two answers: the last good set is loaded from
    disk before the first fetch, so a restart is normally warm already; and
    while the cache is EMPTY the retry is a short backoff rather than the tick
    interval, because minutes of refusing sign-ins is not a schedule anyone
    chose.
    """
    every = float(_cfg("auth.jwks_refresh_s", 300))
    ceiling = float(_cfg("auth.jwks_retry_max_s", 30))
    warm = await asyncio.to_thread(jwt_utils.prime_jwks_from_disk)
    if not warm:
        logger.info("no key cache on disk; the first fetch has to land before anyone can sign in")
    backoff = 1.0
    fetching = True
    while True:
        try:
            got = False
            if fetching:
                try:
                    got = await asyncio.to_thread(jwt_utils.refresh_jwks)
                except jwt_utils.NoKeysPublished as e:
                    if os.environ.get("SUPABASE_JWT_SECRET"):
                        # HS256 tokens verify against the secret and never read
                        # the cache, so there is nothing to wait for. A project
                        # that later publishes keys is picked up on restart.
                        logger.info("the project publishes no signing keys (%s); verifying with SUPABASE_JWT_SECRET", e)
                        fetching = False
            warm = warm or got
            pruned = await pool.execute("DELETE FROM auth_sessions WHERE expires_at < now()")
            if pruned and pruned != "DELETE 0":
                logger.info("swept expired sessions: %s", pruned)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a tick that dies must not take the app
            logger.warning("the key/sweep tick failed; retrying", exc_info=True)
            got = False
        if warm or not fetching:
            backoff = 1.0
            await asyncio.sleep(every)
        else:
            # Nothing can be verified yet. Try again soon, not in five minutes.
            logger.error("still no signing keys; sign-ins are refused until a fetch lands (retry in %.0fs)", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, ceiling)


# --- who is calling ------------------------------------------------------------

# The one authenticated-by-nothing endpoint, so the one that needs its own
# ceiling (12.2.5). In-process and per-IP: this is a single event loop, and the
# point is to bound work an anonymous caller can order, not to be a quota system.
_auth_hits: dict[str, list[float]] = {}


def _check_auth_rate(request: Request) -> None:
    """Bound how often one caller may ask us to verify a token."""
    limit = int(_cfg("quotas.auth_attempts_per_minute", 30))
    if limit <= 0:
        return
    who = request.client.host if request.client else "unknown"
    now = time.monotonic()
    recent = [at for at in _auth_hits.get(who, []) if now - at < 60.0]
    if len(recent) >= limit:
        _auth_hits[who] = recent
        raise ApiError(
            429,
            "quota_exceeded",
            f"Too many sign-in attempts from here. Wait a minute ({limit}/min).",
            retryable=True,
        )
    recent.append(now)
    _auth_hits[who] = recent
    # Bounded memory: an attacker rotating source addresses must not grow this
    # without limit. Anything with no live hits is dropped on the next sweep.
    if len(_auth_hits) > 4096:
        for addr in [a for a, hits in _auth_hits.items() if not hits or now - hits[-1] > 60.0]:
            _auth_hits.pop(addr, None)


async def current_user(request: Request) -> str:
    """Resolve the caller from the session cookie, and origin-check mutations."""
    cookie = request.cookies.get(str(_cfg("auth.cookie_name", "arkos_session")))
    if not cookie:
        raise ApiError(401, "unauthenticated", "No session. Sign in first.")
    try:
        claims = jwt_utils.read_session(cookie)
    except jwt.PyJWTError as e:
        raise ApiError(401, "unauthenticated", f"Session rejected: {e}") from e

    # The cookie's signature says it was ours; this says it still IS (12.2.5).
    # One indexed primary-key lookup, and it is what lets a password change
    # reach a session living in someone else's browser.
    live = await pool.fetchval("SELECT 1 FROM auth_sessions WHERE jti = $1", _uuid(claims["jti"], "session"))
    if live is None:
        raise ApiError(401, "unauthenticated", "That session was signed out. Sign in again.")

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
async def create_auth_session(request: Request, authorization: str | None = Header(default=None)) -> Response:
    """Verify a Supabase token and set the session cookie. The only endpoint that reads a bearer token."""
    _check_auth_rate(request)
    token = jwt_utils.extract_bearer(authorization)
    if not token:
        raise ApiError(401, "unauthenticated", "Send the Supabase access token as Authorization: Bearer.")
    try:
        # Off the loop: the asymmetric path fetches JWKS over the network.
        claims = await jwt_utils.verify_supabase_off_loop(token)
    except jwt.PyJWTError as e:
        raise ApiError(401, "unauthenticated", f"Token rejected: {e}") from e

    # A RECOVERY TOKEN IS NOT A SIGN-IN. The reset link's token is an ordinary
    # access token, so without this the server would trade one for a seven-day
    # cookie — and anything that sees the link (a mail scanner following it, a
    # shared mailbox, a forward) could take the account without changing the
    # password, leaving the owner no signal at all. The client refuses to send
    # one; this is the half that holds when the client is curl.
    if _from_recovery_link(claims):
        raise ApiError(
            401,
            "unauthenticated",
            "That token came from a password-reset link. Set the password, then sign in with it.",
        )

    user_id, email = str(claims["sub"]), claims.get("email")
    await pool.execute(
        """
        INSERT INTO users (id, email, display_name) VALUES ($1, $2, $3)
        ON CONFLICT (id) DO UPDATE
           SET email        = COALESCE(EXCLUDED.email, users.email),
               display_name = COALESCE(EXCLUDED.display_name, users.display_name)
        """,
        _uuid(user_id, "user"),
        email,
        _display_name(claims),
    )
    await _ensure_home_session(user_id)

    cookie, jti, expires = jwt_utils.mint_session(user_id, email)
    await pool.execute(
        "INSERT INTO auth_sessions (jti, user_id, expires_at) VALUES ($1, $2, $3)",
        _uuid(jti, "session"),
        _uuid(user_id, "user"),
        expires,
    )

    out = Response(status_code=204)
    out.set_cookie(
        key=str(_cfg("auth.cookie_name", "arkos_session")),
        value=cookie,
        max_age=int(_cfg("auth.session_ttl_s", 604800)),
        httponly=True,
        secure=bool(_cfg("auth.cookie_secure", True)),
        samesite=str(_cfg("auth.cookie_samesite", "lax")),
        path="/",
    )
    return out


# How Supabase names the recovery/OTP methods in `amr`. A DENYLIST, not an
# allowlist: an unrecognised method must not lock anyone out of signing in.
# Magic link is on it because this app does not use one — if that changes, the
# flow needs its own decision rather than inheriting this refusal.
_RECOVERY_METHODS = {"recovery", "otp", "magiclink", "email_otp", "email_change"}


def _from_recovery_link(claims: dict[str, Any]) -> bool:
    """Whether this token was minted by following an emailed link."""
    amr = claims.get("amr")
    if not isinstance(amr, list):
        return False
    for entry in amr:
        method = entry.get("method") if isinstance(entry, dict) else entry
        if isinstance(method, str) and method.lower() in _RECOVERY_METHODS:
            return True
    return False


def _display_name(claims: dict[str, Any]) -> str | None:
    """What arkos should call this person, from the SIGNED token only.

    `name` is what our sign-up form writes, `full_name` what Google's OIDC
    profile carries. None rather than a fallback: the column is COALESCEd.
    """
    meta = claims.get("user_metadata")
    if not isinstance(meta, dict):
        return None
    for key in ("name", "full_name"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            # Capped because it is rendered.
            return value.strip()[:120]
    return None


async def _ensure_home_session(user_id: str) -> str:
    """Give a user their standing chat, once.

    An ordinary attended session with NO PROJECT, guarded by
    `home_session_id IS NULL` so a second login never makes a second one.
    """
    existing = await pool.fetchval("SELECT home_session_id FROM users WHERE id = $1", _uuid(user_id, "user"))
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
    # other as an ordinary empty one.
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

    The anon key identifies the project and authorizes nothing on its own; it is
    served rather than baked in because it differs per deployment.
    """
    if _proxy_supabase() and _origin:
        # The browser talks to Supabase through `/auth/v1` on this origin.
        return {"supabase_url": _origin, "anon_key": _anon_key()}
    return {"supabase_url": blobs.project_url() or "", "anon_key": _anon_key()}


def _proxy_supabase() -> bool:
    """Whether `/auth/v1/*` on this origin forwards to Supabase Auth.

    For a Supabase the BROWSER cannot reach — a self-hosted one bound to the
    server's loopback, behind a tunnel that carries only this app's port. Off by
    default: proxied, every sign-in reaches GoTrue from this server's address, so
    its per-IP rate limits would be shared by every user.
    """
    return bool(_cfg("auth.proxy_supabase", False))


# What supabase-js sends that GoTrue reads. Nothing else crosses, so our own
# session cookie never reaches Supabase.
_GOTRUE_REQUEST_HEADERS = (
    "apikey",
    "authorization",
    "content-type",
    "accept",
    "x-client-info",
    "x-supabase-api-version",
)


@app.api_route("/auth/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], include_in_schema=False)
async def gotrue_proxy(path: str, request: Request) -> Response:
    """Forward a supabase-js auth call to Supabase Auth, when `auth.proxy_supabase` is on."""
    upstream = blobs.project_url()
    if not _proxy_supabase() or not upstream:
        raise ApiError(404, "not_found", "Supabase Auth is not proxied by this server.")
    if request.method != "GET":
        _check_origin(request)
    headers = {name: request.headers[name] for name in _GOTRUE_REQUEST_HEADERS if name in request.headers}
    try:
        answer = await blobs.http_client().request(
            request.method,
            f"{upstream}/auth/v1/{path}",
            params=request.query_params,
            content=await request.body(),
            headers=headers,
        )
    except httpx.HTTPError as e:
        logger.warning("Supabase Auth unreachable at %s: %s", upstream, e)
        raise ApiError(502, "auth_unreachable", "Sign-in is unavailable: Supabase Auth did not answer.") from e
    kept = {"content-type": answer.headers.get("content-type", "application/json")}
    return Response(content=answer.content, status_code=answer.status_code, headers=kept)


@app.delete("/auth/session", status_code=204)
async def delete_auth_session(request: Request) -> Response:
    """Sign out THIS session: drop its row, then clear the cookie.

    Dropping the row is what makes it a sign-out rather than a suggestion — the
    cookie is self-signed, so a copy taken before this call would otherwise keep
    working for the rest of its seven days.
    """
    cookie = request.cookies.get(str(_cfg("auth.cookie_name", "arkos_session")))
    if not cookie:
        logger.info("sign-out with no session cookie; nothing to revoke")
    else:
        try:
            claims = jwt_utils.read_session(cookie)
            gone = await pool.execute("DELETE FROM auth_sessions WHERE jti = $1", _uuid(claims["jti"], "session"))
            logger.info("sign-out revoked jti %s: %s", claims.get("jti"), gone)
        except (jwt.PyJWTError, ApiError, KeyError):
            # Clearing the cookie is still the right answer, but a revoke that
            # did not happen must not be silent: that is the whole failure this
            # endpoint exists to prevent.
            logger.warning("sign-out could not revoke its session row", exc_info=True)
    out = Response(status_code=204)
    out.delete_cookie(key=str(_cfg("auth.cookie_name", "arkos_session")), path="/")
    return out


@app.post("/auth/sessions/revoke", status_code=204)
async def revoke_all_sessions(authorization: str | None = Header(default=None)) -> Response:
    """Sign this user out EVERYWHERE. Takes a Supabase bearer, not our cookie.

    This is what a password change calls, and it deliberately accepts a token
    `POST /auth/session` refuses — a recovery token included. Proving you hold a
    valid token for an account is enough to END its sessions: revocation only
    ever takes access away, so the failure mode of being too permissive here is
    an unwanted sign-out, while being too strict leaves a stolen cookie alive
    through the one action taken to stop it.
    """
    token = jwt_utils.extract_bearer(authorization)
    if not token:
        raise ApiError(401, "unauthenticated", "Send the Supabase access token as Authorization: Bearer.")
    try:
        claims = await jwt_utils.verify_supabase_off_loop(token)
    except jwt.PyJWTError as e:
        raise ApiError(401, "unauthenticated", f"Token rejected: {e}") from e

    gone = await pool.execute("DELETE FROM auth_sessions WHERE user_id = $1", _uuid(str(claims["sub"]), "user"))
    logger.info("revoked sessions for %s: %s", claims["sub"], gone)
    return Response(status_code=204)


@app.get("/auth/me")
async def auth_me(user_id: str = CurrentUser) -> dict[str, Any]:
    """Who is calling, and which session the app opens for them.

    `home_session_id` is null only for a user whose home session was deleted;
    the next sign-in makes another.
    """
    row = await pool.fetchrow(
        "SELECT id, email, display_name, home_session_id FROM users WHERE id = $1",
        _uuid(user_id, "user"),
    )
    if row is None:
        raise ApiError(401, "unauthenticated", "That user no longer exists.")
    return {
        "user_id": str(row["id"]),
        "email": row["email"],
        # Null when no name is known; the caller falls back to the email.
        "display_name": row["display_name"],
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

    `claims` is `[{folder, subpath?, mode?}]`; absent, it is a write claim on
    every folder the project links. The set is fixed for the session's life.
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
        # No project asked for means new work: it gets a project, and the
        # project gets a folder to keep the work in.
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
        items = [todo_item(step) for step in steps]
        await _append(session_id, TodoEvent(items=items))

    await runner.start(session_id)
    return {"session_id": session_id, "project_id": str(project_id)}


@app.get("/sessions")
async def list_sessions(status: str | None = None, user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """The user's sessions across every project, newest activity first."""
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
    # Read ONCE: `folders` below is a projection of these claims, and two reads
    # can disagree with each other.
    claims = await _claims_of(session_id)
    return {
        **_session_core(row),
        "project_id": str(row["project_id"]) if row["project_id"] else None,
        # The project's LABEL, so the header reads the same however it was opened.
        "project_title": await pool.fetchval(
            "SELECT title FROM projects WHERE id = $1", _uuid(row["project_id"], "project")
        )
        if row["project_id"]
        else None,
        # The FOLDERS this session writes, in link order — where the work lands.
        # A project links folders rather than owning one, so there is no single
        # directory to name.
        "folders": [claim["folder"] for claim in claims],
        # The session's newest plan, exact: `recent_events` is capped, so
        # counting `propose_plan` calls in it drifts on a long session.
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
            "status_rollup": _rollup(r),
            "sessions": r["sessions"],
        }
        for r in rows
    ]


@app.post("/projects", status_code=201)
async def create_project(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Make a project deliberately, rather than as a side effect of starting a session.

    `folders` LINKS folders that already exist in the caller's store, by name;
    with none given, a folder named after the project is made and linked.
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

    # Sentinels excluded, as in `store.folders`: the file that keeps an empty
    # folder alive is not content.
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

    The AGENT sees it from the NEXT session: claims are fixed for a session's
    life (`workspace.claims_for`). Linking twice is the same link.
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
    """Rename a project. The title is a LABEL and nothing durable is keyed by it."""
    await _owned_project(project_id, user_id)
    title = str(body.get("title") or "").strip()
    if not title:
        raise ApiError(400, "invalid_request", "A project needs a name.")
    row = await pool.fetchrow(
        "UPDATE projects SET title = $2, updated_at = now() WHERE id = $1 RETURNING id, title, updated_at",
        _uuid(project_id, "project"),
        title,
    )
    return {
        "id": str(row["id"]),
        "title": row["title"],
        # The links, so a surface can show where the work actually lands.
        "folders": await _folders_of(project_id),
        "updated_at": row["updated_at"].isoformat(),
    }


@app.get("/projects/{project_id}/sessions")
async def list_project_sessions(project_id: str, user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """The project's sessions, most recently active first."""
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

    One query at three scopes: no filter is the whole account, `project_id` that
    project, `session_id` one window. Only a `plan` row carries `version`.
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
            # Only a `call` carries these: the tool that runs if it is approved.
            "tool_name": r["tool_name"],
            "tool_args": json.loads(r["tool_args"]) if isinstance(r["tool_args"], str) else r["tool_args"],
            "created_at": r["created_at"].isoformat(),
            **(await _plan_context(str(r["session_id"])) if r["kind"] == "plan" else {}),
        }
        for r in rows
    ]


async def _plan_context(session_id: str) -> dict[str, Any]:
    """Which version of this session's plan the open row is.

    A plan's version IS its position in the session's history; there is no
    counter column.
    """
    history = await approvals.plan_history(session_id)
    return {"version": len(history)} if history else {}


async def _latest_plan(session_id: str) -> dict[str, Any] | None:
    """The session's newest plan and what became of it, or None if it has none.

    `answer` is the row's verbatim decision: `approve`, `decline`, `superseded`,
    or the feedback that was sent. `steps` comes from the snapshot, not the
    approval the window saw, so the todo seed survives a reload.
    """
    history = await approvals.plan_history(session_id)
    if not history:
        return None
    newest = history[-1]
    args = newest.tool_args or {}
    steps = args.get("steps")
    return {
        "approval_id": newest.id,
        "version": len(history),
        "goal": args.get("goal"),
        "answer": newest.answer,
        "steps": [str(step) for step in steps] if isinstance(steps, list) else [],
    }


# --- the store ------------------------------------------------------------------
#
# ONE flat namespace per user, and a folder is a top-level segment of it. The
# project-scoped route below is a VIEW of it, narrowed to what one project links.


@app.get("/folders")
async def list_folders(user_id: str = CurrentUser) -> list[dict[str, Any]]:
    """Every folder in the caller's store, with how many files are under it.

    There is no folders table: this is the first segment of every path the user
    has, grouped, so a folder appears the moment a file lands under it.
    """
    return [{"name": f.name, "files": f.files} for f in await store.folders(user_id)]


@app.post("/folders", status_code=201)
async def create_folder(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Make a folder durable the moment it is named.

    A folder is not a row but a path segment, so what lands is a zero-byte
    sentinel inside it.
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

    Text is returned decoded; anything that is not UTF-8 comes back `binary`.
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

    A running session whose claim covers the path is written through and reads it
    the same turn. The path must name a folder: the folder is its first segment.
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
    """Move a file or a whole subtree inside the store, in one transaction.

    Refused `409 folder_busy` while a running session holds the write lease on
    either end. Moving a FOLDER is refused — that is a rename, with its own route.
    """
    try:
        src = store.safe_path(str(body.get("from") or ""))
        dst = store.safe_path(str(body.get("to") or ""))
    except ValueError as e:
        raise ApiError(400, "invalid_request", str(e)) from e

    for folder in {store.folder_of(src), store.folder_of(dst)}:
        await _folder_is_free(user_id, folder)

    try:
        moves = await store.move_path(user_id, src, dst)
    except store.MissingPath as e:
        raise ApiError(404, "not_found", str(e)) from e
    except store.StoreError as e:
        raise ApiError(409, "move_refused", str(e)) from e

    if not moves:
        return {"from": src, "to": dst, "moved": []}

    for folder in {store.folder_of(src), store.folder_of(dst)}:
        await _touch_linked_projects(user_id, folder)
    return {
        "from": src,
        "to": dst,
        "moved": [{"from": was, "to": now} for was, now in moves],
    }


@app.post("/files/rename")
async def rename_file(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Rename anything in the store: a file, a directory, or a top-level folder.

    `name` is a name; a `/` in it is refused. A top-level rename carries the
    project links and claims along in one transaction, and is refused
    `409 folder_busy` while a live box has that folder materialized.
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

    # Both ends: the folder it is in, and — for a top-level rename — the name it
    # is becoming, which nothing may be holding either.
    for folder in {store.folder_of(path), store.folder_of(destination)}:
        await _folder_is_free(user_id, folder)

    try:
        moves = await store.rename_path(user_id, path, name)
    except store.MissingPath as e:
        raise ApiError(404, "not_found", str(e)) from e
    except store.StoreError as e:
        raise ApiError(409, "already_exists", str(e)) from e

    if not moves:
        return {"from": path, "to": destination, "moved": []}

    for folder in {store.folder_of(path), store.folder_of(destination)}:
        await _touch_linked_projects(user_id, folder)
    return {
        "from": path,
        "to": destination,
        "moved": [{"from": was, "to": now} for was, now in moves],
    }


async def _folder_is_free(user_id: str, folder: str) -> str | None:
    """Refuse a destructive change to a folder a running session is writing.

    A session holds `folder:{user}:{name}` for as long as it writes that folder.
    Read claims take no lease and need none: their flush discards.

    Raises:
        ApiError: 409 folder_busy, naming the folder.
    """
    holder = await leases.holder(f"folder:{user_id}:{folder}")
    if holder is not None:
        raise ApiError(
            409,
            "folder_busy",
            f"{folder}/ is in use by a running session. Stop it and try again after.",
        )
    return holder


@app.delete("/files")
async def delete_file(body: dict[str, Any] = JsonBody, user_id: str = CurrentUser) -> dict[str, Any]:
    """Delete a file or a whole subtree, and hand back the way to take it back.

    The rows go; the content-addressed BLOBS do not, so undo is exact. A delete
    that empties a folder takes the folder and its project links into the batch.
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

    `batch` names the gesture. `409` when something has since been put at one of
    those paths.
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

    A VIEW of the store, not a tree of its own: `files` rows at store paths.
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

    A running session picks the event up at its next hop, so the 202 carries no
    reply.
    """
    text = str(body.get("text") or "").strip()
    if not text:
        raise ApiError(400, "invalid_request", "An empty message says nothing.")
    row = await _owned_session(session_id, user_id)
    if row["status"] == "awaiting_approval":
        return await _answer_by_message(session_id, text)

    if not runner.is_running(session_id):
        # Before the append, so no user event lands between a call left open by
        # a dead run and its result.
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

    Prose answers a question; a `call` takes exactly `approve`/`decline` and
    appends nothing; a `plan` also takes a reply, which asks for the next one.
    Approving a plan is the only place a session's mode flips to unattended.
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

    if approval.is_goal:
        return await _answer_goal(approval, text)

    if approval.is_plan:
        return await _answer_plan(approval, text, user_id)

    # The UPDATE matches on answered_at IS NULL, so two people answering at once
    # produce one wake.
    answered = await approvals.answer(approval_id, text)
    if answered is None:
        raise ApiError(409, "already_answered", "That question has already been answered.")

    if not answered.gated_call:
        await _append(answered.session_id, UserEvent(text=text, source="human"))
    started = await runner.start(answered.session_id, reason="answered")
    return {"accepted": True, "session_id": answered.session_id, "started": started}


async def _answer_goal(approval: approvals.Approval, text: str) -> dict[str, Any]:
    """Answer the autopilot's opening question, or cancel at it.

    The decline word is the ✕ ON the question: the park closes to `idle` with
    nothing drafted and nothing run. Anything else IS the goal.
    """
    if text.strip().lower() == approvals.DECLINE:
        if await approvals.answer(approval.id, approvals.DECLINE) is None:
            raise ApiError(409, "already_answered", "That question has already been answered.")
        await lifecycle.transition(approval.session_id, "awaiting_approval", "idle", "goal_cancelled")
        return {"accepted": True, "session_id": approval.session_id, "started": False, "mode": "attended"}

    if await approvals.answer(approval.id, text) is None:
        raise ApiError(409, "already_answered", "That question has already been answered.")
    return await _plan_from_goal(approval.session_id, text)


async def _plan_from_goal(session_id: str, goal: str) -> dict[str, Any]:
    """Draft a plan from the goal they just gave: their words, then the handoff behind them.

    Everything after this point is the flow the button used to start with — an
    ordinary ATTENDED turn whose job is to call `propose_plan`.
    """
    await _append(session_id, UserEvent(text=goal, source="human"))
    handoff = prompts.plan_handoff(await runner.read_plan(session_id))
    await _append(session_id, UserEvent(text=handoff, source="system"))
    started = await runner.start(session_id, reason="plan_requested")
    return {"accepted": True, "session_id": session_id, "started": started, "mode": "attended"}


async def _answer_plan(approval: approvals.Approval, text: str, user_id: str) -> dict[str, Any]:
    """Approve, decline or workshop a proposed plan.

    The quota is checked BEFORE the row is answered, so a 429 leaves the plan
    intact.
    """
    verdict = text.strip().lower()
    args = approval.tool_args or {}
    row = await _owned_session(approval.session_id, user_id)
    decision = verdict in (approvals.APPROVE, approvals.DECLINE)
    # The two words are stored NORMALIZED, so the consent row says what was
    # decided rather than how it was typed. Feedback is stored as written.
    recorded = verdict if decision else text

    if verdict == approvals.APPROVE:
        if await runner.plan_folder(approval.session_id) is None:
            # The prompt promises the run a `plan.md` in its first linked
            # folder. Checked BEFORE the row is answered, like the quota.
            raise ApiError(409, "no_folder", "A plan is saved in a folder, and this session was given none.")
        if row["mode"] != "unattended":
            # The only point at which a user's unattended load grows; an
            # already-unattended session is exempt because it grows nothing.
            await _check_unattended_quota(user_id)

    answered = await approvals.answer(approval.id, recorded)
    if answered is None:
        raise ApiError(409, "already_answered", "That plan has already been answered.")

    if verdict == approvals.APPROVE:
        version = len(await approvals.plan_history(answered.session_id))
        # Written before the run starts, so the first materialize carries it
        # into the box.
        await runner.save_plan(answered.session_id, args, version)
        # Mode and status move in ONE conditional UPDATE, inside `start`.
        started = await runner.start(answered.session_id, mode="unattended", reason="plan_approved")
        if not started:
            # Reopen rather than leave a plan stamped approved that nothing ran
            # and nothing can approve again.
            logger.warning("session %s: an approved plan could not start it", answered.session_id)
            await approvals.reopen(answered.id)
            raise ApiError(409, "not_idle", "The session moved before the plan could start it.")
        return {"accepted": True, "session_id": answered.session_id, "started": True, "mode": "unattended"}

    if verdict == approvals.DECLINE:
        await lifecycle.transition(answered.session_id, "awaiting_approval", "idle", "plan_declined")
        return {"accepted": True, "session_id": answered.session_id, "started": False, "mode": "attended"}

    # A reply lands as the human's own turn, followed by the instruction that
    # makes the next turn produce a PLAN rather than a paragraph.
    await _append(answered.session_id, UserEvent(text=text, source="human"))
    await _append(answered.session_id, UserEvent(text=prompts.plan_reply(), source="system"))
    started = await runner.start(answered.session_id, reason="plan_reply")
    return {"accepted": True, "session_id": answered.session_id, "started": started, "mode": "attended"}


@app.post("/sessions/{session_id}/approve", status_code=202)
async def approve_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Ask the human what this run is for. It drafts nothing and runs no turn.

    The button's whole job is the question: arkos asks for the goal and the
    session PARKS on it, whatever the transcript above says. The answer is the
    goal, and `_answer_goal` is where the plan turn starts.
    """
    row = await _owned_session(session_id, user_id)
    if row["mode"] == "unattended":
        raise ApiError(409, "already_unattended", "This session is already running unattended.")
    # A TERMINAL session is a legal starting point: pressing this on a cancelled
    # run is how a continuation gets drafted.
    if row["status"] not in ("idle", "pending") and row["status"] not in lifecycle.TERMINAL:
        raise ApiError(409, "not_idle", f"A session in {row['status']!r} cannot be handed over.")

    # The status is claimed FIRST: that one conditional UPDATE is what makes two
    # presses one park.
    if await lifecycle.transition(session_id, row["status"], "awaiting_approval", "goal_requested") is None:
        raise ApiError(409, "not_idle", "The session moved before it could be asked.")
    try:
        # Arkos's own words, so the question outlives being answered and the model
        # folds the reply as an answer to it.
        await _append(session_id, ContentEvent(text=prompts.GOAL_QUESTION))
        await approvals.create(
            session_id,
            approvals.GOAL_CALL_ID,
            "ask",
            prompts.GOAL_QUESTION,
            tool_name=approvals.GOAL,
        )
    except Exception:
        # A park with nothing to answer is worse than no park: hand the session back.
        await lifecycle.transition(session_id, "awaiting_approval", "idle", "goal_failed")
        raise
    return {"accepted": True, "started": False, "parked": True, "mode": "attended"}


@app.post("/sessions/{session_id}/stop", status_code=202)
async def stop_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Hold a running turn, without ending it.

    The same teardown as cancel, landing differently: `done{stopped}`,
    `running -> idle`, the mode KEPT, the box hibernated rather than reaped.
    """
    row = await _owned_session(session_id, user_id)
    if row["status"] != "running":
        raise ApiError(409, "not_running", f"A session in {row['status']!r} is not running.")
    return {"accepted": True, "stopped": await runner.stop(session_id)}


@app.post("/sessions/{session_id}/resume", status_code=202)
async def resume_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Pick a stopped run back up, with nothing added.

    A plain start: the stop kept the mode, so an idle session that is still
    unattended resumes UNATTENDED, from its plan.
    """
    row = await _owned_session(session_id, user_id)
    if row["status"] != "idle":
        raise ApiError(409, "not_idle", f"A session in {row['status']!r} is not waiting to resume.")
    started = await runner.start(session_id, reason="resumed")
    if not started:
        raise ApiError(409, "not_idle", "The session moved before it could be resumed.")
    return {"accepted": True, "started": True, "mode": row["mode"]}


@app.post("/sessions/{session_id}/cancel", status_code=202)
async def cancel_session(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """End a run for good, from running or from a stop.

    With no live turn the terminal is written directly: `done{cancelled}`, and
    the mode handed back to attended, which is what spends the plan.
    """
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


@app.get("/attention/stream")
async def attention_stream(user_id: str = CurrentUser) -> StreamingResponse:
    """Nudge this human whenever their waiting list moves. One per sign-in.

    A frame carries no approval row: it means "read `/attention` again". Hence
    no `Last-Event-ID` and no replay — a missed nudge costs one stale list.
    """
    return StreamingResponse(
        _attention_frames(user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _attention_frames(user_id: str) -> AsyncIterator[str]:
    """Yield a frame per signal until the client disconnects."""
    keepalive = float(_cfg("harness.sse_keepalive_s", 15))
    async with attention_channel.subscribe(user_id) as queue:
        # The client fetches the list on connect, so this first frame is what
        # makes "subscribed" and "current" the same moment.
        yield 'event: attention\ndata: {"reason":"open"}\n\n'
        while True:
            try:
                signal = await asyncio.wait_for(queue.get(), timeout=keepalive)
            except TimeoutError:
                # Proxies and EventSource drop a stream that stays silent.
                yield ": keepalive\n\n"
                continue
            if signal is CLOSED:
                # A clean end-of-stream; the client reconnects on its own and
                # its opening frame refetches.
                return
            payload = json.dumps({"reason": signal.reason, "session_id": signal.session_id})
            yield f"event: attention\ndata: {payload}\n\n"


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
                if item is CLOSED:
                    # The client reconnects with Last-Event-ID and resumes from
                    # the log.
                    return

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
        yield (
            "event: error\ndata: "
            + json.dumps({"code": "stream_failed", "message": f"{type(e).__name__}: {e}", "retryable": True})
            + "\n\n"
        )


async def _backlog(session_id: str, after_seq: int) -> AsyncIterator[slog.StoredEvent]:
    """Yield every event after `after_seq`, a page at a time until the log runs out.

    One page is not the backlog, and the caller's `sent` only moves forward:
    anything skipped here cannot be recovered on this connection.
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


# The keys the client strips before flattening `payload` up a level (`asEvent`
# in frontend/api.jsx): a payload field named one of these is silently
# overwritten by the envelope, so `test_events` proves no event can carry one.
ENVELOPE_KEYS = frozenset(("seq", "ts", "kind", "version", "payload"))


def _wire(stored: slog.StoredEvent) -> dict[str, Any]:
    """Render one stored event for the wire, adding the seq and ts columns."""
    row = stored.event.to_row()
    return {"seq": stored.seq, "ts": stored.ts.isoformat(), **row}


# --- MCP connections ------------------------------------------------------------


@app.get("/sessions/{session_id}/browser/frames")
async def browser_frames(session_id: str, user_id: str = CurrentUser) -> StreamingResponse:
    """Watch what the browser is looking at, while it looks.

    A side-channel, not the event stream: frames are never appended, never
    replayed and carry no seq. Keyed by (user, session), ownership-checked.
    """
    await _owned_session(session_id, user_id)
    return StreamingResponse(
        _frame_stream(user_id, session_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _frame_stream(user_id: str, session_id: str) -> AsyncIterator[str]:
    """Yield frames until the viewer goes away, with keepalives between pictures."""
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
    """List every configured connector and this user's standing with it.

    Status comes from Composio, not our rows, and a Composio account is per
    TOOLKIT. There is no `scopes` (a managed auth config fixes them) and
    `shares_with` is always empty; it ships so the panel reads one shape.
    """
    client = hands.connectors()
    if client is None:
        return []
    return await client.connections(user_id)


@app.get("/connections/done")
async def connection_done(request: Request, user_id: str = CurrentUser) -> Response:
    """Where Composio lands the browser when consent finishes.

    The return leg carries `status` and `connected_account_id` as query params;
    the toolkit is NOT in the url and is looked up from the account id. Identity
    is the session cookie, so a forged account id settles nothing.
    """
    status = str(request.query_params.get("status") or "")
    account_id = str(request.query_params.get("connected_account_id") or "")
    logger.info("composio consent callback: status=%s account=%s", status, bool(account_id))

    settled: str | None = None
    if status.lower() == "success" and account_id:
        client = hands.connectors()
        if client is not None:
            try:
                settled = await client.reconcile(user_id, account_id)
            except ComposioError as e:
                logger.warning("could not reconcile %s: %s", account_id, e)

    return HTMLResponse(_CLOSE_POPUP.replace("__STATUS__", "connected" if settled else "not connected"))


# The opener re-reads its rows on this postMessage rather than on a timer.
_CLOSE_POPUP = """<!doctype html><meta charset="utf-8"><title>Connected</title>
<body style="font:14px system-ui;padding:2rem;color:#333">
<p>__STATUS__. You can close this window.</p>
<script>
  try { window.opener && window.opener.postMessage({source:"arkos", kind:"connection"}, "*"); } catch (e) {}
  setTimeout(function () { window.close(); }, 400);
</script>
</body>"""


@app.post("/connections/{server}/connect")
async def connect_server(server: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Mint the consent link for one service and record the pending state.

    Nothing is connected here. Already connected, it mints nothing and answers
    `status: connected` with a null `setup_url`; otherwise every press mints a
    FRESH link rather than handing back the unfinished one.
    """
    client = _require_connectors()
    _known_server(client, server)
    try:
        return await client.connect(user_id, server)
    except ComposioError as e:
        raise ApiError(502, "upstream_error", f"Composio refused: {e}", retryable=True) from e


@app.delete("/connections/{server}")
async def disconnect_server(server: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """Revoke one service and say what went with it.

    A Composio connected account is per TOOLKIT, so this takes exactly the one
    service named; the list shape is kept so the panel need not know the backend.
    """
    client = _require_connectors()
    _known_server(client, server)
    try:
        disconnected = await client.disconnect(user_id, server)
    except ComposioError as e:
        raise ApiError(502, "upstream_error", f"Composio refused: {e}", retryable=True) from e
    return {"server": server, "disconnected": disconnected}


def _require_connectors() -> Any:
    client = hands.connectors()
    if client is None:
        raise ApiError(503, "unavailable", "MCP is not configured on this server.")
    return client


def _known_server(client: Any, server: str) -> None:
    """Refuse a prefix that is not one of the configured connectors.

    Google Search fails this on purpose: it is one of ours and has no grant.
    """
    if not client.is_connector(server):
        raise ApiError(404, "not_found", f"No server {server!r} is configured.")


# --- what this session may reach ------------------------------------------------


@app.get("/sessions/{session_id}/tools")
async def session_tools_state(session_id: str, user_id: str = CurrentUser) -> dict[str, Any]:
    """The tool budget for one session: the meter's numbers and a row per connected server.

    `budget` is `llm.max_tools - ours`; `used` is the tool count of the servers
    this session has been given. Google Search is in `ours` and in no row.
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

    A toggle that would put the manifest over `llm.max_tools` is refused HERE,
    with the numbers in the message; the panel refuses it too, before the request.
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

    await session_tools.set_enabled(session_id, row["server"], wanted)

    # Recomputed locally rather than rebuilt: only one server's `enabled` moved,
    # and rebuilding re-ran the whole read on every click.
    servers = [{**r, "enabled": wanted if r["server"] == row["server"] else r["enabled"]} for r in document["servers"]]
    return {
        **document,
        "servers": servers,
        "used": sum(s["tool_count"] for s in servers if s["enabled"]),
    }


async def _tools_document(session_id: str, user_id: str, *, refresh: bool = False) -> dict[str, Any]:
    """Build the meter and the server rows from config, the connections and the toggles.

    `ours` must count what `registry.manifest` counts, which is not what
    `tool_module/tools/` holds: Google Search comes over the gateway and is ours.
    """
    # NOT refreshed by default: this runs on every toggle render and write, and
    # asking the vendor costs ~305ms plus a write transaction per click.
    client = hands.connectors()
    rows = await client.connections(user_id, refresh=refresh) if client is not None else []
    on = set(await session_tools.enabled_servers(session_id))

    ours = len(registry.local_tools())
    if client is not None:
        with contextlib.suppress(Exception):
            ours += len(await client.always(user_id))
    max_tools = int(_cfg("llm.max_tools", 128))
    budget = max(0, max_tools - ours)

    servers = [{**row, "enabled": row["server"] in on} for row in rows]
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

    Never boots anything: a box that has parked or been reaped reads as 404.
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

    Non-UTF-8 comes back `binary`, and a file past `sandbox.browse_max_bytes`
    comes back cut short with `truncated` set.
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
    """Mark a project as changed.

    Separate from `lifecycle.touch_project`, which takes a connection and a
    SESSION id because it runs inside the transaction that moves a session.
    """
    await pool.execute("UPDATE projects SET updated_at = now() WHERE id = $1", _uuid(project_id, "project"))


async def _touch_linked_projects(user_id: str, folder: str) -> None:
    """Mark every project that links this folder as changed.

    Nought, one or several: a file route writes to the store, and a project is
    whatever links the folder the write landed in.
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

    Picking the name and reserving it are ONE operation in the store, under a
    lock, so two projects created at once cannot be given the same name.
    """
    return await store.unique_folder(user_id, base)


def _session_core(row: Any) -> dict[str, Any]:
    """The fields every session projection carries, shaped once."""
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

    `slug` is only the default NAME for the folder the none-case makes, and is
    not uniquified here — `store.unique_folder` resolves collisions.
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
    than held in memory first. Zero bytes is a file like any other.
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

    The home session is excluded: it is created by first login, not asked for.
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


# What a composer message is refused with, per park kind. `ask` is absent on
# purpose: it is the one park a typed message legitimately answers.
_WAITING_ON = {"call": "a tool call", "approval": "an approval", "plan": "a plan"}


async def _answer_by_message(session_id: str, text: str) -> dict[str, Any]:
    """Route a composer message sent to a parked session.

    An `ask` is answered by the message and the session wakes. Consent is NOT:
    an `approval`, a gated `call` and a `plan` are answered only through
    `/approvals/{id}/respond`, where the thing being agreed to is on screen.
    """
    open_questions = await approvals.open_for(session_id)
    if open_questions and open_questions[0].kind in _WAITING_ON:
        waiting = _WAITING_ON[open_questions[0].kind]
        raise ApiError(
            409,
            "awaiting_approval",
            f"This session is waiting on {waiting}. Answer it there, where you can see what it does.",
        )

    if open_questions and open_questions[0].is_goal:
        # Typed in the composer instead of on the question, which is the same act:
        # the words are the goal, and a plan is drafted from them.
        if await approvals.answer(open_questions[0].id, text) is None:
            raise ApiError(409, "already_answered", "That question has already been answered.")
        return await _plan_from_goal(session_id, text)

    if open_questions:
        await approvals.answer(open_questions[0].id, text)
    await _append(session_id, UserEvent(text=text, source="human"))
    started = await runner.start(session_id, reason="answered")
    return {"accepted": True, "started": started}


async def _record_claims(session_id: str, project_id: Any, declared: Any, user_id: str) -> None:
    """Record which FOLDERS this session may touch, fixed for its life.

    Absent, it claims every folder its project links, all write. This runs once,
    at creation, and nothing rewrites it; a session with no project claims none.
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

    # `ord` is the order they were GIVEN, and an explicit column rather than a
    # timestamp because it must be stable: `plan.md` goes to the FIRST folder.
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
        "SELECT folder, subpath, mode FROM session_claims WHERE session_id = $1 ORDER BY ord, folder, subpath",
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

    `sb_publishable_...` is the current format, `SUPABASE_ANON_KEY` the legacy
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

    A malformed id answers `not_found` rather than a 400: "invalid UUID" tells a
    prober it guessed the shape right.
    """
    try:
        return as_uuid(value)
    except ValueError as e:
        raise ApiError(404, "not_found", f"No such {what}.") from e


# --- the app itself --------------------------------------------------------------

# Mounted last, so no API route can be shadowed by a file with the same name.
# Must share the API's origin: the session cookie is SameSite=Lax, which a
# cross-site `EventSource` does not carry.
_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


class _Frontend(StaticFiles):
    """StaticFiles that refuses to let index.html be cached.

    HTML is `no-store`; assets keep the ordinary validators, since `?v=N`
    cache-busts them. Keyed off the PATH: a 304 carries no content-type, so
    testing the response would skip revalidation — the one case that matters.
    """

    async def get_response(self, path: str, scope: Any) -> Response:
        response = await super().get_response(path, scope)
        if path in ("", ".", "index.html") or path.endswith(".html"):
            response.headers["Cache-Control"] = "no-store, must-revalidate"
        return response


if _FRONTEND.is_dir():
    # `html=True` serves index.html for any path the build has no file for,
    # which is what a client-routed page needs.
    app.mount("/app", _Frontend(directory=_FRONTEND, html=True), name="app")
else:  # pragma: no cover - only a broken checkout or a partial image
    logger.error("no frontend/ directory at %s; /app will 404", _FRONTEND)
