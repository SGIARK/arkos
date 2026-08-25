"""MCP, reached through one Arcade MCP Gateway.

Arcade holds the OAuth apps and the users' grants; this side holds an identity
and sends it on every request. Two headers do the work — `Authorization: Bearer
{api_key}` says which project, `Arcade-User-Id: {user}` says whose grant to use —
and the gateway resolves a call against that user's tokens. There is no
connection to create, nothing to PUT, and no credential of the user's on this
box.

WHAT A "SERVER" IS HERE. One gateway url serves every app, and the tools come
back FLAT in a single list, prefixed by app: `Gmail_ListEmails`,
`Linear_CreateIssue`. So a "server" is a PREFIX GROUP, not a url — that prefix
is the identity `user_connections` and `session_tools` are keyed by, and the
gateway url lives in config alone, because recreating the gateway changes the
url while the grants behind it are untouched.

THE TRANSPORT is Streamable HTTP MCP with sessions, measured live before any of
this was written (`docs/implementation_notes.md` § "Arcade gateway — live test
facts"). A bare `tools/list` is refused with `400 missing Mcp-Session-Id`: the
client must `initialize` first, read `Mcp-Session-Id` off the response HEADER,
send `notifications/initialized`, and carry that header on everything after. The
session is minted under the caller's `Arcade-User-Id`, so there is one per user
and it cannot be shared.

`tools/list` IS PAGINATED — 100 per page, with a `nextCursor` — and the loop is
not optional. Reading one page yields 100 of 169 tools and, worse, it yields
them silently: Notion and Outlook Mail simply do not appear, and every symptom
points at the gateway's app selection rather than at the reader. That is exactly
how the first live probe concluded those apps were missing. Page until no cursor
comes back.

CONSENT IS PANEL-FIRST, AND `authorize` IS THE STATUS READ. A tool called before
its app is authorized comes back as a SUCCESSFUL result carrying an
`authorization_url` and `llm_instructions` telling the model to show the link —
which is why the panel gets there first. It uses Arcade's own auth-initiation
call, `POST /v1/tools/authorize`, which mints a consent link without invoking
anything: `completed` means the user has granted this service's scopes,
`pending` carries the url the popup opens. `Arcade_ListApps` is NOT used for
this and the card's expectation that it could was wrong — measured, it reports
PROVIDERS (`arcade-google`) rather than services, so connecting Gmail would make
Google Calendar render connected while every Calendar call still challenged.
`authorize` is scope-aware, so it is per-service by construction.

No envelope guard is built for the case where a challenge reaches the model
anyway; that is a knowing decision recorded in Task 11.10, not an oversight.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

import aiohttp

from agent_module.loop import cap_view
from config_module.loader import cfg as _cfg
from tool_module import connections as conns
from tool_module.connections import CONNECTED, DISCONNECTED
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok

logger = logging.getLogger(__name__)

# Per-request half of the cap; the registry caps the whole call separately.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)

# The gateway's own meta-tools. Harness plumbing, never offered to the model:
# the model does not manage connections, the human does.
META_PREFIX = "Arcade"

# `tools/list` pages at 100. See `list_tools`.
_MAX_PAGES = 20


class ArcadeError(RuntimeError):
    """Raised for any non-auth Arcade API failure."""


def prefix_of(tool_name: str) -> str:
    """Return the app a flat gateway tool name belongs to: `Gmail_ListEmails` -> `Gmail`."""
    return tool_name.split("_", 1)[0]


@dataclass(slots=True)
class _McpSession:
    """One user's live gateway session; `id` is what `Mcp-Session-Id` carries."""

    id: str
    opened_at: float


class ArcadeClient:
    """The wire: the MCP gateway, and the Engine API the panel authorizes through.

    One `aiohttp.ClientSession` for the process, one MCP session per user. Both
    halves speak to `api.arcade.dev` with the same API key; they are different
    endpoints rather than different credentials.
    """

    def __init__(
        self,
        api_key: str,
        gateway_url: str,
        *,
        engine_url: str = "https://api.arcade.dev",
        protocol_version: str = "2025-11-25",
    ):
        if not api_key:
            raise ValueError("ArcadeClient requires an api_key (set ARCADE_API_KEY)")
        if not gateway_url:
            raise ValueError("ArcadeClient requires a gateway url (mcp.gateway_url)")
        self.api_key = api_key
        self.gateway_url = gateway_url.rstrip("/")
        self.engine_url = engine_url.rstrip("/")
        self.protocol_version = protocol_version
        self._http: aiohttp.ClientSession | None = None
        self._http_lock = asyncio.Lock()
        self._sessions: dict[str, _McpSession] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    # ---------- plumbing ----------

    async def http(self) -> aiohttp.ClientSession:
        """Return the process-wide HTTP session, creating it on first use."""
        if self._http is None or self._http.closed:
            async with self._http_lock:
                if self._http is None or self._http.closed:
                    self._http = aiohttp.ClientSession(timeout=_HTTP_TIMEOUT)
        return self._http

    async def close(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None
        self._sessions.clear()

    def _headers(self, user_id: str, *, mcp_session: str | None = None) -> dict[str, str]:
        """The two identity headers, plus the accepts the gateway demands.

        `Accept` must offer BOTH json and event-stream even though the gateway
        answered plain json when this was measured: offering only json is a 406
        from a Streamable HTTP server that decides to frame its reply.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Arcade-User-Id": user_id,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if mcp_session:
            headers["Mcp-Session-Id"] = mcp_session
        return headers

    # ---------- the MCP handshake ----------

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(user_id)
        if lock is None:
            lock = self._session_locks[user_id] = asyncio.Lock()
        return lock

    async def _session(self, user_id: str) -> str:
        """Return this user's `Mcp-Session-Id`, performing the handshake if needed."""
        live = self._sessions.get(user_id)
        if live is not None:
            return live.id
        async with self._lock_for(user_id):
            live = self._sessions.get(user_id)
            if live is not None:
                return live.id
            session_id = await self._handshake(user_id)
            self._sessions[user_id] = _McpSession(session_id, time.monotonic())
            return session_id

    async def _handshake(self, user_id: str) -> str:
        """`initialize` -> the session header -> `notifications/initialized`.

        The session id arrives as a response HEADER, not in the JSON-RPC result,
        which is the one detail that makes a from-scratch client work or not.
        """
        body = {
            "jsonrpc": "2.0",
            "id": uuid.uuid4().hex[:12],
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "arkos", "version": "1"},
            },
        }
        http = await self.http()
        async with http.post(self.gateway_url, json=body, headers=self._headers(user_id)) as resp:
            text = await resp.text()
            if resp.status == 401 or resp.status == 403:
                raise ArcadeError(f"the gateway refused our API key ({resp.status}): {text[:200]}")
            if resp.status >= 400:
                raise ArcadeError(f"initialize {resp.status}: {text[:300]}")
            session_id = resp.headers.get("Mcp-Session-Id")
        if not session_id:
            raise ArcadeError("initialize returned no Mcp-Session-Id header; every later call would be refused")

        # A notification carries no id and expects no result. It is not optional:
        # the server holds the session half-open until it arrives.
        notify = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        async with http.post(
            self.gateway_url, json=notify, headers=self._headers(user_id, mcp_session=session_id)
        ) as resp:
            if resp.status >= 400:
                logger.warning("notifications/initialized returned %s", resp.status)
        return session_id

    def forget_session(self, user_id: str) -> None:
        """Drop a user's MCP session, so the next call re-handshakes."""
        self._sessions.pop(user_id, None)

    # ---------- JSON-RPC ----------

    async def rpc(self, user_id: str, method: str, params: dict[str, Any] | None = None) -> Any:
        """POST one JSON-RPC request as this user and return its `result`.

        Retried ONCE, and only for a dead session. The gateway expires sessions
        on its own schedule, and a expired one is indistinguishable from a
        never-opened one at the call site: both are a 400 naming the header. Any
        other failure is the caller's to report.
        """
        for attempt in (1, 2):
            session_id = await self._session(user_id)
            body = {"jsonrpc": "2.0", "id": uuid.uuid4().hex[:12], "method": method, "params": params or {}}
            http = await self.http()
            async with http.post(
                self.gateway_url, json=body, headers=self._headers(user_id, mcp_session=session_id)
            ) as resp:
                text = await resp.text()
                status = resp.status

            if status in (400, 404) and _is_session_error(text) and attempt == 1:
                logger.info("gateway session for %s expired; re-handshaking", user_id)
                self.forget_session(user_id)
                continue
            if status == 401 or status == 403:
                raise ArcadeError(f"the gateway refused our API key ({status})")
            if status >= 400:
                raise ArcadeError(f"{method} {status}: {text[:300]}")

            data = _parse_rpc(text, method)
            if "error" in data:
                err = data["error"] or {}
                raise ArcadeError(f"{method} rpc error {err.get('code')}: {err.get('message')}")
            return data.get("result", {})
        raise ArcadeError(f"{method}: the gateway session could not be re-established")

    async def list_tools(self, user_id: str) -> list[dict[str, Any]]:
        """Page through `tools/list` until the gateway stops handing back a cursor.

        The page size is 100 and the roster is 169, so a single read returns a
        plausible-looking list that is missing two whole apps. `_MAX_PAGES` is a
        runaway guard, not a limit: a gateway that keeps handing back cursors
        forever is broken, and stopping loudly beats looping forever.
        """
        found: list[dict[str, Any]] = []
        cursor: str | None = None
        for page in range(_MAX_PAGES):
            params = {"cursor": cursor} if cursor else {}
            result = await self.rpc(user_id, "tools/list", params)
            if not isinstance(result, dict):
                break
            tools = result.get("tools")
            if isinstance(tools, list):
                found.extend(tools)
            cursor = result.get("nextCursor") or result.get("next_cursor")
            if not cursor:
                return found
            if page == _MAX_PAGES - 1:
                logger.error(
                    "tools/list still had a cursor after %d pages (%d tools); the roster is truncated",
                    _MAX_PAGES,
                    len(found),
                )
        return found

    async def call_tool(self, user_id: str, name: str, args: dict[str, Any]) -> Any:
        return await self.rpc(user_id, "tools/call", {"name": name, "arguments": args})

    # ---------- the Engine API ----------

    async def _engine(self, method: str, path: str, user_id: str, body: Any = None) -> Any:
        """One call to Arcade's own REST API, which is where authorization lives.

        Separate from the gateway because it is a different surface, not a
        different credential: same key, same host, endpoints the MCP protocol
        has no way to express.
        """
        url = f"{self.engine_url}{path}"
        headers = self._headers(user_id)
        http = await self.http()
        async with http.request(method, url, json=body, headers=headers) as resp:
            text = await resp.text()
            if resp.status == 401 or resp.status == 403:
                raise ArcadeError(f"Arcade refused our API key ({resp.status})")
            if resp.status >= 400:
                raise ArcadeError(f"{method} {path} -> {resp.status}: {text[:300]}")
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ArcadeError(f"{path}: response was not JSON: {text[:200]}") from e

    async def authorize(self, user_id: str, tool_name: str) -> dict[str, Any]:
        """Start authorization for one tool: `{status, url, id, provider_id, scopes}`.

        The dedicated auth-initiation call, and the only honest read of whether
        one SERVICE is connected. It mints a consent link without invoking
        anything, so asking is free: `completed` means this tool's scopes are
        already granted, `pending` carries the url to open. Measured to accept
        both spellings of a tool name, `Gmail.ListEmails` and `Gmail_ListEmails`,
        with identical answers.
        """
        return await self._engine("POST", "/v1/tools/authorize", user_id, {"tool_name": tool_name, "user_id": user_id})

    async def user_connections(self, user_id: str) -> list[dict[str, Any]]:
        """List this user's provider connections, which is where a revoke finds its id."""
        payload = await self._engine("GET", f"/v1/admin/user_connections?user_id={user_id}", user_id)
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("data") or []
            return items if isinstance(items, list) else []
        return payload if isinstance(payload, list) else []

    async def delete_connection(self, user_id: str, connection_id: str) -> None:
        """Revoke one provider connection at Arcade, so the next connect is a fresh grant."""
        await self._engine("DELETE", f"/v1/admin/user_connections/{connection_id}", user_id)


def _is_session_error(text: str) -> bool:
    """Whether a 4xx body is the gateway complaining about `Mcp-Session-Id`."""
    lowered = text.lower()
    return "mcp-session-id" in lowered or "session" in lowered and "id" in lowered


def _parse_rpc(text: str, method: str) -> dict[str, Any]:
    """Parse a response body as JSON, or as the SSE framing the accept header allows."""
    if not text.strip():
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for line in text.splitlines():
        if line.startswith("data: "):
            try:
                candidate = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and ("result" in candidate or "error" in candidate):
                return candidate
    raise ArcadeError(f"{method}: response was neither JSON nor SSE: {text[:200]}")




@dataclass(frozen=True, slots=True)
class ServerTools:
    """One app and the tools it is offering right now.

    `label` is the `mcp_servers:` config key, for logs only; `name` is what a
    human is shown; `server` is the Arcade prefix, and it is the identity
    everything durable is keyed by.
    """

    label: str
    name: str
    server: str
    specs: list[ToolSpec]


@dataclass(frozen=True, slots=True)
class Consent:
    """What `authorize` said about one service for one user.

    `provider_id` is Arcade's own account behind the service — `arcade-google`
    for Gmail, Calendar and Search alike — which is what makes revoking a shared
    act and is why the panel has to warn before it fires.
    """

    server: str
    status: str
    setup_url: str | None = None
    provider_id: str | None = None
    scopes: tuple[str, ...] = ()

    @property
    def connected(self) -> bool:
        return self.status == CONNECTED


class Arcade:
    """The MCP half: one gateway, the apps configured under `mcp_servers:`.

    A config key is a label for logs and display only. Identity is the Arcade
    prefix, which is the vendor's own name for the app and therefore the one
    thing here that neither a config edit nor a new gateway can change.
    """

    def __init__(self, servers: dict[str, dict[str, Any]], mcp_config: dict[str, Any]):
        self.servers = servers or {}
        self.client = ArcadeClient(
            api_key=mcp_config.get("api_key"),
            gateway_url=mcp_config.get("gateway_url"),
            engine_url=mcp_config.get("engine_url", "https://api.arcade.dev"),
            protocol_version=mcp_config.get("protocol_version", "2025-11-25"),
        )
        search = mcp_config.get("search") or {}
        # Google Search rides the same wire and is NOT a connector: SerpAPI-backed,
        # keyed to an app-level secret in Arcade's dashboard, no per-user OAuth. It
        # belongs to buddy, so it is always loaded, counted in `ours`, and appears
        # in neither the session toggles nor the settings panel.
        self.search_server: str | None = search.get("server") or None
        self._ttl_s = float(_cfg("tools.mcp_cache_ttl_s", 3600))
        # user_id -> (fetched_at, {prefix: [raw tool, ...]})
        self._tools: dict[str, tuple[float, dict[str, list[dict[str, Any]]]]] = {}
        self._tool_locks: dict[str, asyncio.Lock] = {}
        # Consent links, in memory only: they expire, and a stale one sends a
        # human to a dead page more confusingly than no link at all.
        self._setup_urls: dict[tuple[str, str], str] = {}

    # ---------- config ----------

    def _by_server(self, server: str) -> tuple[str, dict[str, Any]] | None:
        """Find the config entry for an Arcade prefix, or None if it is not ours to offer."""
        for label, spec in self.servers.items():
            if spec.get("server") == server:
                return label, spec
        return None

    def _display(self, server: str) -> str:
        found = self._by_server(server)
        return (found[1].get("name") or found[0]) if found else server

    def is_connector(self, server: str) -> bool:
        """Whether a prefix is one of the toggleable connectors."""
        return self._by_server(server) is not None

    @property
    def prefixes(self) -> list[str]:
        """Every configured connector's prefix, in config order."""
        return [spec["server"] for spec in self.servers.values() if spec.get("server")]

    # ---------- the gateway's tool list ----------

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        lock = self._tool_locks.get(user_id)
        if lock is None:
            lock = self._tool_locks[user_id] = asyncio.Lock()
        return lock

    async def tools_by_server(self, user_id: str, *, refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
        """Return the gateway's tools grouped by prefix, cached per user on the TTL.

        Cached PER USER rather than per process even though the gateway's
        selection is one selection: the list arrives over a session minted under
        this caller's identity, and one user's reading is not evidence about
        another's. A few copies of a small dict is not a cost worth a guess
        about how the gateway scopes its listing.
        """
        cached = self._tools.get(user_id)
        fresh = cached is not None and (time.monotonic() - cached[0]) < self._ttl_s
        if not refresh and fresh:
            return cached[1]

        async with self._lock_for(user_id):
            cached = self._tools.get(user_id)
            fresh = cached is not None and (time.monotonic() - cached[0]) < self._ttl_s
            if not refresh and fresh:
                return cached[1]
            try:
                raw = await self.client.list_tools(user_id)
            except (ArcadeError, aiohttp.ClientError, TimeoutError) as e:
                # The cached list stands, and the clock re-arms, so a gateway
                # that is down costs one timeout per TTL rather than one per
                # manifest build.
                if cached is not None:
                    self._tools[user_id] = (time.monotonic(), cached[1])
                    logger.warning("tools/list refresh failed for %s: %s", user_id, e)
                    return cached[1]
                raise
            grouped: dict[str, list[dict[str, Any]]] = {}
            for tool in raw:
                name = tool.get("name")
                if name:
                    grouped.setdefault(prefix_of(name), []).append(tool)
            self._tools[user_id] = (time.monotonic(), grouped)
            return grouped

    def invalidate(self, user_id: str) -> None:
        """Drop a user's cached tool list, so the next read is a fresh one."""
        self._tools.pop(user_id, None)

    # ---------- the manifest half ----------

    async def reach(self, user_id: str) -> list[ServerTools]:
        """Return every CONNECTED connector and the tools it currently offers, grouped.

        Grouped rather than flattened because the budget is spent and refused a
        SERVER at a time: half of Gmail in the manifest is a model that thinks it
        can send mail and finds out mid-task that it cannot. `registry.manifest`
        does the choosing; this only reports what is there.

        Connection state is read from the stored rows rather than from Arcade,
        because this runs on every turn and consent is a round trip per service.
        The panel is what refreshes those rows, and a call made against a grant
        that has since died comes back `auth_required`, which corrects them.
        """
        grouped = await self.tools_by_server(user_id)
        stored = await conns.load(user_id)
        out: list[ServerTools] = []
        for label, spec in self.servers.items():
            server = spec.get("server")
            row = stored.get(server)
            if not server or row is None or not row.connected:
                continue
            specs = [_to_spec(tool, spec.get("auto_approve")) for tool in grouped.get(server, [])]
            if specs:
                out.append(ServerTools(label=label, name=spec.get("name", label), server=server, specs=specs))
        return out

    async def always(self, user_id: str) -> list[ToolSpec]:
        """The gateway tools that are OURS: Google Search, and nothing else.

        Always in the manifest, counted in `ours`, absent from the toggles and
        the settings panel. It rides this wire because that is where the SerpAPI
        key is configured, not because it is a connector — there is no per-user
        grant to make, so there is nothing for a human to connect.
        """
        if not self.search_server:
            return []
        grouped = await self.tools_by_server(user_id)
        # Ours, so we assert the posture rather than defaulting to the
        # conservative one every remote tool gets: a search reads the web and
        # changes nothing, and gating it would put an approval card in front of
        # the human every time the agent looked something up.
        return [_to_spec(tool, auto_approve=True, readonly=True) for tool in grouped.get(self.search_server, [])]

    async def specs(self, user_id: str) -> list[ToolSpec]:
        """Every gateway tool this user could reach, named bare and ungated by any session."""
        connectors = [spec for server in await self.reach(user_id) for spec in server.specs]
        return connectors + await self.always(user_id)

    # ---------- the dispatch half ----------

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        """Run one gateway tool, with the `mcp_` prefix already stripped by the registry."""
        server = prefix_of(name)

        if server == META_PREFIX:
            # The gateway's meta-tools manage connections, which is the human's
            # job through the panel. They are never in the manifest, so reaching
            # one means the model invented the name.
            return fail("not_found", f"No tool named {name!r}.")

        if server != self.search_server and not self.is_connector(server):
            return fail("not_found", f"No MCP tool named {name!r} on any server you can reach.")

        if server != self.search_server:
            row = (await conns.load(ctx.user_id)).get(server)
            if row is None or not row.connected:
                return fail("auth_required", self._reconnect_message(server), retryable=False)

        try:
            result = await self.client.call_tool(ctx.user_id, name, args)
        except ArcadeError as e:
            return fail("upstream_error", f"{name} failed: {e}")
        except (aiohttp.ClientError, TimeoutError) as e:
            # aiohttp's total-timeout raises a bare TimeoutError, not a ClientError.
            return fail("upstream_error", f"{name} could not reach the gateway: {type(e).__name__}: {e}")

        return await _envelope(name, result, ctx)

    def _reconnect_message(self, server: str) -> str:
        """Tell the model an app needs authorizing, and where the human does that."""
        return (
            f"{self._display(server)} is not connected. The human has to authorize it "
            f"from the connections panel in Settings. Do not retry this tool."
        )

    # ---------- consent, which is also the status read ----------

    async def _consent_tool(self, user_id: str, server: str) -> str:
        """The tool whose scopes stand for the whole service.

        Named in config, because it is a MEASURED choice: it must ask for scopes
        that cover every tool of the app, or connecting from the panel grants a
        subset and half the service keeps challenging. `scripts/probe_arcade.py`
        authorizes every tool and reports which one covers the union. Falling
        back to whatever the gateway lists first keeps the panel working on a
        newly added app, and says so, rather than refusing to render.
        """
        found = self._by_server(server)
        configured = (found[1].get("consent_tool") if found else None) or None
        if configured:
            return str(configured)

        grouped = await self.tools_by_server(user_id)
        tools = grouped.get(server) or []
        if not tools:
            raise ArcadeError(f"the gateway offers no tools for {server!r}; check the gateway's app selection")
        # Passed through as the gateway spells it. The Engine's own examples name
        # tools `Gmail.ListEmails` while the gateway lists `Gmail_ListEmails`, and
        # `/v1/tools/authorize` was measured to accept both with identical
        # answers — so translating between them would be ceremony with a bug in it.
        chosen = str(tools[0]["name"])
        logger.warning(
            "%s has no consent_tool in config; falling back to %s, whose scopes may not cover the app",
            server,
            chosen,
        )
        return chosen

    async def consent(self, user_id: str, server: str) -> Consent:
        """Ask Arcade whether this user has granted one service, and get the link if not.

        One call answers both questions, which is why it is one call. Arcade
        mints the consent url in response to being asked, so there is no state
        to read first and no challenge to intercept — and because `authorize`
        is scope-aware it answers about the SERVICE, where `Arcade_ListApps`
        would only answer about the provider account behind it.
        """
        tool = await self._consent_tool(user_id, server)
        response = await self.client.authorize(user_id, tool)

        state = str(response.get("status") or "").lower()
        url = response.get("url") or (response.get("authorization") or {}).get("url")
        provider = response.get("provider_id")
        scopes = tuple(response.get("scopes") or ())

        if state == "completed" or (not url and state in ("connected", "authorized")):
            return Consent(server, CONNECTED, None, provider, scopes)
        if not url:
            raise ArcadeError(f"authorize({tool}) returned neither a url nor a completed status: {response}")
        return Consent(server, conns.PENDING, url, provider, scopes)

    async def refresh_status(self, user_id: str) -> dict[str, Consent]:
        """Read every connector's consent state at once, and store what came back.

        Concurrent because they are independent questions and the panel waits on
        all of them; the rows are written in one transaction because a partial
        write would show one service's reading beside another's from minutes ago
        with nothing to say which is which.
        """
        prefixes = self.prefixes
        results = await asyncio.gather(
            *(self.consent(user_id, server) for server in prefixes), return_exceptions=True
        )

        found: dict[str, Consent] = {}
        for server, result in zip(prefixes, results, strict=True):
            if isinstance(result, Consent):
                found[server] = result
                if result.setup_url:
                    self._setup_urls[(user_id, server)] = result.setup_url
                else:
                    self._setup_urls.pop((user_id, server), None)
            else:
                logger.warning("could not read consent for %s: %s", server, result)

        try:
            await conns.sync(user_id, {s: c.status for s, c in found.items()})
        except Exception:
            logger.exception("could not record connection state for %s", user_id)
        return found

    # ---------- the settings panel ----------

    async def connections(self, user_id: str) -> list[dict[str, Any]]:
        """Return one row per configured connector and this user's standing with it.

        Google Search is absent by construction: it is not in `mcp_servers:`, it
        has no grant to make, and a row offering to connect it would be an
        invitation to do nothing.

        `scopes` and `shares_with` are on the row because the panel shows them
        BEFORE it opens a popup and BEFORE it revokes: what a click is about to
        grant, and what a disconnect is about to take away from its siblings.
        """
        live = await self.refresh_status(user_id)
        stored = await conns.load(user_id)
        grouped = await self._tools_or_empty(user_id)
        siblings = self._siblings(live)

        out = []
        for label, spec in self.servers.items():
            server = spec.get("server")
            row = stored.get(server)
            consent = live.get(server)
            status = consent.status if consent else (row.status if row else DISCONNECTED)
            out.append(
                {
                    "server": server,
                    "label": label,
                    "name": spec.get("name", label),
                    "status": status,
                    "tool_count": len(grouped.get(server, [])),
                    "refreshed_at": row.refreshed_at.isoformat() if row and row.refreshed_at else None,
                    "setup_url": self._setup_urls.get((user_id, server)),
                    "scopes": list(consent.scopes) if consent else [],
                    "shares_with": [self._display(s) for s in siblings.get(server, ())],
                }
            )
        return out

    def _siblings(self, live: dict[str, Consent]) -> dict[str, tuple[str, ...]]:
        """Group the connectors that share one Arcade provider account.

        Gmail, Google Calendar and Google Search are all `arcade-google`: one
        sign-in, one revocable connection. Grouped from what `authorize`
        reported rather than from a table in config, because the provider behind
        a service is Arcade's fact to state and ours to read.
        """
        by_provider: dict[str, list[str]] = {}
        for server, consent in live.items():
            if consent.provider_id:
                by_provider.setdefault(consent.provider_id, []).append(server)
        return {
            server: tuple(other for other in group if other != server)
            for group in by_provider.values()
            for server in group
        }

    async def _tools_or_empty(self, user_id: str) -> dict[str, list[dict[str, Any]]]:
        """The grouped tool list, or nothing — the panel renders either way."""
        try:
            return await self.tools_by_server(user_id)
        except (ArcadeError, aiohttp.ClientError, TimeoutError) as e:
            logger.warning("could not read the gateway's tool list for %s: %s", user_id, e)
            return {}

    async def connect(self, user_id: str, server: str) -> dict[str, Any]:
        """Start one service's consent from the panel, and record what came back."""
        consent = await self.consent(user_id, server)
        await conns.mark(user_id, server, consent.status)
        if consent.setup_url:
            self._setup_urls[(user_id, server)] = consent.setup_url
        else:
            self._setup_urls.pop((user_id, server), None)
        return {
            "server": server,
            "status": consent.status,
            "setup_url": consent.setup_url,
            "scopes": list(consent.scopes),
        }

    async def disconnect(self, user_id: str, server: str) -> list[str]:
        """Revoke the grant at Arcade and drop our row. Returns the services taken with it.

        Dropping the row alone would be a lie: the panel would read disconnected
        while Arcade still held a live token and the very next tool call
        succeeded. The revoke is what makes the button mean what it says.

        It is also SHARED. Arcade's connection is per provider account, so
        revoking Gmail revokes Google Calendar and Google Search with it. There
        is no narrower revoke to offer, so the honest design is one button that
        says what it will take — the panel warns from `shares_with` before it
        gets here, and the return value is what actually went.
        """
        live = await self.refresh_status(user_id)
        consent = live.get(server)
        provider = consent.provider_id if consent else None
        taken = [server, *self._siblings(live).get(server, ())] if provider else [server]

        try:
            held = await self.client.user_connections(user_id)
        except ArcadeError as e:
            logger.warning("could not list Arcade connections for %s: %s", user_id, e)
            held = []

        for record in held:
            if provider and str(record.get("provider_id") or "") != provider:
                continue
            connection_id = record.get("id") or record.get("connection_id")
            if connection_id:
                await self.client.delete_connection(user_id, str(connection_id))

        for gone in taken:
            if self.is_connector(gone):
                await conns.forget(user_id, gone)
            self._setup_urls.pop((user_id, gone), None)
        self.invalidate(user_id)
        return taken

    async def close(self) -> None:
        await self.client.close()
        self._tools.clear()
        self._setup_urls.clear()


def _auto_approved(name: str, setting: Any) -> bool:
    """Read `mcp_servers.<label>.auto_approve`: `true` waives the whole app, a list waives those names."""
    if setting is True:
        return True
    if isinstance(setting, (list, tuple, set)):
        return name in setting
    return False


def _to_spec(tool: dict[str, Any], auto_approve: Any = None, *, readonly: bool = False) -> ToolSpec:
    """Convert one `tools/list` entry into a ToolSpec.

    A remote server does not report whether a tool mutates, so no connector's
    tool is marked readonly and every one requires approval. `auto_approve` in
    config waives the approval per app; `readonly` is asserted only for the
    tools that are ours.
    """
    name = tool["name"]
    return ToolSpec(
        name=name,
        description=tool.get("description") or "",
        input_schema=tool.get("inputSchema") or tool.get("input_schema") or {},
        readonly=readonly,
        requires_approval=not _auto_approved(name, auto_approve),
    )


def _render(result: Any) -> tuple[str, bool]:
    """Flatten an MCP `tools/call` result to (text, is_error), falling back to JSON."""
    if not isinstance(result, dict):
        return (result if isinstance(result, str) else json.dumps(result, default=str)), False

    is_error = bool(result.get("isError"))
    blocks = result.get("content")
    if not isinstance(blocks, list):
        return json.dumps(result, default=str), is_error

    parts = []
    for block in blocks:
        if not isinstance(block, dict):
            parts.append(str(block))
        elif block.get("type") == "text":
            parts.append(block.get("text", ""))
        else:
            parts.append(json.dumps(block, default=str))
    return "\n".join(p for p in parts if p), is_error


async def _envelope(name: str, result: Any, ctx: ToolContext) -> ResultEnvelope:
    """Wrap the result, storing the tail as a blob when it is too big to inline."""
    text, is_error = _render(result)
    if is_error:
        return fail("upstream_error", text or f"{name} reported an error with no detail.")

    # The ONE cap rule, from the loop. This read its own copy of the config key
    # and cut to its own length — a third answer to "how big is too big",
    # alongside the loop's and the runner's, which 11.7.5 collapsed into this
    # one function precisely so they cannot drift apart.
    head, total = cap_view(text)
    if total is None or ctx.store_blob is None:
        return ok(text)

    # The blob holds the whole text; the envelope carries the head plus the ref.
    ref = await ctx.store_blob(text)
    return ok(
        f"{head}\n\n[truncated at {len(head)} of {total} chars. "
        f"Read the rest with read_result(ref={ref!r})]",
        ref=ref,
    )
