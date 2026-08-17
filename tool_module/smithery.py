"""
MCP, reached through Smithery Connect.

Smithery holds the credentials; we hold a connection id and a cached tool list
(`connections.py`). A warm tool call is one HTTP request and never opens an OAuth
flow: an unconnected server fails with `auth_required` and the setup URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import aiohttp

from config_module.loader import config
from tool_module import connections as conns
from tool_module.connections import CONNECTED, Connection
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok

logger = logging.getLogger(__name__)

# Per-request half of the cap; the registry caps the whole call separately.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


class AuthRequiredError(Exception):
    """Raised when a connection needs a Smithery-hosted OAuth flow, or config, first."""

    def __init__(self, service: str, setup_url: str | None = None, state: str = "auth_required"):
        self.service = service
        self.setup_url = setup_url
        self.state = state
        super().__init__(f"Connect {service} first: {setup_url}" if setup_url else f"{service} needs setup")


class SmitheryError(RuntimeError):
    """Raised for any non-auth Smithery API failure."""


def _parse_status(raw: Any) -> tuple[str, str | None]:
    """Normalize Smithery's `status`, a bare string or an object, to (state, setup_url)."""
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, dict):
        return raw.get("state", "unknown"), raw.get("setupUrl") or raw.get("authorizationUrl")
    return "unknown", None


class SmitheryClient:
    """Thin REST client owning the one ClientSession every call shares."""

    def __init__(self, api_key: str, namespace: str, base_url: str = "https://api.smithery.ai"):
        if not api_key:
            raise ValueError("SmitheryClient requires an api_key (set SMITHERY_API_KEY)")
        self.api_key = api_key
        self.namespace = namespace
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def session(self) -> aiohttp.ClientSession:
        """Return the process-wide session, creating it on first use."""
        if self._session is None or self._session.closed:
            async with self._lock:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession(timeout=_HTTP_TIMEOUT)
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def upsert(
        self,
        connection_id: str,
        mcp_url: str,
        *,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        return_url: str | None = None,
    ) -> dict[str, Any]:
        """PUT the connection; the caller reads `status` to see whether it is usable."""
        url = f"{self.base_url}/connect/{self.namespace}/{connection_id}"
        body: dict[str, Any] = {"mcpUrl": mcp_url}
        if name:
            body["name"] = name
        meta = dict(metadata or {})
        if return_url:
            # Sent both ways; whichever field the current API honours picks it up.
            body["returnUrl"] = return_url
            meta.setdefault("returnUrl", return_url)
        if meta:
            body["metadata"] = meta
        if headers:
            body["headers"] = headers

        session = await self.session()
        async with session.put(url, json=body, headers=self._headers()) as resp:
            text = await resp.text()
            if resp.status == 401:
                raise AuthRequiredError(service=connection_id)
            if resp.status >= 400:
                raise SmitheryError(f"upsert {resp.status}: {text[:300]}")
            return json.loads(text) if text.strip() else {}

    async def delete(self, connection_id: str) -> None:
        """Remove the connection so the next connect starts a fresh OAuth flow."""
        url = f"{self.base_url}/connect/{self.namespace}/{connection_id}"
        session = await self.session()
        async with session.delete(url, headers=self._headers()) as resp:
            if resp.status not in (200, 204, 404):
                logger.warning("smithery DELETE %s returned %s", connection_id, resp.status)

    async def jsonrpc(self, connection_id: str, method: str, params: dict[str, Any] | None = None) -> Any:
        """POST one JSON-RPC 2.0 request and return its `result` field."""
        url = f"{self.base_url}/connect/{self.namespace}/{connection_id}/mcp"
        headers = self._headers()
        headers["Accept"] = "application/json, text/event-stream"
        body = {"jsonrpc": "2.0", "id": uuid.uuid4().hex[:12], "method": method, "params": params or {}}

        session = await self.session()
        async with session.post(url, json=body, headers=headers) as resp:
            text = await resp.text()
            if resp.status == 401:
                raise AuthRequiredError(service=connection_id)
            if resp.status >= 400:
                raise SmitheryError(f"{method} {resp.status}: {text[:300]}")
            data = _parse_rpc(text, method)

        if "error" in data:
            err = data["error"] or {}
            raise SmitheryError(f"{method} rpc error {err.get('code')}: {err.get('message')}")
        return data.get("result", {})


def _parse_rpc(text: str, method: str) -> dict[str, Any]:
    """Parse a response body as JSON, or as the SSE framing Smithery sometimes uses."""
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
    raise SmitheryError(f"{method}: response was neither JSON nor SSE: {text[:200]}")


@dataclass(slots=True)
class Route:
    """Which connection serves a tool name, and the cached schema for it."""

    owner: str | None
    conn: Connection
    tool: dict[str, Any]


class Smithery:
    """
    The MCP half of the hands, configured by `mcp_servers:` in config.yaml.

    A config key is a label for logs and display only; identity is the `mcp_url`.
    """

    def __init__(self, servers: dict[str, dict[str, Any]], smithery_config: dict[str, Any]):
        self.servers = servers or {}
        self.client = SmitheryClient(
            api_key=smithery_config.get("api_key"),
            namespace=smithery_config.get("namespace", "arkos"),
            base_url=smithery_config.get("base_url", "https://api.smithery.ai"),
        )
        self._ttl_s = float(_cfg("tools.mcp_cache_ttl_s", 3600))
        # mcp_url -> Connection, per user; `None` holds the shared (no-auth) ones.
        self._cache: dict[str | None, dict[str, Connection]] = {}
        self._locks: dict[str | None, asyncio.Lock] = {}
        self._generation: dict[str | None, int] = {}
        # Setup URLs stay in memory: they expire, and a stale one is worse than none.
        self._setup_urls: dict[tuple[str | None, str], str] = {}

    # ---------- config ----------

    def _url(self, label: str) -> str:
        spec = self.servers.get(label)
        if not spec:
            raise SmitheryError(f"no config for server {label!r}")
        return spec["mcp_url"]

    def _label(self, mcp_url: str) -> str:
        """Return the config label for a url, for logs and the settings panel."""
        for label, spec in self.servers.items():
            if spec.get("mcp_url") == mcp_url:
                return label
        return mcp_url

    def _display(self, mcp_url: str) -> str:
        label = self._label(mcp_url)
        return (self.servers.get(label) or {}).get("name", label)

    # ---------- cache ----------

    def _lock_for(self, owner: str | None) -> asyncio.Lock:
        lock = self._locks.get(owner)
        if lock is None:
            lock = self._locks[owner] = asyncio.Lock()
        return lock

    def _invalidate(self, owner: str | None) -> None:
        """Drop the cached rows and bump the generation, so an in-flight `_load` cannot reinstall them."""
        self._generation[owner] = self._generation.get(owner, 0) + 1
        self._cache.pop(owner, None)

    async def _load(self, owner: str | None) -> dict[str, Connection]:
        """Return an owner's connections, reading the DB once per process."""
        cached = self._cache.get(owner)
        if cached is not None:
            return cached

        async with self._lock_for(owner):
            # Re-check under the lock; two concurrent misses would otherwise
            # build divergent Connection graphs.
            cached = self._cache.get(owner)
            if cached is not None:
                return cached
            generation = self._generation.get(owner, 0)
            rows = await conns.load(owner)
            # An invalidation that landed during the read wins over this result.
            if self._generation.get(owner, 0) == generation:
                self._cache[owner] = rows
            return rows

    async def _revalidate(self, owner: str | None, conn: Connection) -> None:
        """Refresh a connected server's tool list once its TTL is up, without a PUT."""
        if not conn.connected or not conn.stale(self._ttl_s):
            return
        try:
            tools = await self._list_tools(conn.connection_id)
        except AuthRequiredError as e:
            # A dead grant, recorded the way call() records one: the status moves
            # and tools_cache stays, so the tool names still resolve to this
            # connection and the model gets `auth_required` with a setup URL
            # rather than `not_found`. Without this the refresh is silent and
            # Settings keeps claiming connected until someone tries a call.
            if e.setup_url:
                self._setup_urls[(owner, conn.mcp_url)] = e.setup_url
            conn.refreshed_at = datetime.now(UTC)
            conn.status = "auth_required"
            try:
                await conns.set_status(owner, conn.mcp_url, "auth_required")
            except Exception:
                logger.exception("could not record dead grant for %s", conn.mcp_url)
            else:
                self._invalidate(owner)
            return
        except (SmitheryError, aiohttp.ClientError, TimeoutError) as e:
            # Keep the cached list, but re-arm anyway or a server that is down
            # costs a timeout on every manifest build.
            conn.refreshed_at = datetime.now(UTC)
            logger.warning("tools/list refresh failed for %s: %s", conn.mcp_url, e)
            return
        conn.tools = tools
        # Re-arm the in-memory clock too, or the cached object stays stale forever.
        conn.refreshed_at = datetime.now(UTC)
        await conns.save(owner, conn.mcp_url, status=CONNECTED, tools=tools)

    async def _list_tools(self, connection_id: str) -> list[dict[str, Any]]:
        result = await self.client.jsonrpc(connection_id, "tools/list", {})
        return result.get("tools", []) if isinstance(result, dict) else []

    # ---------- connecting ----------

    async def connect(self, user_id: str | None, label: str, *, return_url: str | None = None) -> Connection:
        """
        Bring a connection up, writing the row before the PUT.

        Raises AuthRequiredError when the user still has to finish OAuth.
        """
        mcp_url = self._url(label)
        spec = self.servers[label]
        owner = _owner_for(spec, user_id)
        connection_id = await conns.claim(owner, mcp_url)

        response = await self.client.upsert(
            connection_id,
            mcp_url,
            name=spec.get("name", label),
            metadata={"userId": user_id} if owner is not None else None,
            headers=spec.get("headers"),
            return_url=return_url,
        )
        state, setup_url = _parse_status(response.get("status"))

        if state != CONNECTED:
            # Status only, so tools_cache survives: an unconnected state is
            # routine here (every read of GET /connections re-verifies) and a
            # token that is not live yet must not cost a good tool list.
            await conns.set_status(owner, mcp_url, state)
            self._setup_urls[(owner, mcp_url)] = setup_url or ""
            self._invalidate(owner)
            raise AuthRequiredError(service=label, setup_url=setup_url, state=state)

        tools = await self._list_tools(connection_id)
        await conns.save(owner, mcp_url, status=CONNECTED, tools=tools)
        self._setup_urls.pop((owner, mcp_url), None)
        self._invalidate(owner)
        return Connection(mcp_url, connection_id, CONNECTED, tools)

    async def initialize_shared(self) -> None:
        """Connect the no-auth servers at startup, leaving already-connected ones alone."""
        stored = await self._load(None)
        for label, spec in self.servers.items():
            if spec.get("requires_auth"):
                continue
            conn = stored.get(spec["mcp_url"])
            if conn is not None and conn.connected:
                await self._revalidate(None, conn)
                continue
            try:
                await self.connect(None, label)
            except (AuthRequiredError, SmitheryError, aiohttp.ClientError) as e:
                logger.error("smithery: shared server %r did not come up: %s", label, e)

    async def disconnect(self, user_id: str, label: str) -> None:
        """Revoke at Smithery and drop the row, so the next connect is a fresh grant."""
        mcp_url = self._url(label)
        owner = _owner_for(self.servers[label], user_id)
        stored = await self._load(owner)
        conn = stored.get(mcp_url)
        if conn is not None:
            await self.client.delete(conn.connection_id)
        await conns.forget(owner, mcp_url)
        self._setup_urls.pop((owner, mcp_url), None)
        self._invalidate(owner)

    # ---------- the manifest half ----------

    async def _routes(self, user_id: str) -> dict[str, Route]:
        """
        Map every reachable tool name to the connection that owns it.

        The single source of the name-to-connection mapping: `specs` and `call`
        both read it, so the manifest cannot advertise one server's schema while
        dispatch sends the call to another's. Connected wins over disconnected,
        then the user's own connections win over shared ones.
        """
        routes: dict[str, Route] = {}
        for want_connected in (True, False):
            for owner in (user_id, None):
                for conn in list((await self._load(owner)).values()):
                    if conn.connected != want_connected:
                        continue
                    if conn.connected:
                        await self._revalidate(owner, conn)
                    for tool in conn.tools:
                        name = tool.get("name")
                        if name and name not in routes:
                            routes[name] = Route(owner, conn, tool)
        return routes

    async def specs(self, user_id: str) -> list[ToolSpec]:
        """Return every MCP tool this user can reach, named bare; `registry.manifest` prefixes them."""
        routes = await self._routes(user_id)
        return [
            _to_spec(r.tool, self._auto_approve_for(r.conn.mcp_url))
            for r in routes.values()
            if r.conn.connected
        ]

    def _auto_approve_for(self, mcp_url: str) -> Any:
        """The `auto_approve` setting for the server behind a url, absent meaning approve nothing."""
        return (self.servers.get(self._label(mcp_url)) or {}).get("auto_approve")

    # ---------- the dispatch half ----------

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        """Run one MCP tool, with the `mcp_` prefix already stripped by the registry."""
        route = (await self._routes(ctx.user_id)).get(name)
        if route is None:
            return fail("not_found", f"No MCP tool named {name!r} on any server you have connected.")
        owner, conn = route.owner, route.conn

        if not conn.connected:
            return fail("auth_required", self._reconnect_message(owner, conn), retryable=False)

        try:
            result = await self.client.jsonrpc(conn.connection_id, "tools/call", {"name": name, "arguments": args})
        except AuthRequiredError as e:
            if e.setup_url:
                self._setup_urls[(owner, conn.mcp_url)] = e.setup_url
            # Keep tools_cache: the tool names are how a later call resolves to
            # this connection. The write may fail without changing the answer.
            try:
                await conns.set_status(owner, conn.mcp_url, "auth_required")
            except Exception:
                logger.exception("could not record dead grant for %s", conn.mcp_url)
            else:
                self._invalidate(owner)
            return fail("auth_required", self._reconnect_message(owner, conn), retryable=False)
        except SmitheryError as e:
            return fail("upstream_error", f"{name} failed: {e}")
        except (aiohttp.ClientError, TimeoutError) as e:
            # aiohttp's total-timeout raises a bare TimeoutError, not a ClientError.
            return fail("upstream_error", f"{name} could not reach the server: {type(e).__name__}: {e}")

        return await _envelope(name, result, ctx)

    def _reconnect_message(self, owner: str | None, conn: Connection) -> str:
        """Tell the model a server needs authorizing, with the setup URL when we still hold one."""
        service = self._display(conn.mcp_url)
        setup = self._setup_urls.get((owner, conn.mcp_url))
        where = f" here: {setup}" if setup else " from the connections panel in Settings"
        return f"{service} is not connected. The human has to authorize it{where}. Do not retry this tool."

    # ---------- settings panel ----------

    async def connections(self, user_id: str) -> list[dict[str, Any]]:
        """
        Return every configured server and where this user stands with it.

        One row per `mcp_servers:` entry, connected or not, because the panel has
        to offer the ones they have not connected yet. Shared servers are read
        from the shared table: they have no user, and showing them per-user would
        imply a grant that does not exist.
        """
        per_user = await self._load(user_id)
        shared = await self._load(None)
        out = []
        for label, spec in self.servers.items():
            mcp_url = spec["mcp_url"]
            requires_auth = bool(spec.get("requires_auth"))
            owner = user_id if requires_auth else None
            conn = (per_user if requires_auth else shared).get(mcp_url)
            out.append(
                {
                    "server": label,
                    "name": spec.get("name", label),
                    "mcp_url": mcp_url,
                    "requires_auth": requires_auth,
                    "status": conn.status if conn else "disconnected",
                    "tool_count": len(conn.tools) if conn else 0,
                    "refreshed_at": conn.refreshed_at.isoformat() if conn and conn.refreshed_at else None,
                    "setup_url": self._setup_urls.get((owner, mcp_url)),
                }
            )
        return out

    def needs_repair(self, row: dict[str, Any]) -> bool:
        """True for a per-user server that is configured but not usable, so a read can re-verify it."""
        return bool(row["requires_auth"]) and row["status"] != CONNECTED

    async def close(self) -> None:
        await self.client.close()
        self._cache.clear()
        self._setup_urls.clear()


def _owner_for(spec: dict[str, Any], user_id: str | None) -> str | None:
    """
    Return the owner key for a connection: None for shared, else the user_id.

    Raises SmitheryError for a per-user server with no user, rather than handing
    one person's OAuth grant to everybody.
    """
    if not spec.get("requires_auth"):
        return None
    if not user_id:
        raise SmitheryError(f"{spec.get('mcp_url')} requires per-user auth, but no user_id was given")
    return user_id


def _auto_approved(name: str, setting: Any) -> bool:
    """Read `mcp_servers.<label>.auto_approve`: `true` waives the whole server, a list waives those names."""
    if setting is True:
        return True
    if isinstance(setting, (list, tuple, set)):
        return name in setting
    return False


def _to_spec(tool: dict[str, Any], auto_approve: Any = None) -> ToolSpec:
    """
    Convert one cached `tools/list` entry into a ToolSpec.

    A remote server does not tell us whether a tool mutates, so both unknowns
    resolve the safe way: never readonly, so the loop cannot batch a write in
    parallel, and approval required, because an unattended run reaching for
    Gmail or GitHub takes actions that leave the building. Waive it per server
    in config, which is where a decision about a specific server belongs.
    """
    name = tool["name"]
    return ToolSpec(
        name=name,
        description=tool.get("description") or "",
        input_schema=tool.get("inputSchema") or tool.get("input_schema") or {},
        readonly=False,
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

    cap = int(_cfg("tools.result_view_cap_chars", 4000))
    if len(text) <= cap or ctx.store_blob is None:
        return ok(text)

    # `ref` only ever comes from here; without it the tail is simply lost.
    ref = await ctx.store_blob(text)
    head = text[:cap]
    return ok(
        f"{head}\n\n[truncated at {cap} of {len(text)} chars. Read the rest with read_result(ref={ref!r})]",
        ref=ref,
    )
