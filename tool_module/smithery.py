"""
MCP, reached through Smithery Connect.

We never spawn local MCP subprocesses, never hold upstream OAuth tokens, and
never implement a provider's OAuth flow. Smithery keeps the credentials; we keep
a connection id and a cached tool list (`connections.py`).

Endpoints used:
  PUT    {base}/connect/{namespace}/{connection_id}       upsert a connection
  DELETE {base}/connect/{namespace}/{connection_id}       revoke it
  POST   {base}/connect/{namespace}/{connection_id}/mcp   JSON-RPC 2.0 to it

Two rules shape everything below. A warm tool call is ONE HTTP request: the
connection and its tool list come from the stored row, so nothing re-PUTs to
find out where a tool lives. And a tool call never opens an OAuth flow — an
unconnected server fails with `auth_required` carrying the setup URL, because
the model cannot complete a browser redirect and should stop asking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import aiohttp

from config_module.loader import config
from tool_module import connections as conns
from tool_module.connections import CONNECTED, Connection
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok

logger = logging.getLogger(__name__)

# A hung upstream must not hold a hop open. The registry caps the whole call at
# tools.call_timeout_s too; this is the per-request half.
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


class AuthRequiredError(Exception):
    """A connection needs a Smithery-hosted OAuth flow, or config, before it works."""

    def __init__(self, service: str, setup_url: str | None = None, state: str = "auth_required"):
        self.service = service
        self.setup_url = setup_url
        self.state = state
        super().__init__(f"Connect {service} first: {setup_url}" if setup_url else f"{service} needs setup")


class SmitheryError(RuntimeError):
    """Any non-auth Smithery API failure."""


def _parse_status(raw: Any) -> tuple[str, str | None]:
    """Smithery's `status` is a bare string or an object. Normalize to (state, setup_url)."""
    if isinstance(raw, str):
        return raw, None
    if isinstance(raw, dict):
        return raw.get("state", "unknown"), raw.get("setupUrl") or raw.get("authorizationUrl")
    return "unknown", None


class SmitheryClient:
    """Thin REST client. Owns the one ClientSession every call shares."""

    def __init__(self, api_key: str, namespace: str, base_url: str = "https://api.smithery.ai"):
        if not api_key:
            raise ValueError("SmitheryClient requires an api_key (set SMITHERY_API_KEY)")
        self.api_key = api_key
        self.namespace = namespace
        self.base_url = base_url.rstrip("/")
        self._session: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def session(self) -> aiohttp.ClientSession:
        """One session for the process. A session per call re-does TLS every time."""
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
        """PUT the connection. The caller reads `status` to see if it is usable."""
        url = f"{self.base_url}/connect/{self.namespace}/{connection_id}"
        body: dict[str, Any] = {"mcpUrl": mcp_url}
        if name:
            body["name"] = name
        meta = dict(metadata or {})
        if return_url:
            # Sent both ways: whichever field the current API honours picks it up.
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
        """POST one JSON-RPC 2.0 request. Returns the `result` field."""
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
    """JSON, or the SSE framing Smithery sometimes answers with."""
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


class Smithery:
    """
    The MCP half of the hands.

    Config shape (`mcp_servers:` in config.yaml). The key is an in-process label
    for logs and display only — identity is the url, so renaming it is free:

        mcp_servers:
          linear:
            mcp_url: "https://linear.run.tools"
            requires_auth: true
            name: "Linear"
    """

    def __init__(self, servers: dict[str, dict[str, Any]], smithery_config: dict[str, Any]):
        self.servers = servers or {}
        self.client = SmitheryClient(
            api_key=smithery_config.get("api_key"),
            namespace=smithery_config.get("namespace", "arkos"),
            base_url=smithery_config.get("base_url", "https://api.smithery.ai"),
        )
        self._ttl_s = float(_cfg("tools.mcp_cache_ttl_s", 3600))
        # mcp_url -> Connection, per user. `None` holds the shared (no-auth) ones.
        self._cache: dict[str | None, dict[str, Connection]] = {}
        self._locks: dict[str | None, asyncio.Lock] = {}
        self._generation: dict[str | None, int] = {}
        # Setup URLs live in memory only: they expire, and a stale one in the DB
        # is worse than none.
        self._setup_urls: dict[tuple[str | None, str], str] = {}

    # ---------- config ----------

    def _url(self, label: str) -> str:
        spec = self.servers.get(label)
        if not spec:
            raise SmitheryError(f"no config for server {label!r}")
        return spec["mcp_url"]

    def _label(self, mcp_url: str) -> str:
        """Reverse the url to its config label, for logs and the settings panel."""
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
        """
        Drop the cached rows AND bump the generation.

        Popping alone is not enough: a `_load` already awaiting its query would
        install its pre-write snapshot afterwards, and since `_load` only reads
        when the key is absent, that stale entry would never be revisited.
        """
        self._generation[owner] = self._generation.get(owner, 0) + 1
        self._cache.pop(owner, None)

    async def _load(self, owner: str | None) -> dict[str, Connection]:
        """One DB read per owner per process; the rows carry everything else."""
        cached = self._cache.get(owner)
        if cached is not None:
            return cached

        async with self._lock_for(owner):
            # Someone may have filled it while we waited; two concurrent misses
            # would otherwise both query and build divergent Connection graphs.
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
        """Refresh a connected server's tool list once its TTL is up. No PUT."""
        if not conn.connected or not conn.stale(self._ttl_s):
            return
        try:
            tools = await self._list_tools(conn.connection_id)
        except (SmitheryError, AuthRequiredError, aiohttp.ClientError, TimeoutError) as e:
            # The cached list is still the best thing we have; a refresh failing
            # must not take away tools the user already connected. Re-arm anyway,
            # or a server that is down costs a timeout on every manifest build.
            conn.refreshed_at = datetime.now(UTC)
            logger.warning("tools/list refresh failed for %s: %s", conn.mcp_url, e)
            return
        conn.tools = tools
        # Re-arm the in-memory clock too. Writing only the DB leaves the cached
        # object permanently stale, so every later manifest build re-fetches.
        conn.refreshed_at = datetime.now(UTC)
        await conns.save(owner, conn.mcp_url, status=CONNECTED, tools=tools)

    async def _list_tools(self, connection_id: str) -> list[dict[str, Any]]:
        result = await self.client.jsonrpc(connection_id, "tools/list", {})
        return result.get("tools", []) if isinstance(result, dict) else []

    # ---------- connecting ----------

    async def connect(self, user_id: str | None, label: str, *, return_url: str | None = None) -> Connection:
        """
        Bring a connection up, writing the row before the PUT (D24).

        Raises AuthRequiredError when the user still has to finish OAuth; the
        row stays behind holding the id, so finishing it needs no new mint.
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
            await conns.save(owner, mcp_url, status=state, tools=None)
            self._setup_urls[(owner, mcp_url)] = setup_url or ""
            self._invalidate(owner)
            raise AuthRequiredError(service=label, setup_url=setup_url, state=state)

        tools = await self._list_tools(connection_id)
        await conns.save(owner, mcp_url, status=CONNECTED, tools=tools)
        self._setup_urls.pop((owner, mcp_url), None)
        self._invalidate(owner)
        return Connection(mcp_url, connection_id, CONNECTED, tools)

    async def initialize_shared(self) -> None:
        """
        Connect the no-auth servers at startup.

        Anything already `connected` in the DB is left alone: that is what makes
        a restart cost one DB read and zero Smithery PUTs.
        """
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

    async def specs(self, user_id: str) -> list[ToolSpec]:
        """
        Every MCP tool this user can reach, from the stored caches.

        Names are bare here; `registry.manifest` adds the `mcp_` prefix, so a
        remote `read_file` cannot shadow ours.
        """
        specs: list[ToolSpec] = []
        # User-first, matching `_resolve`. If the two disagree the model is shown
        # one server's schema and dispatched to another's, because
        # `registry.manifest` keeps the first of a duplicated name.
        for owner in (user_id, None):
            for conn in list((await self._load(owner)).values()):
                if not conn.connected:
                    continue
                await self._revalidate(owner, conn)
                specs.extend(_to_specs(conn.tools))
        return specs

    # ---------- the dispatch half ----------

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        """
        Run one MCP tool. The `mcp_` prefix is already stripped by the registry.

        Warm path: resolve the url from cache, one POST. Never a PUT — a tool
        call is not the place to start an OAuth flow.
        """
        found = await self._resolve(name, ctx.user_id)
        if found is None:
            return fail("not_found", f"No connected MCP tool named {name!r}.")
        owner, conn = found

        if not conn.connected:
            setup = self._setup_urls.get((owner, conn.mcp_url))
            service = self._display(conn.mcp_url)
            return fail(
                "auth_required",
                f"{service} is not connected yet. The human has to authorize it"
                + (f" here: {setup}" if setup else "")
                + ". Do not retry this tool; tell them what you needed it for.",
                retryable=False,
            )

        try:
            result = await self.client.jsonrpc(conn.connection_id, "tools/call", {"name": name, "arguments": args})
        except AuthRequiredError as e:
            if e.setup_url:
                self._setup_urls[(owner, conn.mcp_url)] = e.setup_url
            # The grant died under us. Mark it so the manifest stops offering it,
            # but KEEP tools_cache: the tool names are how a later call resolves
            # to this connection, and dropping them makes the next attempt a bare
            # not_found instead of this actionable answer. Failing to record the
            # status must not turn "do not retry" into a retryable error, so the
            # write is allowed to fail on its own.
            try:
                await conns.set_status(owner, conn.mcp_url, "auth_required")
            except Exception:
                logger.exception("could not record dead grant for %s", conn.mcp_url)
            else:
                self._invalidate(owner)
            return fail(
                "auth_required",
                f"{self._display(conn.mcp_url)} needs to be reconnected"
                + (f": {e.setup_url}" if e.setup_url else "")
                + ". Do not retry this tool.",
                retryable=False,
            )
        except SmitheryError as e:
            return fail("upstream_error", f"{name} failed: {e}")
        except (aiohttp.ClientError, TimeoutError) as e:
            # aiohttp's total-timeout raises bare TimeoutError, which is NOT a
            # ClientError, so it would otherwise escape the envelope promise.
            return fail("upstream_error", f"{name} could not reach the server: {type(e).__name__}: {e}")

        return await _envelope(name, result, ctx)

    async def _resolve(self, name: str, user_id: str) -> tuple[str | None, Connection] | None:
        """
        Which connection owns this tool. The user's own servers win over shared
        ones, and another user's connections are never reachable from here.
        """
        for owner in (user_id, None):
            for conn in (await self._load(owner)).values():
                if any(t.get("name") == name for t in conn.tools):
                    return owner, conn
        return None

    # ---------- settings panel ----------

    async def status(self, user_id: str) -> list[dict[str, Any]]:
        """Per-user services and whether they are connected, for the UI."""
        stored = await self._load(user_id)
        out = []
        for label, spec in self.servers.items():
            if not spec.get("requires_auth"):
                continue
            conn = stored.get(spec["mcp_url"])
            out.append(
                {
                    "service": label,
                    "name": spec.get("name", label),
                    "connected": bool(conn and conn.connected),
                    "setup_url": self._setup_urls.get((user_id, spec["mcp_url"])),
                }
            )
        return out

    async def close(self) -> None:
        await self.client.close()
        self._cache.clear()
        self._setup_urls.clear()


def _owner_for(spec: dict[str, Any], user_id: str | None) -> str | None:
    """
    Which table a connection belongs in. `None` means shared.

    A `requires_auth` server with no user would otherwise land in
    `shared_connections`, handing one person's OAuth grant to everybody. That is
    a bug in the caller, so it raises rather than guessing.
    """
    if not spec.get("requires_auth"):
        return None
    if not user_id:
        raise SmitheryError(f"{spec.get('mcp_url')} requires per-user auth, but no user_id was given")
    return user_id


def _to_specs(tools: list[dict[str, Any]]) -> list[ToolSpec]:
    """
    A cached `tools/list` entry becomes a ToolSpec.

    `readonly=False` always: a remote server does not tell us whether a tool
    mutates, and guessing wrong runs writes in parallel.
    """
    specs = []
    for tool in tools:
        name = tool.get("name")
        if not name:
            continue
        specs.append(
            ToolSpec(
                name=name,
                description=tool.get("description") or "",
                input_schema=tool.get("inputSchema") or tool.get("input_schema") or {},
                readonly=False,
            )
        )
    return specs


def _render(result: Any) -> tuple[str, bool]:
    """
    Flatten an MCP `tools/call` result to (text, is_error).

    MCP answers with a content list; anything else we hand over as JSON rather
    than dropping it.
    """
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

    # Without this the tail is simply lost: the loop view-caps for the screen but
    # hands the model whatever the envelope holds, and `ref` only ever comes from here.
    ref = await ctx.store_blob(text)
    head = text[:cap]
    return ok(
        f"{head}\n\n[truncated at {cap} of {len(text)} chars — read the rest with read_result(ref={ref!r})]",
        ref=ref,
    )
