"""MCP, reached through one Composio MCP server.

Composio's dialect, where a generic MCP client breaks quietly:

- `x-api-key` only: `Authorization: Bearer` is a 401, and sending both is a 401.
- Identity is a query param on a per-user url, `…/v3/mcp/{server}/mcp?user_id=`;
  it is derived, not minted, and is never handed to a browser.
- The dashboard/SDK url `/v3.1/mcp/{id}` answers 307 with the resolved
  `/v3/mcp/{id}/mcp` in the JSON body and no `Location` header, so nothing
  follows it; this client speaks the resolved form directly.
- No session id (state is url-scoped) and no pagination on `tools/list`.
- Replies to plain JSON-RPC POSTs may arrive as SSE `data:` frames.
- A refused call is a SUCCESSFUL `tools/call` carrying `isError: true`.
- A content block's `text` is itself a JSON string, and the object inside spells
  success `successfull` (three l's) where the REST API spells it `successful`.

One server url serves every toolkit and tools come back flat, prefixed by
toolkit in upper snake; that prefix is the key `user_connections` and
`session_tools` use. A connected account is per TOOLKIT, so disconnecting Gmail
leaves Calendar alone.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import aiohttp

from agent_module.loop import cap_view
from tool_module import connections as conns
from tool_module.connections import CONNECTED
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=90)
# The toolkit prefix Composio puts on its own helper tools. Never in a manifest.
META_PREFIX = "COMPOSIO"


class ComposioError(RuntimeError):
    """A Composio call did not come back usable."""


def prefix_of(tool_name: str) -> str:
    """`GMAIL_FETCH_EMAILS` -> `GMAIL`. The durable key, upper snake."""
    return tool_name.split("_", 1)[0] if "_" in tool_name else tool_name


class ComposioClient:
    """The wire: one MCP server url, made per user, plus the REST API for auth."""

    def __init__(
        self,
        server_id: str,
        api_key: str,
        *,
        base_url: str = "https://backend.composio.dev",
        protocol_version: str = "2025-11-25",
    ):
        self.server_id = server_id
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.protocol_version = protocol_version
        self._http: aiohttp.ClientSession | None = None
        self._lock = asyncio.Lock()

    async def http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            async with self._lock:
                if self._http is None or self._http.closed:
                    self._http = aiohttp.ClientSession(timeout=_HTTP_TIMEOUT)
        return self._http

    async def close(self) -> None:
        if self._http is not None and not self._http.closed:
            await self._http.close()
        self._http = None

    def _headers(self) -> dict[str, str]:
        # `x-api-key` alone: bearer is a 401, and sending both auth modes is a 401.
        return {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }

    def url_for(self, user_id: str) -> str:
        """The resolved per-user MCP url: derived, not minted, and never handed to a browser."""
        return f"{self.base_url}/v3/mcp/{self.server_id}/mcp?user_id={user_id}"

    async def rpc(self, user_id: str, method: str, params: dict[str, Any] | None = None) -> Any:
        """One JSON-RPC call over the per-user url. No session id: state is url-scoped."""
        body = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        http = await self.http()
        async with http.post(self.url_for(user_id), json=body, headers=self._headers()) as resp:
            text = await resp.text()
            if resp.status in (401, 403):
                raise ComposioError(f"Composio refused our API key ({resp.status})")
            if resp.status >= 400:
                raise ComposioError(f"{method} -> {resp.status}: {text[:300]}")
        return _parse_rpc(text, method)

    async def list_tools(self, user_id: str) -> list[dict[str, Any]]:
        """The whole catalogue in ONE page: `tools/list` returns no cursor."""
        result = await self.rpc(user_id, "tools/list", {})
        tools = result.get("tools")
        return tools if isinstance(tools, list) else []

    async def call_tool(self, user_id: str, name: str, args: dict[str, Any]) -> Any:
        return await self.rpc(user_id, "tools/call", {"name": name, "arguments": args})

    # ---------- the REST API, where connecting lives ----------

    async def _rest(self, method: str, path: str, body: Any = None) -> Any:
        url = f"{self.base_url}{path}"
        http = await self.http()
        async with http.request(method, url, json=body, headers=self._headers()) as resp:
            text = await resp.text()
            if resp.status in (401, 403):
                raise ComposioError(f"Composio refused our API key ({resp.status})")
            if resp.status >= 400:
                raise ComposioError(f"{method} {path} -> {resp.status}: {text[:300]}")
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ComposioError(f"{path}: response was not JSON: {text[:200]}") from e

    async def link(self, user_id: str, auth_config_id: str, callback_url: str) -> dict[str, Any]:
        """Mint a consent link for one user and one auth config.

        The browser returns to `callback_url` carrying `status` and
        `connected_account_id` as query params, which `/connections/done` reconciles from.
        """
        return await self._rest(
            "POST",
            "/api/v3/connected_accounts/link",
            {"user_id": user_id, "auth_config_id": auth_config_id, "callback_url": callback_url},
        )

    async def connected_accounts(self, user_id: str) -> list[dict[str, Any]]:
        payload = await self._rest("GET", f"/api/v3/connected_accounts?user_ids={user_id}")
        items = payload.get("items") if isinstance(payload, dict) else None
        return items if isinstance(items, list) else []

    async def delete_connected_account(self, account_id: str) -> None:
        await self._rest("DELETE", f"/api/v3/connected_accounts/{account_id}")


def _parse_rpc(text: str, method: str) -> dict[str, Any]:
    """Unwrap a JSON-RPC reply that may arrive as an SSE `data:` frame."""
    payload: Any = None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            if line.startswith("data: "):
                try:
                    payload = json.loads(line[6:])
                    break
                except json.JSONDecodeError:
                    continue
    if not isinstance(payload, dict):
        raise ComposioError(f"{method}: unreadable reply: {text[:200]}")
    if payload.get("error"):
        raise ComposioError(f"{method}: {json.dumps(payload['error'])[:300]}")
    result = payload.get("result")
    return result if isinstance(result, dict) else {}


@dataclass(frozen=True, slots=True)
class ServerTools:
    """One toolkit and the tools it is offering right now."""

    label: str
    name: str
    server: str
    specs: list[ToolSpec]


@dataclass(frozen=True, slots=True)
class Consent:
    """What Composio says about one toolkit for one user."""

    server: str
    status: str
    setup_url: str | None = None
    account_id: str | None = None

    @property
    def connected(self) -> bool:
        return self.status == CONNECTED


class Composio:
    """The connector backend: what a turn may reach, and what the panel shows."""

    def __init__(self, servers: dict[str, dict[str, Any]], mcp_config: dict[str, Any]):
        self.servers = servers or {}
        self.client = ComposioClient(
            server_id=str(mcp_config.get("server_id") or ""),
            api_key=str(mcp_config.get("api_key") or ""),
            base_url=str(mcp_config.get("base_url") or "https://backend.composio.dev"),
            protocol_version=str(mcp_config.get("protocol_version") or "2025-11-25"),
        )
        self.callback_url = str(mcp_config.get("callback_url") or "")
        self._tools: dict[str, dict[str, list[dict[str, Any]]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._setup_urls: dict[tuple[str, str], str] = {}

    # ---------- the roster ----------

    def _by_server(self, server: str) -> tuple[str, dict[str, Any]] | None:
        for label, spec in self.servers.items():
            if spec.get("server") == server:
                return label, spec
        return None

    def _display(self, server: str) -> str:
        found = self._by_server(server)
        return found[1].get("name", server) if found else server

    def is_connector(self, server: str) -> bool:
        return self._by_server(server) is not None

    def prefixes(self) -> list[str]:
        return [spec["server"] for spec in self.servers.values() if spec.get("server")]

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = self._locks[user_id] = asyncio.Lock()
        return lock

    async def tools_by_server(self, user_id: str, *, refresh: bool = False) -> dict[str, list[dict[str, Any]]]:
        """The catalogue, grouped by toolkit prefix, cached per user (the url is per user)."""
        if not refresh and user_id in self._tools:
            return self._tools[user_id]
        async with self._lock_for(user_id):
            if not refresh and user_id in self._tools:
                return self._tools[user_id]
            tools = await self.client.list_tools(user_id)
            grouped: dict[str, list[dict[str, Any]]] = {}
            for tool in tools:
                grouped.setdefault(prefix_of(str(tool.get("name", ""))), []).append(tool)
            self._tools[user_id] = grouped
            return grouped

    def invalidate(self, user_id: str) -> None:
        self._tools.pop(user_id, None)

    async def _tools_or_empty(self, user_id: str) -> dict[str, list[dict[str, Any]]]:
        try:
            return await self.tools_by_server(user_id)
        except (ComposioError, aiohttp.ClientError, TimeoutError) as e:
            logger.warning("could not read the Composio catalogue for %s: %s", user_id, e)
            return {}

    async def reach(self, user_id: str) -> list[ServerTools]:
        """Every CONNECTED toolkit and the tools it offers, grouped.

        Grouped because the tool budget is spent and refused a whole server at a time.
        Connection state comes from our stored rows, not a round trip: this runs every turn.
        """
        grouped = await self._tools_or_empty(user_id)
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
        """Always empty: no toolkit on this wire is unconditionally ours. `registry.manifest` calls it."""
        return []

    async def specs(self, user_id: str) -> list[ToolSpec]:
        return [spec for server in await self.reach(user_id) for spec in server.specs]

    # ---------- dispatch ----------

    async def call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        """Run one tool, with the `mcp_` prefix already stripped by the registry."""
        server = prefix_of(name)

        if server == META_PREFIX:
            # Composio's own connection-management tools are never in a manifest.
            return fail("not_found", f"No tool named {name!r}.")

        if not self.is_connector(server):
            return fail("not_found", f"No MCP tool named {name!r} on any server you can reach.")

        row = (await conns.load(ctx.user_id)).get(server)
        if row is None or not row.connected:
            return fail("auth_required", self._reconnect_message(server), retryable=False)

        try:
            result = await self.client.call_tool(ctx.user_id, name, args)
        except ComposioError as e:
            return fail("upstream_error", f"{name} failed: {e}")
        except (aiohttp.ClientError, TimeoutError) as e:
            return fail("upstream_error", f"{name} could not reach Composio: {type(e).__name__}: {e}")

        text, is_error = _render(result)
        if is_error and _is_unconnected(text):
            # A dead grant arrives as a successful call carrying isError: correct
            # the row rather than let the model retry into it every turn.
            await conns.mark(ctx.user_id, server, conns.RECONNECT)
            return fail("auth_required", self._reconnect_message(server), retryable=False)
        return await _envelope(name, result, ctx)

    def _reconnect_message(self, server: str) -> str:
        return (
            f"{self._display(server)} is not connected. The human has to authorize it "
            f"from the connections panel in Settings. Do not retry this tool."
        )

    # ---------- consent ----------

    def _auth_config(self, server: str) -> str:
        found = self._by_server(server)
        if not found:
            raise ComposioError(f"{server!r} is not a configured connector")
        auth_config = str(found[1].get("auth_config_id") or "").strip()
        if not auth_config:
            raise ComposioError(
                f"{server} has no `auth_config_id` in config; the panel cannot ask for consent "
                f"without the Composio auth config behind the toolkit"
            )
        return auth_config

    async def consent(self, user_id: str, server: str) -> Consent:
        """Mint a consent link for one toolkit, or report it already connected."""
        live = await self._accounts(user_id)
        found = live.get(server)
        if found and found.status == CONNECTED:
            return Consent(server, CONNECTED, None, found.account_id)

        response = await self.client.link(user_id, self._auth_config(server), self.callback_url)
        url = response.get("redirect_url") or response.get("redirectUrl") or response.get("url")
        account_id = response.get("id") or response.get("connected_account_id")
        if not url:
            raise ComposioError(f"link({server}) returned no consent url: {json.dumps(response)[:300]}")
        return Consent(server, conns.PENDING, str(url), str(account_id) if account_id else None)

    async def _accounts(self, user_id: str) -> dict[str, Consent]:
        """Read Composio's live per-toolkit state, keyed by our prefix."""
        try:
            records = await self.client.connected_accounts(user_id)
        except (ComposioError, aiohttp.ClientError, TimeoutError) as e:
            logger.warning("could not list Composio accounts for %s: %s", user_id, e)
            return {}
        out: dict[str, Consent] = {}
        for record in records:
            toolkit = record.get("toolkit") or {}
            slug = str(toolkit.get("slug") if isinstance(toolkit, dict) else toolkit or "")
            if not slug:
                continue
            out[slug.upper()] = Consent(
                server=slug.upper(),
                status=_status_of(str(record.get("status") or "")),
                account_id=str(record.get("id") or "") or None,
            )
        return out

    async def refresh_status(self, user_id: str) -> dict[str, Consent]:
        """Read every connector's state at once and store what came back."""
        live = await self._accounts(user_id)
        statuses = {
            spec["server"]: (live.get(spec["server"]).status if live.get(spec["server"]) else conns.PENDING)
            for spec in self.servers.values()
            if spec.get("server")
        }
        await conns.sync(user_id, statuses, {s: (live[s].account_id if s in live else None) for s in statuses})
        return live

    async def connections(self, user_id: str, *, refresh: bool = True) -> list[dict[str, Any]]:
        """Every configured connector and this user's standing with it.

        `refresh=True` re-reads Composio and syncs the rows (the settings panel);
        `refresh=False` answers from stored rows and the cached catalogue, no vendor call.
        """
        if refresh:
            live = await self.refresh_status(user_id)
            statuses = {s: c.status for s, c in live.items()}
            accounts = {s: c.account_id for s, c in live.items()}
        else:
            stored = await conns.load(user_id)
            statuses = {s: row.status for s, row in stored.items()}
            accounts = {s: row.connected_account_id for s, row in stored.items()}

        grouped = await self._tools_or_empty(user_id)
        rows: list[dict[str, Any]] = []
        for label, spec in self.servers.items():
            server = spec.get("server")
            if not server:
                continue
            status = statuses.get(server, conns.PENDING)
            rows.append(
                {
                    "server": server,
                    "label": label,
                    "name": spec.get("name", label),
                    "status": status,
                    "tool_count": len(grouped.get(server, [])),
                    "setup_url": self._setup_urls.get((user_id, server)),
                    "account_id": accounts.get(server),
                    # Always empty: grants are per toolkit. The key stays so the
                    # panel reads one shape whatever the backend.
                    "shares_with": [],
                }
            )
        return rows

    async def connect(self, user_id: str, server: str) -> dict[str, Any]:
        """Start one toolkit's consent from the panel, and record what came back."""
        consent = await self.consent(user_id, server)
        await conns.mark(user_id, server, consent.status, consent.account_id)
        if consent.setup_url:
            self._setup_urls[(user_id, server)] = consent.setup_url
        else:
            self._setup_urls.pop((user_id, server), None)
        return {"server": server, "status": consent.status, "setup_url": consent.setup_url}

    async def disconnect(self, user_id: str, server: str) -> list[str]:
        """Delete the connected account at Composio and drop our row.

        Returns exactly one server: a connected account is per toolkit. The list shape is
        the panel's contract, for backends whose grants are shared.
        """
        live = await self._accounts(user_id)
        found = live.get(server)
        if found and found.account_id:
            try:
                await self.client.delete_connected_account(found.account_id)
            except ComposioError as e:
                logger.warning("could not delete Composio account %s: %s", found.account_id, e)
        await conns.forget(user_id, server)
        self._setup_urls.pop((user_id, server), None)
        self.invalidate(user_id)
        return [server]

    async def reconcile(self, user_id: str, account_id: str) -> str | None:
        """Settle one connection from the callback's `connected_account_id`.

        The callback carries no toolkit name, so the toolkit is looked up by account id.
        """
        live = await self._accounts(user_id)
        for server, consent in live.items():
            if consent.account_id == account_id and self.is_connector(server):
                await conns.mark(user_id, server, consent.status, account_id)
                self._setup_urls.pop((user_id, server), None)
                self.invalidate(user_id)
                return server
        return None

    async def close(self) -> None:
        await self.client.close()


def _status_of(raw: str) -> str:
    """Composio's account status -> ours."""
    lowered = raw.strip().upper()
    if lowered == "ACTIVE":
        return CONNECTED
    if lowered in ("INITIATED", "INITIALIZING", "PENDING"):
        return conns.PENDING
    if lowered in ("ERRORED", "EXPIRED", "FAILED"):
        return conns.RECONNECT
    return conns.PENDING


def _is_unconnected(text: str) -> bool:
    """Whether an `isError` body is Composio saying this user has no grant."""
    lowered = text.lower()
    return "no connected account" in lowered or "not connected" in lowered


def _auto_approved(name: str, setting: Any) -> bool:
    if setting is True:
        return True
    if isinstance(setting, (list, tuple, set)):
        return name in setting
    return False


def _to_spec(tool: dict[str, Any], auto_approve: Any = None, *, readonly: bool = False) -> ToolSpec:
    """Convert one `tools/list` entry into a ToolSpec.

    A remote server never reports whether a tool mutates, so nothing is readonly and
    everything requires approval unless config waives it per toolkit.
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
    """Flatten a `tools/call` result to (text, is_error).

    The block's `text` is itself a JSON string whose object spells success `successfull`
    (three l's) or `successful`; a payload saying it failed is an error without `isError`.
    """
    if not isinstance(result, dict):
        return (result if isinstance(result, str) else json.dumps(result, default=str)), False

    is_error = bool(result.get("isError"))
    blocks = result.get("content")
    if not isinstance(blocks, list):
        return json.dumps(result, default=str), is_error

    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        if block.get("type") != "text":
            parts.append(json.dumps(block, default=str))
            continue
        text = block.get("text", "")
        parts.append(text)
        try:
            inner = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(inner, dict):
            said = inner.get("successfull", inner.get("successful"))
            if said is False:
                is_error = True
    return "\n".join(p for p in parts if p), is_error


async def _envelope(name: str, result: Any, ctx: ToolContext) -> ResultEnvelope:
    """Wrap the result, storing the tail as a blob when it is too big to inline."""
    text, is_error = _render(result)
    if is_error:
        return fail("upstream_error", text or f"{name} reported an error with no detail.")

    head, total = cap_view(text)
    if total is None or ctx.store_blob is None:
        return ok(text)

    ref = await ctx.store_blob(text)
    return ok(
        f"{head}\n\n[truncated at {len(head)} of {total} chars. Read the rest with read_result(ref={ref!r})]",
        ref=ref,
    )
