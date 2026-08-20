"""Tool discovery and dispatch, and the one place a turn's tool list is built.

A local tool is a module under `tool_module/tools/` exposing a `TOOLS` list.
MCP tools are namespaced `mcp_*`, and the prefix is stripped on dispatch, so a
remote tool cannot shadow a local name.

`manifest` is the ONLY builder of a turn's tool list, and it cannot overflow.
The provider refuses a request carrying more than `llm.max_tools` schemas
outright — no token is generated, and the error arrives nowhere near the
connection that caused it — so the cap is applied here, to whatever the toggles
say, rather than trusted to whoever wrote them. Ours are always loaded and never
counted against the human's allowance; MCP servers are the only thing deferred,
whole servers at a time, most-recently-enabled first.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from config_module.loader import config
from tool_module import session_tools
from tool_module import tools as tools_package
from tool_module.envelope import ResultEnvelope, Tool, ToolContext, ToolSpec, execute, fail

McpCall = Callable[[str, dict[str, Any], ToolContext], Awaitable[ResultEnvelope]]


class ServerGroup(Protocol):
    """One connected server and its tools, as `manifest` reads them."""

    label: str
    name: str
    mcp_url: str
    specs: list[ToolSpec]


class McpSource(Protocol):
    """What `manifest` needs from the MCP half; `Smithery` satisfies it."""

    async def reach(self, user_id: str) -> list[ServerGroup]: ...


@dataclass(frozen=True, slots=True)
class ServerReach:
    """One server's standing in this turn's manifest.

    Three states, and the prompt says a different thing about each: shipped
    (the model may call it), enabled-but-benched (it was asked for and did not
    fit), and connected-but-off (the human has it, this session does not).
    """

    label: str
    name: str
    mcp_url: str
    tools: int
    enabled: bool
    shipped: bool

    @property
    def benched(self) -> bool:
        """Asked for by the session and left out anyway, to stay under the cap."""
        return self.enabled and not self.shipped


@dataclass(frozen=True, slots=True)
class Manifest:
    """A turn's tool list, and the account of how it came to be that list.

    `specs` is what goes in the request. `servers` is what the system prompt is
    generated from — never the toggles, which can promise a server the cap then
    dropped. Anything reading one without the other will drift.
    """

    specs: list[ToolSpec] = field(default_factory=list)
    servers: list[ServerReach] = field(default_factory=list)
    ours: int = 0
    budget: int = 0
    used: int = 0

    @property
    def benched(self) -> list[ServerReach]:
        return [s for s in self.servers if s.benched]


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


logger = logging.getLogger(__name__)

MCP_PREFIX = "mcp_"

_local: dict[str, Tool] | None = None


def local_tools() -> dict[str, Tool]:
    """Return our own tools, discovered once per process."""
    global _local
    if _local is None:
        found: dict[str, Tool] = {}
        for module_info in pkgutil.iter_modules(tools_package.__path__):
            module = importlib.import_module(f"{tools_package.__name__}.{module_info.name}")
            for tool in getattr(module, "TOOLS", []):
                if tool.spec.name in found:
                    raise RuntimeError(f"duplicate tool name {tool.spec.name!r} in {module_info.name}")
                found[tool.spec.name] = tool
        _local = found
    return _local


def reset() -> None:
    """Drop the discovery cache, for tests."""
    global _local
    _local = None


async def manifest(user_id: str, *, mcp: McpSource | None = None, session_id: str | None = None) -> Manifest:
    """Build the tool list for one turn, and report what it cost to fit.

    Ours are always loaded. An MCP server is reached only when this session has
    been given it — `session_id=None` means no session, so no toggles, so ours
    alone. That default is what makes an accidental over-cap request impossible
    rather than unlikely: a connected server is not a reachable one.

    The cap is then applied to the toggles rather than trusted to them. Servers
    are taken longest-enabled first and the first one that will not fit ends the
    list: a stale toggle set, or a server that doubled its tool list overnight,
    cannot produce a request the provider will reject. Whole servers only —
    shipping half of a server is a model that believes it can post a message and
    discovers otherwise in the middle of a task.
    """
    # ToolSpec is mutable and the local ones are process-cached, so every caller
    # gets its own copy.
    ours = [_copy(t.spec) for t in local_tools().values()]
    budget = max(0, int(_cfg("llm.max_tools", 128)) - len(ours))

    connected = await mcp.reach(user_id) if mcp is not None else []
    enabled = await session_tools.enabled_urls(session_id) if session_id else []
    rank = {url: i for i, url in enumerate(enabled)}

    specs = list(ours)
    taken = {s.name for s in specs}
    shipped: set[str] = set()
    used = 0

    for server in sorted((s for s in connected if s.mcp_url in rank), key=lambda s: rank[s.mcp_url]):
        if used + len(server.specs) > budget:
            # Everything from here on is benched: stopping rather than skipping
            # is what makes "most recently enabled goes first" true. Taking a
            # later, smaller server would keep it while an older one was cut.
            logger.warning(
                "session %s: %s (%d tools) does not fit in %d remaining tool slot(s); benched",
                session_id,
                server.label,
                len(server.specs),
                budget - used,
            )
            break
        for spec in server.specs:
            # The prefix is added unconditionally, including to a remote tool
            # whose own name already starts with it.
            name = f"{MCP_PREFIX}{spec.name}"
            if name in taken:
                logger.warning("dropping MCP tool %s: name collides with %s", spec.name, name)
                continue
            taken.add(name)
            specs.append(_copy(spec, name=name))
        shipped.add(server.mcp_url)
        used += len(server.specs)

    servers = [
        ServerReach(
            label=s.label,
            name=s.name,
            mcp_url=s.mcp_url,
            tools=len(s.specs),
            enabled=s.mcp_url in rank,
            shipped=s.mcp_url in shipped,
        )
        for s in connected
    ]
    return Manifest(specs=specs, servers=servers, ours=len(ours), budget=budget, used=used)


def _copy(spec: ToolSpec, *, name: str | None = None) -> ToolSpec:
    return ToolSpec(
        name=name or spec.name,
        description=spec.description,
        input_schema=dict(spec.input_schema),
        readonly=spec.readonly,
        requires_approval=spec.requires_approval,
    )


class _McpTool:
    """Adapts one remote tool to the `Tool` protocol, so it runs through `execute`."""

    def __init__(self, spec: ToolSpec, bare_name: str, mcp_call: McpCall):
        self.spec = spec
        self._bare = bare_name
        self._call = mcp_call

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        return await self._call(self._bare, args, ctx)


def _mcp_spec(name: str, specs: dict[str, ToolSpec] | None) -> ToolSpec:
    """Return the manifest spec for an mcp_* name, or a conservative stand-in."""
    known = (specs or {}).get(name)
    if known is not None:
        return known
    # An unrecognised remote tool is neither readonly nor pre-approved.
    return ToolSpec(name=name, readonly=False, requires_approval=True)


async def dispatch(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    mcp_call: McpCall | None = None,
    specs: dict[str, ToolSpec] | None = None,
    timeout_s: float = 120.0,
) -> ResultEnvelope:
    """Run one tool by the name the model used.

    Local and MCP tools both go through `envelope.execute`, the single place the
    approval gate, the schema check and the timeout are applied; an mcp_* name is
    wrapped in an adapter to get there.
    """
    if name.startswith(MCP_PREFIX):
        if mcp_call is None:
            return fail("not_found", f"No MCP transport available for {name!r}.")
        tool = _McpTool(_mcp_spec(name, specs), name[len(MCP_PREFIX) :], mcp_call)
        return await execute(name, args, ctx, lookup=lambda _name: tool, timeout_s=timeout_s)

    return await execute(name, args, ctx, lookup=local_tools().get, timeout_s=timeout_s)


def bind(
    ctx: ToolContext,
    *,
    mcp_call: McpCall | None = None,
    tools: Sequence[ToolSpec] | None = None,
    timeout_s: float | None = None,
) -> Callable[[str, dict[str, Any]], Awaitable[ResultEnvelope]]:
    """Adapt `dispatch` to the `(name, args)` shape `run_turn` requires.

    `tools` is the manifest this turn was built with. It carries each remote
    tool's `requires_approval`, which the gate in `execute` reads.
    """
    cap = float(_cfg("tools.call_timeout_s", 120.0)) if timeout_s is None else timeout_s
    specs = {t.name: t for t in tools} if tools else None

    async def dispatch_bound(name: str, args: dict[str, Any]) -> ResultEnvelope:
        return await dispatch(name, args, ctx, mcp_call=mcp_call, specs=specs, timeout_s=cap)

    return dispatch_bound
