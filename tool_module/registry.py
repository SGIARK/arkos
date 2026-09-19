"""Tool discovery and dispatch, and the one place a turn's tool list is built.

The provider rejects outright any request carrying more than `llm.max_tools`
schemas, so `manifest` — the only builder of a turn's list — applies that cap
itself rather than trusting the toggles.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from config_module.loader import cfg as _cfg
from tool_module import session_tools
from tool_module import tools as tools_package
from tool_module.envelope import ResultEnvelope, Tool, ToolContext, ToolSpec, execute, fail

McpCall = Callable[[str, dict[str, Any], ToolContext], Awaitable[ResultEnvelope]]


class ServerGroup(Protocol):
    """One connected server and its tools, as `manifest` reads them."""

    label: str
    name: str
    server: str
    specs: list[ToolSpec]


class McpSource(Protocol):
    """What `manifest` needs from the MCP half; `Composio` satisfies it."""

    async def reach(self, user_id: str) -> list[ServerGroup]: ...

    async def always(self, user_id: str) -> list[ToolSpec]: ...


@dataclass(frozen=True, slots=True)
class ServerReach:
    """One server's standing in this turn's manifest."""

    label: str
    name: str
    server: str
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

    `servers` is what the system prompt is generated from — never the toggles,
    which can promise a server the cap then dropped.
    """

    specs: list[ToolSpec] = field(default_factory=list)
    servers: list[ServerReach] = field(default_factory=list)
    ours: int = 0
    budget: int = 0
    used: int = 0

    @property
    def benched(self) -> list[ServerReach]:
        return [s for s in self.servers if s.benched]


logger = logging.getLogger(__name__)

# The prefix is stripped again on dispatch, so a remote tool can never shadow a local name.
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

    Ours are always loaded; `session_id=None` means no toggles, so ours alone.
    MCP servers are admitted whole or not at all, in `enabled_servers` order.
    """
    # ToolSpec is mutable and the local ones are process-cached, so every caller
    # gets its own copy.
    ours = [_copy(t.spec) for t in local_tools().values()]
    # Prefixed like every other gateway tool so dispatch has ONE route to the
    # gateway; still counted in `ours`, so `budget` is what is left after them.
    ours += [_copy(spec, name=f"{MCP_PREFIX}{spec.name}") for spec in (await mcp.always(user_id) if mcp else [])]
    budget = max(0, int(_cfg("llm.max_tools", 128)) - len(ours))

    connected = await mcp.reach(user_id) if mcp is not None else []
    enabled = await session_tools.enabled_servers(session_id) if session_id else []
    rank = {server: i for i, server in enumerate(enabled)}

    specs = list(ours)
    taken = {s.name for s in specs}
    shipped: set[str] = set()
    used = 0

    for server in sorted((s for s in connected if s.server in rank), key=lambda s: rank[s.server]):
        if used + len(server.specs) > budget:
            # Stop, do not skip: taking a later, smaller server would keep it
            # while an earlier-ranked one was cut.
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
        shipped.add(server.server)
        used += len(server.specs)

    servers = [
        ServerReach(
            label=s.label,
            name=s.name,
            server=s.server,
            tools=len(s.specs),
            enabled=s.server in rank,
            shipped=s.server in shipped,
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
    approval gate, the schema check and the timeout are applied.
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

    `tools` is this turn's manifest; it carries each remote tool's
    `requires_approval`, which the gate in `execute` reads.
    """
    cap = float(_cfg("tools.call_timeout_s", 120.0)) if timeout_s is None else timeout_s
    specs = {t.name: t for t in tools} if tools else None

    async def dispatch_bound(name: str, args: dict[str, Any]) -> ResultEnvelope:
        return await dispatch(name, args, ctx, mcp_call=mcp_call, specs=specs, timeout_s=cap)

    return dispatch_bound
