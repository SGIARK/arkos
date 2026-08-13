"""
Tool discovery and dispatch. A tool is a module under `tool_module/tools/`
exposing a `TOOLS` list. MCP tools are namespaced `mcp_*` and the prefix is
stripped on dispatch, so a remote `read_file` cannot shadow ours.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from config_module.loader import config
from tool_module import tools as tools_package
from tool_module.envelope import ResultEnvelope, Tool, ToolContext, ToolSpec, execute, fail

McpCall = Callable[[str, dict[str, Any], ToolContext], Awaitable[ResultEnvelope]]


class McpSource(Protocol):
    """What `manifest` needs from the MCP half; `Smithery` satisfies it."""

    async def specs(self, user_id: str) -> list[ToolSpec]: ...


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


async def manifest(user_id: str, *, mcp: McpSource | None = None) -> list[ToolSpec]:
    """Return every tool spec a session may use; without `mcp`, only the local ones."""
    # Copies, always: ToolSpec is mutable and the local ones are process-cached,
    # so sharing references lets one session's edit poison every session.
    specs = [_copy(t.spec) for t in local_tools().values()]
    taken = {s.name for s in specs}

    for spec in await mcp.specs(user_id) if mcp is not None else []:
        # Prefix unconditionally: a remote tool genuinely called mcp_foo would
        # otherwise dispatch as foo, a name its server does not have.
        name = f"{MCP_PREFIX}{spec.name}"
        if name in taken:
            logger.warning("dropping MCP tool %s: name collides with %s", spec.name, name)
            continue
        taken.add(name)
        specs.append(_copy(spec, name=name))
    return specs


def _copy(spec: ToolSpec, *, name: str | None = None) -> ToolSpec:
    return ToolSpec(
        name=name or spec.name,
        description=spec.description,
        input_schema=dict(spec.input_schema),
        readonly=spec.readonly,
        requires_approval=spec.requires_approval,
    )


async def dispatch(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    mcp_call: McpCall | None = None,
    timeout_s: float = 120.0,
) -> ResultEnvelope:
    """Run one tool by the name the model used; `mcp_call` handles the mcp_* half."""
    if name.startswith(MCP_PREFIX):
        if mcp_call is None:
            return fail("not_found", f"No MCP transport available for {name!r}.")
        bare = name[len(MCP_PREFIX) :]
        try:
            async with asyncio.timeout(timeout_s):
                result = await mcp_call(bare, args, ctx)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return fail("timeout", f"{name} did not finish within {timeout_s:.0f}s. It may have taken effect.")
        except Exception as e:
            logger.exception("mcp dispatch failed for %s", name)
            return fail("upstream_error", f"{name} failed: {type(e).__name__}: {e}")
        if not isinstance(result, ResultEnvelope):
            return fail("upstream_error", f"{name} returned a malformed result.", retryable=False)
        return result

    return await execute(name, args, ctx, lookup=local_tools().get, timeout_s=timeout_s)


def bind(
    ctx: ToolContext,
    *,
    mcp_call: McpCall | None = None,
    timeout_s: float | None = None,
) -> Callable[[str, dict[str, Any]], Awaitable[ResultEnvelope]]:
    """Adapt `dispatch` to the `(name, args)` shape `run_turn` requires."""
    cap = float(_cfg("tools.call_timeout_s", 120.0)) if timeout_s is None else timeout_s

    async def dispatch_bound(name: str, args: dict[str, Any]) -> ResultEnvelope:
        return await dispatch(name, args, ctx, mcp_call=mcp_call, timeout_s=cap)

    return dispatch_bound
