"""
Tool discovery. Adding a tool is adding a module under `tool_module/tools/`
that exposes a `TOOLS` list; nothing else needs editing.

MCP tools are namespaced `mcp_*` and the prefix is stripped on dispatch, so a
remote `read_file` cannot shadow ours.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import pkgutil
from collections.abc import Awaitable, Callable
from typing import Any

from config_module.loader import config
from tool_module import tools as tools_package
from tool_module.envelope import ResultEnvelope, Tool, ToolContext, ToolSpec, execute, fail

McpCall = Callable[[str, dict[str, Any], ToolContext], Awaitable[ResultEnvelope]]


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


logger = logging.getLogger(__name__)

MCP_PREFIX = "mcp_"

_local: dict[str, Tool] | None = None


def local_tools() -> dict[str, Tool]:
    """Ours, discovered once. Always loaded: an unused tool costs a schema, not a boot."""
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
    """Drop the discovery cache. For tests."""
    global _local
    _local = None


def manifest(mcp_specs: list[ToolSpec] | None = None) -> list[ToolSpec]:
    """
    The whole manifest for a session.

    One manifest: there are no session types, so every session may reach for
    every hand. An MCP tool whose name collides with ours keeps its prefix and
    therefore cannot displace it.
    """
    # Copies, always. ToolSpec is mutable and the local ones are process-cached,
    # so handing out references lets one session's edit poison every session.
    specs = [_copy(t.spec) for t in local_tools().values()]
    taken = {s.name for s in specs}

    for spec in mcp_specs or []:
        # Prefix unconditionally. A remote tool genuinely called mcp_foo would
        # otherwise be dispatched as foo, a name its server does not have.
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
    """
    Run one tool by the name the model used.

    `mcp_call(bare_name, args, ctx) -> ResultEnvelope` handles the mcp_* half.
    Injected rather than imported so the registry does not depend on transport.
    """
    if name.startswith(MCP_PREFIX):
        if mcp_call is None:
            return fail("not_found", f"No MCP transport available for {name!r}.")
        bare = name[len(MCP_PREFIX) :]
        try:
            # The remote tools are the ones that actually hang, so they need the
            # cap more than the local ones do.
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
    """
    Adapt `dispatch` to the `(name, args)` shape `run_turn` requires.

    The loop must not know about contexts or transports, so the session binds
    them here once and hands the loop a two-argument function.
    """
    cap = float(_cfg("tools.call_timeout_s", 120.0)) if timeout_s is None else timeout_s

    async def dispatch_bound(name: str, args: dict[str, Any]) -> ResultEnvelope:
        return await dispatch(name, args, ctx, mcp_call=mcp_call, timeout_s=cap)

    return dispatch_bound
