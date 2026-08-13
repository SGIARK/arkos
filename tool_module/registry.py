"""
Tool discovery. Adding a tool is adding a module under `tool_module/tools/`
that exposes a `TOOLS` list; nothing else needs editing.

MCP tools are namespaced `mcp_*` and the prefix is stripped on dispatch, so a
remote `read_file` cannot shadow ours.
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any

from tool_module import tools as tools_package
from tool_module.envelope import ResultEnvelope, Tool, ToolContext, ToolSpec, execute, fail

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
    specs = [t.spec for t in local_tools().values()]
    for spec in mcp_specs or []:
        if not spec.name.startswith(MCP_PREFIX):
            spec = ToolSpec(
                name=f"{MCP_PREFIX}{spec.name}",
                description=spec.description,
                input_schema=spec.input_schema,
                readonly=spec.readonly,
                requires_approval=spec.requires_approval,
            )
        specs.append(spec)
    return specs


async def dispatch(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    mcp_call: Any = None,
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
            return await mcp_call(bare, args, ctx)
        except Exception as e:
            logger.exception("mcp dispatch failed for %s", name)
            return fail("upstream_error", f"{name} failed: {type(e).__name__}: {e}")

    return await execute(name, args, ctx, lookup=local_tools().get, timeout_s=timeout_s)
