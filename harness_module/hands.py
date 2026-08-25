"""The process-wide MCP transport.

One `Arcade` client is built at startup and shared by every caller; it owns the
per-user gateway sessions and the cached tool roster.
"""

from __future__ import annotations

import logging

from config_module.loader import config
from tool_module.arcade import Arcade

logger = logging.getLogger(__name__)

_arcade: Arcade | None = None


def arcade() -> Arcade | None:
    """Returns the shared client, or None when MCP is not configured."""
    return _arcade


async def start() -> Arcade | None:
    """Builds the shared client.

    Returns None when `mcp.api_key` is unset; the manifest then ships without MCP
    tools, and without Google Search, which rides the same wire.

    Nothing is connected here. Every grant is per user and lives at Arcade, and
    the gateway's tool roster is read on first use per user rather than at boot —
    so a gateway that is down delays one user's first turn instead of holding up
    the whole process starting.
    """
    global _arcade
    if _arcade is not None:
        return _arcade

    mcp_config = config.get("mcp") or {}
    if not mcp_config.get("api_key"):
        logger.warning("mcp.api_key is unset: MCP tools will not be offered")
        return None

    _arcade = Arcade(config.get("mcp_servers") or {}, mcp_config)
    return _arcade


async def stop() -> None:
    """Closes the shared client's HTTP session and clears it."""
    global _arcade
    if _arcade is not None:
        await _arcade.close()
        _arcade = None
