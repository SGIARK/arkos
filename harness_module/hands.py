"""The process-wide MCP transport.

One `Composio` client is built at startup and shared by every caller; it owns
the per-user server urls and the cached tool catalogue. Task 11.10.2 swapped it
in for the gateway that preceded it.
"""

from __future__ import annotations

import logging

from config_module.loader import config
from tool_module.composio_mcp import Composio

logger = logging.getLogger(__name__)

_client: Composio | None = None


def connectors() -> Composio | None:
    """Returns the shared client, or None when MCP is not configured."""
    return _client


async def start() -> Composio | None:
    """Builds the shared client.

    Returns None when `mcp.api_key` is unset; the manifest then ships without MCP
    tools. Google Search no longer rides this wire — it is ours and native since
    11.10.2 — so an unconfigured backend costs connectors and nothing else.

    Nothing is connected here. Every grant is per user and lives at Composio, and
    the tool catalogue is read on first use per user rather than at boot — so a
    backend that is down delays one user's first turn instead of holding up the
    whole process starting.
    """
    global _client
    if _client is not None:
        return _client

    mcp_config = config.get("mcp") or {}
    if not mcp_config.get("api_key"):
        logger.warning("mcp.api_key is unset: MCP tools will not be offered")
        return None
    if not mcp_config.get("server_id"):
        logger.warning("mcp.server_id is unset: MCP tools will not be offered")
        return None

    _client = Composio(config.get("mcp_servers") or {}, mcp_config)
    return _client


async def stop() -> None:
    """Closes the shared client's HTTP session and clears it."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
