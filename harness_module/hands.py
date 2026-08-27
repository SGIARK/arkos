"""The process-wide MCP transport: one shared `Composio` client."""

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
    """Builds the shared client, or returns None when MCP is not configured.

    Connects nothing: grants are per user at Composio and the tool catalogue is
    read on that user's first use, not at boot.
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
