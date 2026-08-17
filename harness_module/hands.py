"""The process-wide MCP transport.

One `Smithery` client is built at startup and shared by every caller; it owns the
in-process connection cache.
"""

from __future__ import annotations

import logging

from config_module.loader import config
from tool_module.smithery import Smithery

logger = logging.getLogger(__name__)

_smithery: Smithery | None = None


def smithery() -> Smithery | None:
    """Returns the shared client, or None when MCP is not configured."""
    return _smithery


async def start() -> Smithery | None:
    """Builds the shared client and brings the no-auth servers up.

    Returns None when `smithery.api_key` is unset; the manifest then ships without MCP
    tools.
    """
    global _smithery
    if _smithery is not None:
        return _smithery

    smithery_config = config.get("smithery") or {}
    if not smithery_config.get("api_key"):
        logger.warning("smithery.api_key is unset: MCP tools will not be offered")
        return None

    _smithery = Smithery(config.get("mcp_servers") or {}, smithery_config)
    try:
        # Shared servers carry a workspace credential, so they come up once at startup.
        await _smithery.initialize_shared()
    except Exception:
        logger.exception("smithery: shared servers did not come up; per-user ones are unaffected")
    return _smithery


async def stop() -> None:
    """Closes the shared client's HTTP session and clears it."""
    global _smithery
    if _smithery is not None:
        await _smithery.close()
        _smithery = None
