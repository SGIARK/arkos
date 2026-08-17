"""
The process-wide MCP transport.

Nothing in production constructed a `Smithery` before this: Task 3 kept the
business logic and Task 7 deleted every route that reached it. One instance is
built at startup and shared, because its point is the in-process connection
cache — a second instance would mean a second cache and a second set of
Smithery PUTs.
"""

from __future__ import annotations

import logging

from config_module.loader import config
from tool_module.smithery import Smithery

logger = logging.getLogger(__name__)

_smithery: Smithery | None = None


def smithery() -> Smithery | None:
    """Return the shared client, or None when MCP is not configured."""
    return _smithery


async def start() -> Smithery | None:
    """
    Build the shared client and bring the no-auth servers up.

    A missing api key is not fatal: MCP is one hand among several, and refusing
    to serve chat because Smithery is unconfigured would be the tail wagging the
    dog. The manifest simply ships without MCP tools.
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
        # Shared servers carry a workspace credential, not a user's, so they come
        # up once here rather than on first use by whoever happens to be first.
        await _smithery.initialize_shared()
    except Exception:
        logger.exception("smithery: shared servers did not come up; per-user ones are unaffected")
    return _smithery


async def stop() -> None:
    """Close the shared client's HTTP session."""
    global _smithery
    if _smithery is not None:
        await _smithery.close()
        _smithery = None
