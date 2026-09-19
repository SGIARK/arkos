"""Which MCP servers one session may reach, stored per session and keyed by `server`.

An absent row reads as off. `server` is the Composio toolkit prefix (`GMAIL`),
never an `mcp_servers:` config label or an mcp url — neither is durable.
"""

from __future__ import annotations

import uuid

from db import pool


def _sid(session_id: str) -> uuid.UUID:
    """Parse a session id as the UUID it must be, raising ValueError otherwise."""
    try:
        return uuid.UUID(str(session_id))
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"session_id must be a UUID, got {session_id!r}") from e


async def enabled_servers(session_id: str) -> list[str]:
    """Return the servers this session has been given, LONGEST-ENABLED FIRST.

    The caller fills up to the cap from the front, so this order IS the drop rule.
    """
    rows = await pool.fetch(
        """
        SELECT server FROM session_tools
         WHERE session_id = $1 AND enabled
         ORDER BY updated_at, server
        """,
        _sid(session_id),
    )
    return [r["server"] for r in rows]


async def set_enabled(session_id: str, server: str, enabled: bool) -> None:
    """Record one server as reachable, or not, for this session.

    `updated_at` moves on every write, including a re-assert of a toggle already
    on; that is the recency the drop rule reads.
    """
    await pool.execute(
        """
        INSERT INTO session_tools (session_id, server, enabled)
        VALUES ($1, $2, $3)
        ON CONFLICT (session_id, server)
        DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = now()
        """,
        _sid(session_id),
        server,
        enabled,
    )
