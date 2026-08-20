"""
Which MCP servers one session may reach, stored per session and keyed by `mcp_url`.

The default is nothing: a server the user has connected is not a server this
session can reach until it is toggled on. An absent row reads as off, so a fresh
session gets exactly our own tools and cannot be the one that puts 164 schemas
in a request.

This module records and reports the toggles and nothing else. Task 11.5 is where
the manifest, the prompt and the loop start OBEYING them; until then a toggle is
honest about state and silent about effect, which is the card boundary drawn on
purpose.
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


async def enabled_urls(session_id: str) -> list[str]:
    """Return the `mcp_url`s this session has been given, LONGEST-ENABLED FIRST.

    The order is load-bearing, not cosmetic. When the cap forces a server to be
    left out, the most recently enabled one goes first — the session keeps the
    reach it has been working with, and loses the thing that was just added. The
    caller reads this list front to back and stops when the next server will not
    fit, so the order here IS the drop rule.
    """
    rows = await pool.fetch(
        """
        SELECT mcp_url FROM session_tools
         WHERE session_id = $1 AND enabled
         ORDER BY updated_at, mcp_url
        """,
        _sid(session_id),
    )
    return [r["mcp_url"] for r in rows]


async def set_enabled(session_id: str, mcp_url: str, enabled: bool) -> None:
    """Record one server as reachable, or not, for this session.

    `updated_at` moves on every write, including one that re-asserts a toggle
    already on, which is what makes it the recency the drop rule reads.
    """
    await pool.execute(
        """
        INSERT INTO session_tools (session_id, mcp_url, enabled)
        VALUES ($1, $2, $3)
        ON CONFLICT (session_id, mcp_url)
        DO UPDATE SET enabled = EXCLUDED.enabled, updated_at = now()
        """,
        _sid(session_id),
        mcp_url,
        enabled,
    )
