"""
Stored MCP connections (D24).

A connection is identified by its `mcp_url` and addressed by a `connection_id`
we mint once and store. There is no formula from config key to id, so renaming a
key under `mcp_servers:` touches no row and prompts no reconnect.

The row is written BEFORE the Smithery PUT: mint, INSERT `pending`, PUT, UPDATE
from the response. A crash between mint and PUT leaves a pending row whose id is
reused on the next attempt, so the id survives every failure. Minting after the
PUT would strand the old connection still holding the user's live OAuth grant.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from db import pool

PENDING = "pending"
CONNECTED = "connected"

_COLUMNS = "mcp_url, connection_id, status, tools_cache, refreshed_at"


@dataclass(slots=True)
class Connection:
    """One stored connection. `tools` is the cached `tools/list` response."""

    mcp_url: str
    connection_id: str
    status: str = PENDING
    tools: list[dict[str, Any]] = field(default_factory=list)
    refreshed_at: datetime | None = None

    @property
    def connected(self) -> bool:
        return self.status == CONNECTED

    def stale(self, ttl_s: float) -> bool:
        """True when the cached tool list is old enough to revalidate."""
        if self.refreshed_at is None:
            return True
        return (datetime.now(UTC) - self.refreshed_at).total_seconds() > ttl_s


def mint_id(mcp_url: str) -> str:
    """
    A fresh connection id. Smithery takes whatever path segment we choose.

    The host is slugified in so the id is still recognisable in Smithery's
    dashboard; the uuid half is what makes it unique.
    """
    host = re.sub(r"^https?://", "", mcp_url).split("/")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")[:40]
    return f"{slug}-{uuid.uuid4().hex[:12]}" if slug else uuid.uuid4().hex


def _uid(user_id: str) -> uuid.UUID:
    """user_id is a Supabase auth sub. Anything else is a bug upstream, not here."""
    try:
        return uuid.UUID(user_id)
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"user_id must be a Supabase auth UUID, got {user_id!r}") from e


def _row(record: Any) -> Connection:
    return Connection(
        mcp_url=record["mcp_url"],
        connection_id=record["connection_id"],
        status=record["status"],
        tools=record["tools_cache"] or [],
        refreshed_at=record["refreshed_at"],
    )


async def load(user_id: str | None) -> dict[str, Connection]:
    """
    Every stored connection keyed by `mcp_url`. `user_id=None` loads the shared ones.

    This single query is what makes a restart cost zero Smithery PUTs: a
    connection already `connected` is usable straight from the row.
    """
    if user_id is None:
        rows = await pool.fetch(f"SELECT {_COLUMNS} FROM shared_connections")
    else:
        rows = await pool.fetch(
            f"SELECT {_COLUMNS} FROM user_connections WHERE user_id = $1",
            _uid(user_id),
        )
    return {r["mcp_url"]: _row(r) for r in rows}


async def claim(user_id: str | None, mcp_url: str) -> str:
    """
    The id to PUT with, written down first.

    Returns the existing id when there already is a row, so a retry after a
    crashed connect reuses it rather than stranding the old one.
    """
    new_id = mint_id(mcp_url)
    if user_id is None:
        return await pool.fetchval(
            """
            INSERT INTO shared_connections (mcp_url, connection_id, status)
            VALUES ($1, $2, $3)
            ON CONFLICT (mcp_url) DO UPDATE SET mcp_url = EXCLUDED.mcp_url
            RETURNING connection_id
            """,
            mcp_url,
            new_id,
            PENDING,
        )
    return await pool.fetchval(
        """
        INSERT INTO user_connections (user_id, mcp_url, connection_id, status)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (user_id, mcp_url) DO UPDATE SET mcp_url = EXCLUDED.mcp_url
        RETURNING connection_id
        """,
        _uid(user_id),
        mcp_url,
        new_id,
        PENDING,
    )


async def save(
    user_id: str | None,
    mcp_url: str,
    *,
    status: str,
    tools: list[dict[str, Any]] | None = None,
) -> None:
    """Record what Smithery came back with. `refreshed_at` is the TTL clock."""
    if user_id is None:
        await pool.execute(
            """
            UPDATE shared_connections SET status = $2, tools_cache = $3, refreshed_at = now()
            WHERE mcp_url = $1
            """,
            mcp_url,
            status,
            tools,
        )
        return
    await pool.execute(
        """
        UPDATE user_connections SET status = $3, tools_cache = $4, refreshed_at = now()
        WHERE user_id = $1 AND mcp_url = $2
        """,
        _uid(user_id),
        mcp_url,
        status,
        tools,
    )


async def forget(user_id: str | None, mcp_url: str) -> None:
    """Drop the row so the next connect mints a fresh id and a fresh OAuth flow."""
    if user_id is None:
        await pool.execute("DELETE FROM shared_connections WHERE mcp_url = $1", mcp_url)
        return
    await pool.execute(
        "DELETE FROM user_connections WHERE user_id = $1 AND mcp_url = $2",
        _uid(user_id),
        mcp_url,
    )
