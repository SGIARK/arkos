"""
Stored MCP connections, keyed by `mcp_url` and addressed by a minted `connection_id`.

The row is written BEFORE the Smithery PUT, so a crash in between leaves a pending
row whose id is reused rather than stranding a connection holding a live OAuth grant.
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
    """One stored connection; `tools` is the cached `tools/list` response."""

    mcp_url: str
    connection_id: str
    status: str = PENDING
    tools: list[dict[str, Any]] = field(default_factory=list)
    refreshed_at: datetime | None = None

    @property
    def connected(self) -> bool:
        return self.status == CONNECTED

    def stale(self, ttl_s: float) -> bool:
        """Return True when the cached tool list is old enough to revalidate."""
        if self.refreshed_at is None:
            return True
        return (datetime.now(UTC) - self.refreshed_at).total_seconds() > ttl_s


def mint_id(mcp_url: str) -> str:
    """Mint a fresh connection id: the host slugified for readability, a uuid for uniqueness."""
    host = re.sub(r"^https?://", "", mcp_url).split("/")[0]
    slug = re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")[:40]
    return f"{slug}-{uuid.uuid4().hex[:12]}" if slug else uuid.uuid4().hex


def _uid(user_id: str) -> uuid.UUID:
    """Parse a user_id as the Supabase auth UUID it must be, raising ValueError otherwise."""
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
    """Return every stored connection keyed by `mcp_url`; `user_id=None` loads the shared ones."""
    if user_id is None:
        rows = await pool.fetch(f"SELECT {_COLUMNS} FROM shared_connections")
    else:
        rows = await pool.fetch(
            f"SELECT {_COLUMNS} FROM user_connections WHERE user_id = $1",
            _uid(user_id),
        )
    return {r["mcp_url"]: _row(r) for r in rows}


async def claim(user_id: str | None, mcp_url: str) -> str:
    """Return the id to PUT with, reusing an existing row's id so a retry strands nothing."""
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
    """Record what Smithery came back with; `refreshed_at` is the TTL clock."""
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


async def set_status(user_id: str | None, mcp_url: str, status: str) -> None:
    """
    Change the status and nothing else.

    Unlike `save(tools=None)` this keeps `tools_cache`, which is how a later call
    still resolves a name to this connection.
    """
    if user_id is None:
        await pool.execute(
            "UPDATE shared_connections SET status = $2, refreshed_at = now() WHERE mcp_url = $1",
            mcp_url,
            status,
        )
        return
    await pool.execute(
        "UPDATE user_connections SET status = $3, refreshed_at = now() WHERE user_id = $1 AND mcp_url = $2",
        _uid(user_id),
        mcp_url,
        status,
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
