"""
Which folders a session may see, and what each claim serializes on.

Claims are bookkeeping, not transfer: this build has no sandbox, so nothing
here moves bytes. The store is the only place a session's files live.
"""

from __future__ import annotations

from dataclasses import dataclass

from db import pool
from db.ids import as_uuid as _uuid


@dataclass(frozen=True, slots=True)
class Claim:
    """One folder, or part of one, that a session may see."""

    user_id: str
    folder: str
    subpath: str = "/"
    mode: str = "write"

    @property
    def prefix(self) -> str:
        """What this claim covers, as a store path prefix."""
        within = (self.subpath or "/").strip("/")
        return f"{self.folder}/{within}" if within else self.folder


async def claims_for(session_id: str) -> list[Claim]:
    """What a session may see, in the order it was declared.

    Claims are FIXED for a session's life: a folder linked mid-run reaches the
    agent at the next session. With none declared, falls back to a write claim
    on every folder the project links.
    """
    rows = await pool.fetch(
        """
        SELECT c.folder, c.subpath, c.mode, s.user_id
          FROM session_claims c JOIN sessions s ON s.id = c.session_id
         WHERE c.session_id = $1
         ORDER BY c.ord, c.folder, c.subpath
        """,
        _uuid(session_id),
    )
    if not rows:
        rows = await pool.fetch(
            """
            SELECT f.folder, '/' AS subpath, 'write' AS mode, s.user_id
              FROM sessions s JOIN project_folders f ON f.project_id = s.project_id
             WHERE s.id = $1
             ORDER BY f.created_at, f.folder
            """,
            _uuid(session_id),
        )
    return [
        Claim(
            user_id=str(r["user_id"]),
            folder=r["folder"],
            subpath=r["subpath"],
            mode=r["mode"],
        )
        for r in rows
    ]


def lease_key(claim: Claim) -> str | None:
    """The folder lease this claim takes, or None. A read claim takes none.

    Per FOLDER, not per project (11.9): two sessions writing the same folder
    serialize even across projects. This is the only place that rule lives.
    """
    return f"folder:{claim.user_id}:{claim.folder}" if claim.mode == "write" else None


def lease_keys(claims: list[Claim]) -> list[str]:
    """The folder leases a claim set takes."""
    return [key for c in claims if (key := lease_key(c))]
