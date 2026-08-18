"""
The per-session e2b sandbox: one box per session, capped per user. The only
module that touches the e2b SDK.

The box follows the session and is destroyed once its flush lands: the disk
holds a cache of the store (D27), so nothing is lost with it. `session_sandboxes`
is the handle table and the pool at once — a row is a slot, and a user may hold
`sandbox.max_concurrent_per_user` of them.

The SDK is imported inside `_create` and `_connect`, so the process starts and
the manifest builds without e2b installed.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from config_module.loader import config
from db import pool

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 300
_DEFAULT_MAX_PER_USER = 3


def _timeout() -> int:
    return int(config.get("sandbox.timeout_seconds") or _DEFAULT_TIMEOUT)


def max_per_user() -> int:
    """How many boxes one user may run at once."""
    return int(config.get("sandbox.max_concurrent_per_user") or _DEFAULT_MAX_PER_USER)


def _template() -> str | None:
    """The e2b template name, or None for the SDK default."""
    name = config.get("sandbox.template")
    return name if name and name != "base" else None


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


async def claim_slot(session_id: str) -> bool:
    """
    Take this session's slot in its user's sandbox pool.

    The row is written before the box boots and dropped when it is reaped, so a
    slot is held for exactly as long as a box can exist. A session that already
    holds one keeps it: the count excludes the caller, so a repeat call renews
    rather than competes with itself.

    Returns:
        False if the user is at `sandbox.max_concurrent_per_user`. The caller
        waits and asks again.
    """
    sid = _uuid(session_id)
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        user_id = await conn.fetchval("SELECT user_id FROM sessions WHERE id = $1", sid)
        if user_id is None:
            raise ValueError(f"session {session_id} does not exist")

        # Serializes this count-then-insert against another session of the same
        # user doing the same thing; released when the transaction ends.
        await conn.execute("SELECT pg_advisory_xact_lock(hashtext($1), 0)", str(user_id))
        held = await conn.fetchval(
            "SELECT count(*) FROM session_sandboxes WHERE user_id = $1 AND session_id <> $2",
            user_id,
            sid,
        )
        if held >= max_per_user():
            return False

        await conn.execute(
            """
            INSERT INTO session_sandboxes (session_id, user_id)
            VALUES ($1, $2)
            ON CONFLICT (session_id) DO UPDATE SET last_used_at = now()
            """,
            sid,
            user_id,
        )
        return True


async def _stored_id(session_id: str) -> str | None:
    return await pool.fetchval("SELECT sandbox_id FROM session_sandboxes WHERE session_id = $1", _uuid(session_id))


async def _remember(session_id: str, sandbox_id: str) -> None:
    """Record the handle against the session's slot, taking one if it has none."""
    await pool.execute(
        """
        INSERT INTO session_sandboxes (session_id, user_id, sandbox_id, last_used_at)
        SELECT s.id, s.user_id, $2, now() FROM sessions s WHERE s.id = $1
        ON CONFLICT (session_id)
        DO UPDATE SET sandbox_id = EXCLUDED.sandbox_id, last_used_at = now()
        """,
        _uuid(session_id),
        sandbox_id,
    )


async def _touch(session_id: str) -> None:
    await pool.execute("UPDATE session_sandboxes SET last_used_at = now() WHERE session_id = $1", _uuid(session_id))


async def release_slot(session_id: str) -> str | None:
    """Free the session's slot, returning the handle it named so the box can still be killed."""
    return await pool.fetchval(
        "DELETE FROM session_sandboxes WHERE session_id = $1 RETURNING sandbox_id", _uuid(session_id)
    )


class SandboxManager:
    """One sandbox per session, wrapping the synchronous e2b SDK in threads."""

    def __init__(self) -> None:
        self._live: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, session_id: str) -> asyncio.Lock:
        lock = self._locks.get(session_id)
        if lock is None:
            lock = self._locks[session_id] = asyncio.Lock()
        return lock

    async def get_or_create(self, session_id: str) -> Any:
        """Return the session's sandbox: the cached handle, the stored one resumed, or a new one."""
        async with self._lock(session_id):
            cached = self._live.get(session_id)
            if cached is not None:
                try:
                    # Validates the handle and resets the idle timer in one call.
                    await asyncio.to_thread(cached.set_timeout, _timeout())
                    return cached
                except Exception as e:  # noqa: BLE001 - e2b raises its own types
                    logger.warning("cached sandbox for session %s is gone (%s); recreating", session_id, e)
                    self._live.pop(session_id, None)

            stored = await _stored_id(session_id)
            if stored:
                resumed = await self._resume(session_id, stored)
                if resumed is not None:
                    return resumed

            sandbox = await asyncio.to_thread(self._create)
            self._live[session_id] = sandbox
            await _remember(session_id, sandbox.sandbox_id)
            logger.info("created sandbox %s for session %s", sandbox.sandbox_id, session_id)
            return sandbox

    async def _resume(self, session_id: str, sandbox_id: str) -> Any | None:
        """Reconnect to a stored sandbox, or return None if it is gone.

        A process that restarted mid-session left its box running; the row
        outlived the handle and says where it is.
        """
        sandbox = await self._connect(sandbox_id)
        if sandbox is None:
            logger.warning("resume failed for session %s; creating a new sandbox", session_id)
            return None
        try:
            await asyncio.to_thread(sandbox.set_timeout, _timeout())
        except Exception as e:  # noqa: BLE001 - e2b raises its own types
            logger.warning("resumed sandbox %s is not usable (%s)", sandbox_id, e)
            return None
        self._live[session_id] = sandbox
        await _touch(session_id)
        logger.info("resumed sandbox %s for session %s", sandbox_id, session_id)
        return sandbox

    async def _connect(self, sandbox_id: str) -> Any | None:
        """Reconnect to a running sandbox by id, or None if it is gone."""
        from e2b_code_interpreter import Sandbox

        try:
            return await asyncio.to_thread(Sandbox.connect, sandbox_id)
        except Exception as e:  # noqa: BLE001 - e2b raises its own types
            logger.warning("could not connect to sandbox %s (%s)", sandbox_id, e)
            return None

    def _create(self) -> Any:
        """Create a sandbox. No environment is passed: credentials stay out of it."""
        from e2b_code_interpreter import Sandbox

        template = _template()
        if template:
            return Sandbox.create(template=template, timeout=_timeout())
        return Sandbox.create(timeout=_timeout())

    async def exec(self, session_id: str, command: str, timeout: int = 120) -> dict[str, Any]:
        """Run a shell command. Returns stdout, stderr and exit_code, including on a non-zero exit."""
        sandbox = await self.get_or_create(session_id)

        def _run() -> dict[str, Any]:
            try:
                result = sandbox.commands.run(command, timeout=timeout)
                return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code}
            except Exception as e:  # noqa: BLE001 - e2b raises on a non-zero exit, carrying the streams
                return {
                    "stdout": getattr(e, "stdout", ""),
                    "stderr": getattr(e, "stderr", str(e)),
                    "exit_code": getattr(e, "exit_code", 1),
                }

        return await asyncio.to_thread(_run)

    async def read_file(self, session_id: str, path: str) -> str:
        sandbox = await self.get_or_create(session_id)
        return await asyncio.to_thread(sandbox.files.read, path)

    async def read_bytes(self, session_id: str, path: str) -> bytes:
        """Read a file as bytes, which a tar transfer out of the sandbox needs."""
        sandbox = await self.get_or_create(session_id)
        return await asyncio.to_thread(lambda: sandbox.files.read(path, format="bytes"))

    async def write_file(self, session_id: str, path: str, content: str | bytes) -> None:
        """Write a file. Bytes go through unchanged, which is what a tar transfer needs."""
        sandbox = await self.get_or_create(session_id)
        await asyncio.to_thread(sandbox.files.write, path, content)

    async def list_dir(self, session_id: str, path: str = "/home/user") -> list[dict[str, Any]]:
        """List a directory as [{name, path, is_dir, size}]."""
        sandbox = await self.get_or_create(session_id)
        entries = await asyncio.to_thread(sandbox.files.list, path)

        def is_dir(entry: Any) -> bool:
            kind = getattr(entry, "type", None)
            return getattr(kind, "value", str(kind)).lower() == "dir"

        return [
            {"name": e.name, "path": e.path, "is_dir": is_dir(e), "size": getattr(e, "size", 0)} for e in entries
        ]

    async def reap(self, session_id: str) -> None:
        """Destroy the session's box and free its slot.

        The slot goes first and carries the handle out with it, so a kill that
        fails leaves a box billing rather than a slot no session can use. The
        caller flushes before it reaps: what is on the disk is a cache of a
        commit that has already landed.
        """
        sandbox = self._live.pop(session_id, None)
        self._locks.pop(session_id, None)
        sandbox_id = await release_slot(session_id)
        if sandbox is None and not sandbox_id:
            return

        try:
            if sandbox is not None:
                await asyncio.to_thread(sandbox.kill)
            else:
                # No handle in this process: the SDK kills by id without a connect.
                from e2b_code_interpreter import Sandbox

                await asyncio.to_thread(Sandbox.kill, sandbox_id)
        except Exception as e:  # noqa: BLE001 - e2b raises its own types
            logger.warning(
                "could not kill sandbox %s for session %s (%s); it bills until its idle timeout",
                sandbox_id,
                session_id,
                e,
            )
            return
        logger.info("reaped sandbox %s for session %s", sandbox_id, session_id)


_manager: SandboxManager | None = None


def manager() -> SandboxManager:
    """Return the process-wide manager, so one session has one sandbox per process."""
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager


def reset() -> None:
    """Drop the manager and its cached handles. For tests."""
    global _manager
    _manager = None
