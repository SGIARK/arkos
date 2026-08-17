"""
The per-user e2b sandbox: created on first use, resumed on later runs, paused
when idle. The only module that touches the e2b SDK.

The SDK is imported inside `_create`, so the process starts and the manifest
builds without e2b installed. The filesystem persists across runs; the running
instance does not.
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


def _timeout() -> int:
    return int(config.get("sandbox.timeout_seconds") or _DEFAULT_TIMEOUT)


def _template() -> str | None:
    """The e2b template name, or None for the SDK default."""
    name = config.get("sandbox.template")
    return name if name and name != "base" else None


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


async def _stored_id(user_id: str) -> str | None:
    return await pool.fetchval("SELECT sandbox_id FROM user_sandboxes WHERE user_id = $1", _uuid(user_id))


async def _remember(user_id: str, sandbox_id: str) -> None:
    await pool.execute(
        """
        INSERT INTO user_sandboxes (user_id, sandbox_id, last_used_at)
        VALUES ($1, $2, now())
        ON CONFLICT (user_id)
        DO UPDATE SET sandbox_id = EXCLUDED.sandbox_id, last_used_at = now()
        """,
        _uuid(user_id),
        sandbox_id,
    )


async def _touch(user_id: str) -> None:
    await pool.execute("UPDATE user_sandboxes SET last_used_at = now() WHERE user_id = $1", _uuid(user_id))


class SandboxManager:
    """One sandbox per user, wrapping the synchronous e2b SDK in threads."""

    def __init__(self) -> None:
        self._live: dict[str, Any] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, user_id: str) -> asyncio.Lock:
        lock = self._locks.get(user_id)
        if lock is None:
            lock = self._locks[user_id] = asyncio.Lock()
        return lock

    async def get_or_create(self, user_id: str) -> Any:
        """Return the user's sandbox: the cached handle, the stored one resumed, or a new one."""
        async with self._lock(user_id):
            cached = self._live.get(user_id)
            if cached is not None:
                try:
                    # Validates the handle and resets the idle timer in one call.
                    await asyncio.to_thread(cached.set_timeout, _timeout())
                    return cached
                except Exception as e:  # noqa: BLE001 - e2b raises its own types
                    logger.warning("cached sandbox for user %s is gone (%s); recreating", user_id, e)
                    self._live.pop(user_id, None)

            stored = await _stored_id(user_id)
            if stored:
                resumed = await self._resume(user_id, stored)
                if resumed is not None:
                    return resumed

            sandbox = await asyncio.to_thread(self._create)
            self._live[user_id] = sandbox
            await _remember(user_id, sandbox.sandbox_id)
            logger.info("created sandbox %s for user %s", sandbox.sandbox_id, user_id)
            return sandbox

    async def _resume(self, user_id: str, sandbox_id: str) -> Any | None:
        """Reconnect to a stored sandbox, or return None if it is gone."""
        from e2b_code_interpreter import Sandbox

        try:
            sandbox = await asyncio.to_thread(Sandbox.connect, sandbox_id)
            await asyncio.to_thread(sandbox.set_timeout, _timeout())
        except Exception as e:  # noqa: BLE001 - e2b raises its own types
            logger.warning("resume failed for user %s (%s); creating a new sandbox", user_id, e)
            return None
        self._live[user_id] = sandbox
        await _touch(user_id)
        logger.info("resumed sandbox %s for user %s", sandbox_id, user_id)
        return sandbox

    def _create(self) -> Any:
        """Create a sandbox. No environment is passed: credentials stay out of it."""
        from e2b_code_interpreter import Sandbox

        template = _template()
        if template:
            return Sandbox.create(template=template, timeout=_timeout())
        return Sandbox.create(timeout=_timeout())

    async def exec(self, user_id: str, command: str, timeout: int = 120) -> dict[str, Any]:
        """Run a shell command. Returns stdout, stderr and exit_code, including on a non-zero exit."""
        sandbox = await self.get_or_create(user_id)

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

    async def read_file(self, user_id: str, path: str) -> str:
        sandbox = await self.get_or_create(user_id)
        return await asyncio.to_thread(sandbox.files.read, path)

    async def write_file(self, user_id: str, path: str, content: str) -> None:
        sandbox = await self.get_or_create(user_id)
        await asyncio.to_thread(sandbox.files.write, path, content)

    async def list_dir(self, user_id: str, path: str = "/home/user") -> list[dict[str, Any]]:
        """List a directory as [{name, path, is_dir, size}]."""
        sandbox = await self.get_or_create(user_id)
        entries = await asyncio.to_thread(sandbox.files.list, path)

        def is_dir(entry: Any) -> bool:
            kind = getattr(entry, "type", None)
            return getattr(kind, "value", str(kind)).lower() == "dir"

        return [
            {"name": e.name, "path": e.path, "is_dir": is_dir(e), "size": getattr(e, "size", 0)} for e in entries
        ]

    async def pause(self, user_id: str) -> None:
        """Hibernate the sandbox. The filesystem persists; the instance stops costing compute."""
        sandbox = self._live.pop(user_id, None)
        if sandbox is None:
            return
        await asyncio.to_thread(sandbox.pause)
        await _touch(user_id)
        logger.info("paused sandbox for user %s", user_id)


_manager: SandboxManager | None = None


def manager() -> SandboxManager:
    """Return the process-wide manager, so one user has one sandbox per process."""
    global _manager
    if _manager is None:
        _manager = SandboxManager()
    return _manager


def reset() -> None:
    """Drop the manager and its cached handles. For tests."""
    global _manager
    _manager = None
