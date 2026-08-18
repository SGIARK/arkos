"""
The agent's filesystem: bytes in a blob store, the tree in Postgres (D27).

Blobs are content-addressed by sha256 and written once, so the same content in
two projects is one blob and a write can never corrupt an existing one. The tree
is `project_files` rows mapping a path to the hash of its content.

`commit_tree` uploads every missing blob before it touches a row, and flips the
rows in one transaction. A crash between the two leaves the previous tree intact
and whole: an orphan blob costs storage, a row pointing at a blob that is not
there costs a file.

Nothing here knows about the sandbox, e2b or tools. The store is the harness's
(D28) and bytes reach a sandbox by being handed to it, never by it reaching in.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from config_module.loader import config
from db import pool

logger = logging.getLogger(__name__)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


@dataclass(frozen=True, slots=True)
class TreeEntry:
    """One file in the tree. `content_hash` addresses the bytes; the row holds none."""

    path: str
    content_hash: str
    size: int
    mtime: datetime


@dataclass(frozen=True, slots=True)
class FileContent:
    """A file on its way into the store."""

    path: str
    content: bytes
    mtime: datetime | None = None


@dataclass(frozen=True, slots=True)
class TreeDiff:
    """What changed between two trees, by hash."""

    added: frozenset[str]
    changed: frozenset[str]
    removed: frozenset[str]

    @property
    def paths(self) -> frozenset[str]:
        return self.added | self.changed | self.removed

    def __bool__(self) -> bool:
        return bool(self.paths)


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def blob_key(content_hash: str) -> str:
    """Where a blob lives: two hex characters of fan-out, then the full hash."""
    prefix = str(_cfg("store.prefix", "arkos")).strip("/")
    return f"{prefix}/blobs/{content_hash[:2]}/{content_hash}"


# --- the blob backend -----------------------------------------------------------


class Blobs(Protocol):
    """Somewhere immutable to keep bytes, addressed by their hash."""

    async def put(self, content_hash: str, content: bytes) -> None: ...

    async def get(self, content_hash: str) -> bytes | None: ...

    async def missing(self, hashes: Iterable[str]) -> set[str]: ...


class FilesystemBlobs:
    """Blobs as files under a root directory.

    The interface above is what an object store implements; this is the local
    one. Writes go to a temporary name and are renamed into place, so a reader
    never sees a partial blob.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _path(self, content_hash: str) -> Path:
        return self.root / blob_key(content_hash)

    async def put(self, content_hash: str, content: bytes) -> None:
        await asyncio.to_thread(self._put, content_hash, content)

    def _put(self, content_hash: str, content: bytes) -> None:
        target = self._path(content_hash)
        if target.exists():
            return  # write-once: the hash is the content, so there is nothing to update
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_suffix(f".{uuid.uuid4().hex}.partial")
        staging.write_bytes(content)
        staging.replace(target)

    async def get(self, content_hash: str) -> bytes | None:
        return await asyncio.to_thread(self._get, content_hash)

    def _get(self, content_hash: str) -> bytes | None:
        target = self._path(content_hash)
        return target.read_bytes() if target.exists() else None

    async def missing(self, hashes: Iterable[str]) -> set[str]:
        wanted = set(hashes)
        return await asyncio.to_thread(lambda: {h for h in wanted if not self._path(h).exists()})


_blobs: Blobs | None = None


def blobs() -> Blobs:
    """Return the process-wide blob backend."""
    global _blobs
    if _blobs is None:
        _blobs = FilesystemBlobs(_cfg("store.root", ".arkos-store"))
    return _blobs


def use_blobs(backend: Blobs | None) -> None:
    """Swap the backend. For tests, and for the day an object store is configured."""
    global _blobs
    _blobs = backend


# --- bytes ------------------------------------------------------------------------


async def put_blob(content: bytes) -> str:
    """Store content and return its hash. Idempotent: the same bytes are the same blob."""
    content_hash = sha256(content)
    await blobs().put(content_hash, content)
    return content_hash


async def get_blob(content_hash: str) -> bytes | None:
    return await blobs().get(content_hash)


async def missing_blobs(hashes: Iterable[str]) -> set[str]:
    """Which of these hashes the backend does not have."""
    return await blobs().missing(hashes)


# --- the tree ---------------------------------------------------------------------


async def read_tree(project_id: str, subpath: str = "/") -> list[TreeEntry]:
    """Read a project's tree, or the part of it under `subpath`."""
    rows = await pool.fetch(
        """
        SELECT path, content_hash, size, mtime
          FROM project_files
         WHERE project_id = $1 AND ($2 = '/' OR path = $3 OR path LIKE $4)
         ORDER BY path
        """,
        _uuid(project_id),
        subpath,
        _relative(subpath),
        f"{_relative(subpath)}/%",
    )
    return [
        TreeEntry(path=r["path"], content_hash=r["content_hash"], size=r["size"], mtime=r["mtime"]) for r in rows
    ]


async def commit_tree(
    project_id: str,
    files: Sequence[FileContent],
    subpath: str = "/",
) -> list[TreeEntry]:
    """
    Replace the tree under `subpath` with `files`.

    Blobs are uploaded before any row is touched, and the rows are flipped in one
    transaction, so an interrupted commit leaves the previous tree readable and
    complete.

    Returns:
        The tree that is now under `subpath`.
    """
    hashed = [(f, sha256(f.content)) for f in files]

    # Blobs first. Uploading one twice is free; a row pointing at a blob that
    # was never uploaded is a lost file.
    for content_hash in await missing_blobs({h for _, h in hashed}):
        content = next(f.content for f, h in hashed if h == content_hash)
        await blobs().put(content_hash, content)

    now = datetime.now(UTC)
    prefix = _relative(subpath)
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        if prefix:
            await conn.execute(
                "DELETE FROM project_files WHERE project_id = $1 AND (path = $2 OR path LIKE $3)",
                _uuid(project_id),
                prefix,
                f"{prefix}/%",
            )
        else:
            await conn.execute("DELETE FROM project_files WHERE project_id = $1", _uuid(project_id))

        for file, content_hash in hashed:
            await conn.execute(
                """
                INSERT INTO project_files (project_id, path, content_hash, size, mtime)
                VALUES ($1, $2, $3, $4, $5)
                """,
                _uuid(project_id),
                file.path,
                content_hash,
                len(file.content),
                file.mtime or now,
            )

    return await read_tree(project_id, subpath)


def diff_tree(before: Sequence[TreeEntry], after: Sequence[TreeEntry]) -> TreeDiff:
    """Compare two trees by hash. Size and mtime do not decide anything."""
    old = {e.path: e.content_hash for e in before}
    new = {e.path: e.content_hash for e in after}
    return TreeDiff(
        added=frozenset(new.keys() - old.keys()),
        changed=frozenset(p for p in old.keys() & new.keys() if old[p] != new[p]),
        removed=frozenset(old.keys() - new.keys()),
    )


def _relative(subpath: str) -> str:
    """Normalize a claim's subpath to a tree prefix. '/' means the whole project."""
    return (subpath or "/").strip("/")


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
