"""
The agent's filesystem: bytes in a blob store, the tree in Postgres (D27).

Blobs are content-addressed by sha256 and written once, so the same content in
two projects is one blob and a write can never corrupt an existing one. The tree
is `project_files` rows mapping a path to the hash of its content.

`commit_tree` uploads every missing blob before it touches a row, and flips the
rows in one transaction. A crash between the two leaves the previous tree intact
and whole: an orphan blob costs storage, a row pointing at a blob that is not
there costs a file.

The user's memory region lives here too, in a table of its own that the mount
path never reads. A session may only append a note to it.

Nothing here knows about the sandbox, e2b or tools. The store is the harness's
(D28) and bytes reach a sandbox by being handed to it, never by it reaching in.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
import tarfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

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


class SupabaseBlobs:
    """Blobs in a Supabase Storage bucket, over its REST API.

    Writes are write-once, so an upload of a hash that is already there is a
    success rather than an overwrite: Supabase reports the duplicate and this
    treats it as the object already being correct, which it is, because the
    name is the hash of the content.

    The URL and secret key come from the environment rather than config.yaml,
    for the same reason E2B_API_KEY does: a `${VAR}` in the yaml makes an unset
    key crash config load for everything, including the parts that do not use it.
    """

    def __init__(self, url: str, secret_key: str, bucket: str, concurrency: int = 8, client: Any = None):
        self.base = url.rstrip("/") + "/storage/v1/object"
        self.bucket = bucket
        # Both headers, which suits either key format: the current secret keys
        # (sb_secret_...) and the legacy service_role JWT.
        self._headers = {"Authorization": f"Bearer {secret_key}", "apikey": secret_key}
        self._client = client
        self._gate = asyncio.Semaphore(concurrency)

    def _client_or_new(self) -> Any:
        import httpx

        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    def _url(self, content_hash: str) -> str:
        return f"{self.base}/{self.bucket}/{blob_key(content_hash)}"

    async def put(self, content_hash: str, content: bytes) -> None:
        client = self._client_or_new()
        async with self._gate:
            response = await client.post(
                self._url(content_hash),
                content=content,
                headers={**self._headers, "Content-Type": "application/octet-stream"},
            )
        if response.status_code in (200, 201):
            return
        # The object already exists. Its name is the hash of its content, so it
        # is the blob we were about to write.
        if response.status_code in (409, 400) and "duplicate" in response.text.lower():
            return
        raise StoreError(f"uploading {content_hash[:12]} failed: {response.status_code} {response.text[:200]}")

    async def get(self, content_hash: str) -> bytes | None:
        client = self._client_or_new()
        async with self._gate:
            response = await client.get(self._url(content_hash), headers=self._headers)
        if _is_absent(response):
            return None
        if response.status_code != 200:
            raise StoreError(f"reading {content_hash[:12]} failed: {response.status_code} {response.text[:200]}")
        return response.content

    async def missing(self, hashes: Iterable[str]) -> set[str]:
        wanted = sorted(set(hashes))
        if not wanted:
            return set()
        client = self._client_or_new()

        async def absent(content_hash: str) -> str | None:
            async with self._gate:
                response = await client.head(self._url(content_hash), headers=self._headers)
            # Anything but a clean 200 counts as absent. A HEAD carries no body
            # to distinguish a miss from an error, and the two mistakes are not
            # equal: calling a present blob missing costs one redundant upload,
            # calling a missing blob present costs the file.
            return None if response.status_code == 200 else content_hash

        found = await asyncio.gather(*(absent(h) for h in wanted))
        return {h for h in found if h is not None}

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None


class StoreError(RuntimeError):
    """Raised when the blob backend refuses a read or a write."""


class MissingPath(StoreError):
    """Raised when an operation names a path the project's tree does not have."""


def _is_absent(response: Any) -> bool:
    """Whether a response means the object is not there.

    Supabase Storage reports a missing object as HTTP 400 carrying a body of
    `{"statusCode": "404", ... "code": "NoSuchKey"}`, so the transport status
    alone does not say.
    """
    if response.status_code == 404:
        return True
    if response.status_code != 400:
        return False
    body = (response.text or "").lower()
    return '"404"' in body or "nosuchkey" in body or "not_found" in body


_blobs: Blobs | None = None


def blobs() -> Blobs:
    """Return the process-wide blob backend, built from `store.backend`."""
    global _blobs
    if _blobs is None:
        _blobs = _build()
    return _blobs


def project_url() -> str | None:
    """The Supabase project URL, from SUPABASE_URL or derived from the database DSN.

    Both DSN shapes carry the project ref: the direct connection puts it in the
    host (`db.<ref>.supabase.co`) and the pooler puts it in the username
    (`postgres.<ref>@aws-0-<region>.pooler.supabase.com`).
    """
    explicit = os.environ.get("SUPABASE_URL")
    if explicit:
        return explicit.rstrip("/")

    dsn = _cfg("database.url", "") or ""
    try:
        parts = urlsplit(dsn)
    except ValueError:
        return None

    host = parts.hostname or ""
    if host.endswith(".supabase.co") and host.startswith("db."):
        return f"https://{host[len('db.'):]}"
    if "pooler.supabase.com" in host and "." in (parts.username or ""):
        return f"https://{parts.username.split('.', 1)[1]}.supabase.co"
    return None


def bucket() -> str:
    """The bucket blobs live in. STORE_BUCKET overrides the configured name."""
    return str(os.environ.get("STORE_BUCKET") or _cfg("store.bucket", "") or "")


def secret_key() -> str | None:
    """The key the store authenticates with.

    `SUPABASE_SECRET_KEY` holds a secret API key (`sb_secret_...`), which is
    revocable and rotatable on its own. `SUPABASE_SERVICE_KEY` is read as a
    fallback for installations still on the legacy service_role JWT, which can
    only be rotated by invalidating every key in the project at once.
    """
    return os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")


def _build() -> Blobs:
    backend = str(_cfg("store.backend", "filesystem")).lower()
    if backend == "filesystem":
        return FilesystemBlobs(_cfg("store.root", ".arkos-store"))
    if backend == "supabase":
        url = project_url()
        key = secret_key()
        name = bucket()
        missing = [
            label
            for label, value in (
                ("SUPABASE_URL (or a Supabase database.url to derive it from)", url),
                ("SUPABASE_SECRET_KEY", key),
                ("store.bucket (or STORE_BUCKET)", name),
            )
            if not value
        ]
        if missing:
            raise StoreError(f"store.backend is 'supabase' but {', '.join(missing)} is unset")
        return SupabaseBlobs(url, key, name)
    raise StoreError(f"unknown store.backend {backend!r}; expected 'filesystem' or 'supabase'")


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


@dataclass(frozen=True, slots=True)
class StoredFile:
    """One file as the tree now holds it: its row id and its entry."""

    id: str
    entry: TreeEntry


def safe_path(name: str) -> str:
    """Normalize an uploaded name to a path inside the project.

    A leading slash reads as project-relative and empty or `.` segments are
    dropped, but `..` is refused rather than resolved: rewriting a path the
    caller asked for into a different one is worse than saying no.

    Raises:
        ValueError: the name climbs out of the project or names nothing.
    """
    raw = (name or "").strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"{name!r} is not a path inside the project")
    return "/".join(parts)


async def put_file(
    project_id: str,
    path: str,
    content: bytes,
    *,
    mtime: datetime | None = None,
) -> StoredFile:
    """
    Put one file in the project's tree, replacing whatever is at that path.

    Blob first, row after, as everywhere else: a crash between the two costs an
    orphan blob rather than a row pointing at bytes that are not there.
    """
    content_hash = await put_blob(content)
    row = await pool.fetchrow(
        """
        INSERT INTO project_files (project_id, path, content_hash, size, mtime)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (project_id, path)
        DO UPDATE SET content_hash = EXCLUDED.content_hash, size = EXCLUDED.size, mtime = EXCLUDED.mtime
        RETURNING id, path, content_hash, size, mtime
        """,
        _uuid(project_id),
        path,
        content_hash,
        len(content),
        mtime or datetime.now(UTC),
    )
    return StoredFile(
        id=str(row["id"]),
        entry=TreeEntry(
            path=row["path"], content_hash=row["content_hash"], size=row["size"], mtime=row["mtime"]
        ),
    )


# A folder with nothing in it. The tree is flat paths and the sandbox round trip
# carries files and only files — materialize writes them, `_sweep` finds them,
# flush commits them — so a directory that is not a file has nowhere to survive:
# the next flush would replace the subtree from what is on disk and the empty
# folder would be gone. A zero-byte sentinel IS a file, so it rides the whole
# pipeline like any other and cannot be silently dropped.
DIR_SENTINEL = ".keep"


def dir_sentinel(path: str) -> str:
    """The sentinel path that makes an empty directory durable."""
    return f"{path}/{DIR_SENTINEL}"


async def move_path(project_id: str, src: str, dst: str) -> list[tuple[str, str]]:
    """
    Move one file, or a whole subtree, to another path in the same project.

    Blobs never move: they are content-addressed and immutable, so a rename is
    a row edit and nothing is re-uploaded. Every row moves in one transaction,
    which is what keeps a half-moved folder from existing.

    Returns:
        The (from, to) pairs that moved, in path order.

    Raises:
        MissingPath: nothing is at `src`.
        StoreError: `dst` is inside `src`, or something already sits at a
            destination path.
    """
    if src == dst:
        return []
    if dst.startswith(f"{src}/"):
        raise StoreError(f"cannot move {src!r} into itself")

    async with (await pool.pool()).acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            """
            SELECT path FROM project_files
             WHERE project_id = $1 AND (path = $2 OR path LIKE $3)
             ORDER BY path
            """,
            _uuid(project_id),
            src,
            f"{src}/%",
        )
        if not rows:
            raise MissingPath(f"nothing to move at {src!r}")

        # `src` itself is a file when a row matches it exactly; otherwise it is a
        # directory and only the part after the prefix is kept.
        moves = [(r["path"], dst if r["path"] == src else dst + r["path"][len(src) :]) for r in rows]

        taken = await conn.fetch(
            "SELECT path FROM project_files WHERE project_id = $1 AND path = ANY($2::text[])",
            _uuid(project_id),
            [to for _, to in moves],
        )
        if taken:
            names = ", ".join(sorted(r["path"] for r in taken))
            raise StoreError(f"something is already at {names}")

        for was, now in moves:
            await conn.execute(
                "UPDATE project_files SET path = $3 WHERE project_id = $1 AND path = $2",
                _uuid(project_id),
                was,
                now,
            )

    # mtime is left alone on purpose: a move does not change what the file says,
    # and materialize decides what to transfer by content hash regardless.
    return moves


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
    return await commit_entries(
        project_id,
        [
            TreeEntry(path=f.path, content_hash=h, size=len(f.content), mtime=f.mtime or now)
            for f, h in hashed
        ],
        subpath,
    )


async def commit_entries(
    project_id: str,
    entries: Sequence[TreeEntry],
    subpath: str = "/",
) -> list[TreeEntry]:
    """
    Replace the tree under `subpath` with entries whose blobs are already stored.

    What a flush uses: only changed files have their bytes uploaded, and every
    row is written from a hash. The blobs are checked before any row moves, so
    the tree still cannot come to point at bytes that are not there.

    Raises:
        StoreError: an entry names a blob the store does not hold.
    """
    absent = await missing_blobs({e.content_hash for e in entries})
    if absent:
        raise StoreError(f"refusing to commit: {len(absent)} blob(s) are not in the store")

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

        for entry in entries:
            await conn.execute(
                """
                INSERT INTO project_files (project_id, path, content_hash, size, mtime)
                VALUES ($1, $2, $3, $4, $5)
                """,
                _uuid(project_id),
                entry.path,
                entry.content_hash,
                entry.size,
                entry.mtime,
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


# --- snapshots --------------------------------------------------------------------
#
# A snapshot is a copy of a project's tree rows and nothing else. The bytes are
# content-addressed and never mutated, so the rows are the whole of the state:
# taking one costs a row copy, and restoring one points the tree back at blobs
# that were always there.


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One saved tree: when it was taken, why, and how big it was."""

    id: str
    project_id: str
    label: str | None
    taken_at: datetime
    files: int


async def snapshot_project(project_id: str, label: str | None = None) -> str:
    """
    Save the project's tree as it stands now.

    The copy happens in one statement inside the same transaction that creates
    the snapshot row, so a snapshot is never half a tree — and an empty project
    snapshots to an empty tree, which is a state worth being able to return to.

    Returns:
        The snapshot's id.
    """
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        snapshot_id = await conn.fetchval(
            "INSERT INTO project_snapshots (project_id, label) VALUES ($1, $2) RETURNING id",
            _uuid(project_id),
            label,
        )
        await conn.execute(
            """
            INSERT INTO snapshot_files (snapshot_id, path, content_hash, size, mtime)
            SELECT $1, path, content_hash, size, mtime FROM project_files WHERE project_id = $2
            """,
            snapshot_id,
            _uuid(project_id),
        )
    return str(snapshot_id)


async def list_snapshots(project_id: str, limit: int = 50) -> list[Snapshot]:
    """The project's snapshots, newest first."""
    rows = await pool.fetch(
        """
        SELECT s.id, s.project_id, s.label, s.taken_at, count(f.path) AS files
          FROM project_snapshots s LEFT JOIN snapshot_files f ON f.snapshot_id = s.id
         WHERE s.project_id = $1
         GROUP BY s.id
         ORDER BY s.taken_at DESC
         LIMIT $2
        """,
        _uuid(project_id),
        max(1, limit),
    )
    return [
        Snapshot(
            id=str(r["id"]),
            project_id=str(r["project_id"]),
            label=r["label"],
            taken_at=r["taken_at"],
            files=r["files"],
        )
        for r in rows
    ]


async def restore_snapshot(snapshot_id: str) -> list[TreeEntry]:
    """
    Put a project's tree back the way the snapshot has it.

    Blobs are checked before any row moves, exactly as a commit checks them: a
    snapshot whose bytes are no longer in the store must not become a tree
    pointing at files that cannot be read. Nothing deletes blobs today, so this
    is a guard against the day something does, not a condition seen in practice.

    A path the project has now and the snapshot does not is deleted, because a
    restore is the tree as it stood and not a merge with what came after.

    Raises:
        StoreError: no such snapshot, or the store no longer holds its blobs.
    """
    row = await pool.fetchrow(
        "SELECT project_id FROM project_snapshots WHERE id = $1", _uuid(snapshot_id)
    )
    if row is None:
        raise StoreError(f"no snapshot {snapshot_id}")
    project_id = row["project_id"]

    saved = await pool.fetch(
        "SELECT path, content_hash, size, mtime FROM snapshot_files WHERE snapshot_id = $1",
        _uuid(snapshot_id),
    )
    absent = await missing_blobs({r["content_hash"] for r in saved})
    if absent:
        raise StoreError(
            f"refusing to restore snapshot {snapshot_id}: {len(absent)} blob(s) are not in the store"
        )

    async with (await pool.pool()).acquire() as conn, conn.transaction():
        await conn.execute("DELETE FROM project_files WHERE project_id = $1", project_id)
        await conn.execute(
            """
            INSERT INTO project_files (project_id, path, content_hash, size, mtime)
            SELECT $2, path, content_hash, size, mtime FROM snapshot_files WHERE snapshot_id = $1
            """,
            _uuid(snapshot_id),
            project_id,
        )

    return await read_tree(str(project_id))


async def prune_snapshots(project_id: str, keep: int) -> int:
    """Delete all but the newest `keep` snapshots of a project.

    Returns:
        How many were deleted.
    """
    result = await pool.execute(
        """
        DELETE FROM project_snapshots
         WHERE project_id = $1
           AND id NOT IN (
               SELECT id FROM project_snapshots
                WHERE project_id = $1
                ORDER BY taken_at DESC
                LIMIT $2
           )
        """,
        _uuid(project_id),
        max(0, keep),
    )
    return int(result.rsplit(" ", 1)[-1] or 0)


# --- memory -----------------------------------------------------------------------
#
# The user's own region, `{user}/memory/` in the layout, kept in `memory_files`
# rather than in the project tree: memory is keyed by user, a project tree is
# keyed by project. Whether it may ever be mounted is D30 and open; today it is
# reached only through the calls below, and no claim can name it.
#
# The write discipline is the transcript's. A session appends a note and never
# rewrites one; the curated core is replaced whole, under a lock, so two
# sessions curating at once cannot interleave into nonsense.


# The curated core, and the directory one appended note lands in. Both are
# relative to the user's region.
MEMORY_CORE = "MEMORY.md"
NOTES_DIR = "notes"

# The advisory lock's namespace, so a memory lock cannot collide with any other
# advisory lock this database grows later.
_MEMORY_LOCK = 8808

# One statement for both writers: a note that has never been written before, and
# a core that is written over every time it is curated.
_MEMORY_UPSERT = """
    INSERT INTO memory_files (user_id, path, content_hash, size, mtime, body)
    VALUES ($1, $2, $3, $4, $5, $6)
    ON CONFLICT (user_id, path) DO UPDATE
        SET content_hash = EXCLUDED.content_hash,
            size = EXCLUDED.size,
            mtime = EXCLUDED.mtime,
            body = EXCLUDED.body
"""


@dataclass(frozen=True, slots=True)
class Note:
    """One note a session appended: where it lives, what it says, when it landed."""

    path: str
    text: str
    written_at: datetime


@dataclass(frozen=True, slots=True)
class Hit:
    """One search result, and how well it matched."""

    path: str
    text: str
    written_at: datetime
    rank: float

    @property
    def is_core(self) -> bool:
        return self.path == MEMORY_CORE


async def append_note(user_id: str, text: str) -> str:
    """
    Add one note to the user's memory. This is how a session records a fact.

    Each note is a file of its own, named for the moment it landed and a random
    suffix. Two sessions appending at once cannot collide or overwrite: nothing
    on this path reads a file in order to write it back, so no lock is needed.

    Returns:
        The note's path inside the region.
    """
    now = datetime.now(UTC)
    path = f"{NOTES_DIR}/{now.strftime('%Y%m%dT%H%M%S%f')}-{uuid.uuid4().hex[:8]}.md"
    content = text.encode()
    await pool.execute(
        _MEMORY_UPSERT, _uuid(user_id), path, await put_blob(content), len(content), now, text
    )
    return path


async def update_memory(user_id: str, text: str) -> None:
    """
    Replace the curated core with `text`, one writer at a time.

    The core is the one memory file that is rewritten rather than appended, so
    it is the one with a gate: the write takes a transaction-scoped advisory
    lock on the user, and a second curation waits for the first to finish. One
    upsert would be atomic without it; the lock is here because curation grows
    into read-then-write — the background compactor reading the notes before it
    replaces the core — and that is the version a gate has to already exist for.
    It is released with the transaction, including one that dies.

    Whoever calls this is the compactor. For now that is the model, reading the
    core and the notes and rewriting the core; the background job that will do
    it unattended is a later card and needs no other entry point than this.
    """
    # Blobs first, rows last, as everywhere in the store — and it keeps the lock
    # off an upload to another service.
    content = text.encode()
    content_hash = await put_blob(content)

    async with (await pool.pool()).acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1, hashtext($2))", _MEMORY_LOCK, str(user_id))
        await conn.execute(
            _MEMORY_UPSERT,
            _uuid(user_id),
            MEMORY_CORE,
            content_hash,
            len(content),
            datetime.now(UTC),
            text,
        )


async def read_memory(user_id: str) -> str:
    """The curated core, or '' when nothing has written one yet."""
    body = await pool.fetchval(
        "SELECT body FROM memory_files WHERE user_id = $1 AND path = $2",
        _uuid(user_id),
        MEMORY_CORE,
    )
    return body or ""


async def read_notes(user_id: str) -> list[Note]:
    """Every note the user has, oldest first — the name carries the order."""
    rows = await pool.fetch(
        """
        SELECT path, body, mtime
          FROM memory_files
         WHERE user_id = $1 AND path LIKE $2
         ORDER BY path
        """,
        _uuid(user_id),
        f"{NOTES_DIR}/%",
    )
    return [Note(path=r["path"], text=r["body"], written_at=r["mtime"]) for r in rows]


async def search_memory(user_id: str, query: str, limit: int = 10) -> list[Hit]:
    """
    Full-text search over one user's memory, core and notes alike.

    `websearch_to_tsquery` because the query is written by the model in the
    shape a person would type: bare words, quoted phrases, `or`. A query that
    parses to nothing matches nothing rather than everything.
    """
    rows = await pool.fetch(
        """
        SELECT path, body, mtime, ts_rank(tsv, q) AS rank
          FROM memory_files, websearch_to_tsquery('english', $2) AS q
         WHERE user_id = $1 AND tsv @@ q
         ORDER BY rank DESC, mtime DESC
         LIMIT $3
        """,
        _uuid(user_id),
        query,
        max(1, limit),
    )
    return [
        Hit(path=r["path"], text=r["body"], written_at=r["mtime"], rank=float(r["rank"])) for r in rows
    ]


# --- moving bytes -----------------------------------------------------------------


def build_tar(files: Sequence[tuple[str, bytes]]) -> bytes:
    """Pack (path, content) pairs into an uncompressed tar.

    One archive per materialize keeps the transfer to a single write and a
    single extract, whatever the file count.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, content in files:
            info = tarfile.TarInfo(name=path)
            info.size = len(content)
            info.mtime = 0  # a stable archive for the same content
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def slug(title: str, fallback: str) -> str:
    """A directory name for a project. Mounted names are read by the model as context."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return cleaned[:48] or fallback


def covers(subpath: str, path: str) -> bool:
    """Whether a claim on `subpath` includes `path`. Paths are project-relative."""
    prefix = _relative(subpath)
    return not prefix or path == prefix or path.startswith(prefix + "/")


def _relative(subpath: str) -> str:
    """Normalize a claim's subpath to a tree prefix. '/' means the whole project."""
    return (subpath or "/").strip("/")


def _uuid(value: str) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))
