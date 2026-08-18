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
import os
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
