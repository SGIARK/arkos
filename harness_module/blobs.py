"""
Bytes, addressed by the hash of their content.

Write-once and immutable: the name IS the content, so two paths holding the same
bytes are one blob, a write can never corrupt an existing one, and a retry after
a partial upload is safe. Nothing here knows about paths, users, trees or
sandboxes — `store.py` maps paths to hashes and this maps hashes to bytes.

Split out of `store.py` in 11.8.8, which was three modules wearing one filename.
The import direction is one-way and stays that way: blobs <- store <- workspace.

**The HTTP client is bound to the event loop that made it**, which is why its
lifecycle lives here rather than inside the backend that uses it. `httpx`
sockets register with a running loop; pytest-asyncio gives each test its own, so
a client cached across two of them raises "Event loop is closed" on the second —
a flake that was on the 11.7.5 list and moved rather than went away every time
somebody cached the client somewhere new. `_client_for_loop` keys the cache by
the RUNNING loop and closes the one it replaces, so the lifetime is a fact about
the loop rather than about who remembered to reset a global.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import tarfile
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from config_module.loader import cfg as _cfg

logger = logging.getLogger(__name__)


class StoreError(RuntimeError):
    """Raised when the blob backend refuses a read or a write."""


class MissingPath(StoreError):
    """Raised when an operation names a path the tree does not have."""


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
        # An injected client is a test's, and it owns it.
        self._injected = client
        self._gate = asyncio.Semaphore(concurrency)

    def _client_or_new(self) -> Any:
        if self._injected is not None:
            return self._injected
        return _client_for_loop()

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
        """Close an injected client. The shared one belongs to its loop, not here."""
        if self._injected is not None and not self._injected.is_closed:
            await self._injected.aclose()
        self._injected = None


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



# --- the HTTP client, one per running loop ---------------------------------------
#
# `httpx` binds its sockets to the loop that opened them, so a client cached in a
# module global outlives the loop it belongs to and the next loop finds it dead —
# "Event loop is closed", from a connection pool nobody thought about. Keying the
# cache by the RUNNING loop makes the lifetime follow the thing it actually
# depends on, and closing the one being replaced means a swap leaks nothing.

_clients: dict[Any, Any] = {}


def _client_for_loop() -> Any:
    """The HTTP client belonging to the loop that is running now."""
    import httpx

    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is not None and not client.is_closed:
        return client
    client = httpx.AsyncClient(timeout=30.0)
    _clients[loop] = client
    # Loops that have gone leave nothing behind: their clients are already dead
    # with them, and the entry is only a reference to collect.
    for stale in [key for key in _clients if key.is_closed()]:
        _clients.pop(stale, None)
    return client


async def close_clients() -> None:
    """Close the client for the running loop. Called from the app's lifespan."""
    loop = asyncio.get_running_loop()
    client = _clients.pop(loop, None)
    if client is not None and not client.is_closed:
        await client.aclose()


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
