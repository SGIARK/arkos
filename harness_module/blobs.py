"""
Bytes, addressed by the hash of their content.

Write-once and immutable. Imports run one way: blobs <- store <- workspace.
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
    prefix = str(_cfg("store.prefix", "buddy")).strip("/")
    return f"{prefix}/blobs/{content_hash[:2]}/{content_hash}"


# --- the blob backend -----------------------------------------------------------


class Blobs(Protocol):
    """Somewhere immutable to keep bytes, addressed by their hash."""

    async def put(self, content_hash: str, content: bytes) -> None: ...

    async def get(self, content_hash: str) -> bytes | None: ...

    async def missing(self, hashes: Iterable[str]) -> set[str]: ...


class FilesystemBlobs:
    """Blobs as files under a root directory.

    Writes are renamed into place, so a reader never sees a partial blob.
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
            return  # write-once: the name is the content, so there is nothing to update
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

    URL and secret key come from the environment, not config.yaml: a `${VAR}`
    there makes an unset key crash config load for everything.
    """

    def __init__(self, url: str, secret_key: str, bucket: str, concurrency: int = 8, client: Any = None):
        self.base = url.rstrip("/") + "/storage/v1/object"
        self.bucket = bucket
        # Both headers, to suit either key format: secret keys (sb_secret_...)
        # and the legacy service_role JWT.
        self._headers = {"Authorization": f"Bearer {secret_key}", "apikey": secret_key}
        # An injected client belongs to the test that passed it.
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
        # A reported duplicate is a success: the name is the hash of the content,
        # so the existing object is the blob we were about to write.
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
            # A HEAD carries no body to tell a miss from an error, so anything but 200
            # counts as absent: a false "present" costs the file, a false "missing" one upload.
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

    Supabase Storage reports a missing object as HTTP 400 with a body of
    `{"statusCode": "404", ... "code": "NoSuchKey"}`, so the status alone does not say.
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

    Both DSN shapes carry the project ref: the direct connection in the host
    (`db.<ref>.supabase.co`), the pooler in the username (`postgres.<ref>@...`).
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
        return f"https://{host[len('db.') :]}"
    if "pooler.supabase.com" in host and "." in (parts.username or ""):
        return f"https://{parts.username.split('.', 1)[1]}.supabase.co"
    return None


def bucket() -> str:
    """The bucket blobs live in. STORE_BUCKET overrides the configured name."""
    return str(os.environ.get("STORE_BUCKET") or _cfg("store.bucket", "") or "")


def secret_key() -> str | None:
    """The key the store authenticates with; SUPABASE_SERVICE_KEY is the legacy fallback."""
    return os.environ.get("SUPABASE_SECRET_KEY") or os.environ.get("SUPABASE_SERVICE_KEY")


def _build() -> Blobs:
    backend = str(_cfg("store.backend", "filesystem")).lower()
    if backend == "filesystem":
        return FilesystemBlobs(_cfg("store.root", ".buddy-store"))
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
    """Swap the backend."""
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
# `httpx` binds its sockets to the loop that opened them, so a client outliving its
# loop raises "Event loop is closed". Key the cache by the RUNNING loop.

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
    # Clients of closed loops are already dead; the entry is only a reference to drop.
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
    """Pack (path, content) pairs into an uncompressed tar."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, content in files:
            info = tarfile.TarInfo(name=path)
            info.size = len(content)
            info.mtime = 0  # deterministic archive for identical content
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()
