"""
The agent's filesystem: bytes in a blob store, the tree in Postgres (D27).

ONE FLAT NAMESPACE PER USER. The tree is `files` rows mapping `(user_id, path)`
to the hash of the path's content, and a FOLDER is the first segment of a path —
derived, never a row, unique per user by construction, and alive exactly as long
as a file exists under it (11.9). No project owns a folder; projects LINK
folders, which is `project_folders`, and deleting a project deletes its links
and no files.

Blobs are content-addressed by sha256 and written once, so the same content at
two paths is one blob and a write can never corrupt an existing one.

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

from config_module.loader import cfg as _cfg
from db import pool
from db.ids import as_uuid as _uuid

logger = logging.getLogger(__name__)




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
#
# ONE FLAT NAMESPACE PER USER, and a folder is the first segment of a path. It
# is derived and never stored: it exists exactly as long as a file exists under
# it, its name is unique per user because `(user_id, path)` is, and no table
# holds it. Nothing here takes a project id — a project links folders, it does
# not own them (11.9).
#
# `prefix` narrows a read or a commit to part of that namespace. "" or "/" is
# the whole store; "triage" is one folder; "triage/receipts" is a subtree.


async def read_tree(user_id: str, prefix: str = "/") -> list[TreeEntry]:
    """Read the user's tree, or the part of it under `prefix`. Paths are full store paths."""
    rows = await pool.fetch(
        """
        SELECT path, content_hash, size, mtime
          FROM files
         WHERE user_id = $1 AND ($2 = '' OR path = $2 OR path LIKE $3)
         ORDER BY path
        """,
        _uuid(user_id),
        _relative(prefix),
        f"{_relative(prefix)}/%",
    )
    return [
        TreeEntry(path=r["path"], content_hash=r["content_hash"], size=r["size"], mtime=r["mtime"]) for r in rows
    ]


@dataclass(frozen=True, slots=True)
class StoredFile:
    """One file as the tree now holds it: its row id and its entry."""

    id: str
    entry: TreeEntry


@dataclass(frozen=True, slots=True)
class Folder:
    """A top-level segment of the store, and how many files are under it.

    Not a row and never was. This is the answer to a GROUP BY, which is why a
    folder cannot be renamed by editing something and cannot exist while empty
    of everything — including the sentinel that keeps a named-but-unfilled one
    alive.
    """

    name: str
    files: int


def folder_of(path: str) -> str:
    """The folder a store path belongs to: its first segment."""
    return path.split("/", 1)[0]


async def folders(user_id: str) -> list[Folder]:
    """Every folder in the user's store, alphabetically, with its file count.

    The count excludes sentinels: a folder that has been named and not filled
    reads as 0 files, which is what it holds, rather than as 1 file nobody put
    there.
    """
    rows = await pool.fetch(
        """
        SELECT split_part(path, '/', 1) AS name,
               count(*) FILTER (WHERE path NOT LIKE '%/' || $2) AS files
          FROM files
         WHERE user_id = $1
         GROUP BY 1
         ORDER BY 1
        """,
        _uuid(user_id),
        DIR_SENTINEL,
    )
    return [Folder(name=r["name"], files=int(r["files"])) for r in rows]


async def unique_folder(user_id: str, base: str) -> str:
    """A folder name not already taken in this user's store.

    The none-case of creating a project: no links were picked, so a folder named
    after the project is made for it. Folder names are unique per user by
    construction, so the collision has to be resolved before the name is used
    rather than caught after.
    """
    taken = {f.name for f in await folders(user_id)}
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


def safe_path(name: str) -> str:
    """Normalize a name to a path inside the store.

    A leading slash reads as store-relative and empty or `.` segments are
    dropped, but `..` is refused rather than resolved: rewriting a path the
    caller asked for into a different one is worse than saying no.

    Raises:
        ValueError: the name climbs out of the store or names nothing.
    """
    raw = (name or "").strip().replace("\\", "/")
    parts = [part for part in raw.split("/") if part not in ("", ".")]
    if not parts or ".." in parts:
        raise ValueError(f"{name!r} is not a path inside the store")
    return "/".join(parts)


def in_folder(path: str) -> str:
    """Return `path` if it names a file inside a folder, else refuse.

    Every file in the store is in exactly one folder, because the folder IS the
    first segment. A file at the top level would be its own folder holding
    nothing, which no claim could mount and no header could show, so the store
    does not accept one.

    Raises:
        ValueError: the path has no folder segment.
    """
    if "/" not in path:
        raise ValueError(f"{path!r} is not inside a folder — every file in the store lives in one")
    return path


async def put_file(
    user_id: str,
    path: str,
    content: bytes,
    *,
    mtime: datetime | None = None,
) -> StoredFile:
    """
    Put one file in the user's store, replacing whatever is at that path.

    Blob first, row after, as everywhere else: a crash between the two costs an
    orphan blob rather than a row pointing at bytes that are not there.
    """
    in_folder(path)
    content_hash = await put_blob(content)
    row = await pool.fetchrow(
        """
        INSERT INTO files (user_id, path, content_hash, size, mtime)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (user_id, path)
        DO UPDATE SET content_hash = EXCLUDED.content_hash, size = EXCLUDED.size, mtime = EXCLUDED.mtime
        RETURNING id, path, content_hash, size, mtime
        """,
        _uuid(user_id),
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
# pipeline like any other and cannot be silently dropped. It is also what makes
# a top-level folder exist before anything has been put in it, which is the
# none-case of creating a project.
DIR_SENTINEL = ".keep"


def dir_sentinel(path: str) -> str:
    """The sentinel path that makes an empty directory durable."""
    return f"{path}/{DIR_SENTINEL}"


async def move_path(user_id: str, src: str, dst: str) -> list[tuple[str, str]]:
    """
    Move one file, or a whole subtree, to another path in the user's store.

    Blobs never move: they are content-addressed and immutable, so a rename is
    a row edit and nothing is re-uploaded. Every row moves in one transaction,
    which is what keeps a half-moved folder from existing. Moving BETWEEN
    folders is an ordinary move now — one store, one namespace — and needs no
    copy and no second project.

    **A DIRECTORY may be moved OUT to the top level**, where it becomes a folder
    of its own: `triage/inbox -> inbox`. That is not a special case bolted on, it
    is what the model already says — a folder IS a top-level path segment, so
    promoting a directory to the first position makes one, and demoting a folder
    into another path would be a rename, which has its own route. A FILE cannot
    go there: it would be its own folder holding nothing.

    The file/directory question is answered by the ROWS, not by the string, so
    it is settled inside the transaction after the lookup.

    Returns:
        The (from, to) pairs that moved, in path order.

    Raises:
        MissingPath: nothing is at `src`.
        StoreError: `src` is a top-level folder, a FILE is sent to the top
            level, `dst` is inside `src`, or something already sits at a
            destination path.
    """
    if src == dst:
        return []
    if "/" not in src:
        # A path-prefix rewrite that also has to move every live claim and every
        # mounted path underneath it. It is a RENAME and `rename_path` is where
        # it lives, with the checks it needs. Doing it here by accident, because
        # a folder happens to be a path prefix, would move the ground under a
        # running session silently.
        raise StoreError(f"{src!r} is a folder; renaming or moving one is not something this can do")
    if dst.startswith(f"{src}/"):
        raise StoreError(f"cannot move {src!r} into itself")

    async with (await pool.pool()).acquire() as conn, conn.transaction():
        if "/" not in dst:
            # Bound for the top level. Only a directory may go: the rows say
            # which this is, because an exact match means `src` names a file.
            is_file = await conn.fetchval(
                "SELECT 1 FROM files WHERE user_id = $1 AND path = $2", _uuid(user_id), src
            )
            if is_file:
                raise StoreError(
                    f"the store's top level holds folders, not files: {dst!r} needs a folder to go in"
                )
        moves = await _rewrite_prefix(conn, user_id, src, dst)

    # mtime is left alone on purpose: a move does not change what the file says,
    # and materialize decides what to transfer by content hash regardless.
    return moves


async def _rewrite_prefix(conn: Any, user_id: str, src: str, dst: str) -> list[tuple[str, str]]:
    """Move every row at or under `src` to the same position under `dst`.

    The shared core of a move and a rename — they differ in which paths they
    allow, not in what they do to the rows. Runs inside the caller's transaction,
    so a rename that also rewrites links and claims does all three or none.

    Raises:
        MissingPath: nothing is at `src`.
        StoreError: something already sits at a destination path.
    """
    rows = await conn.fetch(
        """
        SELECT path FROM files
         WHERE user_id = $1 AND (path = $2 OR path LIKE $3)
         ORDER BY path
        """,
        _uuid(user_id),
        src,
        f"{src}/%",
    )
    if not rows:
        raise MissingPath(f"nothing at {src!r}")

    # `src` itself is a file when a row matches it exactly; otherwise it is a
    # directory and only the part after the prefix is kept.
    moves = [(r["path"], dst if r["path"] == src else dst + r["path"][len(src) :]) for r in rows]

    taken = await conn.fetch(
        "SELECT path FROM files WHERE user_id = $1 AND path = ANY($2::text[])",
        _uuid(user_id),
        [to for _, to in moves],
    )
    if taken:
        names = ", ".join(sorted(r["path"] for r in taken))
        raise StoreError(f"something is already at {names}")

    for was, now in moves:
        await conn.execute(
            "UPDATE files SET path = $3 WHERE user_id = $1 AND path = $2",
            _uuid(user_id),
            was,
            now,
        )
    return moves


def renamed_to(path: str, name: str) -> str:
    """The path `path` becomes when its LAST SEGMENT is renamed to `name`.

    Raises:
        ValueError: the name is empty, carries a separator, or is a relative
            step. A rename changes what a thing is called; moving it somewhere
            else is a move, and letting a name contain `/` would quietly be one.
    """
    clean = (name or "").strip().strip("/")
    if not clean or "/" in clean or clean in (".", ".."):
        raise ValueError(f"{name!r} is not a name")
    parts = path.split("/")
    parts[-1] = clean
    return "/".join(parts)


async def rename_path(user_id: str, path: str, name: str) -> list[tuple[str, str]]:
    """
    Rename the last segment of a path: a file, a directory, or a top-level folder.

    A rename is a path-prefix rewrite and nothing is re-uploaded, exactly as a
    move is. What makes a TOP-LEVEL folder different is that its name is written
    down in two more places — the projects that link it and the claims that
    mount it — and a rewrite that moved only the paths would leave a project
    linking a folder that no longer exists and a session claiming one. All three
    move in ONE transaction, which is why this reaches past `files`: the folder
    name is duplicated in exactly three tables, and this is the one operation
    that changes it.

    It does NOT touch a live sandbox. The caller checks that no box holds the
    folder first, because a box that materialized `~/store/<old>/` would flush
    its work back under the old name and resurrect the folder this just renamed.

    Returns:
        The (from, to) pairs that moved, in path order.

    Raises:
        ValueError: `name` is not a name.
        MissingPath: nothing is at `path`.
        StoreError: something already sits at the destination.
    """
    destination = renamed_to(path, name)
    if destination == path:
        return []

    async with (await pool.pool()).acquire() as conn, conn.transaction():
        # The NAME must be free, not merely the paths under it. `_rewrite_prefix`
        # refuses a collision path by path, which would let `triage -> notes`
        # succeed by merging two folders whenever their files happened not to
        # clash — silently, and with no way back. A name that is taken is taken:
        # merging is a thing someone might want, but it is not a rename.
        taken = await conn.fetchval(
            "SELECT 1 FROM files WHERE user_id = $1 AND (path = $2 OR path LIKE $3) LIMIT 1",
            _uuid(user_id),
            destination,
            f"{destination}/%",
        )
        if taken:
            raise StoreError(f"{destination!r} is already taken")

        moves = await _rewrite_prefix(conn, user_id, path, destination)

        if "/" not in path:
            # A top-level folder. Its name is also in the links and the claims.
            await conn.execute(
                """
                UPDATE project_folders SET folder = $3
                 WHERE folder = $2
                   AND project_id IN (SELECT id FROM projects WHERE user_id = $1)
                """,
                _uuid(user_id),
                path,
                destination,
            )
            await conn.execute(
                """
                UPDATE session_claims SET folder = $3
                 WHERE folder = $2
                   AND session_id IN (SELECT id FROM sessions WHERE user_id = $1)
                """,
                _uuid(user_id),
                path,
                destination,
            )
    return moves


@dataclass(frozen=True, slots=True)
class Deletion:
    """What one delete gesture removed, and the handle that takes it back."""

    batch: str
    path: str
    files: int
    unlinked: int
    # The folders that ceased to exist, because their last file went with it.
    folders: tuple[str, ...] = ()


async def delete_path(user_id: str, path: str) -> Deletion:
    """
    Delete a file or a whole subtree, keeping everything needed to undo it.

    The rows move to `deleted_files`; the BLOBS are untouched, because they are
    content-addressed, immutable and never collected. That is the whole reason
    undo can be exact: nothing was destroyed, only unlisted.

    A folder exists exactly as long as a file exists under it, so a delete that
    empties one takes the folder with it — and the links that named it, which
    are recorded in the same batch and come back with it. Nothing else in the
    system may hold a link to a folder that is not there.

    The caller checks first that no live box has the affected folder mounted;
    a box holding it would put the files back at its next flush.

    Returns:
        The deletion, whose `batch` is what `undo_delete` takes.

    Raises:
        MissingPath: nothing is at `path`.
    """
    batch = uuid.uuid4()
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            """
            DELETE FROM files
             WHERE user_id = $1 AND (path = $2 OR path LIKE $3)
            RETURNING id, path, content_hash, size, mtime, created_at
            """,
            _uuid(user_id),
            path,
            f"{path}/%",
        )
        if not rows:
            raise MissingPath(f"nothing to delete at {path!r}")

        for row in rows:
            await conn.execute(
                """
                INSERT INTO deleted_files
                       (id, user_id, path, content_hash, size, mtime, created_at, batch)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                row["id"],
                _uuid(user_id),
                row["path"],
                row["content_hash"],
                row["size"],
                row["mtime"],
                row["created_at"],
                batch,
            )

        # Which folders the delete emptied. Asked of the tree AFTER the rows are
        # gone, so it is the truth rather than a prediction: a folder is derived,
        # and this is what it now derives to.
        touched = {folder_of(row["path"]) for row in rows}
        emptied = [
            folder
            for folder in sorted(touched)
            if not await conn.fetchval(
                "SELECT 1 FROM files WHERE user_id = $1 AND split_part(path, '/', 1) = $2 LIMIT 1",
                _uuid(user_id),
                folder,
            )
        ]

        unlinked = 0
        if emptied:
            dropped = await conn.fetch(
                """
                DELETE FROM project_folders f
                 USING projects p
                 WHERE p.id = f.project_id AND p.user_id = $1 AND f.folder = ANY($2::text[])
                RETURNING f.project_id, f.folder
                """,
                _uuid(user_id),
                emptied,
            )
            for link in dropped:
                await conn.execute(
                    "INSERT INTO deleted_links (batch, project_id, folder) VALUES ($1, $2, $3)",
                    batch,
                    link["project_id"],
                    link["folder"],
                )
            unlinked = len(dropped)

    return Deletion(
        batch=str(batch),
        path=path,
        files=len(rows),
        unlinked=unlinked,
        folders=tuple(emptied),
    )


async def undo_delete(user_id: str, batch: str) -> Deletion:
    """
    Put back exactly what one delete gesture removed.

    The same rows under the same ids, pointing at the same blobs — which are
    still in the store, since nothing collects them. The links that went with
    the folders come back too.

    Raises:
        MissingPath: no such batch for this user, or it has already been undone.
        StoreError: something now occupies a path the delete freed. Refusing
            beats overwriting: whatever is there was put there afterwards, and
            it is not this batch's to replace.
    """
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        rows = await conn.fetch(
            "SELECT id, path, content_hash, size, mtime, created_at FROM deleted_files "
            "WHERE user_id = $1 AND batch = $2 ORDER BY path",
            _uuid(user_id),
            _uuid(batch),
        )
        if not rows:
            raise MissingPath("there is nothing to undo")

        taken = await conn.fetch(
            "SELECT path FROM files WHERE user_id = $1 AND path = ANY($2::text[])",
            _uuid(user_id),
            [r["path"] for r in rows],
        )
        if taken:
            names = ", ".join(sorted(r["path"] for r in taken))
            raise StoreError(f"something is already at {names}")

        for row in rows:
            await conn.execute(
                """
                INSERT INTO files (id, user_id, path, content_hash, size, mtime, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,
                row["id"],
                _uuid(user_id),
                row["path"],
                row["content_hash"],
                row["size"],
                row["mtime"],
                row["created_at"],
            )

        links = await conn.fetch(
            "SELECT project_id, folder FROM deleted_links WHERE batch = $1", _uuid(batch)
        )
        for link in links:
            await conn.execute(
                "INSERT INTO project_folders (project_id, folder) VALUES ($1, $2) "
                "ON CONFLICT DO NOTHING",
                link["project_id"],
                link["folder"],
            )

        await conn.execute("DELETE FROM deleted_links WHERE batch = $1", _uuid(batch))
        await conn.execute(
            "DELETE FROM deleted_files WHERE user_id = $1 AND batch = $2", _uuid(user_id), _uuid(batch)
        )

    return Deletion(
        batch=str(batch),
        path=rows[0]["path"],
        files=len(rows),
        unlinked=len(links),
        folders=tuple(sorted({folder_of(r["path"]) for r in rows})),
    )


async def commit_tree(
    user_id: str,
    contents: Sequence[FileContent],
    prefix: str = "/",
) -> list[TreeEntry]:
    """
    Replace the tree under `prefix` with `contents`.

    Blobs are uploaded before any row is touched, and the rows are flipped in one
    transaction, so an interrupted commit leaves the previous tree readable and
    complete.

    Returns:
        The tree that is now under `prefix`.
    """
    hashed = [(f, sha256(f.content)) for f in contents]

    # Blobs first. Uploading one twice is free; a row pointing at a blob that
    # was never uploaded is a lost file.
    for content_hash in await missing_blobs({h for _, h in hashed}):
        content = next(f.content for f, h in hashed if h == content_hash)
        await blobs().put(content_hash, content)

    now = datetime.now(UTC)
    return await commit_entries(
        user_id,
        [
            TreeEntry(path=f.path, content_hash=h, size=len(f.content), mtime=f.mtime or now)
            for f, h in hashed
        ],
        prefix,
    )


async def commit_entries(
    user_id: str,
    entries: Sequence[TreeEntry],
    prefix: str = "/",
) -> list[TreeEntry]:
    """
    Replace the tree under `prefix` with entries whose blobs are already stored.

    What a flush uses: only changed files have their bytes uploaded, and every
    row is written from a hash. The blobs are checked before any row moves, so
    the tree still cannot come to point at bytes that are not there.

    Raises:
        StoreError: an entry names a blob the store does not hold.
    """
    absent = await missing_blobs({e.content_hash for e in entries})
    if absent:
        raise StoreError(f"refusing to commit: {len(absent)} blob(s) are not in the store")

    scope = _relative(prefix)
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        if scope:
            await conn.execute(
                "DELETE FROM files WHERE user_id = $1 AND (path = $2 OR path LIKE $3)",
                _uuid(user_id),
                scope,
                f"{scope}/%",
            )
        else:
            await conn.execute("DELETE FROM files WHERE user_id = $1", _uuid(user_id))

        for entry in entries:
            await conn.execute(
                """
                INSERT INTO files (user_id, path, content_hash, size, mtime)
                VALUES ($1, $2, $3, $4, $5)
                """,
                _uuid(user_id),
                entry.path,
                entry.content_hash,
                entry.size,
                entry.mtime,
            )

    return await read_tree(user_id, prefix)


def diff_tree(before: Sequence[TreeEntry], after: Sequence[TreeEntry]) -> TreeDiff:
    """Compare two trees by hash. Size and mtime do not decide anything."""
    old = {e.path: e.content_hash for e in before}
    new = {e.path: e.content_hash for e in after}
    return TreeDiff(
        added=frozenset(new.keys() - old.keys()),
        changed=frozenset(p for p in old.keys() & new.keys() if old[p] != new[p]),
        removed=frozenset(old.keys() - new.keys()),
    )


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
    """A folder name from a title. Mounted names are read by the model as context.

    Used for the folder a project with no links makes for itself, and for
    `projects.slug`, which is now nothing but the default that name comes from.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return cleaned[:48] or fallback


def covers(prefix: str, path: str) -> bool:
    """Whether a claim on `prefix` includes `path`. Both are full store paths."""
    scope = _relative(prefix)
    return not scope or path == scope or path.startswith(scope + "/")


def _relative(prefix: str) -> str:
    """Normalize a prefix to a tree prefix. '/' or '' means the whole store."""
    return (prefix or "").strip("/")


