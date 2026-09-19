"""
The agent's filesystem: the TREE. Bytes live in `blobs`, keyed by their hash.

One flat namespace per user: a row maps `(user_id, path)` to a content hash, and
a FOLDER is the first segment of a path — derived, never a row. Imports go one
way: blobs <- store <- workspace.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from db import pool
from db.ids import as_uuid as _uuid
from harness_module.blobs import (
    Blobs,
    FilesystemBlobs,
    MissingPath,
    StoreError,
    SupabaseBlobs,
    blob_key,
    blobs,
    build_tar,
    get_blob,
    missing_blobs,
    put_blob,
    sha256,
    use_blobs,
)

logger = logging.getLogger(__name__)

# Blob calls are re-exported here; `blobs.py` owns them, this is only a doorway.
__all__ = [
    "Blobs",
    "Deletion",
    "FileContent",
    "FilesystemBlobs",
    "Folder",
    "MissingPath",
    "StoreError",
    "StoredFile",
    "SupabaseBlobs",
    "TreeEntry",
    "blob_key",
    "blobs",
    "build_tar",
    "commit_entries",
    "commit_tree",
    "covers",
    "delete_path",
    "dir_sentinel",
    "folder_of",
    "folders",
    "get_blob",
    "in_folder",
    "missing_blobs",
    "move_path",
    "put_blob",
    "put_file",
    "read_tree",
    "rename_path",
    "renamed_to",
    "safe_path",
    "sha256",
    "slug",
    "undo_delete",
    "unique_folder",
    "use_blobs",
]


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


# --- the tree ---------------------------------------------------------------------
#
# A folder is the first segment of a path: derived, never stored, unique per user
# because `(user_id, path)` is. `prefix` narrows a read or a commit to part of the
# namespace; "" or "/" is the whole store.


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
    return [TreeEntry(path=r["path"], content_hash=r["content_hash"], size=r["size"], mtime=r["mtime"]) for r in rows]


@dataclass(frozen=True, slots=True)
class StoredFile:
    """One file as the tree now holds it: its row id and its entry."""

    id: str
    entry: TreeEntry


@dataclass(frozen=True, slots=True)
class Folder:
    """A top-level segment of the store, and how many files are under it."""

    name: str
    files: int


def folder_of(path: str) -> str:
    """The folder a store path belongs to: its first segment."""
    return path.split("/", 1)[0]


async def folders(user_id: str) -> list[Folder]:
    """Every folder in the user's store, alphabetically, with its file count."""
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


# Advisory-lock namespace reserved for folder naming; collides with no other lock.
_NAMING_LOCK = 8809


async def unique_folder(user_id: str, base: str) -> str:
    """Reserve a folder name not already taken in this user's store, and return it.

    The advisory lock is transaction-scoped and held across BOTH the read and the
    sentinel write, so two concurrent callers cannot reserve the same name.
    """
    async with (await pool.pool()).acquire() as conn, conn.transaction():
        await conn.execute("SELECT pg_advisory_xact_lock($1, hashtext($2))", _NAMING_LOCK, str(user_id))
        rows = await conn.fetch(
            "SELECT DISTINCT split_part(path, '/', 1) AS name FROM files WHERE user_id = $1",
            _uuid(user_id),
        )
        taken = {r["name"] for r in rows}
        name = base
        n = 2
        while name in taken:
            name = f"{base}-{n}"
            n += 1
        # Reserved INSIDE the lock: the sentinel is what makes the folder exist.
        content_hash = await put_blob(b"")
        await conn.execute(
            """
            INSERT INTO files (user_id, path, content_hash, size, mtime)
            VALUES ($1, $2, $3, 0, now())
            ON CONFLICT (user_id, path) DO NOTHING
            """,
            _uuid(user_id),
            dir_sentinel(name),
            content_hash,
        )
        return name


def safe_path(name: str) -> str:
    """Normalize a name to a path inside the store; `..` is refused, not resolved.

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

    Blob first, row after: a crash between the two costs an orphan blob rather
    than a row pointing at bytes that are not there.
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
        entry=TreeEntry(path=row["path"], content_hash=row["content_hash"], size=row["size"], mtime=row["mtime"]),
    )


# The sandbox round trip carries files and only files, so an empty folder survives
# only as a zero-byte file — and this is also what makes an unfilled folder exist.
DIR_SENTINEL = ".keep"


def dir_sentinel(path: str) -> str:
    """The sentinel path that makes an empty directory durable."""
    return f"{path}/{DIR_SENTINEL}"


async def move_path(user_id: str, src: str, dst: str) -> list[tuple[str, str]]:
    """
    Move one file, or a whole subtree, to another path in the user's store.

    Blobs never move: a move is a row edit, all rows in one transaction. A
    DIRECTORY may move out to the top level, becoming a folder; a FILE may not.
    File-vs-directory is answered by the ROWS, inside the transaction.

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
        # A top-level folder also lives in the links and the claims: that is
        # `rename_path`, which rewrites all three.
        raise StoreError(f"{src!r} is a folder; renaming or moving one is not something this can do")
    if dst.startswith(f"{src}/"):
        raise StoreError(f"cannot move {src!r} into itself")

    async with (await pool.pool()).acquire() as conn, conn.transaction():
        if "/" not in dst:
            # Only a directory may go to the top level; an exact row match means
            # `src` names a file.
            is_file = await conn.fetchval("SELECT 1 FROM files WHERE user_id = $1 AND path = $2", _uuid(user_id), src)
            if is_file:
                raise StoreError(f"the store's top level holds folders, not files: {dst!r} needs a folder to go in")
        moves = await _rewrite_prefix(conn, user_id, src, dst)

    # mtime is left alone: a move changes no content, and materialize transfers by hash.
    return moves


async def _rewrite_prefix(conn: Any, user_id: str, src: str, dst: str) -> list[tuple[str, str]]:
    """Move every row at or under `src` to the same position under `dst`.

    Runs inside the CALLER's transaction, so a rename that also rewrites links
    and claims does all three or none.

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

    # An exact row match means `src` is a file; otherwise only the suffix is kept.
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
        ValueError: the name is empty, carries a separator, or is a relative step.
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

    A top-level folder's name is duplicated in exactly three tables — `files`,
    `project_folders`, `session_claims` — and all three move in ONE transaction.
    It does NOT touch a live sandbox: the caller must first check that no box
    holds the folder, or that box's next flush resurrects the old name.

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
        # The NAME must be free, not merely the paths under it: `_rewrite_prefix`
        # checks collisions path by path, which would silently merge two folders.
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
            # A top-level folder: its name is also in the links and the claims.
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

    Rows move to `deleted_files`; the BLOBS are untouched, so undo is exact. A
    delete that empties a folder also drops the links naming it, in the same
    batch. The caller must first check that no live box has the folder mounted,
    or its next flush puts the files back.

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

        # Asked AFTER the rows are gone, so it is what the tree now derives to.
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

    The same rows under the same ids, pointing at the same blobs, plus the links
    that went with the folders.

    Raises:
        MissingPath: no such batch for this user, or it has already been undone.
        StoreError: something now occupies a path the delete freed.
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

        links = await conn.fetch("SELECT project_id, folder FROM deleted_links WHERE batch = $1", _uuid(batch))
        for link in links:
            await conn.execute(
                "INSERT INTO project_folders (project_id, folder) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                link["project_id"],
                link["folder"],
            )

        await conn.execute("DELETE FROM deleted_links WHERE batch = $1", _uuid(batch))
        await conn.execute("DELETE FROM deleted_files WHERE user_id = $1 AND batch = $2", _uuid(user_id), _uuid(batch))

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

    Blobs are uploaded before any row is touched and the rows flip in one
    transaction, so an interrupted commit leaves the previous tree whole.

    Returns:
        The tree that is now under `prefix`.
    """
    hashed = [(f, sha256(f.content)) for f in contents]

    # Blobs first: uploading one twice is free, a row pointing at a missing blob
    # is a lost file.
    for content_hash in await missing_blobs({h for _, h in hashed}):
        content = next(f.content for f, h in hashed if h == content_hash)
        await blobs().put(content_hash, content)

    now = datetime.now(UTC)
    return await commit_entries(
        user_id,
        [TreeEntry(path=f.path, content_hash=h, size=len(f.content), mtime=f.mtime or now) for f, h in hashed],
        prefix,
    )


async def commit_entries(
    user_id: str,
    entries: Sequence[TreeEntry],
    prefix: str = "/",
) -> list[TreeEntry]:
    """
    Replace the tree under `prefix` with entries whose blobs are already stored.

    Every blob is checked present before any row moves, so the tree cannot come
    to point at bytes that are not there.

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


def slug(title: str, fallback: str) -> str:
    """A folder name from a title. Mounted names are read by the model as context."""
    cleaned = re.sub(r"[^a-z0-9]+", "-", (title or "").lower()).strip("-")
    return cleaned[:48] or fallback


def covers(prefix: str, path: str) -> bool:
    """Whether a claim on `prefix` includes `path`. Both are full store paths."""
    scope = _relative(prefix)
    return not scope or path == scope or path.startswith(scope + "/")


def _relative(prefix: str) -> str:
    """Normalize a prefix to a tree prefix. '/' or '' means the whole store."""
    return (prefix or "").strip("/")
