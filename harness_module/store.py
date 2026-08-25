"""
The agent's filesystem: the TREE. Bytes live in `blobs`, keyed by their hash.

ONE FLAT NAMESPACE PER USER. A row maps `(user_id, path)` to the hash of that
path's content, and a FOLDER is the first segment of a path — derived, never a
row, unique per user by construction, and alive exactly as long as a file exists
under it (11.9). No project owns a folder; projects LINK folders, which is
`project_folders`, and deleting a project deletes its links and no files.

`commit_tree` uploads every missing blob before it touches a row, and flips the
rows in one transaction. A crash between the two leaves the previous tree intact
and whole: an orphan blob costs storage, a row pointing at a blob that is not
there costs a file.

Nothing here knows about the sandbox, e2b or tools. The store is the harness's
(D28) and bytes reach a sandbox by being handed to it, never by it reaching in.

11.8.8 split this file by idea. `blobs.py` is content-addressed bytes and the
HTTP client that carries them; `memory.py` is the user's notes and curated core,
which is keyed by user, mounts nowhere, and shared nothing with the tree but a
filename. The imports go one way — blobs <- store <- workspace — and this module
re-exports the blob calls it is the natural caller of, so the tree's own users
do not have to know where bytes are kept.
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

# Re-exported so a caller that reads the tree and then wants the bytes has one
# import. `blobs.py` is the owner; this is a doorway, not a second copy.
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


# The advisory lock two concurrent folder-namings contend on, in a namespace of
# its own so it cannot collide with any other advisory lock this database grows.
_NAMING_LOCK = 8809


async def unique_folder(user_id: str, base: str) -> str:
    """Reserve a folder name not already taken in this user's store, and return it.

    The none-case of creating a project: no links were picked, so a folder named
    after the project is made for it. Folder names are unique per user by
    construction — they are segments of unique paths — so the collision is
    resolved before the name is used rather than caught after.

    Check-then-act needs a gate, and this one takes it: two projects created at
    the same moment both read the same set of taken names and both picked the
    same one, silently, because reserving the folder is a separate write. The
    lock is transaction-scoped and held across BOTH the read and the sentinel
    that reserves it, so the second caller sees the first one's folder.
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
        # Reserved INSIDE the lock: the sentinel is what makes the folder exist,
        # so a name returned without it is a name the next caller may also pick.
        # The blob is written first, as everywhere — an empty one, already there.
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


