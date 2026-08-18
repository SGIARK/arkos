"""
Filling and emptying the sandbox's cache of the store.

The sandbox disk holds a copy of the claimed subtrees, nothing more (D27).
`materialize` puts them there when the session takes its box; `flush` reads back
what changed and commits it. Between the two the sandbox is free to be
destroyed: whatever survives there is a warm start, not the record.

Bytes move store -> harness -> sandbox and back. The sandbox is never given a
credential and never reaches the store itself (D28), and it is asked nothing
about its own contents that is not verified: both directions hash the files on
disk and compare against the tree, so a stale or tampered record cannot decide
what is transferred, kept or deleted.

Claimed project subtrees are the only thing this module mounts. The user's
memory region is a different table in the store and nothing here reads it, so
memory does not reach a box today. Whether it ever should is D30, open: this is
the default posture, not a rule the code is built around.

A flush may only commit against a workspace that proves it was materialized.
`materialize` leaves a sentinel in the box and records its nonce against the
session's slot; `flush` reads both and aborts unless they agree. The proof is
about the box, not its contents, so a session that deleted every file still
commits that deletion, while an empty, replaced or foreign box commits nothing.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import posixpath
import shlex
import tarfile
import uuid as _uuid_module
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from db import pool
from harness_module import store

logger = logging.getLogger(__name__)

# Where claimed projects appear inside the sandbox.
MOUNT_ROOT = "/home/user/projects"

# The sentinel, outside MOUNT_ROOT so no sweep of the claimed mounts can see it.
SENTINEL = "/home/user/.ark/materialized.json"

_STAGING_TAR = "/tmp/arkos-materialize.tar"
_FLUSH_TAR = "/tmp/arkos-flush.tar"


@dataclass(frozen=True, slots=True)
class Claim:
    """One project subtree a session may see, and where it is mounted."""

    project_id: str
    slug: str
    subpath: str = "/"
    mode: str = "write"

    @property
    def mount(self) -> str:
        return posixpath.join(MOUNT_ROOT, self.slug)


@dataclass(frozen=True, slots=True)
class Materialized:
    """What ended up in the sandbox, and what it cost to put there."""

    manifest: dict[str, str]
    transferred: int
    bytes_sent: int
    removed: tuple[str, ...] = ()
    nonce: str = ""


class SandboxIO(Protocol):
    """The part of the sandbox this module uses. Keyed by session: the box is the session's."""

    async def exec(self, session_id: str, command: str, timeout: int = ...) -> dict[str, Any]: ...

    async def write_file(self, session_id: str, path: str, content: Any) -> None: ...

    async def read_file(self, session_id: str, path: str) -> str: ...

    async def read_bytes(self, session_id: str, path: str) -> bytes: ...


async def claims_for(session_id: str) -> list[Claim]:
    """What a session may see, in the order it was declared.

    A session with no declared claims gets a write claim on its own project,
    which is what every session created before claims existed had in effect.
    """
    rows = await pool.fetch(
        """
        SELECT c.project_id, c.subpath, c.mode, p.title
          FROM session_claims c JOIN projects p ON p.id = c.project_id
         WHERE c.session_id = $1
         ORDER BY c.project_id, c.subpath
        """,
        _uuid(session_id),
    )
    if not rows:
        rows = await pool.fetch(
            """
            SELECT s.project_id, '/' AS subpath, 'write' AS mode, p.title
              FROM sessions s JOIN projects p ON p.id = s.project_id
             WHERE s.id = $1
            """,
            _uuid(session_id),
        )
    return [
        Claim(
            project_id=str(r["project_id"]),
            slug=store.slug(r["title"], str(r["project_id"])[:8]),
            subpath=r["subpath"],
            mode=r["mode"],
        )
        for r in rows
    ]


def lease_keys(claims: list[Claim]) -> list[str]:
    """The project leases a claim set takes. Read claims take none."""
    return [f"project:{c.project_id}" for c in claims if c.mode == "write"]


async def materialize(sandbox: SandboxIO, session_id: str, claims: list[Claim]) -> Materialized:
    """
    Put the claimed subtrees in the sandbox, transferring only what is missing.

    A fresh sandbox receives everything. A resumed one receives only the files
    whose contents differ from the tree, and has anything the tree does not
    contain deleted — including a file another session removed while this
    sandbox slept, which would otherwise be resurrected by the next flush.
    """
    wanted: dict[str, str] = {}
    contents: dict[str, tuple[str, str]] = {}
    for claim in claims:
        for entry in await store.read_tree(claim.project_id, claim.subpath):
            mounted = posixpath.join(claim.mount, entry.path)
            wanted[mounted] = entry.content_hash
            contents[mounted] = (claim.project_id, entry.content_hash)

    # What is actually on disk, by hash. The sandbox's own record of what it
    # holds is not consulted: it is stale the moment a flush commits, and a
    # file it forgot about is exactly the file a deletion needs removed.
    on_disk = await _sweep(sandbox, session_id, claims)
    stale = [path for path, digest in wanted.items() if on_disk.get(path) != digest]
    removed = tuple(sorted(set(on_disk) - set(wanted)))

    payload: list[tuple[str, bytes]] = []
    for path in sorted(stale):
        _, content_hash = contents[path]
        blob = await store.get_blob(content_hash)
        if blob is None:
            # The tree names a blob the store does not hold. Skipping it beats
            # writing a file whose contents are a guess.
            logger.error("blob %s for %s is missing from the store", content_hash[:12], path)
            continue
        payload.append((path.lstrip("/"), blob))

    if removed:
        await _remove(sandbox, session_id, removed)

    bytes_sent = 0
    if payload:
        archive = store.build_tar(payload)
        bytes_sent = len(archive)
        await sandbox.write_file(session_id, _STAGING_TAR, archive)
        result = await sandbox.exec(
            session_id,
            f"mkdir -p {shlex.quote(MOUNT_ROOT)} && tar xf {shlex.quote(_STAGING_TAR)} -C / "
            f"&& rm -f {shlex.quote(_STAGING_TAR)}",
        )
        if result["exit_code"] != 0:
            raise store.StoreError(f"materialize failed to extract: {result['stderr'][:200]}")

    nonce = await _seal(sandbox, session_id, claims, wanted)
    logger.info(
        "materialized %d file(s) for session %s (%d transferred, %d removed)",
        len(wanted),
        session_id,
        len(payload),
        len(removed),
    )
    return Materialized(
        manifest=wanted, transferred=len(payload), bytes_sent=bytes_sent, removed=removed, nonce=nonce
    )


def _tree_hash(manifest: dict[str, str]) -> str:
    """One hash over the materialized tree: path and content hash, in path order."""
    body = "\n".join(f"{path}:{digest}" for path, digest in sorted(manifest.items()))
    return hashlib.sha256(body.encode()).hexdigest()


async def _seal(sandbox: SandboxIO, session_id: str, claims: list[Claim], manifest: dict[str, str]) -> str:
    """Write the sentinel into the box and record its nonce against the session's slot.

    The box is written first: a nonce recorded for a sentinel that never landed
    refuses the next flush, which is the safe direction to fail.

    Raises:
        StoreError: the session holds no slot, so there is nowhere to record the
            nonce and nothing may be committed from this box later.
    """
    nonce = _uuid_module.uuid4().hex
    payload = {
        "nonce": nonce,
        "tree_hash": _tree_hash(manifest),
        "claims": [{"project_id": c.project_id, "subpath": c.subpath, "mount": c.mount} for c in claims],
    }
    await sandbox.write_file(session_id, SENTINEL, json.dumps(payload))
    recorded = await pool.execute(
        "UPDATE session_sandboxes SET workspace_nonce = $2 WHERE session_id = $1",
        _uuid(session_id),
        nonce,
    )
    if not recorded.endswith(" 1"):
        raise store.StoreError(f"session {session_id} holds no sandbox slot to seal")
    return nonce


async def _verify_seal(sandbox: SandboxIO, session_id: str, manifest: dict[str, str] | None) -> None:
    """Refuse a flush from a box that cannot prove it is the one that was materialized.

    Raises:
        StoreError: the sentinel is missing, unreadable, or names a different
            workspace. The caller keeps the box and its leases: the disk may
            still hold the only copy of the work.
    """
    expected = await pool.fetchval(
        "SELECT workspace_nonce FROM session_sandboxes WHERE session_id = $1", _uuid(session_id)
    )
    try:
        raw = await sandbox.read_file(session_id, SENTINEL)
        sealed = json.loads(raw)
    except Exception as e:  # noqa: BLE001 - a missing or corrupt sentinel is one answer: no proof
        raise store.StoreError(
            f"refusing to flush session {session_id}: the sandbox carries no materialize sentinel ({e})"
        ) from e

    if not expected or sealed.get("nonce") != expected:
        raise store.StoreError(
            f"refusing to flush session {session_id}: the sandbox was materialized for another workspace"
        )
    if manifest is not None and sealed.get("tree_hash") != _tree_hash(manifest):
        raise store.StoreError(
            f"refusing to flush session {session_id}: the sandbox holds a different tree than the flush expects"
        )


@dataclass(frozen=True, slots=True)
class Flushed:
    """What a flush moved back into the store."""

    committed: int
    uploaded: int
    discarded: tuple[str, ...] = ()


async def flush(
    sandbox: SandboxIO,
    session_id: str,
    claims: list[Claim],
    manifest: dict[str, str] | None = None,
) -> Flushed:
    """
    Commit what the sandbox changed back to the store.

    One sha256sum sweep says what is on disk now and the tree says what the store
    holds. Only the differences have their bytes read back, and every file's row
    is written from a hash, so an unchanged file costs nothing. A path the tree
    has and the sweep does not is a deletion, which falls out of replacing the
    subtree rather than needing a rule.

    `manifest` is accepted for callers that already have the materialized map;
    when absent the tree is read instead, which is the same answer.

    A read claim commits nothing. Its edits are discarded and named in the
    return value, because silently keeping or silently dropping them are both
    worse than saying which.

    Raises:
        StoreError: the box cannot prove it is the one that was materialized. An
        empty sweep of a replaced box would otherwise commit an empty tree, which
        is a deletion of the project.
    """
    await _verify_seal(sandbox, session_id, manifest)
    current = await _sweep(sandbox, session_id, claims)
    if manifest is None:
        manifest = {}
        for claim in claims:
            for entry in await store.read_tree(claim.project_id, claim.subpath):
                manifest[posixpath.join(claim.mount, entry.path)] = entry.content_hash
    changed = sorted(path for path, digest in current.items() if manifest.get(path) != digest)

    contents: dict[str, bytes] = await _read_out(sandbox, session_id, changed) if changed else {}

    committed = 0
    uploaded = 0
    discarded: list[str] = []
    for claim in claims:
        under = {p: h for p, h in current.items() if p.startswith(claim.mount + "/")}
        if claim.mode != "write":
            # Measured against what the store holds now, not against what was
            # materialized: a file uploaded mid-run and written through is not an
            # edit being dropped, and saying it was would be a false alarm.
            stored = {
                posixpath.join(claim.mount, e.path): e.content_hash
                for e in await store.read_tree(claim.project_id, claim.subpath)
            }
            discarded.extend(sorted(p for p, digest in under.items() if stored.get(p) != digest))
            continue

        # Sizes for files whose bytes were never read back come from the tree
        # they were materialized from.
        previous = {e.path: e for e in await store.read_tree(claim.project_id, claim.subpath)}
        now = datetime.now(UTC)

        entries: list[store.TreeEntry] = []
        for path, digest in sorted(under.items()):
            relative = posixpath.relpath(path, claim.mount)
            body = contents.get(path)
            if body is not None:
                entries.append(
                    store.TreeEntry(
                        path=relative,
                        content_hash=await store.put_blob(body),
                        size=len(body),
                        mtime=now,
                    )
                )
                uploaded += 1
                continue

            known = previous.get(relative)
            if known is None or known.content_hash != digest:
                # Unchanged by the manifest, yet the tree does not agree. Read
                # it rather than record a size that would be a guess.
                extra = await _read_out(sandbox, session_id, [path])
                body = extra.get(path, b"")
                entries.append(
                    store.TreeEntry(
                        path=relative, content_hash=await store.put_blob(body), size=len(body), mtime=now
                    )
                )
                uploaded += 1
                continue

            entries.append(known)

        await store.commit_entries(claim.project_id, entries, claim.subpath)
        committed += len(entries)

    if discarded:
        logger.warning("discarded %d edit(s) under a read claim in session %s", len(discarded), session_id)
    logger.info("flushed %d file(s) for session %s (%d uploaded)", committed, session_id, uploaded)
    return Flushed(committed=committed, uploaded=uploaded, discarded=tuple(discarded))


async def _sweep(sandbox: SandboxIO, session_id: str, claims: list[Claim]) -> dict[str, str]:
    """Hash every file under the claimed mounts, in one command."""
    mounts = " ".join(shlex.quote(c.mount) for c in claims)
    if not mounts:
        return {}
    result = await sandbox.exec(
        session_id,
        f"find {mounts} -type f -exec sha256sum {{}} + 2>/dev/null || true",
    )
    found: dict[str, str] = {}
    for line in (result.get("stdout") or "").splitlines():
        digest, _, path = line.partition("  ")
        if len(digest) == 64 and path:
            found[path.strip()] = digest
    return found


async def _read_out(sandbox: SandboxIO, session_id: str, paths: list[str]) -> dict[str, bytes]:
    """Tar the changed files and read the archive back in one transfer."""
    quoted = " ".join(shlex.quote(p.lstrip("/")) for p in paths)
    result = await sandbox.exec(
        session_id, f"tar cf {shlex.quote(_FLUSH_TAR)} -C / {quoted}"
    )
    if result["exit_code"] != 0:
        raise store.StoreError(f"flush could not archive the changes: {result['stderr'][:200]}")

    archive = await sandbox.read_bytes(session_id, _FLUSH_TAR)
    await sandbox.exec(session_id, f"rm -f {shlex.quote(_FLUSH_TAR)}")

    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                extracted = tar.extractfile(member)
                if extracted is not None:
                    out["/" + member.name.lstrip("/")] = extracted.read()
    return out


async def _remove(sandbox: SandboxIO, session_id: str, paths: tuple[str, ...]) -> None:
    """Delete files the tree no longer has, so a resumed sandbox does not keep them."""
    quoted = " ".join(shlex.quote(p) for p in paths)
    await sandbox.exec(session_id, f"rm -f {quoted}")


def _uuid(value: str) -> _uuid_module.UUID:
    if isinstance(value, _uuid_module.UUID):
        return value
    return _uuid_module.UUID(str(value))


async def write_through(sandbox: SandboxIO, project_id: str, path: str, content: bytes) -> list[str]:
    """
    Put an uploaded file into every live box that has this project materialized.

    A box is written to only if the file falls inside a claim it mounted; every
    other box needs nothing, because the store has the file and the next
    materialize brings it in. Parked sessions are left asleep for the same
    reason. Failures are logged rather than raised: the store already holds the
    file, so the upload stands either way.

    Returns:
        The sessions whose box now has the file.
    """
    rows = await pool.fetch(
        """
        SELECT p.session_id
          FROM session_sandboxes p
          JOIN sessions s ON s.id = p.session_id
         WHERE p.workspace_nonce IS NOT NULL
           AND p.sandbox_id IS NOT NULL
           AND p.expires_at > now()
           AND s.status = 'running'
           AND s.user_id = (SELECT user_id FROM projects WHERE id = $1)
        """,
        _uuid(project_id),
    )

    written: list[str] = []
    for row in rows:
        session_id = str(row["session_id"])
        for claim in await claims_for(session_id):
            if claim.project_id != project_id or not store.covers(claim.subpath, path):
                continue
            try:
                await sandbox.write_file(session_id, posixpath.join(claim.mount, path), content)
            except Exception:  # noqa: BLE001 - the store has the file; the box is a cache
                logger.exception("could not write %s through to the box of session %s", path, session_id)
            else:
                written.append(session_id)
            break
    if written:
        logger.info("wrote %s through to %d live box(es)", path, len(written))
    return written
