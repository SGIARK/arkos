"""
Filling and emptying the sandbox's cache of the store: materialize, flush, seal.

Bytes move store -> harness -> sandbox and back; the sandbox never holds a
credential (D28), and both directions hash the files on disk rather than trust
any record of what the box contains.
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
from db.ids import as_uuid as _uuid
from harness_module import store

logger = logging.getLogger(__name__)

# A mounted path is `MOUNT_ROOT + "/" + <store path>`: one namespace, not two.
MOUNT_ROOT = "/home/user/store"

# Outside MOUNT_ROOT so no sweep of the claimed mounts can pick it up.
SENTINEL = "/home/user/.ark/materialized.json"

_STAGING_TAR = "/tmp/arkos-materialize.tar"
_FLUSH_TAR = "/tmp/arkos-flush.tar"


@dataclass(frozen=True, slots=True)
class Claim:
    """One folder, or part of one, that a session may see — and where it is mounted."""

    user_id: str
    folder: str
    subpath: str = "/"
    mode: str = "write"

    @property
    def prefix(self) -> str:
        """What this claim covers, as a store path prefix."""
        within = (self.subpath or "/").strip("/")
        return f"{self.folder}/{within}" if within else self.folder

    @property
    def mount(self) -> str:
        """The folder's directory in the box: `~/store/<folder>/`."""
        return posixpath.join(MOUNT_ROOT, self.folder)

    @property
    def mounted_prefix(self) -> str:
        """Where this claim's own subtree sits in the box.

        Narrower than `mount` for a subpath claim, so a sweep cannot carry files
        the claim does not cover into a commit that replaces them.
        """
        return posixpath.join(MOUNT_ROOT, self.prefix)


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

    Claims are FIXED for a session's life: a folder linked mid-run reaches the
    agent at the next session. With none declared, falls back to a write claim
    on every folder the project links.
    """
    rows = await pool.fetch(
        """
        SELECT c.folder, c.subpath, c.mode, s.user_id
          FROM session_claims c JOIN sessions s ON s.id = c.session_id
         WHERE c.session_id = $1
         ORDER BY c.ord, c.folder, c.subpath
        """,
        _uuid(session_id),
    )
    if not rows:
        rows = await pool.fetch(
            """
            SELECT f.folder, '/' AS subpath, 'write' AS mode, s.user_id
              FROM sessions s JOIN project_folders f ON f.project_id = s.project_id
             WHERE s.id = $1
             ORDER BY f.created_at, f.folder
            """,
            _uuid(session_id),
        )
    return [
        Claim(
            user_id=str(r["user_id"]),
            folder=r["folder"],
            subpath=r["subpath"],
            mode=r["mode"],
        )
        for r in rows
    ]


def lease_key(claim: Claim) -> str | None:
    """The folder lease this claim takes, or None. A read claim takes none.

    Per FOLDER, not per project (11.9): two sessions writing the same folder
    serialize even across projects. This is the only place that rule lives.
    """
    return f"folder:{claim.user_id}:{claim.folder}" if claim.mode == "write" else None


def lease_keys(claims: list[Claim]) -> list[str]:
    """The folder leases a claim set takes."""
    return [key for c in claims if (key := lease_key(c))]


async def materialize(sandbox: SandboxIO, session_id: str, claims: list[Claim]) -> Materialized:
    """
    Put the claimed subtrees in the sandbox, transferring only what is missing.

    A resumed sandbox also has anything the tree no longer holds deleted, so the
    next flush cannot resurrect a file another session removed.
    """
    wanted: dict[str, str] = {}
    for claim in claims:
        for entry in await store.read_tree(claim.user_id, claim.prefix):
            # A store path already carries its folder segment, so the mount root
            # is the whole mapping.
            wanted[posixpath.join(MOUNT_ROOT, entry.path)] = entry.content_hash

    # Hashed from disk, never from the box's own record: that record is stale
    # the moment a flush commits.
    on_disk = await _sweep(sandbox, session_id, claims)
    stale = [path for path, digest in wanted.items() if on_disk.get(path) != digest]
    removed = tuple(sorted(set(on_disk) - set(wanted)))

    payload: list[tuple[str, bytes]] = []
    for path in sorted(stale):
        content_hash = wanted[path]
        blob = await store.get_blob(content_hash)
        if blob is None:
            raise store.StoreError(f"cannot materialize {path}: blob {content_hash[:12]} is missing")
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
    return Materialized(manifest=wanted, transferred=len(payload), bytes_sent=bytes_sent, removed=removed, nonce=nonce)


def _tree_hash(manifest: dict[str, str]) -> str:
    """One hash over the materialized tree: path and content hash, in path order."""
    body = "\n".join(f"{path}:{digest}" for path, digest in sorted(manifest.items()))
    return hashlib.sha256(body.encode()).hexdigest()


async def _seal(sandbox: SandboxIO, session_id: str, claims: list[Claim], manifest: dict[str, str]) -> str:
    """Write the sentinel into the box and record its nonce against the session's slot.

    Order matters: a nonce recorded for a sentinel that never landed refuses the
    next flush, which is the safe direction to fail.

    Raises:
        StoreError: the session holds no sandbox slot to record the nonce against.
    """
    nonce = _uuid_module.uuid4().hex
    payload = {
        "nonce": nonce,
        "tree_hash": _tree_hash(manifest),
        "claims": [{"folder": c.folder, "subpath": c.subpath, "mount": c.mount} for c in claims],
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
            workspace. The caller keeps the box and its leases — the disk may
            hold the only copy of the work.
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

    `manifest` is optional; when absent the tree is read instead, which is the
    same answer. A read claim commits nothing — its edits are discarded and
    named in the return value.

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
            for entry in await store.read_tree(claim.user_id, claim.prefix):
                manifest[posixpath.join(MOUNT_ROOT, entry.path)] = entry.content_hash
    changed = sorted(path for path, digest in current.items() if manifest.get(path) != digest)

    contents: dict[str, bytes] = await _read_out(sandbox, session_id, changed) if changed else {}

    committed = 0
    uploaded = 0
    discarded: list[str] = []
    for claim in claims:
        under = {p: h for p, h in current.items() if p.startswith(claim.mounted_prefix + "/")}
        if claim.mode != "write":
            # Measured against what the store holds NOW, so a file written
            # through mid-run is not reported as a dropped edit.
            stored = {
                posixpath.join(MOUNT_ROOT, e.path): e.content_hash
                for e in await store.read_tree(claim.user_id, claim.prefix)
            }
            discarded.extend(sorted(p for p, digest in under.items() if stored.get(p) != digest))
            continue

        # Sizes for files whose bytes were never read back come from the tree.
        previous = {e.path: e for e in await store.read_tree(claim.user_id, claim.prefix)}
        now = datetime.now(UTC)

        entries: list[store.TreeEntry] = []
        for path, digest in sorted(under.items()):
            relative = posixpath.relpath(path, MOUNT_ROOT)
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
                # Unchanged by the manifest yet the tree disagrees: read it
                # rather than record a size that would be a guess.
                extra = await _read_out(sandbox, session_id, [path])
                body = extra.get(path, b"")
                entries.append(
                    store.TreeEntry(path=relative, content_hash=await store.put_blob(body), size=len(body), mtime=now)
                )
                uploaded += 1
                continue

            entries.append(known)

        await store.commit_entries(claim.user_id, entries, claim.prefix)
        committed += len(entries)

    if discarded:
        logger.warning("discarded %d edit(s) under a read claim in session %s", len(discarded), session_id)
    logger.info("flushed %d file(s) for session %s (%d uploaded)", committed, session_id, uploaded)
    return Flushed(committed=committed, uploaded=uploaded, discarded=tuple(discarded))


async def _sweep(sandbox: SandboxIO, session_id: str, claims: list[Claim]) -> dict[str, str]:
    """Hash every file under the claimed mounts, in one command."""
    mounts = " ".join(shlex.quote(c.mounted_prefix) for c in claims)
    if not mounts:
        return {}
    result = await sandbox.exec(
        session_id,
        f"mkdir -p {mounts} && find {mounts} -type f -exec sha256sum {{}} +",
    )
    if result["exit_code"] != 0:
        raise store.StoreError(f"cannot scan sandbox files: {result['stderr'][:200]}")
    found: dict[str, str] = {}
    for line in (result.get("stdout") or "").splitlines():
        digest, _, path = line.partition("  ")
        if len(digest) == 64 and path:
            found[path.strip()] = digest
    return found


async def _read_out(sandbox: SandboxIO, session_id: str, paths: list[str]) -> dict[str, bytes]:
    """Tar the changed files and read the archive back in one transfer."""
    quoted = " ".join(shlex.quote(p.lstrip("/")) for p in paths)
    result = await sandbox.exec(session_id, f"tar cf {shlex.quote(_FLUSH_TAR)} -C / {quoted}")
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


async def _live_boxes(user_id: str) -> list[str]:
    """The sessions whose box is awake and holding some of this user's work."""
    rows = await pool.fetch(
        """
        SELECT p.session_id
          FROM session_sandboxes p
          JOIN sessions s ON s.id = p.session_id
         WHERE p.workspace_nonce IS NOT NULL
           AND p.sandbox_id IS NOT NULL
           AND p.expires_at > now()
           AND s.status = 'running'
           AND s.user_id = $1
        """,
        _uuid(user_id),
    )
    return [str(r["session_id"]) for r in rows]


async def write_through(sandbox: SandboxIO, user_id: str, path: str, content: bytes) -> list[str]:
    """
    Put an uploaded file into every live box that has its folder materialized.

    Failures are logged rather than raised: the store already holds the file, so
    the upload stands either way.

    Returns:
        The sessions whose box now has the file.
    """
    written: list[str] = []
    for session_id in await _live_boxes(user_id):
        for claim in await claims_for(session_id):
            if not store.covers(claim.prefix, path):
                continue
            try:
                await sandbox.write_file(session_id, posixpath.join(MOUNT_ROOT, path), content)
            except Exception:  # noqa: BLE001 - the store has the file; the box is a cache
                logger.exception("could not write %s through to the box of session %s", path, session_id)
            else:
                written.append(session_id)
            break
    if written:
        logger.info("wrote %s through to %d live box(es)", path, len(written))
    return written
