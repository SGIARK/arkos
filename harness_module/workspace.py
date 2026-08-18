"""
Filling and emptying the sandbox's cache of the store.

The sandbox disk holds a copy of the claimed subtrees, nothing more (D27).
`materialize` puts them there when the lease is taken; `flush` reads back what
changed and commits it. Between the two the sandbox is free to be destroyed:
whatever survives there is a warm start, not the record.

Bytes move store -> harness -> sandbox and back. The sandbox is never given a
credential and never reaches the store itself (D28), and it is asked nothing
about its own contents that is not verified: both directions hash the files on
disk and compare against the tree, so a stale or tampered record cannot decide
what is transferred, kept or deleted.
"""

from __future__ import annotations

import io
import logging
import posixpath
import shlex
import tarfile
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from harness_module import store

logger = logging.getLogger(__name__)

# Where claimed projects appear inside the sandbox.
MOUNT_ROOT = "/home/user/projects"

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


class SandboxIO(Protocol):
    """The part of the sandbox this module uses."""

    async def exec(self, user_id: str, command: str, timeout: int = ...) -> dict[str, Any]: ...

    async def write_file(self, user_id: str, path: str, content: Any) -> None: ...

    async def read_file(self, user_id: str, path: str) -> str: ...

    async def read_bytes(self, user_id: str, path: str) -> bytes: ...


async def materialize(sandbox: SandboxIO, user_id: str, claims: list[Claim]) -> Materialized:
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
    on_disk = await _sweep(sandbox, user_id, claims)
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
        await _remove(sandbox, user_id, removed)

    bytes_sent = 0
    if payload:
        archive = store.build_tar(payload)
        bytes_sent = len(archive)
        await sandbox.write_file(user_id, _STAGING_TAR, archive)
        result = await sandbox.exec(
            user_id,
            f"mkdir -p {shlex.quote(MOUNT_ROOT)} && tar xf {shlex.quote(_STAGING_TAR)} -C / "
            f"&& rm -f {shlex.quote(_STAGING_TAR)}",
        )
        if result["exit_code"] != 0:
            raise store.StoreError(f"materialize failed to extract: {result['stderr'][:200]}")

    logger.info(
        "materialized %d file(s) for user %s (%d transferred, %d removed)",
        len(wanted),
        user_id,
        len(payload),
        len(removed),
    )
    return Materialized(manifest=wanted, transferred=len(payload), bytes_sent=bytes_sent, removed=removed)


@dataclass(frozen=True, slots=True)
class Flushed:
    """What a flush moved back into the store."""

    committed: int
    uploaded: int
    discarded: tuple[str, ...] = ()


async def flush(
    sandbox: SandboxIO,
    user_id: str,
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
    """
    current = await _sweep(sandbox, user_id, claims)
    if manifest is None:
        manifest = {}
        for claim in claims:
            for entry in await store.read_tree(claim.project_id, claim.subpath):
                manifest[posixpath.join(claim.mount, entry.path)] = entry.content_hash
    changed = sorted(path for path, digest in current.items() if manifest.get(path) != digest)

    contents: dict[str, bytes] = await _read_out(sandbox, user_id, changed) if changed else {}

    committed = 0
    uploaded = 0
    discarded: list[str] = []
    for claim in claims:
        under = {p: h for p, h in current.items() if p.startswith(claim.mount + "/")}
        if claim.mode != "write":
            discarded.extend(sorted(p for p in under if manifest.get(p) != under[p]))
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
                extra = await _read_out(sandbox, user_id, [path])
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
        logger.warning("discarded %d edit(s) under a read claim for user %s", len(discarded), user_id)
    logger.info("flushed %d file(s) for user %s (%d uploaded)", committed, user_id, uploaded)
    return Flushed(committed=committed, uploaded=uploaded, discarded=tuple(discarded))


async def _sweep(sandbox: SandboxIO, user_id: str, claims: list[Claim]) -> dict[str, str]:
    """Hash every file under the claimed mounts, in one command."""
    mounts = " ".join(shlex.quote(c.mount) for c in claims)
    if not mounts:
        return {}
    result = await sandbox.exec(
        user_id,
        f"find {mounts} -type f -exec sha256sum {{}} + 2>/dev/null || true",
    )
    found: dict[str, str] = {}
    for line in (result.get("stdout") or "").splitlines():
        digest, _, path = line.partition("  ")
        if len(digest) == 64 and path:
            found[path.strip()] = digest
    return found


async def _read_out(sandbox: SandboxIO, user_id: str, paths: list[str]) -> dict[str, bytes]:
    """Tar the changed files and read the archive back in one transfer."""
    quoted = " ".join(shlex.quote(p.lstrip("/")) for p in paths)
    result = await sandbox.exec(
        user_id, f"tar cf {shlex.quote(_FLUSH_TAR)} -C / {quoted}"
    )
    if result["exit_code"] != 0:
        raise store.StoreError(f"flush could not archive the changes: {result['stderr'][:200]}")

    archive = await sandbox.read_bytes(user_id, _FLUSH_TAR)
    await sandbox.exec(user_id, f"rm -f {shlex.quote(_FLUSH_TAR)}")

    out: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile():
                extracted = tar.extractfile(member)
                if extracted is not None:
                    out["/" + member.name.lstrip("/")] = extracted.read()
    return out


async def _remove(sandbox: SandboxIO, user_id: str, paths: tuple[str, ...]) -> None:
    """Delete files the tree no longer has, so a resumed sandbox does not keep them."""
    quoted = " ".join(shlex.quote(p) for p in paths)
    await sandbox.exec(user_id, f"rm -f {quoted}")
