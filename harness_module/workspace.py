"""
Filling and emptying the sandbox's cache of the store.

The sandbox disk holds a copy of the claimed subtrees, nothing more (D27).
`materialize` puts them there when the lease is taken; `flush` reads back what
changed and commits it. Between the two the sandbox is free to be destroyed:
whatever survives there is a warm start, not the record.

Bytes move store -> harness -> sandbox and back. The sandbox is never given a
credential and never reaches the store itself (D28), so the manifest below is
the only thing it is trusted with, and only as a hint about what it already has.
"""

from __future__ import annotations

import json
import logging
import posixpath
import shlex
from dataclasses import dataclass
from typing import Any, Protocol

from harness_module import store

logger = logging.getLogger(__name__)

# Where claimed projects appear inside the sandbox.
MOUNT_ROOT = "/home/user/projects"

# What was materialized last time, so a resumed sandbox transfers only the
# difference. A hint: every path in it is verified against the tree before it
# is trusted to be current.
MANIFEST_PATH = "/home/user/.arkos/manifest.json"

_STAGING_TAR = "/tmp/arkos-materialize.tar"


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


async def materialize(sandbox: SandboxIO, user_id: str, claims: list[Claim]) -> Materialized:
    """
    Put the claimed subtrees in the sandbox, transferring only what is missing.

    A fresh sandbox receives everything. A resumed one receives the difference
    between its manifest and the tree as it stands now, and has files the tree
    no longer contains removed.
    """
    wanted: dict[str, str] = {}
    contents: dict[str, tuple[str, str]] = {}
    for claim in claims:
        for entry in await store.read_tree(claim.project_id, claim.subpath):
            mounted = posixpath.join(claim.mount, entry.path)
            wanted[mounted] = entry.content_hash
            contents[mounted] = (claim.project_id, entry.content_hash)

    present = await _read_manifest(sandbox, user_id)
    stale = [path for path, digest in wanted.items() if present.get(path) != digest]
    removed = tuple(sorted(set(present) - set(wanted)))

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

    await _write_manifest(sandbox, user_id, wanted)
    logger.info(
        "materialized %d file(s) for user %s (%d transferred, %d removed)",
        len(wanted),
        user_id,
        len(payload),
        len(removed),
    )
    return Materialized(manifest=wanted, transferred=len(payload), bytes_sent=bytes_sent, removed=removed)


async def _read_manifest(sandbox: SandboxIO, user_id: str) -> dict[str, str]:
    """What the sandbox says it already has. Absent or unreadable means nothing."""
    try:
        raw = await sandbox.read_file(user_id, MANIFEST_PATH)
    except Exception:
        return {}
    try:
        loaded = json.loads(raw)
    except (TypeError, ValueError):
        logger.warning("manifest in the sandbox for user %s is unreadable; treating it as empty", user_id)
        return {}
    return {str(k): str(v) for k, v in loaded.items()} if isinstance(loaded, dict) else {}


async def _write_manifest(sandbox: SandboxIO, user_id: str, manifest: dict[str, str]) -> None:
    await sandbox.exec(user_id, f"mkdir -p {shlex.quote(posixpath.dirname(MANIFEST_PATH))}")
    await sandbox.write_file(user_id, MANIFEST_PATH, json.dumps(manifest, sort_keys=True))


async def _remove(sandbox: SandboxIO, user_id: str, paths: tuple[str, ...]) -> None:
    """Delete files the tree no longer has, so a resumed sandbox does not keep them."""
    quoted = " ".join(shlex.quote(p) for p in paths)
    await sandbox.exec(user_id, f"rm -f {quoted}")
