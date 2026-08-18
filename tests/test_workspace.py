"""Materializing the store into a sandbox: everything claimed, only what is missing.

Runs against a real Postgres; the sandbox is a fake that actually applies the tar,
so "byte-identical" is checked against real extracted files rather than a call log.
The box is keyed by session, so that is what the fakes are handed.
"""

from __future__ import annotations

import io
import posixpath
import tarfile
import uuid

import pytest
import pytest_asyncio

from db import pool
from harness_module import store, workspace
from tests.dbgate import require_db

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


class FakeSandbox:
    """A sandbox whose filesystem is a dict, and whose tar/rm commands really run."""

    def __init__(self):
        self.files: dict[str, bytes] = {}
        self.commands: list[str] = []
        self.writes: list[tuple[str, int]] = []

    async def write_file(self, session_id, path, content):
        blob = content.encode() if isinstance(content, str) else content
        self.files[path] = blob
        self.writes.append((path, len(blob)))

    async def read_file(self, session_id, path):
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path].decode()

    async def exec(self, session_id, command, timeout=120):
        self.commands.append(command)
        if command.startswith("rm -f "):
            for token in command[len("rm -f ") :].split():
                self.files.pop(token.strip("'"), None)
        if "tar xf" in command:
            source = next(p for p in self.files if p.endswith(".tar"))
            with tarfile.open(fileobj=io.BytesIO(self.files[source])) as archive:
                for member in archive.getmembers():
                    self.files["/" + member.name] = archive.extractfile(member).read()
            self.files.pop(source, None)
        return {"stdout": "", "stderr": "", "exit_code": 0}


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    store.use_blobs(None)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _project(title: str = "Taxes") -> tuple[str, str]:
    """A session and a project of its own, which is what a claim set names."""
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, $2) RETURNING id", user_id, title
    )
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, project_id, mode, status) VALUES ($1, $2, 'attended', 'idle') RETURNING id",
        user_id,
        project_id,
    )
    return str(session_id), str(project_id)


def _file(path: str, content: str) -> store.FileContent:
    return store.FileContent(path=path, content=content.encode())


def _claim(project_id: str, slug: str = "taxes", **kw) -> workspace.Claim:
    return workspace.Claim(project_id=project_id, slug=slug, **kw)


# --- a fresh sandbox ---------------------------------------------------------------


async def test_a_fresh_sandbox_materializes_byte_identical_to_the_store():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("src/main.py", "print(1)\n"), _file("README.md", "# Taxes\n")])
    sandbox = _sweeping(FakeSandbox())

    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/src/main.py"] == b"print(1)\n"
    assert sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/README.md"] == b"# Taxes\n"
    assert result.transferred == 2


async def test_everything_arrives_in_one_write_and_one_extract():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file(f"f{i}.txt", str(i)) for i in range(10)])
    sandbox = _sweeping(FakeSandbox())

    await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    tar_writes = [p for p, _ in sandbox.writes if p.endswith(".tar")]
    assert len(tar_writes) == 1, "one archive per materialize, whatever the file count"
    assert len([c for c in sandbox.commands if "tar xf" in c]) == 1


async def test_two_claims_mount_side_by_side():
    session_id, first = await _project("Taxes")
    _, second = await _project("Notes")
    await store.commit_tree(first, [_file("a.txt", "1")])
    await store.commit_tree(second, [_file("b.txt", "2")])
    sandbox = _sweeping(FakeSandbox())

    await workspace.materialize(
        sandbox, session_id, [_claim(first, "taxes"), _claim(second, "notes")]
    )

    assert sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] == b"1"
    assert sandbox.files[f"{workspace.MOUNT_ROOT}/notes/b.txt"] == b"2"


async def test_only_the_claimed_subtree_is_mounted():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("src/a.py", "1"), _file("secrets/b.txt", "2")])
    sandbox = _sweeping(FakeSandbox())

    await workspace.materialize(sandbox, session_id, [_claim(project_id, subpath="/src")])

    assert f"{workspace.MOUNT_ROOT}/taxes/src/a.py" in sandbox.files
    assert not any("secrets" in p for p in sandbox.files)


# --- a resumed sandbox --------------------------------------------------------------


async def test_a_resumed_sandbox_transfers_only_what_changed():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "one"), _file("b.txt", "two"), _file("c.txt", "three")])
    sandbox = _sweeping(FakeSandbox())
    await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    await store.commit_tree(project_id, [_file("a.txt", "one"), _file("b.txt", "CHANGED"), _file("c.txt", "three")])
    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert result.transferred == 1, "an unchanged file was sent again"
    assert sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/b.txt"] == b"CHANGED"


async def test_a_resumed_sandbox_with_nothing_changed_transfers_nothing():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "one")])
    sandbox = _sweeping(FakeSandbox())
    await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert result.transferred == 0
    assert result.bytes_sent == 0


async def test_a_file_the_tree_no_longer_has_is_removed_from_the_sandbox():
    """The manifest is a hint about what is there; the tree decides what should be."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("keep.txt", "1"), _file("gone.txt", "2")])
    sandbox = _sweeping(FakeSandbox())
    await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    await store.commit_tree(project_id, [_file("keep.txt", "1")])
    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert f"{workspace.MOUNT_ROOT}/taxes/gone.txt" not in sandbox.files
    assert f"{workspace.MOUNT_ROOT}/taxes/keep.txt" in sandbox.files
    assert result.removed == (f"{workspace.MOUNT_ROOT}/taxes/gone.txt",)


async def test_a_sandbox_claiming_to_have_files_it_does_not_is_not_believed_about_content():
    """The manifest is compared to hashes, so a stale entry is re-sent."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "real")])
    sandbox = _sweeping(FakeSandbox())
    mounted = f"{workspace.MOUNT_ROOT}/taxes/a.txt"
    sandbox.files[mounted] = b"something else entirely"

    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert result.transferred == 1
    assert sandbox.files[mounted] == b"real"


async def test_a_sandbox_is_asked_nothing_about_its_own_contents():
    """Both directions hash the files on disk rather than trusting a record."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    sandbox = _sweeping(FakeSandbox())

    await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert not any("manifest" in p for p in sandbox.files)


async def test_materialize_reports_the_tree_it_put_there():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    sandbox = _sweeping(FakeSandbox())

    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert result.manifest == {f"{workspace.MOUNT_ROOT}/taxes/a.txt": store.sha256(b"1")}


# --- degenerate cases ----------------------------------------------------------------


async def test_an_empty_project_materializes_nothing_and_does_not_fail():
    session_id, project_id = await _project()
    sandbox = _sweeping(FakeSandbox())

    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert result.transferred == 0
    assert result.manifest == {}


async def test_a_tree_row_whose_blob_is_gone_skips_that_file_rather_than_guessing():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1"), _file("b.txt", "2")])
    await pool.execute(
        "UPDATE project_files SET content_hash = $2 WHERE project_id = $1 AND path = 'b.txt'",
        uuid.UUID(project_id),
        "f" * 64,
    )
    sandbox = _sweeping(FakeSandbox())

    result = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    assert result.transferred == 1
    assert f"{workspace.MOUNT_ROOT}/taxes/a.txt" in sandbox.files
    assert f"{workspace.MOUNT_ROOT}/taxes/b.txt" not in sandbox.files


async def test_a_failed_extract_is_raised():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    sandbox = FakeSandbox()

    async def failing_exec(session_id, command, timeout=120):
        sandbox.commands.append(command)
        return {"stdout": "", "stderr": "tar: disk full", "exit_code": 2}

    sandbox.exec = failing_exec

    with pytest.raises(store.StoreError, match="disk full"):
        await workspace.materialize(sandbox, session_id, [_claim(project_id)])


async def test_the_mount_name_is_a_slug_of_the_project_title():
    assert store.slug("My Tax Return 2026", "fallback") == "my-tax-return-2026"
    assert store.slug("", "abc-123") == "abc-123"
    assert store.slug("///", "abc-123") == "abc-123"


async def test_nothing_in_the_transfer_carries_a_credential():
    """Bytes flow store -> harness -> sandbox; the sandbox is given no way to reach the store."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    sandbox = _sweeping(FakeSandbox())

    await workspace.materialize(sandbox, session_id, [_claim(project_id)])
    everything = " ".join(sandbox.commands) + " " + " ".join(str(v) for v in sandbox.files.values())

    for secret in ("sb_secret", "SUPABASE", "service_role", "apikey", "Authorization"):
        assert secret not in everything, f"{secret} reached the sandbox"


async def test_a_claim_knows_where_it_mounts():
    claim = workspace.Claim(project_id="p", slug="taxes")

    assert claim.mount == posixpath.join(workspace.MOUNT_ROOT, "taxes")


# --- flushing back ------------------------------------------------------------------


def _sweeping(sandbox: FakeSandbox) -> FakeSandbox:
    """Give the fake a real sha256sum sweep and a real tar-out."""
    import hashlib

    async def exec_(session_id, command, timeout=120):
        sandbox.commands.append(command)
        if command.startswith("find "):
            # shlex.quote leaves an ordinary path unquoted, so accept both forms.
            mounts = [t.strip("'") for t in command.split() if t.strip("'").startswith("/home/")]
            lines = [
                f"{hashlib.sha256(body).hexdigest()}  {path}"
                for path, body in sorted(sandbox.files.items())
                if any(path.startswith(mount + "/") for mount in mounts)
            ]
            return {"stdout": "\n".join(lines), "stderr": "", "exit_code": 0}
        if command.startswith("tar cf"):
            wanted = [t.strip("'") for t in command.split(" -C / ")[1].split()]
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w") as archive:
                for relative in wanted:
                    body = sandbox.files.get("/" + relative)
                    if body is None:
                        continue
                    info = tarfile.TarInfo(name=relative)
                    info.size = len(body)
                    archive.addfile(info, io.BytesIO(body))
            sandbox.files["/tmp/arkos-flush.tar"] = buffer.getvalue()
            return {"stdout": "", "stderr": "", "exit_code": 0}
        if command.startswith("rm -f "):
            for token in command[len("rm -f ") :].split():
                sandbox.files.pop(token.strip("'"), None)
        if "tar xf" in command:
            source = next(p for p in sandbox.files if p.endswith("materialize.tar"))
            with tarfile.open(fileobj=io.BytesIO(sandbox.files[source])) as archive:
                for member in archive.getmembers():
                    sandbox.files["/" + member.name] = archive.extractfile(member).read()
            sandbox.files.pop(source, None)
        return {"stdout": "", "stderr": "", "exit_code": 0}

    async def read_bytes(session_id, path):
        return sandbox.files[path]

    sandbox.exec = exec_
    sandbox.read_bytes = read_bytes
    return sandbox


async def test_a_file_written_in_the_sandbox_reaches_the_store():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "original")])
    sandbox = _sweeping(FakeSandbox())
    materialized = await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] = b"edited in the sandbox"
    sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/new.txt"] = b"brand new"
    result = await workspace.flush(sandbox, session_id, [_claim(project_id)], materialized.manifest)

    tree = {e.path: e for e in await store.read_tree(project_id)}
    assert result.uploaded == 2
    assert await store.get_blob(tree["a.txt"].content_hash) == b"edited in the sandbox"
    assert await store.get_blob(tree["new.txt"].content_hash) == b"brand new"
    assert tree["a.txt"].size == len(b"edited in the sandbox")


async def test_a_second_session_reads_what_the_first_wrote_byte_identical():
    """The store is the handover, not the sandbox."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("notes.md", "v1")])
    first = _sweeping(FakeSandbox())
    manifest = (await workspace.materialize(first, session_id, [_claim(project_id)])).manifest
    first.files[f"{workspace.MOUNT_ROOT}/taxes/notes.md"] = b"v2 written by session one"
    await workspace.flush(first, session_id, [_claim(project_id)], manifest)

    second = _sweeping(FakeSandbox())
    await workspace.materialize(second, session_id, [_claim(project_id)])

    assert second.files[f"{workspace.MOUNT_ROOT}/taxes/notes.md"] == b"v2 written by session one"


async def test_a_flush_uploads_only_what_changed():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1"), _file("b.txt", "2"), _file("c.txt", "3")])
    sandbox = _sweeping(FakeSandbox())
    manifest = (await workspace.materialize(sandbox, session_id, [_claim(project_id)])).manifest

    sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/b.txt"] = b"changed"
    result = await workspace.flush(sandbox, session_id, [_claim(project_id)], manifest)

    assert result.uploaded == 1, "an unchanged file was uploaded again"
    assert result.committed == 3, "the tree must still describe every file"


async def test_a_flush_with_no_edits_uploads_nothing():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    sandbox = _sweeping(FakeSandbox())
    manifest = (await workspace.materialize(sandbox, session_id, [_claim(project_id)])).manifest

    result = await workspace.flush(sandbox, session_id, [_claim(project_id)], manifest)

    assert result.uploaded == 0
    assert [e.path for e in await store.read_tree(project_id)] == ["a.txt"]


async def test_a_file_deleted_in_the_sandbox_leaves_the_tree():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("keep.txt", "1"), _file("delete.txt", "2")])
    sandbox = _sweeping(FakeSandbox())
    manifest = (await workspace.materialize(sandbox, session_id, [_claim(project_id)])).manifest

    del sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/delete.txt"]
    await workspace.flush(sandbox, session_id, [_claim(project_id)], manifest)

    assert [e.path for e in await store.read_tree(project_id)] == ["keep.txt"]


async def test_a_read_claim_commits_nothing_and_says_what_it_dropped():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "original")])
    sandbox = _sweeping(FakeSandbox())
    claims = [_claim(project_id, mode="read")]
    manifest = (await workspace.materialize(sandbox, session_id, claims)).manifest

    sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] = b"edited anyway"
    result = await workspace.flush(sandbox, session_id, claims, manifest)

    tree = await store.read_tree(project_id)
    assert result.committed == 0
    assert result.discarded == (f"{workspace.MOUNT_ROOT}/taxes/a.txt",)
    assert await store.get_blob(tree[0].content_hash) == b"original"


async def test_a_process_that_dies_before_the_flush_leaves_the_last_tree_intact():
    """Nothing is committed until the flush runs, so a crash costs the edits, not the project."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "committed")])
    sandbox = _sweeping(FakeSandbox())
    await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] = b"never flushed"
    # The process stops here. Nothing calls flush.

    fresh = _sweeping(FakeSandbox())
    await workspace.materialize(fresh, session_id, [_claim(project_id)])

    assert fresh.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] == b"committed"


async def test_losing_the_sandbox_entirely_loses_nothing_that_was_flushed():
    """e2b's own persistence is a warm start, not the record."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "v1")])
    sandbox = _sweeping(FakeSandbox())
    manifest = (await workspace.materialize(sandbox, session_id, [_claim(project_id)])).manifest
    sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] = b"v2"
    await workspace.flush(sandbox, session_id, [_claim(project_id)], manifest)

    del sandbox  # the sandbox is destroyed outright
    replacement = _sweeping(FakeSandbox())
    await workspace.materialize(replacement, session_id, [_claim(project_id)])

    assert replacement.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] == b"v2"


async def test_an_archive_that_cannot_be_built_is_raised_not_swallowed():
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])
    sandbox = _sweeping(FakeSandbox())
    manifest = (await workspace.materialize(sandbox, session_id, [_claim(project_id)])).manifest
    sandbox.files[f"{workspace.MOUNT_ROOT}/taxes/a.txt"] = b"changed"

    working = sandbox.exec

    async def failing(session_id_, command, timeout=120):
        if command.startswith("tar cf"):
            return {"stdout": "", "stderr": "tar: cannot read", "exit_code": 2}
        return await working(session_id_, command, timeout)

    sandbox.exec = failing

    with pytest.raises(store.StoreError, match="cannot read"):
        await workspace.flush(sandbox, session_id, [_claim(project_id)], manifest)


# --- deletions must survive a warm sandbox ------------------------------------------


async def test_a_deletion_by_another_session_is_not_resurrected_by_a_warm_sandbox():
    """A creates and flushes; B deletes and flushes; A's warm sandbox must not bring it back."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("keep.txt", "1")])

    # Session A materializes and adds a file.
    a = _sweeping(FakeSandbox())
    manifest_a = (await workspace.materialize(a, session_id, [_claim(project_id)])).manifest
    a.files[f"{workspace.MOUNT_ROOT}/taxes/doomed.txt"] = b"created by A"
    await workspace.flush(a, session_id, [_claim(project_id)], manifest_a)
    assert "doomed.txt" in [e.path for e in await store.read_tree(project_id)]

    # Session B, its own sandbox, deletes it and flushes.
    b = _sweeping(FakeSandbox())
    manifest_b = (await workspace.materialize(b, session_id, [_claim(project_id)])).manifest
    del b.files[f"{workspace.MOUNT_ROOT}/taxes/doomed.txt"]
    await workspace.flush(b, session_id, [_claim(project_id)], manifest_b)
    assert "doomed.txt" not in [e.path for e in await store.read_tree(project_id)]

    # A's sandbox is still warm and still has the file on disk.
    assert f"{workspace.MOUNT_ROOT}/taxes/doomed.txt" in a.files
    manifest_a2 = (await workspace.materialize(a, session_id, [_claim(project_id)])).manifest

    assert f"{workspace.MOUNT_ROOT}/taxes/doomed.txt" not in a.files, "the deletion was undone"
    await workspace.flush(a, session_id, [_claim(project_id)], manifest_a2)
    tree = [e.path for e in await store.read_tree(project_id)]
    assert "doomed.txt" not in tree, "the deletion came back through the tree"


async def test_a_deletion_survives_a_sandbox_that_kept_no_record():
    """Deletions are derived from the disk, so there is no record to lose."""
    session_id, project_id = await _project()
    await store.commit_tree(project_id, [_file("keep.txt", "1"), _file("doomed.txt", "2")])
    sandbox = _sweeping(FakeSandbox())
    await workspace.materialize(sandbox, session_id, [_claim(project_id)])

    await store.commit_tree(project_id, [_file("keep.txt", "1")])
    manifest = (await workspace.materialize(sandbox, session_id, [_claim(project_id)])).manifest

    assert f"{workspace.MOUNT_ROOT}/taxes/doomed.txt" not in sandbox.files
    await workspace.flush(sandbox, session_id, [_claim(project_id)], manifest)
    assert [e.path for e in await store.read_tree(project_id)] == ["keep.txt"]
