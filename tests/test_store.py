"""Blobs and trees: content addressing, and a commit that cannot half-happen.

The tree is ONE flat namespace per user (11.9): `files (user_id, path)`, where a
folder is the first segment of a path and no project owns one.

Runs against a real Postgres with the migrations applied; blobs go to a tmp_path.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from db import pool
from harness_module import store
from tests.dbgate import require_db

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    await require_db()
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    store.use_blobs(None)
    await pool.execute("DELETE FROM sessions WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM deleted_files WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _user() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(user_id)


def _file(path: str, content: str) -> store.FileContent:
    return store.FileContent(path=path, content=content.encode())


# --- blobs -------------------------------------------------------------------------


async def test_a_blob_round_trips_under_its_hash():
    content_hash = await store.put_blob(b"alpha")

    assert content_hash == store.sha256(b"alpha")
    assert await store.get_blob(content_hash) == b"alpha"


async def test_the_same_content_is_one_blob(tmp_path):
    first = await store.put_blob(b"shared")
    second = await store.put_blob(b"shared")

    assert first == second
    on_disk = list(tmp_path.rglob("*"))
    assert len([p for p in on_disk if p.is_file()]) == 1


async def test_the_same_content_in_two_stores_is_one_blob(tmp_path):
    first, second = await _user(), await _user()

    await store.commit_tree(first, [_file("one/a.txt", "identical")])
    await store.commit_tree(second, [_file("two/b/other.txt", "identical")])

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1, "content addressing did not dedup across users"


async def test_a_missing_blob_reads_as_none():
    assert await store.get_blob("0" * 64) is None


async def test_missing_blobs_reports_only_what_is_absent():
    present = await store.put_blob(b"here")

    missing = await store.missing_blobs([present, "1" * 64])

    assert missing == {"1" * 64}


# --- trees --------------------------------------------------------------------------


async def test_a_commit_writes_the_tree_and_the_bytes():
    user_id = await _user()

    entries = await store.commit_tree(
        user_id, [_file("proj/src/main.py", "print(1)"), _file("proj/README.md", "hi")]
    )

    assert [e.path for e in entries] == ["proj/README.md", "proj/src/main.py"]
    body = await store.get_blob(next(e.content_hash for e in entries if e.path == "proj/README.md"))
    assert body == b"hi"
    assert next(e.size for e in entries if e.path == "proj/src/main.py") == len("print(1)")


async def test_a_commit_replaces_the_tree_it_covers():
    user_id = await _user()
    await store.commit_tree(user_id, [_file("p/keep.txt", "1"), _file("p/drop.txt", "2")])

    await store.commit_tree(user_id, [_file("p/keep.txt", "1")])

    assert [e.path for e in await store.read_tree(user_id)] == ["p/keep.txt"]


async def test_a_commit_is_idempotent_on_retry():
    user_id = await _user()
    files = [_file("p/a.txt", "same"), _file("p/b.txt", "same too")]

    first = await store.commit_tree(user_id, files)
    second = await store.commit_tree(user_id, files)

    assert [(e.path, e.content_hash, e.size) for e in first] == [
        (e.path, e.content_hash, e.size) for e in second
    ]
    assert await pool.fetchval(
        "SELECT count(*) FROM files WHERE user_id = $1", uuid.UUID(user_id)
    ) == 2


async def test_a_commit_that_dies_before_the_rows_leaves_the_old_tree_whole(monkeypatch):
    """Blobs first, rows last: an interrupted commit is a no-op on the tree."""
    user_id = await _user()
    await store.commit_tree(user_id, [_file("p/original.txt", "before")])

    real_pool = pool.pool

    async def die(*a, **kw):
        raise RuntimeError("the process stopped here")

    monkeypatch.setattr(pool, "pool", die)
    with pytest.raises(RuntimeError):
        await store.commit_tree(user_id, [_file("p/replacement.txt", "after")])
    monkeypatch.setattr(pool, "pool", real_pool)

    tree = await store.read_tree(user_id)

    assert [e.path for e in tree] == ["p/original.txt"], "the tree was left half-written"
    assert await store.get_blob(next(e.content_hash for e in tree)) == b"before"
    # The blob for the abandoned commit is uploaded and orphaned, which is the
    # side the invariant errs on.
    assert await store.get_blob(store.sha256(b"after")) == b"after"


async def test_a_prefix_commit_leaves_the_rest_of_the_tree_alone():
    """A commit scoped to one folder is what a flush of one claim does."""
    user_id = await _user()
    await store.commit_tree(user_id, [_file("code/a.py", "1"), _file("docs/b.md", "2")])

    await store.commit_tree(user_id, [_file("code/c.py", "3")], prefix="code")

    assert [e.path for e in await store.read_tree(user_id)] == ["code/c.py", "docs/b.md"]


async def test_reading_a_prefix_returns_only_that_subtree():
    user_id = await _user()
    await store.commit_tree(user_id, [_file("code/a.py", "1"), _file("docs/b.md", "2")])

    assert [e.path for e in await store.read_tree(user_id, "code")] == ["code/a.py"]


async def test_an_empty_commit_empties_the_tree():
    user_id = await _user()
    await store.commit_tree(user_id, [_file("p/a.txt", "1")])

    await store.commit_tree(user_id, [])

    assert await store.read_tree(user_id) == []


# --- folders ------------------------------------------------------------------------
#
# A folder is the first segment of a path. It is never a row, which is what
# makes every property below fall out rather than need enforcing.


async def test_a_folder_is_the_first_segment_of_a_path():
    user_id = await _user()
    await store.commit_tree(
        user_id, [_file("triage/a.txt", "1"), _file("triage/deep/b.txt", "2"), _file("notes/c.md", "3")]
    )

    assert [(f.name, f.files) for f in await store.folders(user_id)] == [("notes", 1), ("triage", 2)]
    assert store.folder_of("triage/deep/b.txt") == "triage"


async def test_a_folder_exists_exactly_as_long_as_a_file_lives_under_it():
    user_id = await _user()
    await store.put_file(user_id, "ski-trip/plan.md", b"go")
    assert [f.name for f in await store.folders(user_id)] == ["ski-trip"]

    await store.commit_tree(user_id, [], prefix="ski-trip")

    assert await store.folders(user_id) == [], "the folder outlived the last file under it"


async def test_a_named_but_unfilled_folder_is_kept_by_its_sentinel_and_counts_zero():
    """The none-case of creating a project: a folder reserved before anything is in it."""
    user_id = await _user()

    await store.put_file(user_id, store.dir_sentinel("ski-trip"), b"")

    assert [(f.name, f.files) for f in await store.folders(user_id)] == [("ski-trip", 0)]


async def test_a_new_folder_name_does_not_collide_with_one_that_exists():
    user_id = await _user()
    await store.put_file(user_id, "notes/a.md", b"1")

    assert await store.unique_folder(user_id, "notes") == "notes-2"
    assert await store.unique_folder(user_id, "other") == "other"


async def test_a_file_must_live_inside_a_folder():
    """A top-level file would be its own folder holding nothing: no mount, no header."""
    user_id = await _user()

    with pytest.raises(ValueError):
        await store.put_file(user_id, "loose.txt", b"nowhere")


async def test_moving_between_folders_is_a_row_edit_and_moves_no_blob(tmp_path):
    user_id = await _user()
    await store.put_file(user_id, "triage/receipt.pdf", b"bytes")
    before = [p for p in tmp_path.rglob("*") if p.is_file()]

    moved = await store.move_path(user_id, "triage/receipt.pdf", "archive/receipt.pdf")

    assert moved == [("triage/receipt.pdf", "archive/receipt.pdf")]
    assert [e.path for e in await store.read_tree(user_id)] == ["archive/receipt.pdf"]
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == before


async def test_renaming_a_folder_is_refused_here():
    """It is a path-prefix rewrite that also moves live claims and mounts: its own card."""
    user_id = await _user()
    await store.put_file(user_id, "triage/a.txt", b"1")

    with pytest.raises(store.StoreError, match="renaming or moving one"):
        await store.move_path(user_id, "triage", "sorted")


async def test_a_directory_can_be_moved_out_to_the_top_level():
    """Dragging a folder to the edge promotes it: `triage/inbox` becomes `inbox`.

    Not a special case — it is the model. A folder IS a top-level path segment,
    so putting a directory in the first position makes one of it.
    """
    user_id = await _user()
    for path in ("triage/inbox/a.md", "triage/inbox/deep/b.md", "triage/other.md"):
        await store.put_file(user_id, path, b"x")

    moved = await store.move_path(user_id, "triage/inbox", "inbox")

    assert [e.path for e in await store.read_tree(user_id)] == [
        "inbox/a.md",
        "inbox/deep/b.md",
        "triage/other.md",
    ]
    assert len(moved) == 2
    # And it is a folder now, on its own, beside the one it came out of.
    assert [(f.name, f.files) for f in await store.folders(user_id)] == [("inbox", 2), ("triage", 1)]


async def test_a_file_cannot_be_moved_out_to_the_top_level():
    """The top level holds FOLDERS. A file there would be its own folder holding
    nothing — unmountable, unlinkable, and a phantom row in every folder picker.

    The two refusals say different things on purpose: this one is "it needs a
    folder to go in", and moving a folder is "that is not something this can
    do". One message for both left dragging a file out of a folder explaining
    itself as a folder rename, which is not what was attempted."""
    user_id = await _user()
    await store.put_file(user_id, "triage/a.txt", b"1")

    with pytest.raises(store.StoreError, match="top level holds folders"):
        await store.move_path(user_id, "triage/a.txt", "a.txt")

    assert [e.path for e in await store.read_tree(user_id)] == ["triage/a.txt"]


# --- renaming ------------------------------------------------------------------------


async def test_renaming_a_file_keeps_it_where_it_is():
    """A rename changes what a thing is CALLED. Where it lives is a move."""
    user_id = await _user()
    stored = await store.put_file(user_id, "triage/draft.md", b"words")

    moved = await store.rename_path(user_id, "triage/draft.md", "final.md")

    assert moved == [("triage/draft.md", "triage/final.md")]
    tree = await store.read_tree(user_id)
    assert [e.path for e in tree] == ["triage/final.md"]
    # Same row, same blob: nothing was re-uploaded and an open reader follows it.
    assert await pool.fetchval(
        "SELECT path FROM files WHERE id = $1", uuid.UUID(stored.id)
    ) == "triage/final.md"


async def test_renaming_a_directory_takes_everything_under_it():
    user_id = await _user()
    for path in ("triage/inbox/a.md", "triage/inbox/deep/b.md", "triage/other.md"):
        await store.put_file(user_id, path, b"x")

    await store.rename_path(user_id, "triage/inbox", "archive")

    assert [e.path for e in await store.read_tree(user_id)] == [
        "triage/archive/a.md",
        "triage/archive/deep/b.md",
        "triage/other.md",
    ]


async def test_a_name_is_a_name_and_not_a_path():
    """`/` in a rename would quietly make it a move, which has its own rules."""
    user_id = await _user()
    await store.put_file(user_id, "triage/a.md", b"1")

    for bad in ("", "   ", "notes/a.md", "..", "/"):
        with pytest.raises(ValueError):
            await store.rename_path(user_id, "triage/a.md", bad)

    assert [e.path for e in await store.read_tree(user_id)] == ["triage/a.md"]


async def test_renaming_onto_something_that_exists_is_refused_whole():
    user_id = await _user()
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "triage/b.md", b"2")

    with pytest.raises(store.StoreError, match="already taken"):
        await store.rename_path(user_id, "triage/a.md", "b.md")

    assert [e.path for e in await store.read_tree(user_id)] == ["triage/a.md", "triage/b.md"]


async def test_renaming_a_folder_onto_another_is_refused_rather_than_merging_them():
    """Their files not clashing is not permission to fold two folders into one."""
    user_id = await _user()
    await store.put_file(user_id, "triage/a.md", b"1")
    await store.put_file(user_id, "notes/x.md", b"2")

    with pytest.raises(store.StoreError, match="already taken"):
        await store.rename_path(user_id, "triage", "notes")

    assert [f.name for f in await store.folders(user_id)] == ["notes", "triage"]


async def test_renaming_a_top_level_folder_carries_its_links_and_claims():
    """The folder's name is written in three places, and all three move at once.

    A rewrite that moved only the paths would leave a project linking a folder
    that no longer exists and a session claiming one — the folder would vanish
    from the working-files pane and the next materialize would mount nothing.
    """
    user_id = await _user()
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'work') RETURNING id", uuid.UUID(user_id)
    )
    session_id = await pool.fetchval(
        "INSERT INTO sessions (user_id, project_id, mode, status) VALUES ($1, $2, 'attended', 'idle') "
        "RETURNING id",
        uuid.UUID(user_id),
        project_id,
    )
    await pool.execute(
        "INSERT INTO project_folders (project_id, folder) VALUES ($1, 'triage')", project_id
    )
    await pool.execute(
        "INSERT INTO session_claims (session_id, folder, mode) VALUES ($1, 'triage', 'write')", session_id
    )
    await store.put_file(user_id, "triage/deep/a.md", b"1")

    await store.rename_path(user_id, "triage", "sorted")

    assert [e.path for e in await store.read_tree(user_id)] == ["sorted/deep/a.md"]
    assert [f.name for f in await store.folders(user_id)] == ["sorted"]
    assert await pool.fetchval(
        "SELECT folder FROM project_folders WHERE project_id = $1", project_id
    ) == "sorted"
    assert await pool.fetchval(
        "SELECT folder FROM session_claims WHERE session_id = $1", session_id
    ) == "sorted"


async def test_renaming_a_nested_directory_leaves_links_and_claims_alone():
    """Only the FIRST segment is a folder. Renaming below it changes no name anybody holds."""
    user_id = await _user()
    project_id = await pool.fetchval(
        "INSERT INTO projects (user_id, title) VALUES ($1, 'work') RETURNING id", uuid.UUID(user_id)
    )
    await pool.execute(
        "INSERT INTO project_folders (project_id, folder) VALUES ($1, 'triage')", project_id
    )
    await store.put_file(user_id, "triage/inbox/a.md", b"1")

    await store.rename_path(user_id, "triage/inbox", "archive")

    assert await pool.fetchval(
        "SELECT folder FROM project_folders WHERE project_id = $1", project_id
    ) == "triage"


async def test_the_way_to_a_new_top_level_folder_is_to_make_it_then_move_into_it():
    """Which is the whole flow: reserve the folder, then drag the file across."""
    user_id = await _user()
    await store.put_file(user_id, "triage/a.txt", b"1")

    await store.put_file(user_id, store.dir_sentinel("archive"), b"")
    moved = await store.move_path(user_id, "triage/a.txt", "archive/a.txt")

    assert moved == [("triage/a.txt", "archive/a.txt")]
    # And `triage` is gone with its last file, which is what "a folder exists
    # exactly as long as files exist under it" means when you watch it happen.
    assert [f.name for f in await store.folders(user_id)] == ["archive"]


# --- the Supabase backend -------------------------------------------------------------


def _supabase(handler) -> store.SupabaseBlobs:
    """A backend wired to a mock transport, so the HTTP contract is tested without a bucket."""
    import httpx

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return store.SupabaseBlobs("https://ref.supabase.co", "service-key", "files", client=client)


async def test_an_upload_addresses_the_blob_by_its_hash_and_authenticates():
    seen = {}

    def handler(request):
        import httpx

        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["apikey"] = request.headers.get("apikey")
        seen["body"] = request.content
        return httpx.Response(200, json={"Key": "ok"})

    content_hash = store.sha256(b"alpha")
    await _supabase(handler).put(content_hash, b"alpha")

    assert seen["url"] == f"https://ref.supabase.co/storage/v1/object/files/{store.blob_key(content_hash)}"
    assert seen["auth"] == "Bearer service-key"
    assert seen["apikey"] == "service-key"
    assert seen["body"] == b"alpha"


async def test_uploading_a_blob_that_is_already_there_succeeds():
    """Write-once: the name is the hash of the content, so a duplicate is the same blob."""

    def handler(request):
        import httpx

        return httpx.Response(400, json={"error": "Duplicate", "message": "The resource already exists"})

    await _supabase(handler).put("a" * 64, b"alpha")


async def test_an_upload_that_fails_is_raised_not_swallowed():
    def handler(request):
        import httpx

        return httpx.Response(500, text="storage is unwell")

    with pytest.raises(store.StoreError):
        await _supabase(handler).put("a" * 64, b"alpha")


async def test_a_download_returns_bytes_and_a_miss_returns_none():
    def handler(request):
        import httpx

        return httpx.Response(200, content=b"beta") if "bb" in str(request.url) else httpx.Response(404)

    backend = _supabase(handler)

    assert await backend.get("bb" + "0" * 62) == b"beta"
    assert await backend.get("cc" + "0" * 62) is None


async def test_missing_asks_only_about_the_hashes_it_was_given():
    asked = []

    def handler(request):
        import httpx

        asked.append(str(request.url).rsplit("/", 1)[-1])
        return httpx.Response(404 if str(request.url).endswith("2") else 200)

    present, absent = "a" * 63 + "1", "b" * 63 + "2"
    missing = await _supabase(handler).missing([present, absent])

    assert missing == {absent}
    assert sorted(asked) == sorted([present, absent])


async def test_selecting_supabase_without_credentials_fails_loudly(monkeypatch):
    """A store that cannot reach its bucket must say so at startup, not at first write."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.setattr(store, "_cfg", lambda key, default: "supabase" if key == "store.backend" else default)

    with pytest.raises(store.StoreError, match="SUPABASE_URL"):
        store._build()


async def test_a_secret_api_key_is_preferred_over_the_legacy_service_role(monkeypatch):
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_current")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "eyJlegacy")

    assert store.secret_key() == "sb_secret_current"


async def test_the_legacy_service_role_key_still_works(monkeypatch):
    """An installation that has not migrated yet keeps running."""
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "eyJlegacy")

    assert store.secret_key() == "eyJlegacy"


async def test_an_unknown_backend_is_refused(monkeypatch):
    monkeypatch.setattr(store, "_cfg", lambda key, default: "carrier-pigeon" if key == "store.backend" else default)

    with pytest.raises(store.StoreError, match="carrier-pigeon"):
        store._build()


async def test_the_project_url_is_derived_from_a_direct_dsn(monkeypatch):
    """The DSN already carries the project ref, so the URL need not be configured twice."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setattr(
        store,
        "_cfg",
        lambda key, default: "postgresql://postgres:pw@db.abcdefg.supabase.co:5432/postgres"
        if key == "database.url"
        else default,
    )

    assert store.project_url() == "https://abcdefg.supabase.co"


async def test_the_project_url_is_derived_from_a_pooler_dsn(monkeypatch):
    """The pooler moves the ref into the username."""
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setattr(
        store,
        "_cfg",
        lambda key, default: "postgresql://postgres.abcdefg:pw@aws-0-eu-west-2.pooler.supabase.com:6543/postgres"
        if key == "database.url"
        else default,
    )

    assert store.project_url() == "https://abcdefg.supabase.co"


async def test_an_explicit_url_wins_over_the_dsn(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://storage.example.com/")
    monkeypatch.setattr(
        store,
        "_cfg",
        lambda key, default: "postgresql://postgres:pw@db.abcdefg.supabase.co:5432/postgres"
        if key == "database.url"
        else default,
    )

    assert store.project_url() == "https://storage.example.com"


async def test_a_non_supabase_dsn_derives_nothing(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.setattr(
        store,
        "_cfg",
        lambda key, default: "postgresql://user:pw@localhost:5432/arkos" if key == "database.url" else default,
    )

    assert store.project_url() is None


async def test_a_missing_object_reported_as_a_400_is_still_a_miss():
    """Supabase answers a missing object with 400 and a body saying 404."""

    def handler(request):
        import httpx

        return httpx.Response(
            400, json={"statusCode": "404", "error": "not_found", "message": "Object not found", "code": "NoSuchKey"}
        )

    assert await _supabase(handler).get("a" * 64) is None


async def test_a_genuine_400_is_still_an_error():
    def handler(request):
        import httpx

        return httpx.Response(400, json={"error": "InvalidRequest", "message": "malformed key"})

    with pytest.raises(store.StoreError):
        await _supabase(handler).get("a" * 64)


async def test_a_head_that_is_not_a_clean_200_counts_as_missing():
    """A redundant upload is cheap; a file believed present and absent is not."""

    def handler(request):
        import httpx

        return httpx.Response(400)

    assert await _supabase(handler).missing(["a" * 64]) == {"a" * 64}
