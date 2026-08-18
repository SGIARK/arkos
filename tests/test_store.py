"""Blobs and trees: content addressing, and a commit that cannot half-happen.

Runs against a real Postgres with migration 0001 applied; blobs go to a tmp_path.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from db import pool
from harness_module import store

pytestmark = pytest.mark.asyncio

_seeded: list[uuid.UUID] = []


@pytest_asyncio.fixture(autouse=True)
async def _db(tmp_path):
    try:
        await pool.fetchval("SELECT 1")
    except Exception as e:  # noqa: BLE001 - any connection failure means skip
        await pool.close()
        pytest.skip(f"needs the arkos database (migrations applied): {e}")
    store.use_blobs(store.FilesystemBlobs(tmp_path))
    yield
    store.use_blobs(None)
    await pool.execute("DELETE FROM projects WHERE user_id = ANY($1::uuid[])", _seeded)
    await pool.execute("DELETE FROM users WHERE id = ANY($1::uuid[])", _seeded)
    _seeded.clear()
    await pool.close()


async def _project() -> str:
    user_id = uuid.uuid4()
    await pool.execute("INSERT INTO users (id) VALUES ($1)", user_id)
    _seeded.append(user_id)
    return str(
        await pool.fetchval("INSERT INTO projects (user_id, title) VALUES ($1, 'p') RETURNING id", user_id)
    )


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


async def test_the_same_content_in_two_projects_stores_one_blob(tmp_path):
    first, second = await _project(), await _project()

    await store.commit_tree(first, [_file("a.txt", "identical")])
    await store.commit_tree(second, [_file("b/other.txt", "identical")])

    files = [p for p in tmp_path.rglob("*") if p.is_file()]
    assert len(files) == 1, "content addressing did not dedup across projects"


async def test_a_missing_blob_reads_as_none():
    assert await store.get_blob("0" * 64) is None


async def test_missing_blobs_reports_only_what_is_absent():
    present = await store.put_blob(b"here")

    missing = await store.missing_blobs([present, "1" * 64])

    assert missing == {"1" * 64}


# --- trees --------------------------------------------------------------------------


async def test_a_commit_writes_the_tree_and_the_bytes():
    project_id = await _project()

    entries = await store.commit_tree(project_id, [_file("src/main.py", "print(1)"), _file("README.md", "hi")])

    assert [e.path for e in entries] == ["README.md", "src/main.py"]
    body = await store.get_blob(next(e.content_hash for e in entries if e.path == "README.md"))
    assert body == b"hi"
    assert next(e.size for e in entries if e.path == "src/main.py") == len("print(1)")


async def test_a_commit_replaces_the_tree_it_covers():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("keep.txt", "1"), _file("drop.txt", "2")])

    await store.commit_tree(project_id, [_file("keep.txt", "1")])

    assert [e.path for e in await store.read_tree(project_id)] == ["keep.txt"]


async def test_a_commit_is_idempotent_on_retry():
    project_id = await _project()
    files = [_file("a.txt", "same"), _file("b.txt", "same too")]

    first = await store.commit_tree(project_id, files)
    second = await store.commit_tree(project_id, files)

    assert [(e.path, e.content_hash, e.size) for e in first] == [
        (e.path, e.content_hash, e.size) for e in second
    ]
    assert await pool.fetchval(
        "SELECT count(*) FROM project_files WHERE project_id = $1", uuid.UUID(project_id)
    ) == 2


async def test_a_commit_that_dies_before_the_rows_leaves_the_old_tree_whole(monkeypatch):
    """Blobs first, rows last: an interrupted commit is a no-op on the tree."""
    project_id = await _project()
    await store.commit_tree(project_id, [_file("original.txt", "before")])

    real_pool = pool.pool

    async def die(*a, **kw):
        raise RuntimeError("the process stopped here")

    monkeypatch.setattr(pool, "pool", die)
    with pytest.raises(RuntimeError):
        await store.commit_tree(project_id, [_file("replacement.txt", "after")])
    monkeypatch.setattr(pool, "pool", real_pool)

    tree = await store.read_tree(project_id)

    assert [e.path for e in tree] == ["original.txt"], "the tree was left half-written"
    assert await store.get_blob(next(e.content_hash for e in tree)) == b"before"
    # The blob for the abandoned commit is uploaded and orphaned, which is the
    # side the invariant errs on.
    assert await store.get_blob(store.sha256(b"after")) == b"after"


async def test_a_subpath_commit_leaves_the_rest_of_the_tree_alone():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("src/a.py", "1"), _file("docs/b.md", "2")])

    await store.commit_tree(project_id, [_file("src/c.py", "3")], subpath="/src")

    assert [e.path for e in await store.read_tree(project_id)] == ["docs/b.md", "src/c.py"]


async def test_reading_a_subpath_returns_only_that_subtree():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("src/a.py", "1"), _file("docs/b.md", "2")])

    assert [e.path for e in await store.read_tree(project_id, "/src")] == ["src/a.py"]


async def test_an_empty_commit_empties_the_tree():
    project_id = await _project()
    await store.commit_tree(project_id, [_file("a.txt", "1")])

    await store.commit_tree(project_id, [])

    assert await store.read_tree(project_id) == []


# --- diff ---------------------------------------------------------------------------


async def test_diff_reports_added_changed_and_removed_by_hash():
    project_id = await _project()
    before = await store.commit_tree(
        project_id, [_file("same.txt", "x"), _file("edit.txt", "1"), _file("gone.txt", "z")]
    )
    after = await store.commit_tree(
        project_id, [_file("same.txt", "x"), _file("edit.txt", "2"), _file("new.txt", "n")]
    )

    diff = store.diff_tree(before, after)

    assert diff.added == {"new.txt"}
    assert diff.changed == {"edit.txt"}
    assert diff.removed == {"gone.txt"}
    assert "same.txt" not in diff.paths


async def test_an_unchanged_tree_diffs_to_nothing():
    project_id = await _project()
    tree = await store.commit_tree(project_id, [_file("a.txt", "1")])

    assert not store.diff_tree(tree, tree)


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
