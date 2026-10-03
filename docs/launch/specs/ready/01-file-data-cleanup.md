# // Feature Spec: File Data Cleanup (NMORGAN)

*Ready 01 · Wave 1 · no blockers. The one primitive that account deletion and retention both need: delete a file's bytes, but only when nobody else still uses them.*

---

**Sources**
- `harness_module/blobs.py` (`Blobs`, `FilesystemBlobs`, `SupabaseBlobs._url`, `_is_absent`, `use_blobs`, `put_blob`, `get_blob`)
- `harness_module/store.py` (import direction blobs → store; the store re-exports blob calls)
- `db/migrations/0015_user_store.sql` (`files.content_hash`), `0016_delete_undo.sql` (`deleted_files.content_hash`, "blob GC is still nobody's card"), `0005_memory.sql` (`memory_files.content_hash`), `0018_drop_snapshots.sql`
- `harness_module/memory.py` (`append_note`, `update_memory`: both `put_blob` and store the hash)
- Used by: ready 05 (Account Deletion), ready 06 (Session Delete & Retention)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
Nothing ever deletes a file's bytes. `0016` says so outright: blobs "are never collected". So "delete" in arkos removes the tree row and keeps the content forever, and the Privacy Policy can't promise deletion. A naive delete would be worse, because blobs are content-addressed and shared: two users uploading the same PDF share one object. Success: a hash can be deleted from either backend, and `delete_unreferenced` removes exactly the hashes no row still names.

---

# Technical Background
`blob_key(hash)` is `{prefix}/blobs/{hh}/{sha256}`, with no owner in the path. `SupabaseBlobs` speaks Storage REST at `{url}/storage/v1/object/{bucket}/{key}` via `_url`, with `put` (POST), `get` (GET) and `missing` (HEAD). `_is_absent(response)` already turns a Storage 404/400-not-found into "absent". `FilesystemBlobs` keeps files under a root and backs the test suite through `use_blobs`.

After `0018` dropped the snapshot tables, three live tables hold a blob hash: `files.content_hash`, `deleted_files.content_hash` and `memory_files.content_hash`. The memory region is blob-backed too: `memory.append_note` and `memory.update_memory` both call `put_blob` and write the hash, and `memory_files.body` is only a searchable copy of the text. `result_blobs.content` is plain TEXT in Postgres and holds no hash.

---

# Proposed Approach
1. `delete` on the `Blobs` protocol and both backends. "Already absent" is success, so every caller can retry.
2. `store.delete_unreferenced(hashes)` checks all three hash columns in one query and deletes the rest. It lives in `store.py` because it reads the tree, and blobs → store is the allowed import direction.

**Not in scope:** a full-bucket orphan scan (`Blobs` has no list call, and this only deletes hashes a caller can name); the callers themselves (ready 05, 06).

---

# Implementation Plan

## Task 1: `Blobs.delete`
**Problem:** neither backend can remove an object.
**Done when:** `delete(hash)` removes the object on both backends; deleting a hash that isn't there returns without raising; any other Storage error raises `StoreError`.
**Touch point:** `harness_module/blobs.py`: `Blobs` protocol, `FilesystemBlobs.delete` (unlink via `asyncio.to_thread`, `missing_ok=True`), `SupabaseBlobs.delete` (`client.delete(self._url(hash))` under `self._gate`), and a module-level `delete_blob(hash)` beside `get_blob`.
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** batching many deletes into one request.
**Acceptance test:** `test_delete_blob_twice_is_harmless`.

```python
class Blobs(Protocol):
    async def delete(self, content_hash: str) -> None: ...

async def delete_blob(content_hash: str) -> None: ...
```

---

## Task 2: `store.delete_unreferenced`
**Problem:** callers need "delete these hashes unless someone else still uses them".
**Done when:** given hashes where some are still in `files`, `deleted_files` or `memory_files` for any user, only the unreferenced ones are deleted, and the count returned equals the number deleted.
**Touch point:** `harness_module/store.py` (new function; re-export `delete_blob` with the other blob calls).
**Priority:** High | **Effort:** <2h | **Blockers:** Task 1
**Out of scope:** deciding *when* to call it.
**Acceptance test:** `test_shared_hash_is_kept`, `test_every_hash_column_is_checked`.

```python
async def delete_unreferenced(hashes: Iterable[str]) -> int:
    """Delete each hash that no `files`, `deleted_files` or `memory_files` row still names; return how many went."""
    # SELECT h FROM unnest($1::text[]) h
    #  WHERE NOT EXISTS (SELECT 1 FROM files WHERE content_hash = h)
    #    AND NOT EXISTS (SELECT 1 FROM deleted_files WHERE content_hash = h)
    #    AND NOT EXISTS (SELECT 1 FROM memory_files WHERE content_hash = h)
```

---

# Tests
New `tests/test_blobs.py` for Task 1, and `tests/test_store.py` for Task 2. Both use `FilesystemBlobs` through `use_blobs`. The Supabase backend gets a mocked `httpx` client, injected through the constructor's `client` argument.

## Test 1: `test_delete_blob_twice_is_harmless`
**What it verifies:** the first delete removes the bytes (`get_blob` returns `None`); the second raises nothing. The Supabase variant maps a 404 to success and a 500 to `StoreError`.
**Why this matters:** account deletion retries after partial failure, so delete has to be idempotent.

## Test 2: `test_shared_hash_is_kept`
**What it verifies:** hash H in user A's `files`, user B's `deleted_files` and user C's `memory_files`; `delete_unreferenced([H])` returns 0 and the bytes remain, and removing the three references one at a time keeps the bytes until the last.
**Why this matters:** content addressing means one user's delete can destroy another user's file.

## Test 3: `test_every_hash_column_is_checked`
**What it verifies:** the set of columns named `content_hash` in `information_schema.columns` equals `{files, deleted_files, memory_files}`.
**Why this matters:** a future table storing hashes would silently turn this into data loss. The test forces that table to be added to the query.

---

# Open Questions
1. *Verify in Task 1, not a decision:* Supabase Storage's single-object `DELETE` at the `_url` path. Confirm the status code for a missing object and whether `_is_absent` already covers it; record it in Implementation Notes.

---

# Implementation Notes
