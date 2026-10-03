# // Feature Spec: Deletion (NMORGAN)

*Ready 02 · Parts A–C before launch (waves 1–2), Part D after launch. One spec for how arkos represents deletion and carries it out: files, sessions, projects, accounts, expired data, and the bytes and third-party grants behind them. It replaces the earlier File Data Cleanup, Account Deletion, Session Delete & Retention, Project Delete and Inactive Account Sweep specs.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §1.2, §4.3; `docs/launch/DECISIONS.md` (D7, D16)
- `db/migrations/0000_migration_0.sql` (`sessions`, `session_events` "audit, never pruned", `result_blobs`, `approvals`, `resource_leases`, `user_sandboxes`), `0001_store.sql` (`session_claims`), `0002_session_sandboxes.sql`, `0005_memory.sql` (`memory_files.content_hash`), `0008_home_session.sql`, `0010_session_tools.sql`, `0015_user_store.sql` (`files`, `project_folders`), `0016_delete_undo.sql` (`deleted_files`, `deleted_links`), `0018_drop_snapshots.sql`, `0023_auth_sessions.sql`, `0024_users_are_supabase_accounts.sql`
- `harness_module/blobs.py` (`Blobs`, `FilesystemBlobs`, `SupabaseBlobs._url`, `_is_absent`, `use_blobs`, `project_url`, `secret_key`), `harness_module/store.py` (`delete_path`, `undo_delete`), `harness_module/memory.py` (`append_note`, `update_memory`)
- `harness_module/runner.py` (`cancel`), `harness_module/lifecycle.py` (`TERMINAL`), `harness_module/hands.py` (`connectors`), `tool_module/composio_mcp.py` (`Composio.disconnect`, `Composio._accounts`), `harness_module/system_log.py` (`start`, `prune`, `record`)
- `harness_module/api.py` (`create_auth_session`, `_ensure_home_session`, `_owned_session`, `_owned_project`, `lifespan`, `_keys_and_sweep`)
- `frontend/views.jsx` (`SettingsModal`), `frontend/lookingglass.jsx` (`SessionDetail`, `LookingGlassView` project card `pc-rename-btn`), `frontend/components.jsx` (`FileTree`), `frontend/app.jsx` (`authNotice`)
- `tests/test_store_schema.py` (`test_deleting_a_project_deletes_its_links_and_no_files`), `tests/test_store.py`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-02

---

# Problem
A user can delete a file, and that's all, and even that only hides it. Sessions, projects and accounts can't be deleted at all. Deleting a user fails on a foreign key (`sessions.user_id`, `user_sandboxes.user_id`), and deleting a project with sessions fails the same way (`sessions.project_id`). File bytes are never collected (`0016`: blobs "are never collected"), so every deleted upload lives forever. Disconnecting an account can silently leave a live Composio grant. The Privacy Policy can't promise deletion or a retention window until all of this works. Success: every deletable thing has one endpoint, one representation, and one test proving nothing is left behind.

---

# Technical Background
**Rows.** Postgres foreign keys already do most of the work. Everything under `sessions` cascades: `session_events`, `result_blobs`, `approvals`, `resource_leases`, `session_tools`, `session_sandboxes`, `session_claims`. Everything a user owns directly cascades from `users`: `projects`, `files`, `deleted_files`, `memory_files`, `user_connections`, `auth_sessions`, `session_sandboxes`. And `0024` makes `users.id` a foreign key to `auth.users(id) ON DELETE CASCADE`, so deleting the Supabase account deletes our row and everything under it. Three links break the chain:
- `sessions.user_id` and `user_sandboxes.user_id`: `REFERENCES users(id)`, NO ACTION. `0024`'s header says it fixed "two breaks"; these are the other two.
- `sessions.project_id`: NO ACTION, so a project with sessions can't be deleted. NO ACTION is checked at end of statement, so a user delete that takes projects and sessions together doesn't trip it.

`users.home_session_id` is `ON DELETE SET NULL`, and `_ensure_home_session` makes a new home session at the next sign-in.

**Files.** `store.delete_path` moves rows from `files` to `deleted_files` under a `batch`, with the dropped project links in `deleted_links`; `undo_delete` restores a batch exactly. Nothing ever expires them.

**Bytes.** Blobs are content-addressed at `{prefix}/blobs/{hh}/{sha256}`, with no owner in the path, so two users uploading the same bytes share one object. Three live tables hold a hash: `files.content_hash`, `deleted_files.content_hash` and `memory_files.content_hash` (memory writes go through `put_blob`; `memory_files.body` is only a searchable copy). `result_blobs.content` is TEXT in Postgres and holds no hash. `Blobs` has `put`, `get`, `missing` and no `delete`. `SupabaseBlobs._url` addresses one object, and `_is_absent` already maps Storage's 404, and its 400-with-NoSuchKey, to "absent".

**Third parties.** `Composio.disconnect` reads the user's live accounts from Composio (`_accounts`), calls `client.delete_connected_account`, catches any `ComposioError` with a warning, and forgets our row regardless. Fine for a panel button; wrong for account deletion, where a swallowed failure orphans the grant. `runner.cancel` ends a live or parked run. Supabase Auth owns the account itself.

**Timers.** `system_log.start` calls `prune()` once at boot (30-day log retention); `_keys_and_sweep` drops expired `auth_sessions`. Run-at-boot is the pattern for retention here; there's no scheduler.

---

# Proposed Approach

## The deletion model

| What | Who starts it | How it's represented | Bytes | Third parties | Undo |
|---|---|---|---|---|---|
| File or folder | user, file tree | rows move to `deleted_files` (+ `deleted_links`) under one `batch` | kept while undo lasts, then collected if unreferenced | — | 30 days |
| Session | user | `DELETE FROM sessions`; events, approvals, results, tools, claims and sandbox slot cascade | `result_blobs` are rows, gone with it | its sandbox released | none |
| Project | user | `DELETE FROM projects`; folder links cascade; sessions detach (`project_id` → null) | none (projects own no files) | — | none |
| Account | user, typing their email | cookies killed → runs cancelled → grants revoked → Supabase auth user deleted, which cascades every row | collected where no other user's row references them | Composio accounts deleted; Supabase account deleted | none |
| Expired undo | boot job | `deleted_files` rows past the window deleted | collected if unreferenced | — | — |
| Inactive account (after launch) | boot job | notice, then the account path above | same as account | same as account | — |

**Kept on purpose:** consent records (`legal_acceptances`, ready 05) have no foreign key and survive account deletion as evidence of agreement. `system_events` stays on its existing 30-day prune.

## Invariants
1. **One root per delete.** Each delete removes one row (`sessions`, `projects`, `auth.users`) and foreign keys do the rest. No handwritten multi-table deletes.
2. **Revoke outside state before deleting the rows that point at it.** A failed revoke stops the delete and leaves everything in place.
3. **Bytes go last,** and only for hashes no `files`, `deleted_files` or `memory_files` row still names.
4. **Every step is idempotent,** so a retry after a partial failure finishes the job.
5. **Only files have undo.** Everything else is final and says so before the click.
6. **Timers delete only expired things:** undo past its window, logs, expired auth sessions, and (after launch) noticed inactive accounts.

**Defaults:** undo window `retention.deleted_files_days: 30` (D7 is still open; config ships 30 and the key comes from ready 01). Project delete detaches sessions; to delete them instead, use `ON DELETE CASCADE` in Task 3 and change the confirm copy. Consent records are kept. Inactive accounts are kept at launch.

**Not in scope:** grace periods or undo for accounts, sessions or projects; admin-initiated deletion; bulk delete; a full-bucket orphan scan (`Blobs` has no list call, so only hashes a caller can name are deleted); pruning `session_events` on a timer.

---

# Implementation Plan

**Order inside this spec:**
- **Part A** and **Task 4** have no blockers (wave 1).
- **Task 5** needs only Task 3.
- **Tasks 6–7** need Part A.
- **Tasks 8–9** need Task 2 and ready 01's `retention:` block (wave 2).
- **Part D** is after launch.

## Part A — Primitives

## Task 1: `Blobs.delete`
**Problem:** neither backend can remove an object.
**Done when:** `delete(hash)` removes the object on both backends; deleting a hash that isn't there returns without raising; any other Storage error raises `StoreError`.
**Touch point:** `harness_module/blobs.py`: `Blobs` protocol, `FilesystemBlobs.delete` (unlink via `asyncio.to_thread`, `missing_ok=True`), `SupabaseBlobs.delete` (`client.delete(self._url(hash))` under `self._gate`, absent via `_is_absent`), module-level `delete_blob`.
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** batched deletes.
**Acceptance test:** `test_delete_blob_twice_is_harmless`.

```python
class Blobs(Protocol):
    async def delete(self, content_hash: str) -> None: ...

async def delete_blob(content_hash: str) -> None: ...
```

---

## Task 2: `store.delete_unreferenced`
**Problem:** callers need "delete these hashes unless anyone still uses them".
**Done when:** given hashes where some are still in `files`, `deleted_files` or `memory_files` for any user, only the unreferenced ones are deleted, and the returned count equals the number deleted.
**Touch point:** `harness_module/store.py` (new function; re-export `delete_blob` with the other blob calls).
**Priority:** High | **Effort:** <2h | **Blockers:** Task 1
**Out of scope:** when to call it.
**Acceptance test:** `test_shared_hash_is_kept`, `test_every_hash_column_is_checked`.

```python
async def delete_unreferenced(hashes: Iterable[str]) -> int:
    """Delete each hash that no files, deleted_files or memory_files row still names; return how many went."""
    # SELECT h FROM unnest($1::text[]) h
    #  WHERE NOT EXISTS (SELECT 1 FROM files         WHERE content_hash = h)
    #    AND NOT EXISTS (SELECT 1 FROM deleted_files WHERE content_hash = h)
    #    AND NOT EXISTS (SELECT 1 FROM memory_files  WHERE content_hash = h)
```

---

## Task 3: Fix the foreign keys deletion depends on
**Problem:** deleting a user fails on `sessions.user_id` / `user_sandboxes.user_id`, and deleting a project with sessions fails on `sessions.project_id`.
**Done when:** deleting a user with a session, a sandbox row, a project, files, memory and a connection succeeds and leaves zero rows for that user in every user-linked table; deleting a project with two sessions leaves both sessions with null `project_id`, its links gone, and its files intact.
**Touch point:** new `db/migrations/NNNN_deletion_foreign_keys.sql` (next free number at merge; the repo is at 0024); `tests/test_store_schema.py`.
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** `legal_acceptances` (deliberately unlinked).
**Acceptance test:** `test_deleting_a_user_leaves_nothing`, `test_no_deletion_chain_fk_is_no_action`, `test_deleting_a_project_detaches_its_sessions`.

```sql
BEGIN;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_user_id_fkey;
ALTER TABLE sessions ADD CONSTRAINT sessions_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE user_sandboxes DROP CONSTRAINT IF EXISTS user_sandboxes_user_id_fkey;
ALTER TABLE user_sandboxes ADD CONSTRAINT user_sandboxes_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_project_id_fkey;
ALTER TABLE sessions ADD CONSTRAINT sessions_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL;
COMMIT;
```
The header comment corrects `0024`'s "two breaks". Migrations are append-only, so `0024` itself isn't edited.

---

## Part B — User-facing deletes

## Task 4: Delete a session
**Problem:** a user can't remove one conversation.
**Done when:** the owner gets 204, and the session's events, approvals and result blobs are gone; a non-owner gets 404; a `running` session gets `409 running`; deleting the home session nulls `users.home_session_id`, and the next sign-in makes a new one.
**Touch point:** `harness_module/api.py` (new `delete_session`, using `_owned_session`); `frontend/api.jsx` (new `api.deleteSession`); `frontend/lookingglass.jsx` (`SessionDetail`: two-click confirm, matching `SettingsModal`'s disconnect pattern); `frontend/index.html` (`?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** none (everything under `sessions` already cascades)
**Out of scope:** bulk delete.
**Acceptance test:** `test_delete_session_takes_its_transcript`, `test_running_session_cannot_be_deleted`.

---

## Task 5: Delete a project
**Problem:** projects pile up forever.
**Done when:** `DELETE /projects/{project_id}` returns 204 for the owner, 404 for anyone else, and `409 running` while any of its sessions is `running`; the project card has a delete control beside rename, with a two-click confirm reading "delete project — its conversations and files stay".
**Touch point:** `harness_module/api.py` (new `delete_project`, using `_owned_project`); `frontend/api.jsx` (new `api.deleteProject`); `frontend/lookingglass.jsx` (`LookingGlassView` project card, beside `pc-rename-btn`); `frontend/index.html` (`?v=`).
**Priority:** Medium | **Effort:** <4h | **Blockers:** Task 3
**Out of scope:** moving sessions into another project.
**Acceptance test:** `test_delete_project_refuses_while_running`; in the browser pane, delete a project and find its session under desk.

---

## Task 6: The account deletion orchestrator
**Problem:** deletion spans Postgres, the blob store, Composio, e2b and Supabase Auth, and a half-done run must be finishable.
**Done when:** `delete_account(user_id)` performs the steps below in order; a `ComposioError` stops it before any row is deleted and raises `DeletionBlocked`; a live Composio account with no `user_connections` row is still revoked; a second call on the same user is a no-op.
**Touch point:** `harness_module/account.py` (new or extended: ready 03 may have created it); `tool_module/composio_mcp.py` (new `Composio.revoke_all`, which raises; `disconnect` is unchanged).
**Priority:** High | **Effort:** 1 day | **Blockers:** Tasks 2, 3
**Out of scope:** HTTP (Task 7).
**Acceptance test:** `test_delete_account_removes_everything`, `test_delete_account_stops_when_composio_refuses`, `test_delete_account_twice_is_harmless`.

```python
class DeletionBlocked(RuntimeError): ...

async def delete_account(user_id: str) -> None:
    # 1. DELETE FROM auth_sessions WHERE user_id = $1                     -- every cookie dies now
    # 2. runner.cancel(s) for each session whose status is not in lifecycle.TERMINAL
    # 3. client = hands.connectors(); if client: await client.revoke_all(user_id)
    #    ComposioError (listing or deleting) -> raise DeletionBlocked (rows untouched)
    # 4. hashes = content_hash FROM files, deleted_files, memory_files WHERE user_id = $1
    # 5. Supabase Admin API DELETE {project_url()}/auth/v1/admin/users/{id} with secret_key();
    #    404 counts as done; no project_url -> DELETE FROM users WHERE id = $1
    # 6. store.delete_unreferenced(hashes)
    # system_log.record(...) at each step

# tool_module/composio_mcp.py
async def revoke_all(self, user_id: str) -> list[str]:
    """Delete every live Composio account for this user, raising ComposioError on the first failure."""
```
Step 5 runs before step 6 on purpose: once the rows are gone, the reference check is honest.

---

## Task 7: Account deletion endpoint and Settings UI
**Problem:** users can't trigger it.
**Done when:** `DELETE /auth/me` with `{"confirm_email": "<account email>"}` (case-insensitive) returns 204 and clears the cookie; a mismatch returns 422; `DeletionBlocked` returns `502 upstream_error` with a retry message. Settings has a danger-zone block that says what gets deleted, points to "download my data", requires the email, and lands on sign-in with "your account has been deleted".
**Touch point:** `harness_module/api.py` (new `delete_me` on `CurrentUser`, never consent-guarded); `frontend/api.jsx` (new `api.deleteAccount`); `frontend/views.jsx` (`SettingsModal`); `frontend/app.jsx` (the notice via the existing `authNotice`); `frontend/index.html` (`?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 6
**Out of scope:** a confirmation email.
**Acceptance test:** `test_delete_me_requires_matching_email`; in the browser pane, delete a test account and sign in again with the same email, which gives a fresh account.

---

## Part C — Retention

## Task 8: Purge expired undo records and their bytes
**Problem:** deleted files and their bytes are kept forever.
**Done when:** at startup, `deleted_files` rows older than the window are removed, `deleted_links` rows whose `batch` has no remaining `deleted_files` go too, and each freed hash no live row references is deleted from the blob store; rows inside the window are untouched.
**Touch point:** new `harness_module/retention.py`; `harness_module/api.py` `lifespan` (beside `sweep_interrupted`, inside `contextlib.suppress` like its neighbours).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2, ready 01 (`retention:` block)
**Out of scope:** a periodic scheduler. Deploys restart often enough in beta.
**Acceptance test:** `test_purge_keeps_recent_undo`, `test_purge_frees_only_unreferenced_bytes`.

```python
async def purge_deleted_files(now: datetime | None = None) -> int:
    """Drop undo records past retention.deleted_files_days and collect their unshared blobs."""
```

---

## Task 9: Say the undo window
**Problem:** users don't know undo expires.
**Done when:** the undo control in the file tree reads "undo (kept 30 days)", with the number read from `limits.deleted_files_days` in `/auth/config`.
**Touch point:** `frontend/components.jsx` (`FileTree`, the `undone`/`undo` block); `frontend/index.html` (`?v=`).
**Priority:** Low | **Effort:** <1h | **Blockers:** ready 01 (`limits.deleted_files_days`)
**Out of scope:** a trash view.
**Acceptance test:** visible in the browser pane after deleting a file.

---

## Part D — After launch

## Task 10: Track last sign-in
**Problem:** there's no activity signal to judge inactivity by.
**Done when:** each `POST /auth/session` sets `users.last_sign_in_at = now()` and clears `deletion_notice_at`; existing rows are backfilled with `created_at`.
**Touch point:** new `db/migrations/NNNN_last_sign_in.sql`; `harness_module/api.py` (`create_auth_session` upsert). Per sign-in, not per request: `current_user` must stay one read.
**Priority:** Low | **Effort:** <2h | **Blockers:** none (safe to land before launch; it only records)
**Out of scope:** the sweep.
**Acceptance test:** `test_sign_in_stamps_and_clears_notice`.

```sql
ALTER TABLE users ADD COLUMN last_sign_in_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN deletion_notice_at TIMESTAMPTZ;
UPDATE users SET last_sign_in_at = created_at WHERE last_sign_in_at IS NULL;
```

---

## Task 11: Inactive-account notice and sweep
**Problem:** unused accounts keep conversations, files, memory and live grants forever.
**Done when:** with `retention.inactive_account_days` set, the boot sweep sends one notice to each account past the window (never a second), and calls `delete_account` for each account noticed more than `retention.inactive_notice_days` ago with no sign-in since; with it null, the sweep does nothing.
**Touch point:** `harness_module/retention.py` (new `sweep_inactive_accounts`); `harness_module/api.py` (`lifespan`); `config_module/config.yaml` (`retention.inactive_account_days: null`, `retention.inactive_notice_days: 30`); the mail sender chosen in D16.
**Priority:** Low | **Effort:** 1 day | **Blockers:** Tasks 6, 10; D16 (arkos sends no email of its own today)
**Out of scope:** reminder emails.
**Acceptance test:** `test_sweep_notices_then_deletes`, `test_sweep_is_off_by_default`.

```python
async def sweep_inactive_accounts(now: datetime | None = None) -> tuple[int, int]:
    """(noticed, deleted). Off while retention.inactive_account_days is null."""
```

---

# Tests
Mock Composio (`hands.connectors`), the Supabase Admin call and `runner.cancel`. Use `FilesystemBlobs` via `use_blobs`; the Supabase backend gets a mocked `httpx` client through its constructor's `client` argument. New `tests/test_blobs.py`, `tests/test_account.py` and `tests/test_retention.py`; additions to `tests/test_store.py`, `tests/test_store_schema.py` and `tests/test_api.py`.

## Test 1: `test_delete_blob_twice_is_harmless`
**What it verifies:** the first delete removes the bytes; the second raises nothing; on Supabase a 404 is success and a 500 raises `StoreError`.
**Why this matters:** account deletion retries after partial failure.

## Test 2: `test_shared_hash_is_kept`
**What it verifies:** hash H in user A's `files`, user B's `deleted_files` and user C's `memory_files`; `delete_unreferenced([H])` keeps the bytes until the last reference goes.
**Why this matters:** content addressing makes the naive delete destroy someone else's data.

## Test 3: `test_every_hash_column_is_checked`
**What it verifies:** the set of tables with a `content_hash` column in `information_schema.columns` equals `{files, deleted_files, memory_files}`.
**Why this matters:** a new table holding hashes would silently turn cleanup into data loss; this forces it into the query.

## Test 4: `test_deleting_a_user_leaves_nothing`
**What it verifies:** with a row seeded in every user-linked table (discovered from `information_schema`), `DELETE FROM users` succeeds and each table has zero rows for that user.
**Why this matters:** the next table anyone adds is covered without anyone remembering.

## Test 5: `test_no_deletion_chain_fk_is_no_action`
**What it verifies:** every foreign key targeting `users`, `projects` or `sessions` has `delete_rule` `CASCADE` or `SET NULL`.
**Why this matters:** these are the breaks `0024` missed.

## Test 6: `test_deleting_a_project_detaches_its_sessions`
**What it verifies:** both sessions survive with null `project_id`, `project_folders` is empty for the project, and `files` is unchanged.
**Why this matters:** the promise is that nothing but the project goes.

## Test 7: `test_delete_session_takes_its_transcript`
**What it verifies:** after 204, `session_events`, `approvals` and `result_blobs` have zero rows for that session.
**Why this matters:** the cascade is the promise.

## Test 8: `test_running_session_cannot_be_deleted`
**What it verifies:** a `running` session gets 409 and survives.
**Why this matters:** deleting under a live turn races the runner's writes.

## Test 9: `test_delete_project_refuses_while_running`
**What it verifies:** a project with a `running` session gets 409 and survives.
**Why this matters:** detaching under a live turn changes which folders its claims resolve against.

## Test 10: `test_delete_account_removes_everything`
**What it verifies:** after `delete_account`, the user's rows are gone, unshared blobs are gone, and `revoke_all` and the Admin API were each called once.
**Why this matters:** this is the whole guarantee in one assertion set.

## Test 11: `test_delete_account_stops_when_composio_refuses`
**What it verifies:** a `ComposioError` from `revoke_all` raises `DeletionBlocked`, and `users` and `user_connections` are intact.
**Why this matters:** a deleted row with a live grant is an OAuth token we can no longer revoke.

## Test 12: `test_delete_account_twice_is_harmless`
**What it verifies:** the second call raises nothing and makes no Composio or Admin API calls.
**Why this matters:** retry is the recovery path.

## Test 13: `test_delete_me_requires_matching_email`
**What it verifies:** a wrong `confirm_email` returns 422 and deletes nothing; the right one returns 204, and `GET /auth/me` then returns 401.
**Why this matters:** one mis-click mustn't end an account.

## Test 14: `test_purge_keeps_recent_undo`
**What it verifies:** with `now` injected, a 29-day-old undo row survives and a 31-day-old one is gone.
**Why this matters:** the policy's number must be the code's number.

## Test 15: `test_purge_frees_only_unreferenced_bytes`
**What it verifies:** an expired undo row whose hash is still in `files` keeps its blob.
**Why this matters:** the same trap as Test 2, through the retention path.

## Test 16: `test_sign_in_stamps_and_clears_notice`
**What it verifies:** a sign-in sets `last_sign_in_at` and nulls `deletion_notice_at`.
**Why this matters:** coming back must cancel the countdown.

## Test 17: `test_sweep_notices_then_deletes`
**What it verifies:** with `now` injected, an inactive account gets exactly one notice; 31 days later `delete_account` is called once; an account that signed in after its notice isn't touched.
**Why this matters:** deleting without notice, or deleting someone who came back, is the failure.

## Test 18: `test_sweep_is_off_by_default`
**What it verifies:** with `inactive_account_days: null`, no notice is sent and nothing is deleted.
**Why this matters:** Part D ships dark until it's switched on.

---

# Open Questions
1. *Verify in Task 1, not a decision:* the status Supabase Storage returns for a single-object `DELETE` on a missing key, and that `_is_absent` covers it. Record it in Implementation Notes.
2. *Verify in Task 6:* the Supabase Admin API path and which key it accepts (`SUPABASE_SECRET_KEY` vs the legacy service key).
3. *Verify in Tasks 4 and 6:* does `runner.cancel` (or the session-row cascade) free the e2b box for an idle session holding a slot? If not, call `sandbox_manager.release_slot` and kill the box first. Check `docs/e2b-local-migration-spec` on origin as well: if the sandbox moves to a local backend, use whatever release call that branch lands.

---

# Implementation Notes
