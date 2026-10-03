# // Feature Spec: Account Deletion (NMORGAN)

*Ready 05 · Wave 2 · blocked only by ready 01. A user can delete their account, and it actually goes: rows, unshared bytes, third-party grants, and the Supabase auth user.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §1.2, §4
- `db/migrations/0000_migration_0.sql` (`sessions`, `user_sandboxes`), `0008_home_session.sql`, `0024_users_are_supabase_accounts.sql`
- `harness_module/runner.py` (`cancel`), `harness_module/lifecycle.py` (`TERMINAL`), `harness_module/hands.py` (`connectors`), `tool_module/connections.py` (`load`)
- `harness_module/blobs.py` (`project_url`, `secret_key`), `harness_module/store.py` (`delete_unreferenced`, from ready 01)
- `harness_module/api.py` (`disconnect_server`, `delete_auth_session`), `frontend/views.jsx` (`SettingsModal`), `tests/test_store_schema.py`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
There's no way to delete an account, and the database would refuse if we tried. `sessions.user_id` and `user_sandboxes.user_id` are `REFERENCES users(id)` with no `ON DELETE`, and every user has a home session, so deleting any real user fails on a foreign key. A cascade on `user_connections` would also leave the Composio grant alive. Success: `DELETE /auth/me` leaves no row, no unshared blob, no Composio grant and no Supabase auth user for that account, and it's safe to retry.

---

# Technical Background
`0024` made `users.id` a foreign key to `auth.users(id) ON DELETE CASCADE` and fixed `projects` and `user_connections`. Its header says the chain "had two breaks"; it had four. Everything below `sessions` (`session_events`, `result_blobs`, `approvals`, `resource_leases`, `session_tools`, `session_sandboxes`, `session_claims`) already cascades. `sessions.project_id` is NO ACTION, which is checked at end of statement, so deleting a user removes projects and sessions together without tripping it. `users.home_session_id` is `ON DELETE SET NULL`.

`DELETE /connections/{server}` calls `hands.connectors().disconnect(user_id, server)`, which revokes at Composio first. `connections.load(user_id)` returns the user's rows keyed by server. `runner.cancel(session_id)` ends a live or parked run.

---

# Proposed Approach
1. A migration closes the two missing links, and a test that discovers tables from `information_schema` keeps the chain closed.
2. `account.delete_account(user_id)` runs fixed, idempotent steps: kill every cookie, cancel runs, revoke grants, collect hashes, delete the Supabase auth user (which cascades everything), then `store.delete_unreferenced`. A retry after a partial failure finishes the job.
3. `DELETE /auth/me` requires the account email typed back.

No `deleting_at` column: deleting `auth_sessions` first kills every cookie at once, which is the same guarantee.

**Default:** consent records (`legal_acceptances`, ready 07) have no foreign key, so they survive deletion and this spec doesn't touch them. Deleting them too would be one migration there.

**Not in scope:** a grace period or undo; admin-initiated deletion; inactive-account deletion (deferred 01); `sessions.project_id` semantics (ready 11).

---

# Implementation Plan

## Task 1: Close the cascade chain
**Problem:** deleting a user with any session raises a foreign key violation.
**Done when:** deleting a `users` row that has a session, a sandbox row, a project, files, memory and a connection succeeds, and every table with a `user_id` column (or a chain to `sessions`/`projects`) has zero rows for that user.
**Touch point:** new `db/migrations/NNNN_user_delete_cascade.sql` (next free number at merge; the repo is at 0024); `tests/test_store_schema.py`.
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** `sessions.project_id`.
**Acceptance test:** `test_deleting_a_user_leaves_nothing`, `test_no_user_chain_fk_is_no_action`.

```sql
BEGIN;
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_user_id_fkey;
ALTER TABLE sessions ADD CONSTRAINT sessions_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE user_sandboxes DROP CONSTRAINT IF EXISTS user_sandboxes_user_id_fkey;
ALTER TABLE user_sandboxes ADD CONSTRAINT user_sandboxes_user_id_fkey
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
COMMIT;
```
The header comment corrects `0024`'s "two breaks". Migrations are append-only, so `0024` itself isn't edited.

---

## Task 2: The deletion orchestrator
**Problem:** deletion spans Postgres, the blob store, Composio, e2b and Supabase Auth, and a half-done run must be finishable.
**Done when:** `delete_account(user_id)` performs the steps below in order; a `ComposioError` stops it before any row is deleted and raises `DeletionBlocked`; a second call on the same user is a no-op.
**Touch point:** `harness_module/account.py` (new or extended: ready 03 may have created it).
**Priority:** High | **Effort:** 1 day | **Blockers:** Task 1, ready 01
**Out of scope:** HTTP (Task 3).
**Acceptance test:** `test_delete_account_removes_everything`, `test_delete_account_stops_when_composio_refuses`, `test_delete_account_twice_is_harmless`.

```python
class DeletionBlocked(RuntimeError): ...

async def delete_account(user_id: str) -> None:
    # 1. DELETE FROM auth_sessions WHERE user_id = $1          -- every cookie dies now
    # 2. runner.cancel(s) for each session whose status is not in lifecycle.TERMINAL
    # 3. for server in connections.load(user_id): hands.connectors().disconnect(user_id, server)
    #    ComposioError -> raise DeletionBlocked (rows untouched, grant not orphaned)
    # 4. hashes = content_hash FROM files, deleted_files, memory_files WHERE user_id = $1
    # 5. Supabase Admin API DELETE {project_url()}/auth/v1/admin/users/{id} with secret_key();
    #    404 counts as done; no project_url -> DELETE FROM users WHERE id = $1
    # 6. store.delete_unreferenced(hashes)
    # system_log.record(...) at each step
```
Step 5 runs before step 6 on purpose: once the rows are gone, the reference check is honest.

---

## Task 3: Endpoint and Settings UI
**Problem:** users can't trigger it.
**Done when:** `DELETE /auth/me` with `{"confirm_email": "<account email>"}` (case-insensitive) returns 204 and clears the cookie; a mismatch returns 422; `DeletionBlocked` returns `502 upstream_error` with a retry message. Settings has a danger-zone block that says what gets deleted, points to "download my data", requires the email, and lands on sign-in with "your account has been deleted".
**Touch point:** `harness_module/api.py` (new `delete_me` on `CurrentUser`, never consent-guarded); `frontend/api.jsx` (new `api.deleteAccount`); `frontend/views.jsx` (`SettingsModal`); `frontend/app.jsx` (the post-deletion notice via the existing `authNotice`); `frontend/index.html` (`?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2
**Out of scope:** a confirmation email.
**Acceptance test:** `test_delete_me_requires_matching_email`; in the browser pane, delete a test account and sign in again with the same email, which gives a fresh account.

---

# Tests
Mock `hands.connectors`, the Supabase Admin call and `runner.cancel`; use `FilesystemBlobs` via `use_blobs`.

## Test 1: `test_deleting_a_user_leaves_nothing`
**What it verifies:** after seeding a row in every user-linked table (discovered from `information_schema`), `DELETE FROM users` succeeds and each table has zero rows for that user.
**Why this matters:** the next table anyone adds is covered without anyone remembering.

## Test 2: `test_no_user_chain_fk_is_no_action`
**What it verifies:** every FK targeting `users` has `delete_rule` `CASCADE` or `SET NULL`.
**Why this matters:** it's the bug `0024` claimed to have fixed.

## Test 3: `test_delete_account_stops_when_composio_refuses`
**What it verifies:** a `ComposioError` from `disconnect` raises `DeletionBlocked`, and the `users` and `user_connections` rows are intact.
**Why this matters:** a deleted row with a live grant is an OAuth token we can no longer revoke.

## Test 4: `test_delete_account_twice_is_harmless`
**What it verifies:** the second call raises nothing and makes no Composio or Admin API calls.
**Why this matters:** retry is the recovery path.

## Test 5: `test_delete_me_requires_matching_email`
**What it verifies:** a wrong `confirm_email` returns 422 and deletes nothing; the right one returns 204, and `GET /auth/me` then returns 401.
**Why this matters:** one mis-click mustn't end an account.

---

# Open Questions
1. *Verify in Task 2, not a decision:* the Supabase Admin API path and which key it accepts (`SUPABASE_SECRET_KEY` vs the legacy service key). Check against the project; record it in Implementation Notes.
2. *Verify in Task 2:* does `runner.cancel` free the e2b box for an idle session holding a slot? If not, add `sandbox_manager.release_slot` plus a kill to step 2.

---

# Implementation Notes
