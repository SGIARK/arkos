# // Feature Spec: Session Delete & Retention (NMORGAN)

*Ready 06 · Wave 2 · blocked by ready 01 and 02. Users can delete a conversation, and deleted files stop living forever.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §4.3; `docs/launch/DECISIONS.md` (D7)
- `harness_module/api.py` (`_owned_session`, `lifespan`), `harness_module/runner.py` (`is_running`)
- `harness_module/system_log.py` (`start`, `prune`), the run-at-boot pattern
- `harness_module/store.py` (`delete_path`, `undo_delete`, `delete_unreferenced` from ready 01), `db/migrations/0016_delete_undo.sql`
- `frontend/lookingglass.jsx` (`SessionDetail`), `frontend/components.jsx` (`FileTree`: the `undone`/`undo` block)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
A user can't delete a single conversation. Deleted files sit in `deleted_files` forever and their bytes are never collected, so "delete" in the file tree is really "hide". The Privacy Policy draft states a per-session delete right and an undo window that don't exist. Success: `DELETE /sessions/{id}` works, and a startup job purges undo records past the window along with bytes nothing else references.

---

# Technical Background
`_owned_session(session_id, user_id)` loads a session row or 404s. It's the ownership check every session handler uses. Everything under `sessions` cascades. `users.home_session_id` is `ON DELETE SET NULL`, and `_ensure_home_session` makes a new one at the next `POST /auth/session`.

The `session_events` table comment says "audit, never pruned". That describes the system; a user deleting their own conversation is a different act, and this spec adds no automatic event pruning.

`store.delete_path` moves rows to `deleted_files` under a `batch`, and `undo_delete` restores a batch. `system_log.start` calls `prune()` once at boot, which is the retention pattern here: run at startup, no scheduler.

---

# Proposed Approach
1. `DELETE /sessions/{session_id}`: owner only, `409` while `status = 'running'`, otherwise one `DELETE FROM sessions`, and the cascade does the rest.
2. `harness_module/retention.py` `purge_deleted_files()`: drop `deleted_files` rows older than `retention.deleted_files_days` (config from ready 02), then `store.delete_unreferenced` on their hashes. Called from `lifespan`.
3. The undo control says how long undo lasts, reading `limits.deleted_files_days` from `/auth/config` (ready 02).

**Decisions applied (defaults from `DECISIONS.md`):** D7 → 30 days, in config.

**Not in scope:** deleting projects (ready 11); deleting inactive accounts (deferred 01, post-launch); pruning `session_events` on a timer.

---

# Implementation Plan

## Task 1: Delete a session
**Problem:** a user can't remove one conversation.
**Done when:** the owner gets 204, and the session's events, approvals and result blobs are gone; a non-owner gets 404; a `running` session gets `409 running`; deleting the home session nulls `users.home_session_id`, and the next sign-in makes a new one.
**Touch point:** `harness_module/api.py` (new `delete_session`, using `_owned_session`); `frontend/api.jsx` (new `api.deleteSession`); `frontend/lookingglass.jsx` (`SessionDetail`: delete with a two-click confirm, the same pattern as `SettingsModal`'s disconnect); `frontend/index.html` (`?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** bulk delete.
**Acceptance test:** `test_delete_session_takes_its_transcript`, `test_running_session_cannot_be_deleted`.

---

## Task 2: Purge expired undo records and their bytes
**Problem:** deleted files and their bytes are kept forever.
**Done when:** at startup, `deleted_files` rows older than the window are removed, `deleted_links` rows whose `batch` has no remaining `deleted_files` go too, and each freed hash no live row references is deleted from the blob store. Rows inside the window are untouched.
**Touch point:** new `harness_module/retention.py`; `harness_module/api.py` `lifespan` (call it beside `sweep_interrupted`, inside `contextlib.suppress` like its neighbours).
**Priority:** High | **Effort:** <4h | **Blockers:** ready 01, ready 02 (the `retention:` block)
**Out of scope:** a periodic scheduler. Deploys restart often enough in beta.
**Acceptance test:** `test_purge_keeps_recent_undo`, `test_purge_frees_only_unreferenced_bytes`.

```python
async def purge_deleted_files(now: datetime | None = None) -> int:
    """Drop undo records past retention.deleted_files_days and collect their unshared blobs."""
```

---

## Task 3: Say the window
**Problem:** users don't know undo expires.
**Done when:** the undo control in the file tree reads "undo (kept 30 days)", with the number read from `/auth/config`.
**Touch point:** `frontend/components.jsx` (`FileTree`); `frontend/index.html` (`?v=`).
**Priority:** Low | **Effort:** <1h | **Blockers:** ready 02 (`limits.deleted_files_days`)
**Out of scope:** a trash view.
**Acceptance test:** visible in the browser pane after deleting a file.

---

# Tests

## Test 1: `test_delete_session_takes_its_transcript`
**What it verifies:** after 204, `session_events`, `approvals` and `result_blobs` have zero rows for that session id.
**Why this matters:** the cascade is the promise.

## Test 2: `test_running_session_cannot_be_deleted`
**What it verifies:** a session with `status = 'running'` gets 409 and survives.
**Why this matters:** deleting under a live turn races the runner's writes.

## Test 3: `test_purge_keeps_recent_undo`
**What it verifies:** with `now` injected, a 29-day-old row survives and a 31-day-old row is gone.
**Why this matters:** the policy's number must be the code's number.

## Test 4: `test_purge_frees_only_unreferenced_bytes`
**What it verifies:** an expired undo row whose hash is still in `files` keeps its blob.
**Why this matters:** the same content-addressing trap as account deletion.

---

# Open Questions
None. D7 ships with its default, and project delete is its own spec (ready 11).

---

# Implementation Notes
