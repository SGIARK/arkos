# // Feature Spec: Data Export (NMORGAN)

*Ready 03 · Wave 1 · no blockers. One button that downloads everything arkos holds about you as a ZIP.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §4.1
- `harness_module/api.py` (`CurrentUser`, `_check_auth_rate` as the limiter pattern, `ApiError`)
- `harness_module/store.py` (`read_tree`), `harness_module/blobs.py` (`get_blob`), `harness_module/memory.py` (`read_memory`)
- `db/migrations/0000_migration_0.sql` (`sessions`, `session_events`, `approvals`), `0005_memory.sql` (`memory_files`), `0015_user_store.sql` (`project_folders`)
- `tool_module/connections.py` (`load`), `frontend/views.jsx` (`SettingsModal`)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
Users can't get their data out. The Privacy Policy draft promises access and portability, the Terms tell users to keep backups, and account deletion says "export first". None of that is true until this exists. Success: `GET /auth/me/export` returns a ZIP of every category in plan §4.1 that we hold for the user, with no credential in it.

---

# Technical Background
Everything is already addressable per user. `store.read_tree(user_id)` lists the file tree and `blobs.get_blob(hash)` returns bytes. `memory.read_memory(user_id)` returns the curated core; notes are `memory_files` rows. Sessions are `sessions WHERE user_id`, with `session_events` in `seq` order and their `approvals`. `connections.load(user_id)` returns `server`, `status`, `refreshed_at` and `connected_account_id`, and the last of those is a Composio handle that must not leave.

`_check_auth_rate` is the in-process sliding-window limiter already in `api.py`. Export copies its shape, keyed on `user_id`.

---

# Proposed Approach
`account.export_archive(user_id)` builds the ZIP into a `tempfile.SpooledTemporaryFile` (spills to disk past 64 MB): blob fetches run on the loop, `zipfile` writes run in `asyncio.to_thread`. Stdlib `zipfile` can't write to a socket; a spooled file bounds memory with no new dependency. `harness_module/account.py` is created here if ready 02 hasn't created it yet; whichever lands first creates it.

```
account.json          id, email, display_name, created_at, consent records if the table exists
projects.json         projects and linked folders
sessions.json         session rows
sessions/<id>.jsonl   events, oldest first
approvals.json
memory/MEMORY.md      core
memory/notes.jsonl
files/<path>          current tree bytes
connections.json      server, status, refreshed_at. Never connected_account_id
README.txt            what each file is, and that operational logs are excluded
```

**Defaults:** `system_events` (30-day operational logs) are excluded, and the README says so; `quotas.export_max_mb: 2048`. Both are one-line changes.

**Not in scope:** emailed async exports; files sitting in undo (`deleted_files`).

---

# Implementation Plan

## Task 1: Archive builder
**Problem:** nothing assembles one user's data in one place.
**Done when:** for a seeded user with 2 projects, 3 sessions, a memory core and 2 notes, 3 files and 1 connection, the archive holds every path above with matching counts; `connections.json` has no `connected_account_id`; exceeding `quotas.export_max_mb` raises `ExportTooLarge`.
**Touch point:** `harness_module/account.py` (new or extended: `export_archive`, `ExportTooLarge`); `config_module/config.yaml` (`quotas.export_max_mb: 2048`).
**Priority:** High | **Effort:** 1 day | **Blockers:** none
**Out of scope:** HTTP.
**Acceptance test:** `test_export_contains_every_section`, `test_export_carries_no_credentials`.

```python
class ExportTooLarge(RuntimeError): ...

async def export_archive(user_id: str) -> IO[bytes]:
    """A rewound spooled file holding the ZIP; the caller streams and closes it."""
```
Consent records: include them if `legal_acceptances` exists (`to_regclass`), so this spec doesn't depend on ready 05.

---

## Task 2: Endpoint and button
**Problem:** users can't ask for it.
**Done when:** `GET /auth/me/export` returns `application/zip` with `Content-Disposition: attachment; filename="arkos-export-YYYY-MM-DD.zip"`; a second call within 10 minutes returns `429 quota_exceeded`; `ExportTooLarge` returns 413; Settings has "download my data".
**Touch point:** `harness_module/api.py` (new `export_me` on `CurrentUser`; `_export_hits` limiter); `frontend/views.jsx` (`SettingsModal`: a plain `<a href="/auth/me/export" download>`, so the cookie rides along); `frontend/index.html` (`?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 1
**Out of scope:** a progress indicator.
**Acceptance test:** `test_export_is_rate_limited`; the ZIP downloads and opens in the browser pane.

---

# Tests
In new `tests/test_account.py` (Task 1) and `tests/test_api.py` (Task 2), with `FilesystemBlobs` via `use_blobs`.

## Test 1: `test_export_contains_every_section`
**What it verifies:** each layout path exists, and each session's JSONL line count equals its seeded event count.
**Why this matters:** a silently missing section is a broken access right.

## Test 2: `test_export_carries_no_credentials`
**What it verifies:** no file in the archive contains `connected_account_id` or the session cookie name.
**Why this matters:** exports get forwarded around, and one mustn't work as a key.

## Test 3: `test_export_is_rate_limited`
**What it verifies:** 200, then 429 within the window for the same user; a different user still gets 200.
**Why this matters:** export is the most expensive read we have.

---

# Open Questions
None.

---

# Implementation Notes
