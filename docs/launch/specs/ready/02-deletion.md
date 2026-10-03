# // Feature Spec: Deletion (NMORGAN)

*Ready 02 · How arkos deletes things, and the work to get there. Parts A–C before launch, Part D after.*

---

**Sources**
- Database: `db/migrations/` (0000, 0005, 0015, 0016, 0024)
- Storage: `harness_module/blobs.py`, `harness_module/store.py`, `harness_module/memory.py`
- Connected accounts: `tool_module/composio_mcp.py`
- API and startup: `harness_module/api.py`
- Screens: `frontend/views.jsx`, `frontend/lookingglass.jsx`, `frontend/components.jsx`
- Decisions: `docs/launch/DECISIONS.md` (D7 retention, D16 email)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
Today a user can only delete files, and even those aren't really gone: they're hidden, and their contents stay in storage forever. Conversations, projects and accounts can't be deleted at all, and the database would refuse an account or project delete if we tried. Disconnecting Gmail or another account can leave arkos's access to it switched on. Until this is fixed, the Privacy Policy can't promise deletion. Success: users can delete each kind of thing, and a test proves nothing is left behind.

---

# Technical Background
Two terms used below:
- **Cascade:** a database rule that deletes child records automatically when their parent is deleted.
- **Blob:** the stored contents of a file. Identical files share one blob, even between users, so a blob can only be removed once nothing points at it.

What's true today:
- Most data already cascades from its user or its conversation. Three links don't, and they block deletion: conversation → user, sandbox → user, conversation → project.
- Deleting a file moves it to an undo list. Nothing ever empties that list.
- Blobs can be written but never deleted. Three tables point at them: files, the undo list, and the agent's memory.
- The existing "disconnect" button quietly ignores a failed revoke at Composio and forgets the connection anyway.
- Cleanup jobs run once when the server starts (that's how the 30-day log cleanup works). There's no scheduler.

---

# Proposed Approach

## What deletion means

| Thing | What happens | Undo? |
|---|---|---|
| File or folder | Moved to undo for 30 days, then removed along with its contents | Yes, 30 days |
| Conversation | Removed with its messages and approvals | No |
| Project | Removed; its conversations and files stay | No |
| Account | Connected accounts are revoked first, then the account and everything in it is removed | No |
| Inactive account (after launch) | Warned by email, then deleted like an account | No |

Kept on purpose: consent records (proof the user agreed to the Terms) and 30 days of server logs.

## Rules every delete follows
1. Delete one parent record and let cascades remove the rest.
2. Revoke outside access (Composio) first. If that fails, stop and change nothing.
3. Remove blobs last, and only when nothing else points at them.
4. Every step is safe to run twice.
5. Only files have undo. Everything else warns before the click.

**Defaults:** 30-day undo (`retention.deleted_files_days`; D7 is still open). Deleting a project keeps its conversations.

**Not in scope:** undo for accounts, projects or conversations; admin deletes; bulk delete.

---

# Implementation Plan
**Order:** Tasks 1–4 can start now. Tasks 5–9 follow. Part D waits until after launch.

## Part A — Groundwork

## Task 1: Let storage delete a blob
**Problem:** blobs can't be deleted.
**Done when:** deleting a blob removes it, and deleting one that's already gone is not an error.
**Touch point:** `harness_module/blobs.py` (add `delete` to both storage backends).
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** deleting many at once.
**Acceptance test:** Test 1.

---

## Task 2: Delete only unused blobs
**Problem:** a blob may be shared, so it can't be deleted blindly.
**Done when:** `delete_unreferenced(hashes)` removes only blobs that no file, undo entry or memory note points at.
**Touch point:** `harness_module/store.py`.
**Priority:** High | **Effort:** <2h | **Blockers:** Task 1
**Out of scope:** scanning all of storage for strays.
**Acceptance test:** Tests 2–3.

---

## Task 3: Fix the three database links
**Problem:** deleting a user or a project fails.
**Done when:** deleting a user removes all their data; deleting a project keeps its conversations (unlinked) and its files.
**Touch point:** one new migration in `db/migrations/` (next free number).
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** editing old migrations.
**Acceptance test:** Tests 4–6.

```sql
ALTER TABLE sessions       -- conversation → user: cascade
    DROP CONSTRAINT sessions_user_id_fkey,
    ADD CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE user_sandboxes -- sandbox → user: cascade
    DROP CONSTRAINT user_sandboxes_user_id_fkey,
    ADD CONSTRAINT user_sandboxes_user_id_fkey FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE;
ALTER TABLE sessions       -- conversation → project: unlink, don't delete
    DROP CONSTRAINT sessions_project_id_fkey,
    ADD CONSTRAINT sessions_project_id_fkey FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL;
```

---

## Part B — Delete buttons

## Task 4: Delete a conversation
**Problem:** a user can't remove a conversation.
**Done when:** the owner can delete it; anyone else gets "not found"; a conversation that's still running must be stopped first.
**Touch point:** `harness_module/api.py` (`DELETE /sessions/{id}`); `frontend/lookingglass.jsx` (button with a confirm click).
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** deleting several at once.
**Acceptance test:** Tests 7–8.

---

## Task 5: Delete a project
**Problem:** projects pile up forever.
**Done when:** the owner can delete a project; the confirm says "its conversations and files stay"; a project with a running conversation can't be deleted until it's stopped.
**Touch point:** `harness_module/api.py` (`DELETE /projects/{id}`); `frontend/lookingglass.jsx` (project card).
**Priority:** Medium | **Effort:** <4h | **Blockers:** Task 3
**Out of scope:** moving conversations to another project.
**Acceptance test:** Test 9.

---

## Task 6: Delete an account (behind the scenes)
**Problem:** an account touches the database, file storage, Composio and Supabase, and a half-finished delete must be finishable.
**Done when:** `delete_account` runs these steps in order, and running it again finishes or does nothing:
1. Sign the user out everywhere.
2. Stop their running conversations.
3. Revoke every connection Composio holds for them, including any arkos lost track of. If one fails, stop.
4. Note which blobs they used.
5. Delete their Supabase account, which removes all their records.
6. Remove the blobs nobody else uses.
**Touch point:** new `harness_module/account.py`; `tool_module/composio_mcp.py` (add `revoke_all`, which reports failures instead of hiding them).
**Priority:** High | **Effort:** 1 day | **Blockers:** Tasks 2, 3
**Out of scope:** the screen (Task 7).
**Acceptance test:** Tests 10–12.

---

## Task 7: Delete an account (the screen)
**Problem:** users have no way to start it.
**Done when:** Settings has a "delete account" section that says what will be removed and suggests downloading data first; the user types their email to confirm; afterwards they land on sign-in with "your account has been deleted". A wrong email changes nothing.
**Touch point:** `harness_module/api.py` (`DELETE /auth/me`); `frontend/views.jsx` (Settings).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 6
**Out of scope:** a confirmation email.
**Acceptance test:** Test 13; in the browser, delete a test account, then sign up again with the same email.

---

## Part C — Cleanup

## Task 8: Empty the undo list after 30 days
**Problem:** deleted files are kept forever.
**Done when:** at server start, undo entries older than 30 days are removed, with their saved project links and any blobs nobody uses. Newer entries are untouched.
**Touch point:** new `harness_module/retention.py`; `harness_module/api.py` (run it at startup).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2; ready 01 (the setting)
**Out of scope:** a scheduler.
**Acceptance test:** Tests 14–15.

---

## Task 9: Show how long undo lasts
**Problem:** users don't know undo expires.
**Done when:** the file tree's undo button reads "undo (kept 30 days)", using the configured number.
**Touch point:** `frontend/components.jsx`.
**Priority:** Low | **Effort:** <1h | **Blockers:** ready 01
**Out of scope:** a trash view.
**Acceptance test:** check it in the browser after deleting a file.

---

## Part D — After launch

## Task 10: Record each sign-in
**Problem:** there's no way to tell an account is inactive.
**Done when:** each sign-in saves the time and cancels any pending inactivity warning.
**Touch point:** one new migration; `harness_module/api.py` (sign-in).
**Priority:** Low | **Effort:** <2h | **Blockers:** none
**Out of scope:** the sweep.
**Acceptance test:** Test 16.

---

## Task 11: Delete inactive accounts
**Problem:** unused accounts keep their data and connected-account access forever.
**Done when:** once switched on, accounts unused for 12 months get one warning email and are deleted 30 days later unless the user signs in. Off by default.
**Touch point:** `harness_module/retention.py`; `config_module/config.yaml`.
**Priority:** Low | **Effort:** 1 day | **Blockers:** Tasks 6, 10; an email sender (D16)
**Out of scope:** reminder emails.
**Acceptance test:** Tests 17–18.

---

# Tests
Composio, Supabase's admin API and stopping conversations are mocked; storage uses the local-disk backend.

| # | Test | Proves |
|---|---|---|
| 1 | `test_delete_blob_twice_is_harmless` | Deleting a blob twice doesn't fail |
| 2 | `test_shared_hash_is_kept` | A blob another user still needs survives |
| 3 | `test_every_hash_column_is_checked` | Only files, undo and memory point at blobs; a new table that does fails this test |
| 4 | `test_deleting_a_user_leaves_nothing` | No table keeps any of a deleted user's data |
| 5 | `test_no_deletion_chain_fk_is_no_action` | No database link can block a delete again |
| 6 | `test_deleting_a_project_detaches_its_sessions` | A project delete keeps its conversations and files |
| 7 | `test_delete_session_takes_its_transcript` | A conversation's messages and approvals go with it |
| 8 | `test_running_session_cannot_be_deleted` | A running conversation is refused |
| 9 | `test_delete_project_refuses_while_running` | A project with a running conversation is refused |
| 10 | `test_delete_account_removes_everything` | Records, unused blobs and connections are all gone |
| 11 | `test_delete_account_stops_when_composio_refuses` | A failed revoke stops the delete with nothing changed |
| 12 | `test_delete_account_twice_is_harmless` | Retrying a delete is safe |
| 13 | `test_delete_me_requires_matching_email` | A wrong email deletes nothing |
| 14 | `test_purge_keeps_recent_undo` | Undo entries under 30 days old survive |
| 15 | `test_purge_frees_only_unreferenced_bytes` | Expired undo doesn't remove a blob still in use |
| 16 | `test_sign_in_stamps_and_clears_notice` | Signing in resets the inactivity clock |
| 17 | `test_sweep_notices_then_deletes` | One warning, then deletion; anyone who signs in is spared |
| 18 | `test_sweep_is_off_by_default` | Nothing happens until it's switched on |

---

# Open Questions
Checks for whoever builds this. None needs a decision.
1. Task 1: what Supabase Storage returns when deleting a file that's already gone.
2. Task 6: Supabase's admin endpoint and key for deleting a user.
3. Tasks 4 and 6: whether stopping a conversation also frees its sandbox. If the sandbox moves off e2b (branch `docs/e2b-local-migration-spec`), use that branch's release call.

---

# Implementation Notes
