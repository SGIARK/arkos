# // Feature Spec: Deletion (VBAKER)

*Ready 02 · How arkos deletes things, and the work to get there. Tasks 1–5 before launch, Task 6 after.*

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
Today users can only delete files, and even those stay in storage forever. Conversations, projects and accounts can't be deleted, and the database would refuse an account or project delete. Disconnecting an account can leave arkos's access to it switched on. Until this is fixed, the Privacy Policy can't promise deletion. Success: users can delete each kind of thing, and a test proves nothing is left behind.

---

# Technical Background
What's true today:
- Three database links block deletion: conversation → user, sandbox → user, conversation → project.
- Blobs can be written but never deleted. Files, the undo list and the agent's memory all point at them.
- Identical files share one blob, even between users, so a blob can only be removed once nothing points at it.
- The "disconnect" button ignores a failed revoke at Composio and forgets the connection anyway.

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

Kept on purpose: consent records and 30 days of server logs.

## Rules every delete follows
1. Delete one parent record and let cascades remove the rest.
2. Revoke outside access (Composio) first. If that fails, stop and change nothing.
3. Remove blobs last, and only when nothing else points at them.
4. Every step is safe to run twice.
5. Only files have undo. Everything else warns before the click.

**Defaults:** 30-day undo (`retention.deleted_files_days`; D7 is still open). Deleting a project keeps its conversations.

**Not in scope:** undo for accounts, projects or conversations; admin or bulk deletes; a trash view; a scheduler; confirmation or reminder emails.

---

# Implementation Plan

| # | Task | Effort | Needs |
|---|---|---|---|
| 1 | Delete unused blobs | ½ day | — |
| 2 | Fix the three database links | ½ day | — |
| 3 | Delete conversations and projects | 1 day | Task 2 (projects only) |
| 4 | Delete an account | 1½ days | Tasks 1, 2 |
| 5 | Expire undo after 30 days | ½ day | Task 1; ready 01 |
| 6 | Delete inactive accounts (after launch) | 1 day | Task 4; an email sender (D16) |

## Task 1: Delete unused blobs
**Problem:** blobs can't be deleted, and a shared one mustn't be.
**Done when:** storage can delete a blob (a missing one isn't an error), and `delete_unreferenced` removes only blobs that no file, undo entry or memory note points at.
**Touch point:** `harness_module/blobs.py` (delete in both storage backends); `harness_module/store.py` (`delete_unreferenced`).
**Priority:** High | **Effort:** ½ day | **Blockers:** none
**Out of scope:** scanning all of storage for strays.
**Acceptance test:** Tests 1–3.

---

## Task 2: Fix the three database links
**Problem:** deleting a user or a project fails.
**Done when:** deleting a user removes all their data; deleting a project keeps its conversations (unlinked) and its files.
**Touch point:** one new migration in `db/migrations/` (next free number).
**Priority:** High | **Effort:** ½ day | **Blockers:** none
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

## Task 3: Delete conversations and projects
**Problem:** neither can be removed.
**Done when:** the owner can delete either with a confirm click, and anyone else gets "not found"; anything still running must be stopped first; a project's confirm says "its conversations and files stay", and they do.
**Touch point:** `harness_module/api.py` (`DELETE /sessions/{id}`, `DELETE /projects/{id}`); `frontend/lookingglass.jsx` (the two buttons).
**Priority:** High | **Effort:** 1 day | **Blockers:** Task 2 (project delete only)
**Out of scope:** bulk delete.
**Acceptance test:** Tests 7–9.

---

## Task 4: Delete an account
**Problem:** an account spans the database, file storage, Composio and Supabase, and a half-finished delete must be finishable.
**Done when:** in Settings, the user types their email to confirm (a wrong email does nothing). arkos then:
1. Signs them out everywhere.
2. Stops their running conversations.
3. Revokes every Composio connection, including any arkos lost track of. If one fails, it stops.
4. Notes which blobs they used.
5. Deletes their Supabase account, which removes all their records.
6. Removes the blobs nobody else uses.

The user lands on sign-in with "your account has been deleted". Running it again finishes the job or does nothing.
**Touch point:** new `harness_module/account.py`; `tool_module/composio_mcp.py` (`revoke_all`, which reports failures); `harness_module/api.py` (`DELETE /auth/me`); `frontend/views.jsx` (Settings).
**Priority:** High | **Effort:** 1½ days | **Blockers:** Tasks 1, 2
**Out of scope:** a confirmation email.
**Acceptance test:** Tests 10–13; in the browser, delete a test account, then sign up again with the same email.

---

## Task 5: Expire undo after 30 days
**Problem:** deleted files are kept forever, and users don't know undo expires.
**Done when:** at server start, undo entries older than 30 days are removed with their saved project links and unused blobs; newer ones are untouched; the undo button reads "undo (kept 30 days)".
**Touch point:** new `harness_module/retention.py`; `harness_module/api.py` (run it at startup); `frontend/components.jsx` (the label).
**Priority:** High | **Effort:** ½ day | **Blockers:** Task 1; ready 01 (the setting)
**Out of scope:** a trash view.
**Acceptance test:** Tests 14–15.

---

## Task 6: Delete inactive accounts (after launch)
**Problem:** unused accounts keep their data and connected-account access forever.
**Done when:** each sign-in records its time; once switched on, accounts unused for 12 months get one warning email and are deleted 30 days later unless the user signs in. Off by default.
**Touch point:** one new migration; `harness_module/api.py` (sign-in); `harness_module/retention.py`; `config_module/config.yaml`.
**Priority:** Low | **Effort:** 1 day | **Blockers:** Task 4; an email sender (D16)
**Out of scope:** reminder emails.
**Acceptance test:** Tests 16–18.

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
2. Task 4: Supabase's admin endpoint and key for deleting a user.
3. Tasks 3 and 4: whether stopping a conversation also frees its sandbox. If the sandbox moves off e2b (branch `docs/e2b-local-migration-spec`), use that branch's release call.

---

# Implementation Notes
