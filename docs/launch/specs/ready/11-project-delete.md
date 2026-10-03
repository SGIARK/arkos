# // Feature Spec: Project Delete (NMORGAN)

*Ready 11 · Wave 1 · no blockers. A user can delete a project: its folder links go, its files and its conversations stay.*

---

**Sources**
- `db/migrations/0000_migration_0.sql` (`sessions.project_id REFERENCES projects(id)`, NO ACTION), `0015_user_store.sql` ("deleting a project deletes its links and nothing else"), `0016_delete_undo.sql` (`deleted_links`)
- `harness_module/api.py` (`_owned_project`, `rename_project`, `list_projects`)
- `frontend/lookingglass.jsx` (`LookingGlassView`: the project card's rename button `pc-rename-btn`), `frontend/api.jsx` (`renameProject`)
- `tests/test_store_schema.py` (`test_deleting_a_project_deletes_its_links_and_no_files`)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-02

---

# Problem
A user can't delete a project. There's no `DELETE /projects/{id}`, and even a direct delete fails once the project has a session, because `sessions.project_id` is NO ACTION. Projects pile up in the grid forever. Success: the owner can delete a project; its folder links go, its files stay (they belong to the user, not the project), and its sessions stay, detached.

---

# Technical Background
`0015` split files from projects: a project links folders through `project_folders` (`ON DELETE CASCADE`) and owns no files. `test_deleting_a_project_deletes_its_links_and_no_files` already pins that rule with a direct SQL delete; it passes only because that test's project has no sessions. `deleted_links.project_id` cascades too. `_owned_project(project_id, user_id)` is the ownership check. The home session has `project_id IS NULL`, so it's unaffected.

---

# Proposed Approach
Detach, not delete: deleting a project already deletes its links and nothing else, and conversations are the user's to delete one at a time (ready 06). Change `sessions.project_id` to `ON DELETE SET NULL`, so the database does it and every path agrees; add the endpoint and a two-click delete on the project card. Detached sessions show in the account-wide lists and simply have no project. To make a project delete take its sessions instead, use `ON DELETE CASCADE` in Task 1 and change the confirm copy in Task 2.

**Not in scope:** bulk delete; undo.

---

# Implementation Plan

## Task 1: Let a project go
**Problem:** a project with sessions can't be deleted.
**Done when:** deleting a project that has two sessions and a folder link succeeds; the sessions remain with `project_id` null; the link is gone; the files remain.
**Touch point:** new `db/migrations/NNNN_project_delete_detaches.sql`; `tests/test_store_schema.py`.
**Priority:** Medium | **Effort:** <2h | **Blockers:** none
**Out of scope:** the endpoint.
**Acceptance test:** `test_deleting_a_project_detaches_its_sessions`.

```sql
ALTER TABLE sessions DROP CONSTRAINT IF EXISTS sessions_project_id_fkey;
ALTER TABLE sessions ADD CONSTRAINT sessions_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE SET NULL;
```

---

## Task 2: Endpoint and UI
**Problem:** users can't trigger it.
**Done when:** `DELETE /projects/{project_id}` returns 204 for the owner, 404 for anyone else, and `409 running` while any of its sessions is `running`; the project card has a delete control beside rename, with a two-click confirm reading "delete project — its conversations and files stay".
**Touch point:** `harness_module/api.py` (new `delete_project`, using `_owned_project`); `frontend/api.jsx` (new `api.deleteProject`); `frontend/lookingglass.jsx` (`LookingGlassView` project card); `frontend/index.html` (`?v=`).
**Priority:** Medium | **Effort:** <4h | **Blockers:** Task 1
**Out of scope:** moving sessions into another project.
**Acceptance test:** `test_delete_project_refuses_while_running`; in the browser pane, delete a project and find its session under desk.

---

# Tests

## Test 1: `test_deleting_a_project_detaches_its_sessions`
**What it verifies:** after the delete, both sessions exist with null `project_id`, `project_folders` is empty for the project, and `files` is unchanged.
**Why this matters:** the promise is that nothing but the project goes.

## Test 2: `test_delete_project_refuses_while_running`
**What it verifies:** a project with a `running` session returns 409 and survives.
**Why this matters:** detaching under a live turn changes which folders its claims resolve against.

---

# Open Questions
None.

---

# Implementation Notes
