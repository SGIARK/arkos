# // Feature Spec: Data Export (NMORGAN)

*Ready 03 · Wave 1 · no blockers. One button that downloads everything arkos holds about you as a ZIP.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §4.1 (what we store)
- Storage: `harness_module/store.py`, `harness_module/blobs.py`, `harness_module/memory.py`
- Connections: `tool_module/connections.py`
- Server and screen: `harness_module/api.py`, `frontend/views.jsx`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
Users can't get their data out. The Privacy Policy promises they can, the Terms tell them to keep backups, and account deletion tells them to download first. None of that is true until this exists. Success: one download contains everything arkos keeps about the user, and no credentials.

---

# Technical Background
- Everything is already stored per user: files, conversations and their messages, approvals, projects, the agent's memory, and connected accounts.
- One stored field is off limits: the Composio account id kept with each connection is effectively a key to that account.
- The server already limits sign-in attempts per address with a small in-memory counter. The export rate limit copies that pattern, per user.

---

# Proposed Approach
Build the ZIP into a temporary file that spills to disk past 64 MB, then send it. That keeps memory use flat without a new library. The account module is shared with ready 02; whichever lands first creates it.

| In the ZIP | What it is |
|---|---|
| `account.json` | email, name, sign-up date, consent records (if that table exists yet) |
| `projects.json` | projects and their linked folders |
| `sessions.json`, `sessions/<id>.jsonl` | every conversation and its messages, oldest first |
| `approvals.json` | every approval asked and answered |
| `memory/` | the agent's memory and notes |
| `files/` | every current file |
| `connections.json` | which accounts are connected and when; never the account ids |
| `README.txt` | what each file is, and that server logs aren't included |

**Defaults:** server logs (kept 30 days) are left out; exports are capped at 2 GB (`quotas.export_max_mb`).

**Not in scope:** emailed exports; files sitting in undo.

---

# Implementation Plan

## Task 1: Build the archive
**Problem:** nothing gathers one user's data in one place.
**Done when:** for a test user with projects, conversations, memory, files and a connection, the archive has every section above with the right counts, contains no account ids, and an export over the cap fails with a clear error.
**Touch point:** `harness_module/account.py` (new, or shared with ready 02); `config_module/config.yaml` (export cap).
**Priority:** High | **Effort:** 1 day | **Blockers:** none
**Out of scope:** the download itself.
**Acceptance test:** Tests 1–2.

---

## Task 2: The download button
**Problem:** users can't ask for it.
**Done when:** `GET /auth/me/export` downloads `arkos-export-<date>.zip`; asking again within 10 minutes is refused; an oversize export says to contact support; Settings has "download my data".
**Touch point:** `harness_module/api.py` (new endpoint and limit); `frontend/views.jsx` (Settings); `frontend/index.html` (bump `?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 1
**Out of scope:** a progress bar.
**Acceptance test:** Test 3; download and open the ZIP in the browser.

---

# Tests
Storage uses the local-disk backend.

| # | Test | Proves |
|---|---|---|
| 1 | `test_export_contains_every_section` | Every section is present with the right counts |
| 2 | `test_export_carries_no_credentials` | No account id or session cookie ends up in the ZIP |
| 3 | `test_export_is_rate_limited` | A second export within 10 minutes is refused; another user isn't affected |

---

# Open Questions
None.

---

# Implementation Notes
