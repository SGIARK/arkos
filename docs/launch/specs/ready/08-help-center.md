# // Feature Spec: Help Center (VBAKER)

*Ready 08 · Wave 3 · after ready 02 and 03, because it describes deletion and export. Eleven short guide pages, written only from facts checked against the code.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §7.11 (page list), §10 (checked facts)
- Settings the pages quote: `config_module/config.yaml`
- Screens to link from: `frontend/views.jsx`, `frontend/lookingglass.jsx`, `frontend/components.jsx`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
The only guidance in arkos is the sign-in pitch and a few empty-screen lines like "no projects yet". New users have to discover projects, conversations, autopilot, approvals, per-conversation tool switches and the tool limit by trial and error. Some of it looks like a bug, such as a connected account switched off automatically, or a conversation waiting on an approval. Success: each guide page exists, every fact on it is checked, and the app links to it where people get stuck.

---

# Technical Background
Things the pages must explain, all checked in the code:
- Projects link folders of files; conversations work inside them. Inside the agent's sandbox, folders appear at `~/store/<folder>/`, and only that path is kept.
- Tools for connected accounts are switched on per conversation. Past 128 tools in total, arkos switches whole accounts off (most recently enabled first) and says so in the conversation.
- Sign-up and reset emails can only be resent once every 60 seconds.
- Plan §10 lists the rest, with where each fact comes from.

---

# Proposed Approach
Plain HTML pages in `frontend/help/`, using the legal pages' stylesheet. Each ends with "Still stuck? {{SUPPORT_EMAIL}}". A fact not in plan §10 gets checked in the code and added there first. Policy questions (for example, "does arkos train on my data?") link to the matching Privacy Policy section instead of answering, so this spec never waits on a decision.

| Page | Covers |
|---|---|
| Start here, Getting started | What arkos is; sign up, first chat, first project, first approval |
| Projects and conversations | Projects, conversations, autopilot, stopping, deleting |
| Approvals and autopilot | What waits for you, what always asks, declining |
| Connecting accounts | The seven services, per-conversation switches, the tool limit |
| Files and memory | Folders, uploads (25 MB), undo, what the agent remembers |
| Limits | Every number, read from config |
| Your data | Export, deleting things, how long things are kept |
| Responsible use | Academic integrity, what not to upload, checking the agent's work |
| FAQ, Troubleshooting | 10–15 common questions; missing emails, expired links, blocked popups |

**Not in scope:** screenshots (they go stale with every UI change), in-app tours, search.

---

# Implementation Plan

## Task 1: First-hour pages
**Problem:** new users have no walkthrough.
**Done when:** Start here, Getting started, Projects and conversations, Approvals and autopilot, and Connecting accounts cover their rows; the seven services are named as in config; the tool limit is explained under a `#tool-limit` heading.
**Touch point:** five new pages in `frontend/help/`.
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 01
**Out of scope:** the reference pages.
**Acceptance test:** Test 1.

---

## Task 2: Reference and troubleshooting pages
**Problem:** limits, data handling and common failures aren't written down.
**Done when:** the remaining six pages cover their rows; each number on the Limits page is tagged with the config setting it comes from; Your data describes export and deletion as built.
**Touch point:** six new pages in `frontend/help/`.
**Priority:** Medium | **Effort:** 1 day | **Blockers:** Task 1; ready 02 (Tasks 1–5) and ready 03 merged
**Out of scope:** the Privacy Policy itself.
**Acceptance test:** Test 2.

---

## Task 3: Link from where people get stuck
**Problem:** a guide nobody finds doesn't help.
**Done when:** the desk's "no projects yet" and a new empty state on the projects view (it has none today) say "new here? start with the guide"; the empty approvals view links the approvals page; the check-your-email screen links troubleshooting; the "tools switched off" notice in a conversation links the tool-limit section.
**Touch point:** `frontend/views.jsx` (desk, approvals, email screen); `frontend/lookingglass.jsx` (projects view); `frontend/components.jsx` (conversation notices); `frontend/index.html` (bump `?v=`).
**Priority:** Medium | **Effort:** <2h | **Blockers:** Tasks 1–2
**Out of scope:** changing the notices themselves.
**Acceptance test:** in the browser, each link opens the right page and section.

---

# Tests
These live in `tests/test_legal_pages.py`, shared with drafting 01.

| # | Test | Proves |
|---|---|---|
| 1 | `test_help_pages_exist_and_link_up` | Every page exists and every link between pages works |
| 2 | `test_limits_page_matches_config` | The Limits page can't drift from the real settings |

---

# Open Questions
None. The support address is D16; policy questions link to the Privacy Policy.

---

# Implementation Notes
