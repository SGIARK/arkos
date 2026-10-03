# // Feature Spec: Help Center (NMORGAN)

*Ready 09 · Wave 3 · built after ready 03, 05 and 06, because it describes their flows. Eleven static guide pages, written only from verified facts.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §7.11 (page list), §10 (verified facts)
- `frontend/app.jsx` (`NAV`), `frontend/views.jsx` (`DeskView`, `ApprovalsView`, `MailSent`), `frontend/lookingglass.jsx` (`LookingGlassView`, `SessionTools`), `frontend/components.jsx` (`Empty`, `StreamEvent`)
- `config_module/config.yaml` (`quotas`, `budgets`, `sandbox`, `memory`, `llm.max_tools`, `mcp_servers`)
- Depends on: ready 02 (page shell)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
The only guidance in the product is the three-line pitch in `AuthAside` and empty states like "no projects yet". New users have to find projects, sessions, autopilot, approvals, per-session tool toggles, folder claims and the 128-tool cap by trial and error. Some of that, like a connector silently benched or a session parked on an approval, looks like a bug. Success: each §7.11 page exists, every factual sentence traces to §10 or to code the author opened, and the product links to it where people get stuck.

---

# Technical Background
`NAV = ["desk", "approvals", "files", "projects"]`, with settings as a modal. A project links folders (`project_folders`); a session claims folders (`session_claims`), and inside the sandbox they appear at `~/store/<folder>/`, the only durable path. Tool toggles are per session (`SessionTools`). Past `llm.max_tools` (128), whole servers are benched most-recently-enabled first, and a `status` event says so. Resend and reset emails wait 60 s (`RESEND_WAIT`). §10 lists the rest with their sources.

---

# Proposed Approach
Plain HTML in `frontend/help/`, sharing `legal/legal.css`, each page ending "Still stuck? {{SUPPORT_EMAIL}}". If a fact isn't in §10, verify it in code and add it to §10 in the same PR. Two PRs by audience: first-hour pages, then reference. Questions that touch policy (for example "does arkos train on my data?") get one sentence that links the matching Privacy Policy section rather than answering, so this spec never waits on D8 or any other decision.

**Not in scope:** screenshots (they rot with every UI change), tours, search.

---

# Implementation Plan

## Task 1: First-hour pages
**Problem:** new users have no walkthrough.
**Done when:** `index.html`, `getting-started.html`, `projects-and-sessions.html`, `approvals-and-autopilot.html` and `connecting-accounts.html` cover their §7.11 rows; the seven connectors are named exactly as `mcp_servers.*.name`; the benching behaviour and the 128 cap are explained on the connections page under `#tool-limit`.
**Touch point:** `frontend/help/{index,getting-started,projects-and-sessions,approvals-and-autopilot,connecting-accounts}.html`.
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 02
**Out of scope:** reference pages.
**Acceptance test:** `test_help_pages_exist_and_link_up`.

---

## Task 2: Reference and troubleshooting pages
**Problem:** limits, data handling and failure modes are undocumented.
**Done when:** `files-and-memory.html`, `limits.html`, `your-data.html`, `responsible-use.html`, `faq.html` (10–15 Q&As) and `troubleshooting.html` cover their §7.11 rows; `limits.html` marks each number `data-config="<dotted.key>"`; `your-data.html` describes export (ready 03), session delete and undo (ready 06), and account delete (ready 05) as built, and links the Privacy Policy for everything else.
**Touch point:** `frontend/help/{files-and-memory,limits,your-data,responsible-use,faq,troubleshooting}.html`.
**Priority:** Medium | **Effort:** 1 day | **Blockers:** Task 1; ready 03, 05, 06 merged
**Out of scope:** the Privacy Policy itself.
**Acceptance test:** `test_limits_page_matches_config`.

---

## Task 3: Link from where people get stuck
**Problem:** a guide nobody finds doesn't help.
**Done when:** `DeskView`'s "no projects yet" and `LookingGlassView`'s empty list add "new here? start with the guide"; `ApprovalsView`'s empty state links `approvals-and-autopilot.html`; `MailSent` links the troubleshooting anchor for missing email; a benching `status` line in `StreamEvent` links `connecting-accounts.html#tool-limit`.
**Touch point:** `frontend/views.jsx` (`DeskView`, `ApprovalsView`, `MailSent`), `frontend/lookingglass.jsx` (`LookingGlassView`), `frontend/components.jsx` (`StreamEvent`), `frontend/index.html` (`?v=`).
**Priority:** Medium | **Effort:** <2h | **Blockers:** Tasks 1–2
**Out of scope:** changing the benching event payload.
**Acceptance test:** in the browser pane, each link opens its page at the right anchor.

---

# Tests
In `tests/test_legal_pages.py`, shared with drafting 01.

## Test 1: `test_help_pages_exist_and_link_up`
**What it verifies:** every §7.11 page exists and links `help/index.html`, and every relative link under `frontend/help/` resolves to a file.
**Why this matters:** a broken help link is the worst kind.

## Test 2: `test_limits_page_matches_config`
**What it verifies:** each `data-config` element's text equals that config value.
**Why this matters:** limits change in config, and the page must fail CI rather than drift.

---

# Open Questions
None. `{{SUPPORT_EMAIL}}` is D16, and policy questions link out to the Privacy Policy.

---

# Implementation Notes
