# // Feature Spec: Legal Documents (NMORGAN)

*Drafting 01 · start once ready 01's page shell exists. Write every legal page now, with placeholders where a decision is open. Publishing waits for the decisions and a lawyer's review, and ready 09 enforces that.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §0.2 (placeholders), §4 (what we store and who gets it), §7.1–7.10 (outlines), §10 (checked facts)
- Decisions: `docs/launch/DECISIONS.md` (D1, D2, D3, D6, D7, D8, D9, D10, D16, D17)
- Facts to disclose: `config_module/config.yaml`, `frontend/index.html`, `LICENSE`, `NOTICE`

**Status:** Drafting | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
arkos stores accounts, conversations, files, the agent's memory and content from connected accounts, and shares them with six outside services. No document says so. No contract limits liability for an agent that can send email and merge code as the user. Without these pages, the agreement screen (ready 05) records agreement to nothing. Success: every page in plan §7.1–7.10 exists as a draft ready for a lawyer, every fact in it is checked, and every open decision is marked.

---

# Technical Background
What the pages must disclose, as configured today:
- The AI model is OpenAI's gpt-4.1-mini, unless D3 moves to a self-hosted model.
- Files and accounts live in Supabase; connected accounts (Gmail, Outlook, GitHub, Linear, Notion, Google Calendar, Google Drive) go through Composio; web search goes through SerpAPI; the agent's sandbox runs on e2b; its browser runs on Browserless.
- Server logs are kept 30 days. The list of actions that always need approval is in config.
- Until ready 07 lands, every page load also contacts unpkg and Google Fonts.

---

# Proposed Approach
Four pull requests, one per group of pages. Each page uses ready 01's shell, carries its version number, and opens with a plain "short version" box that says it isn't the binding text.

**Open decisions in the text:**
- A missing value (a name, an age, an email) is a placeholder from plan §0.2, like `{{OPERATOR_LEGAL_NAME}}`.
- A choice between two paragraphs is written both ways: the default shows, the other sits in a hidden comment.
- Every such spot is tagged with its decision number (`data-decision="D3"`), so resolving a decision means searching for its tag.

Every factual sentence must trace to plan §4 or §10. A new fact gets checked in the code and added to §10 first.

**Done means drafted, not published.** Final text needs the decisions answered and a lawyer's review (D17).

**Not in scope:** help pages (ready 08), in-app warnings (ready 06), the legal review itself.

---

# Implementation Plan

## Task 1: Privacy Policy, subprocessors, cookies
**Problem:** nothing says what's collected, who gets it, or for how long.
**Done when:** the Privacy Policy covers every heading in plan §7.2; the subprocessors page lists each provider in §4.2; the cookies page lists the login cookie, Supabase's sign-in storage, the theme and banner settings, and the font and script services (unless ready 07 has merged); Google's required "Limited Use" statement is included; the retention section says consent records and inactive accounts are kept; D3, D7, D8 and D9 are tagged.
**Touch point:** `frontend/legal/privacy.html`, `subprocessors.html`, `cookies.html`.
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 01 Task 1
**Out of scope:** answering any decision.
**Acceptance test:** Tests 1–2.

---

## Task 2: Terms of Service and Acceptable Use
**Problem:** no agreement says who's responsible for approved actions, or disclaims mistakes by the AI.
**Done when:** the Terms have every §7.1 section in order, with the warranty and liability sections in bold capitals; eligibility says 18 or older, a confirmed email and one account per person, open to anyone; the Acceptable Use Policy lists every §7.3 item, including what not to upload and MIT's academic integrity rules; D1, D2, D6, D8 and D10 are tagged.
**Touch point:** `frontend/legal/terms.html`, `acceptable-use.html`.
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 01 Task 1
**Out of scope:** choosing courts or arbitration (D10).
**Acceptance test:** Tests 1–2.

---

## Task 3: How arkos acts on your behalf
**Problem:** the approval system and its limits aren't explained anywhere, and they're central to who is responsible for what.
**Done when:** the page covers plan §7.4; every always-ask action is listed by a readable name; the session limits match config (6 steps / 5 minutes normally, 15 steps / 30 minutes on autopilot).
**Touch point:** `frontend/legal/ai-and-agents.html`.
**Priority:** High | **Effort:** <4h | **Blockers:** ready 01 Task 1
**Out of scope:** generating the page from config.
**Acceptance test:** Test 3.

---

## Task 4: Supporting pages
**Problem:** there's no copyright takedown process, vulnerability policy, accessibility statement or open-source notice.
**Done when:** the copyright page has the six parts a takedown notice needs, a repeat-infringer policy, and `{{LEGAL_EMAIL}}` and `{{OPERATOR_LEGAL_NAME}}` as the contact; the security page and a new `SECURITY.md` share one "how to report" section with `{{SECURITY_EMAIL}}`; the accessibility page lists known gaps; the licenses page has the AGPL notice, the `NOTICE` text, every outside script and font, and a table of Python package licenses; the changelog has its first entry.
**Touch point:** `frontend/legal/copyright.html`, `security.html`, `accessibility.html`, `licenses.html`, `changelog.html`; new `SECURITY.md`.
**Priority:** Medium | **Effort:** 1 day | **Blockers:** ready 01 Task 1
**Out of scope:** registering a copyright agent (it's on the launch checklist in drafting 02).
**Acceptance test:** Test 4.

---

# Tests
In `tests/test_legal_pages.py`, reading the pages with Python's built-in HTML parser.

| # | Test | Proves |
|---|---|---|
| 1 | `test_legal_page_versions_match_config` | The version users accept is the text they read |
| 2 | `test_placeholders_and_decisions_are_known` | Every placeholder and decision tag exists in the plan and `DECISIONS.md` |
| 3 | `test_destructive_list_matches_disclosure` | Every always-ask action in config is disclosed |
| 4 | `test_licenses_page_names_every_frontend_asset` | Every outside script and font has its license listed |

---

# Open Questions
Checks for whoever builds this. None needs a decision.
1. Task 1: read the Supabase project's region and backup period from its dashboard, and add both to plan §4 before Task 1 merges.

---

# Implementation Notes
