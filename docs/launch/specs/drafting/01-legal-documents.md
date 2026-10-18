# // Feature Spec: Legal Documents (VBAKER)

*Drafting 01 · start once ready 01's page shell exists. Write every legal page now, with placeholders where a decision is open. Publishing waits for the decisions and a lawyer's review. Ready 09 blocks launch while placeholders or open decisions remain.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §0.2 (placeholders), §4 (what we store and who gets it), §7.1–7.10 (outlines), §10 (checked facts)
- Decisions: `docs/launch/DECISIONS.md` (D1, D2, D3, D6, D7, D8, D9, D10, D16, D17)
- Facts to disclose: `config_module/config.yaml`, `frontend/index.html`, `frontend/app.jsx` and `frontend/lookingglass.jsx` (local storage), `LICENSE`, `NOTICE`
- Privacy Policy tone: opengrades.mit.edu/privacy

**Status:** Drafting | **Author:** Vin Baker | **Last updated:** 2026-10-14

---

# Problem
arkos stores accounts, conversations, files, the agent's memory and content from connected accounts, and shares them with the outside services in plan §4.2. No document says so. No contract limits liability for an agent that can send email and merge code as the user. Without these pages, the agreement screen (ready 05) records agreement to nothing. Success: every page in plan §7.1–7.10 exists as a draft ready for a lawyer (the Privacy Policy in a short form), every fact in it is checked, and every open decision is marked.

---

# Technical Background
What the pages must disclose, as configured today:
- The AI model is OpenAI's gpt-4.1-mini, unless D3 moves to a self-hosted model.
- Files and accounts live in Supabase; connected accounts (Gmail, Outlook, GitHub, Linear, Notion, Google Calendar, Google Drive) go through Composio; web search goes through SerpAPI; the agent's sandbox runs on e2b; its browser runs on Browserless. Which of these are self-hosted in production isn't recorded yet (Open Question 2).
- arkos's own log (`system_events`) drops rows older than 30 days each time the server starts (`system_log.retain_days`), not on a timer. Logs kept by the hosting provider and Supabase aren't pruned by arkos. The list of actions that always need approval is in config.
- Until ready 07 lands, every page load also contacts unpkg and Google Fonts.
- The browser keeps one cookie, `arkos_session`. The Supabase client stores no sign-in token (`persistSession: false` in `frontend/api.jsx`). Local storage holds `arkos-theme` and `arkos-canvas`. Ready 06 and ready 10 add dismissals for the beta banner, the connect explainer and the cookie notice.
- The browser library's analytics (PostHog) are switched off by ready 10 Task 1. Until that merges, they would have to be disclosed.

---

# Proposed Approach
Four pull requests, one per group of pages. Each page uses ready 01's shell. The Terms and Privacy Policy carry the version from config in `<meta name="arkos-doc-version">` (plan §5.2); the other pages carry a "last updated" date. Each opens with a plain "short version" box that says it isn't the binding text. The Privacy Policy is the exception: it is already the short version, one page in the style of opengrades.mit.edu/privacy, so it gets no box and doesn't follow plan §7.2's seventeen headings.

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
**Problem:** nothing says what's collected, who gets it, or for how long, and the landing page (ready 10) links here.
**Done when:** the text describes arkos as it will be at launch, once every ready spec has merged.
1. `privacy.html` is one plain-language page with these sections in order, each with a fixed anchor for help pages to link to: What we collect (`#collect`) · How we use it (`#use`) · AI features (`#ai`) · Who sees your data (`#who-sees`) · Sharing (`#sharing`) · Keeping and deleting data (`#retention`) · Your rights (`#rights`) · Contact (`#contact`).
2. **What we collect** covers every category in plan §4.1. **How we use it:** to run the service, keep it safe and prevent abuse, and nothing else unless D8 allows research use (tagged D8). **AI features:** what reaches the model (messages, files, web pages and connected-account content the agent reads); the provider `{{LLM_PROVIDER}}` and its retention terms from plan §4.2 (D3); the default that user data isn't used to train models (D8); Google's "Limited Use" statement. **Who sees your data** covers human access (D9). **Sharing:** no sale and no ads; the providers on the subprocessors page; legal requirements; a business transfer; and anything the user directs the agent to send.
3. **Keeping and deleting data** gives plan §4.3's periods (D7), worded to match the code: arkos's own logs and deleted files go at the first server start after 30 days; the hosting provider's and Supabase's logs follow their own terms, as recorded in §4.2; consent records are kept after an account is deleted, as proof of agreement; disconnecting an account asks the provider to revoke arkos's access, and users can also revoke it in the provider's settings.
4. **Your rights:** see and export your data, correct it, delete conversations, projects or the whole account, disconnect accounts, and withdraw consent by deleting the account. Requests by email are answered within 30 days, from the account's own address. **Contact:** `{{PRIVACY_EMAIL}}` (D16), `{{OPERATOR_LEGAL_NAME}}` and `{{OPERATOR_POSTAL_ADDRESS}}` (D1).
5. It closes with one-line notes:
   - Age: 18 or older (D5).
   - Where data is processed: the regions recorded in §4.2 (Open Question 1).
   - Security: the measures in plan §7.2 item 11, and that no system is perfectly secure.
   - EEA and UK: legal bases and the right to complain.
   - California: no sale or sharing.
   - Students: arkos isn't an MIT education-records system (D2).
   - Cookies: a link to the cookies page.
   - Changes: each version is dated, and the agreement screen (ready 05) asks again after a change.
6. The policy is under 1,500 words, counting the text inside `<main>` without HTML comments.
7. The subprocessors page lists every outside provider in plan §4.2 that is in use at launch. It leaves out self-hosted services and the rows §4.2 marks as removed.
8. The cookies page is one table (name · essential or preference · what it's for · how long) listing `arkos_session` and every `arkos-…` local storage key the frontend sets. It also lists the font and script services until ready 07 merges, and ends: "arkos uses no analytics or advertising cookies. If that changes, we'll ask first."

**Touch point:** `frontend/legal/privacy.html`, `subprocessors.html`, `cookies.html`; `docs/launch/LAUNCH_READINESS_PLAN.md` §4.2 and §10 (record the facts first).
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 01 Tasks 1–2. Publishing also waits for ready 02, 03, 05, 06 and 10, whose features the text describes.
**Out of scope:** answering any decision; making consent records expire; making the disconnect button stop when a revoke fails.
**Acceptance test:** Tests 1, 2, 5 and 6.

---

## Task 2: Terms of Service and Acceptable Use
**Problem:** no agreement says who's responsible for approved actions, or disclaims mistakes by the AI.
**Done when:** the Terms have every §7.1 section in order, with the warranty and liability sections in bold capitals; eligibility says 18 or older, a confirmed email and one account per person, open to anyone; the Acceptable Use Policy lists every §7.3 item, including what not to upload and MIT's academic integrity rules; D1, D2, D6, D8, D10 and D16 are tagged.
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
**Done when:** the copyright page has the six parts a takedown notice needs, a repeat-infringer policy, and `{{LEGAL_EMAIL}}` (D16) and `{{OPERATOR_LEGAL_NAME}}` (D1) as the contact; the security page and a new `SECURITY.md` share one "how to report" section with `{{SECURITY_EMAIL}}` (D16); the accessibility page lists known gaps; the licenses page has the AGPL notice, the `NOTICE` text, every outside script and font, and a table of Python package licenses; the changelog has its first entry.
**Touch point:** `frontend/legal/copyright.html`, `security.html`, `accessibility.html`, `licenses.html`, `changelog.html`; new `SECURITY.md`.
**Priority:** Medium | **Effort:** 1 day | **Blockers:** ready 01 Task 1
**Out of scope:** registering a copyright agent (it's on the launch checklist in drafting 02).
**Acceptance test:** Test 4.

---

# Tests
In `tests/test_legal_pages.py`, reading the pages with Python's built-in HTML parser.

| # | Test | Proves |
|---|---|---|
| 1 | `test_legal_page_versions_match_config` | Each written page's `arkos-doc-version` matches config (Privacy from Task 1, Terms from Task 2) |
| 2 | `test_placeholders_and_decisions_are_known` | Every placeholder and decision tag exists in the plan and `DECISIONS.md` |
| 3 | `test_destructive_list_matches_disclosure` | Every always-ask action in config is disclosed |
| 4 | `test_licenses_page_names_every_frontend_asset` | Every outside script and font has its license listed |
| 5 | `test_privacy_policy_is_short_and_complete` | The policy has the eight anchored sections in order and stays under 1,500 words inside `<main>` |
| 6 | `test_cookies_page_lists_every_storage_key` | Every `"arkos-…"` string literal in `frontend/` (outside `vendor/`) appears in the cookies table |

---

# Open Questions
Checks for whoever builds this. None needs a decision.
1. Task 1: for each provider in plan §4.2, record where it processes data. For Supabase, also record the backup period. Do this before Task 1 merges.
2. Task 1: record how production actually runs, since `docker-compose.yml` looks out of date. Check where `DB_URL` points, whether Supabase is hosted, whether Browserless is self-hosted or cloud, and who the hosting provider is. Record all of it in plan §4.2.

---

# Implementation Notes
- 2026-10-14: Task 1 drafted: `privacy.html` (917 words), `subprocessors.html`, `cookies.html`. Provider regions and Browserless hosting show as "Draft —" cells until Open Questions 1–2 are answered; ready 09 flags them. The D3 self-hosted wording sits in an HTML comment beside the AI paragraph.
