# // Feature Spec: Landing Page and Cookie Notice (VBAKER)

*Ready 10 · this week · needs ready 01 Tasks 1–2. A static page at `/` that visitors see before signing in, with a privacy summary, contact details and legal links below, plus a one-time cookie notice. It also switches off the browser library's analytics, so "no analytics" is true.*

---

**Sources**
- Server: `harness_module/api.py` (the `/app` mount, `_Frontend`)
- Screens: `frontend/views.jsx` (sign-in pitch), `frontend/api.jsx` (Supabase client), `frontend/app.jsx` and `frontend/lookingglass.jsx` (local storage)
- Browser tool: `tool_module/browser/tool.py`; `requirements.txt` (`browser-use>=0.12.0,<0.13`)
- Approvals: `config_module/config.yaml` (`tools.destructive`), `harness_module/runner.py`, `tool_module/composio_mcp.py`
- Pages: `frontend/legal/`, `frontend/help/` and their stylesheet (from ready 01)
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §4.1, §4.2, §7.6; `docs/launch/DECISIONS.md` (D1, D2, D3, D16)
- References: trybuddynow.com (layout), opengrades.mit.edu/privacy (tone)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-14

---

# Problem
Nothing is served at `/`. Visitors have to know to go to `/app`, where the first thing they see is a sign-in form, with no privacy policy, contact details or cookie notice anywhere. Reviewers asked for a landing page first, with the privacy policy and contact information below it. Separately, the agent's browser tool sends each task's text and the URLs it visits to the browser library's analytics service, so a "no analytics" promise would be false today. Success: `/` explains arkos, links to sign-in, privacy and contact, and tells first-time visitors once what arkos keeps in their browser, and every word of that is true.

---

# Technical Background
What's true today:
- The app is mounted only at `/app` (`api.py`). There is no route for `/`, and files in `frontend/` are reachable only under `/app/…`.
- `_Frontend` sends `no-store` for HTML under `/app`. CSS and JS keep normal caching and rely on a `?v=` suffix to update.
- Signing in is kept by one cookie, `arkos_session` (HttpOnly, 7 days). The Supabase client runs with `persistSession: false`, so no sign-in token is stored in the browser. Local storage holds two settings, `arkos-theme` and `arkos-canvas`. Ready 06 will add banner dismissals.
- The app still loads fonts and scripts from Google Fonts and unpkg until ready 07 lands.
- browser-use reads `ANONYMIZED_TELEMETRY` (default true) each time it's used. The first browser task sets up the analytics, which then send the task text, visited URLs and final result to PostHog's EU servers. It also checks PyPI for a newer version on every run (`BROWSER_USE_VERSION_CHECK`). Nothing in the repo sets either. Cloud sync needs a browser-use login and doesn't run here.
- Connected-account tools ask for approval unless a toolkit sets `auto_approve`, and none does. Autopilot answers those approvals itself, except for the `tools.destructive` list, which covers every send, delete and merge tool in the roster. `browser_task` has no gate; it only tells the model to ask first.
- The sign-in pitch says arkos "works on a real computer and browser". The sandbox is headless (a shell and a filesystem), and the browser is separate.

---

# Proposed Approach
1. **Analytics off.** The browser tool sets `ANONYMIZED_TELEMETRY` and `BROWSER_USE_VERSION_CHECK` to `false`, after its imports and before any browser task runs. This is not overridable, because the public pages promise it. It ships before anything public says "no analytics".
2. **The page.** `frontend/landing/index.html`, plain HTML and CSS that works without JavaScript and loads nothing from other sites. `GET /` returns it directly, so the address stays `/`. Its own CSS, JS and images use absolute `/app/…` paths with `?v=`. Links may also use ready 01's short addresses, `mailto:`, and the source URL.
3. **Top to bottom**, after the trybuddynow.com layout:
   - Header: the arkos mark and "sign in" (`/app/`).
   - Hero: "a digital intern still on the job next week." Under it: "arkos works in its own sandbox and browser and keeps your files between runs. Actions in your connected accounts wait for your approval unless you turn on autopilot, and sending email, deleting and merging always wait for you." Buttons: "get started" and "sign in", both to `/app/`. The sign-in screen's pitch is changed to the same sentence.
   - Three short blocks: approvals for connected accounts, a fresh sandbox per conversation, files that persist.
   - **Your data:** what's stored (account, conversations, files, the agent's notes); the AI model is `{{LLM_PROVIDER}}` (D3); no selling and no ads; "to delete or export your data, email `{{PRIVACY_EMAIL}}`" (D16). Then "Read the privacy policy" (`/privacy`).
   - **Contact:** `{{SUPPORT_EMAIL}}` for help, `{{PRIVACY_EMAIL}}` for privacy, `{{SECURITY_EMAIL}}` for security reports (D16).
   - Footer: Terms · Privacy · Cookies · Acceptable use · Help · Source (AGPL-3.0, the same URL as `legal.source_url`), `{{MIT_RELATIONSHIP_STATEMENT}}` (D2), "© `{{OPERATOR_LEGAL_NAME}}`" (D1).
4. **The cookie notice.** `frontend/legal/cookie-notice.js`, loaded by the landing page and every page in `frontend/legal/` and `frontend/help/`. It brings its own styles, so it looks the same everywhere. On a first visit it shows a bar at the bottom: "arkos uses one cookie to keep you signed in and remembers a few settings in your browser, like your theme. No ads, no analytics." It has a "Cookie policy" link (`/app/legal/cookies.html`) and an OK button. OK stores `arkos-cookie-ack` in local storage.

**Defaults:** contact addresses, the AI provider, the operator name and the MIT line stay placeholders (D16, D3, D1, D2); `DECISIONS.md` lists this page under each. Ready 09's placeholder scan includes this page.

**Not in scope:** the notice inside the app at `/app`; accept and reject choices (only needed if analytics are added, plan §7.6); the privacy policy text (drafting 01); removing the app's Google Fonts and unpkg requests (ready 07); self-serve delete and export (ready 02, 03); adding calendar invites and comment tools to `tools.destructive`.

---

# Implementation Plan

## Task 1: Analytics off, landing page and cookie notice
**Problem:** nothing is at `/`, no one is told what's stored in their browser, and browser tasks are reported to PostHog.
**Done when:**
1. The analytics change merges first, in its own commit. After `tool_module/browser/tool.py` is imported, `ANONYMIZED_TELEMETRY` and `BROWSER_USE_VERSION_CHECK` are `"false"`, even if they were set to true.
2. `GET /` returns the landing page with `Cache-Control: no-store`, and `/app/` still returns the app.
3. The page has every section above, in that order. "get started" and "sign in" go to `/app/`. Every `/app/…` path on the page exists, and the Source link equals `legal.source_url`.
4. Nothing on the page, including its CSS, loads from another host.
5. At 375 px wide there's no sideways scrolling, and the footer links can be clicked with the notice showing. With JavaScript off, every section and link still shows.
6. The landing, legal and help pages all load `cookie-notice.js`. A first visit shows the bar, OK hides it, and a reload keeps it hidden. With storage blocked it shows every time without an error. Tab reaches its link and button.
7. The sign-in pitch in `views.jsx` uses the landing hero's sentence.
8. If drafting 01 Task 1 has merged, `cookies.html` lists `arkos-cookie-ack`.

**Touch point:** `tool_module/browser/tool.py` (two settings after the imports); new `frontend/landing/` (`index.html`, `landing.css`); new `frontend/legal/cookie-notice.js`; the `frontend/legal/` and `frontend/help/` pages (one script tag each); `harness_module/api.py` (the `/` route); `frontend/views.jsx` (pitch text); `frontend/index.html` (bump `?v=`); `frontend/legal/cookies.html` (one row, per item 8).
**Priority:** High | **Effort:** 1 day | **Blockers:** ready 01 Tasks 1–2
**Out of scope:** illustrations or screenshots; a separate marketing domain.
**Acceptance test:** Tests 1–5. Then, in the browser at 375 px, signed out: the sections appear in order, the bar shows on the first visit, OK hides it, it stays hidden after a reload, and it returns after clearing site data. The sign-in screen shows the new sentence.

---

# Tests
Test 1 goes in `tests/test_browser.py` and imports `tool.py` in a fresh subprocess. Test 2 uses `tests/test_api.py`'s client. Tests 3–5 read files only, in a new `tests/test_public_pages.py` with no database.

| # | Test | Proves |
|---|---|---|
| 1 | `test_browser_analytics_forced_off` | With both settings unset, or set to true, importing the tool leaves them `"false"` |
| 2 | `test_root_serves_landing_page` | `/` returns the landing page with `no-store`; `/app/` still returns the app |
| 3 | `test_landing_loads_nothing_offsite` | No script, stylesheet, font or CSS `url()`/`@import` on the page points at another host |
| 4 | `test_landing_paths_exist` | Every `/app/…` path on the page maps to a real file, and the Source link equals `legal.source_url` |
| 5 | `test_cookie_notice_on_public_pages` | The landing, legal and help pages all load the notice script |

---

# Open Questions
Checks for whoever builds this. None needs a decision.
1. Task 1: the repo has no proxy config. Check the deployed proxy's rules for `/`, and record what you find in Implementation Notes.

---

# Implementation Notes
*Runtime surprises, bugs found during execution, and lessons. Add entries here as work lands so the next person inherits the context.*
- 2026-10-14: built. `/` also answers `HEAD` (uptime checks and proxies probe with it; GET-only returned 405). The notice's element id is `cookie-notice-bar`, not an `arkos-` name, so the cookies-page test only sees storage keys. On phones the footer gets 180px of bottom padding so its last lines clear the notice. Open Question 1 (deployed proxy rules for `/`) is still unchecked: the deployment isn't reachable from here.
