# // Feature Spec: Legal Surface & Public Config (NMORGAN)

*Ready 01 · Wave 1 · no blockers. The empty pages and links every legal and help page needs, plus one place the sign-in screen reads public settings from.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §5.1–5.2; `docs/launch/DECISIONS.md` (D2, D5, D7)
- Server: `harness_module/api.py`; settings: `config_module/config.yaml`
- Screens: `frontend/views.jsx`, `frontend/app.jsx`, `frontend/components.jsx`, `frontend/api.jsx`, `frontend/index.html`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
A visitor who isn't signed in sees only the sign-in pitch. There's no Terms or Privacy link to give Google's sign-in setup, no footer, no "get the source" link (arkos's open-source license, the AGPL, requires one), and nowhere to put the legal and help pages. Several other specs also need public settings from the server. Success: every legal and help address opens while signed out, the app links to them, and the public settings live in one place.

---

# Technical Background
One term used below:
- **Public config:** `GET /auth/config`, the one settings request the sign-in screen makes before anyone logs in. Today it returns the Supabase address and public key.

What's true today:
- Anything in `frontend/` is already served at `/app/…`, so new HTML pages need no server code. Only short addresses like `/terms` do.
- Every script and stylesheet link in `index.html` ends in `?v=49`. Any change to frontend files must bump that number, or browsers keep old copies.

---

# Proposed Approach
1. Plain HTML pages in `frontend/legal/` and `frontend/help/`, readable without JavaScript, sharing one stylesheet.
2. Every policy number lives in `config.yaml`, so a decision changes one value: document versions, minimum age, source link, undo days.
3. Public config gains those values, the autopilot time limit and the build id, added once here. Ready 02, 05 and 06 only read them.
4. Short addresses: `/terms`, `/privacy`, `/acceptable-use`, `/legal`, `/help`.
5. A row of links on the sign-in screens and an "about & legal" section in Settings.

**Decided:** minimum age 18 (D5). **Defaults:** undo kept 30 days (D7 still open); the footer shows a placeholder for the MIT relationship statement (D2 still open); the build id comes from `ARKOS_BUILD_SHA`, or "dev".

**Not in scope:** the page text (drafting 01), help text (ready 08), consent (ready 05).

---

# Implementation Plan

## Task 1: Page shell and placeholder pages
**Problem:** there's no styled page to drop text into, and links added later would go nowhere.
**Done when:** every page listed in plan §5.1 exists, shows its title and "Draft — content pending", and reads well at phone width in light and dark themes.
**Touch point:** new `frontend/legal/` (stylesheet, template, index, one placeholder per page); new `frontend/help/index.html`.
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** real text.
**Acceptance test:** Test 1; check in the browser at phone width.

---

## Task 2: Settings, short addresses, public config
**Problem:** no stable public addresses, and no single place for other specs to read settings from.
**Done when:** the five short addresses redirect to real pages, and public config returns the document versions, minimum age, source link, undo days, autopilot time limit and build id.
**Touch point:** `config_module/config.yaml` (new `legal` and `retention` sections); `harness_module/api.py` (redirects and public config).
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** enforcing anything.
**Acceptance test:** Tests 2–3.

```yaml
legal:
  terms_version: "2026-10-01"
  privacy_version: "2026-10-01"
  min_age: 18
  source_url: "https://github.com/SGIARK/arkos"
retention:
  deleted_files_days: 30
```

---

## Task 3: Links in the app
**Problem:** users can't reach the documents from inside arkos.
**Done when:** the sign-in, reset and check-your-email screens show Terms · Privacy · Acceptable use · Help · Source; Settings has "about & legal" with the same links, the build id, and "arkos is free software under the AGPL-3.0 — get the source"; the side rail has a help link that opens in a new tab; the keyboard can reach every link.
**Touch point:** `frontend/components.jsx` (new `LegalLinks`); `frontend/views.jsx` (sign-in screens, Settings); `frontend/app.jsx` (side rail); `frontend/api.jsx` (read public config once); `frontend/index.html` (bump `?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2
**Out of scope:** making help a tab inside the app.
**Acceptance test:** in the browser, signed out and signed in, every link opens its page.

---

# Tests
No mocks needed.

| # | Test | Proves |
|---|---|---|
| 1 | `test_every_legal_page_exists` | Every planned page file is there, so no footer link is dead |
| 2 | `test_short_legal_urls_redirect_to_real_files` | Each short address lands on a real page |
| 3 | `test_auth_config_carries_public_policy` | The sign-in screen gets versions, limits and build id without logging in |

---

# Open Questions
None. Unknowns are placeholders or config defaults tracked in `DECISIONS.md`.

---

# Implementation Notes
