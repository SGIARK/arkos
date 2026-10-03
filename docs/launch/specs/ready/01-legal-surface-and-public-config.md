# // Feature Spec: Legal Surface & Public Config (NMORGAN)

*Ready 01 · Wave 1 · no blockers. The page shell every legal and help page drops into, the links that reach it, and the one public config endpoint every other spec reads from.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §5.1, §5.2; `docs/launch/DECISIONS.md` (D2, D5, D7)
- `harness_module/api.py` (`auth_config`, `_Frontend`, the `/app` mount)
- `frontend/views.jsx` (`Login`, `MailSent`, `ResetPassword`, `AuthAside`, `SettingsModal`)
- `frontend/app.jsx` (`App`, rail `foot`), `frontend/components.jsx`, `frontend/api.jsx`, `frontend/index.html`
- `config_module/config.yaml` (`budgets.unattended`)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
A signed-out visitor can read nothing but the pitch in `AuthAside`. There's no URL to give Google's OAuth consent screen, no footer, no home for the AGPL source offer, and no place for the legal and help pages other specs write. Four other specs also each want to add a field to `GET /auth/config`, which would make them conflict. Success: every legal and help URL resolves signed-out; the auth screens and Settings link to them; and `/auth/config` carries every public value the other specs need, added once, here.

---

# Technical Background
`app.mount("/app", _Frontend(directory=_FRONTEND, html=True))` serves all of `frontend/`, and `_Frontend` sets `no-store` on `.html`. So `frontend/legal/privacy.html` is already live at `/app/legal/privacy.html`; only the short URLs need code.

`GET /auth/config` is public and returns `{supabase_url, anon_key}`. It's the one unauthenticated JSON call the sign-in screen already makes, so every public value rides on it.

Every script and stylesheet in `frontend/index.html` carries `?v=49`, and any `.jsx`/`.css` change bumps it everywhere.

---

# Proposed Approach
1. Static pages under `frontend/legal/` and `frontend/help/`: plain HTML, readable with JS off, one `legal.css` on top of `styles.css`.
2. Config owns every policy number, so decisions change a value, not code: a `legal:` block (versions, minimum age, source URL) and a `retention:` block.
3. `/auth/config` gains `legal`, `limits` and `build` in one change. Ready 02, 07 and 08 read these fields and never edit the endpoint.
4. Short 307 redirects (`/terms`, `/privacy`, `/acceptable-use`, `/legal`, `/help`) as stable URLs for consent screens and email.
5. One `LegalLinks` component on the three auth screens and in a new "about & legal" section in Settings.

**Decisions applied (defaults from `DECISIONS.md`):** D5 → `legal.min_age: 18`. D7 → `retention.deleted_files_days: 30`. D2 → the footer renders `{{MIT_RELATIONSHIP_STATEMENT}}` until it's resolved. Build SHA comes from `ARKOS_BUILD_SHA`, falling back to `"dev"`; that's an engineering default, not a decision.

**Not in scope:** document text (drafting 01), help text (ready 08), consent (ready 05).

---

# Implementation Plan

## Task 1: Page shell and stubs
**Problem:** no styled container for long pages, and every link added later would 404.
**Done when:** every file in plan §5.1 under `frontend/legal/` and `frontend/help/` exists and renders a title plus "Draft — content pending" at 375px with no horizontal scroll, in both themes.
**Touch point:** new `frontend/legal/legal.css`, `frontend/legal/_template.html`, `frontend/legal/index.html`, a stub per §5.1 page, `frontend/help/index.html`.
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** real copy.
**Acceptance test:** `test_every_legal_page_exists`; manual 375px check in the browser pane.

```html
<meta name="arkos-doc-version" content="2026-10-01" />
<link rel="stylesheet" href="../styles.css?v=50" /><link rel="stylesheet" href="legal.css?v=50" />
<body class="legal"><header><a href="/app/">arkos</a></header>
<main><h1>…</h1><p class="meta">Effective {{EFFECTIVE_DATE}} · Version 2026-10-01</p>…</main>
<footer><!-- LegalLinks markup -->{{MIT_RELATIONSHIP_STATEMENT}}</footer></body>
```
Theme comes from `localStorage["arkos-theme"]` in a four-line inline script wrapped in try/catch; with no JS it stays light.

---

## Task 2: Config, short URLs, public config
**Problem:** no stable public URLs, and no single place other specs can read policy values from.
**Done when:** the five short URLs return 307 to existing files, and `GET /auth/config` returns the shape below with values read from config.
**Touch point:** `config_module/config.yaml` (new `legal:` and `retention:` blocks); `harness_module/api.py` (a `_SHORT_URLS` loop with `app.add_api_route`, registered above the `/app` mount; `auth_config`).
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** enforcing anything.
**Acceptance test:** `test_short_legal_urls_redirect_to_real_files`, `test_auth_config_carries_public_policy`.

```yaml
legal:
  terms_version: "2026-10-01"
  privacy_version: "2026-10-01"
  min_age: 18                       # D5
  source_url: "https://github.com/SGIARK/arkos"
retention:
  deleted_files_days: 30            # D7
```
```python
# GET /auth/config
{
    "supabase_url": ..., "anon_key": ...,
    "legal": {"terms_version": ..., "privacy_version": ..., "min_age": ..., "source_url": ...},
    "limits": {"deleted_files_days": ..., "unattended_wall_clock_s": ...},  # budgets.unattended.wall_clock_s
    "build": os.environ.get("ARKOS_BUILD_SHA") or "dev",
}
```

---

## Task 3: Links in the product
**Problem:** a user can't reach the documents from inside the app.
**Done when:** the auth screens show Terms · Privacy · Acceptable use · Help · Source; Settings has "about & legal" with the same links, the build SHA, and "arkos is free software under the AGPL-3.0 — get the source"; the rail `foot` has `help`, opening `/help` in a new tab; `api.config()` fetches `/auth/config` once and caches it.
**Touch point:** `frontend/components.jsx` (new `LegalLinks`); `frontend/views.jsx` (`Login`, `MailSent`, `ResetPassword`, `SettingsModal`); `frontend/app.jsx` (rail `foot`); `frontend/api.jsx` (new `api.config`); `frontend/index.html` (`?v=50`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2
**Out of scope:** adding `help` to `NAV`, since help lives outside the app.
**Acceptance test:** in the browser pane, signed out and signed in, each link opens its page, and Tab reaches every link.

```jsx
function LegalLinks({ compact = false, build = null }) { /* <nav className="legal-links"> of <a target="_blank" rel="noopener"> */ }
```

---

# Tests

## Test 1: `test_every_legal_page_exists`
**What it verifies:** every §5.1 path is a file under `frontend/`.
**Why this matters:** `html=True` can mask a missing page, so a dead footer link has to fail CI.

## Test 2: `test_short_legal_urls_redirect_to_real_files`
**What it verifies:** each short URL answers 307, and its `Location` maps to an existing file.
**Why this matters:** Google's consent screen and every email point here.

## Test 3: `test_auth_config_carries_public_policy`
**What it verifies:** unauthenticated `GET /auth/config` returns `legal`, `limits` and `build`, equal to the config values (and `"dev"` with the env var unset).
**Why this matters:** ready 02, 05 and 06 read these fields and never touch the endpoint.

---

# Open Questions
None. Every unknown here is a placeholder or a config default tracked in `DECISIONS.md`.

---

# Implementation Notes
