# // Feature Spec: Security Headers & security.txt (NMORGAN)

*Ready 04 · Wave 1 · no blockers. The cheap, standard browser protections, and a published contact for security reports.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §3 (risks R12, R17); `docs/launch/DECISIONS.md` (D16)
- Server: `harness_module/api.py`; settings: `config_module/config.yaml`
- Pages: `frontend/index.html`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
arkos sends no security headers. Another site could show the approvals page inside a hidden frame and trick a user into clicking "approve" (clickjacking). Security researchers have no published way to report a problem. Massachusetts law expects reasonable safeguards, and these are the cheapest ones. Success: every response carries the headers below, and `/.well-known/security.txt` answers.

---

# Technical Background
Two terms used below:
- **Security headers:** instructions the server sends with every response, telling the browser what not to allow.
- **CSP (content security policy):** the header listing where scripts, styles and data may come from.

What's true today:
- The server adds no headers of its own beyond the standard cross-origin (CORS) one.
- Scripts load from unpkg and fonts from Google until ready 07 moves them in-house, so the CSP must allow both for now.
- Three things run script the CSP would normally block: the in-browser JSX compiler (Babel), the small page that closes the "connect account" popup, and the theme switch on the legal pages. Until the frontend is prebuilt, the CSP has to allow them, so its real value at launch is blocking framing.
- Local development runs on plain http, so the "always use https" header must only be sent in production.

---

# Proposed Approach
One small piece of server code (middleware) adds these headers to every response, pages and API alike:

| Header | What it does |
|---|---|
| `X-Frame-Options: DENY` and CSP `frame-ancestors 'none'` | No site can frame arkos |
| `X-Content-Type-Options: nosniff` | Browsers don't guess file types |
| `Referrer-Policy: strict-origin-when-cross-origin` | Other sites see only arkos's domain, not full addresses |
| `Permissions-Policy` | Camera, microphone and location are off |
| `Strict-Transport-Security` | Browsers always use https (production only) |
| CSP | Limits where scripts, styles, fonts and data come from |

The CSP starts in "report only" mode: the browser logs what it would have blocked but blocks nothing. It switches to enforced after a clean check in the browser.

**Default:** the security contact is a placeholder until the inbox exists (D16); ready 09 refuses launch while it is.

**Not in scope:** hosting scripts and fonts ourselves (ready 07); prebuilding the frontend; two-factor sign-in.

---

# Implementation Plan

## Task 1: Header middleware
**Problem:** no browser protections are sent.
**Done when:** a page and an API response both carry every header in the table; the CSP is report-only while `security.csp_enforce` is false; the https header appears only when `auth.cookie_secure` is true; in the browser, signing in, running a conversation and connecting an account (the popup must close itself) produce no CSP warnings.
**Touch point:** new `harness_module/headers.py`; `harness_module/api.py` (register it); `config_module/config.yaml` (new `security` section).
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** collecting CSP reports.
**Acceptance test:** Tests 1–2, plus the browser check.

---

## Task 2: security.txt
**Problem:** there's no published security contact.
**Done when:** `/.well-known/security.txt` returns the contact, an expiry date, a link to `/app/legal/security.html` and the preferred language, all from config.
**Touch point:** `harness_module/api.py`.
**Priority:** Medium | **Effort:** <1h | **Blockers:** none
**Out of scope:** signing the file.
**Acceptance test:** Test 3.

---

# Tests
No mocks needed.

| # | Test | Proves |
|---|---|---|
| 1 | `test_security_headers_on_api_and_static` | Pages and API responses both get the headers |
| 2 | `test_hsts_follows_cookie_secure` | The https header is off in local development |
| 3 | `test_security_txt_is_served` | The file is served with all four fields |

---

# Open Questions
Checks for whoever builds this. None needs a decision.
1. Task 1: whether the production web server (Caddy) already sets any of these headers. If it does, set them in one place only; two CSPs combine and break pages.
2. Task 1: whether the live browser view loads images from another address. Adjust the CSP from the report-only warnings.

---

# Implementation Notes
