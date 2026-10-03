# // Feature Spec: Security Headers & security.txt (NMORGAN)

*Ready 04 · Wave 1 · no blockers. The cheap "reasonable security" basics. Self-hosting fonts and scripts is a separate spec (ready 12).*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §3 (R12, R17); `docs/launch/DECISIONS.md` (D16)
- `harness_module/api.py` (`app`, the `CORSMiddleware` setup, the `/app` mount)
- `frontend/index.html` (unpkg scripts, Google Fonts), `frontend/api.jsx` (`supabaseClient`)
- `config_module/config.yaml` (`auth.cookie_secure`, `app.public_url`)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
The app sends no security headers. Any site can frame the approvals page and clickjack its approve buttons, browsers get no sniffing or referrer rules, and a researcher has no published contact. Massachusetts 201 CMR 17.00 expects reasonable technical safeguards, and these are the cheapest. Success: every response carries the header set below, and `/.well-known/security.txt` answers.

---

# Technical Background
`app` has one middleware, a `CORSMiddleware` restricted to `app.public_url`. Three things in the tree run script the CSP would otherwise block: `@babel/standalone` compiles and runs the `text/babel` scripts in the page; the `/connections/done` handler returns `_CLOSE_POPUP`, an HTML page whose inline `<script>` posts back to the opener and closes the popup; and ready 02's legal pages set the theme from a short inline script. So `script-src` needs `'unsafe-inline'` and `'unsafe-eval'` until the frontend is prebuilt. That's a known gap, written down rather than hidden: the CSP's real protection at launch is `frame-ancestors 'none'` (clickjacking) plus the origin limits on connect, font and style. Scripts come from `unpkg.com` and fonts from Google Fonts today, so the CSP allows them until ready 12 lands. supabase-js talks to the project URL from `/auth/config`, so `connect-src` includes that origin.

`auth.cookie_secure` is false only for plain-HTTP local dev. HSTS follows it, or a local run pins localhost to HTTPS for a year.

---

# Proposed Approach
1. One small ASGI middleware in `harness_module/headers.py` sets the headers on every response, API and static. CSP ships report-only (`security.csp_enforce: false`), then flips to enforced after a clean browser pass.
2. `GET /.well-known/security.txt` from config.

**Decisions applied (defaults from `DECISIONS.md`):** D16 → `security.contact: "mailto:{{SECURITY_EMAIL}}"` until the inbox exists; ready 10 fails launch while it's a placeholder.

**Not in scope:** vendoring assets (ready 12), removing Babel-in-browser, MFA, proxy-level rate limiting.

---

# Implementation Plan

## Task 1: Header middleware
**Problem:** no clickjacking, sniffing, referrer or transport protections.
**Done when:** `GET /app/` and `GET /auth/config` both carry `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`, `Permissions-Policy: camera=(), microphone=(), geolocation=()`, and a CSP with `frame-ancestors 'none'`, sent as `Content-Security-Policy-Report-Only` while `security.csp_enforce` is false. HSTS is present only when `auth.cookie_secure` is true.
**Touch point:** new `harness_module/headers.py`; `harness_module/api.py` (`app.add_middleware` beside CORS); `config_module/config.yaml` (new `security:` block).
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** a CSP violation-report endpoint.
**Acceptance test:** `test_security_headers_on_api_and_static`, `test_hsts_follows_cookie_secure`; in the browser pane, sign in, run a session, and complete a connector popup (it must close itself), with no report-only violations in the console.

```python
def content_security_policy(supabase_origin: str) -> str:
    """default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' https://unpkg.com;
    style-src 'self' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com;
    connect-src 'self' <supabase_origin>; img-src 'self' data:; frame-ancestors 'none';
    base-uri 'self'; form-action 'self'"""
```
```yaml
security:
  csp_enforce: false
  contact: "mailto:{{SECURITY_EMAIL}}"   # D16
  txt_expires: "2027-10-01T00:00:00Z"
```

---

## Task 2: security.txt
**Problem:** no published vulnerability contact.
**Done when:** `GET /.well-known/security.txt` returns `text/plain` with `Contact` (from `security.contact`), `Expires` (from `security.txt_expires`), `Policy: <app.public_url>/app/legal/security.html`, and `Preferred-Languages: en`.
**Touch point:** `harness_module/api.py` (new handler registered above the `/app` mount).
**Priority:** Medium | **Effort:** <1h | **Blockers:** none (the policy page exists as a stub from ready 02, or 404s until then, which doesn't break the file)
**Out of scope:** PGP signing.
**Acceptance test:** `test_security_txt_is_served`.

---

# Tests
New `tests/test_headers.py` for the middleware; `tests/test_api.py` for the route.

## Test 1: `test_security_headers_on_api_and_static`
**What it verifies:** the four headers plus the CSP header are on both an API route and a static page.
**Why this matters:** static files bypass route code, so the middleware has to cover both.

## Test 2: `test_hsts_follows_cookie_secure`
**What it verifies:** HSTS is present with `cookie_secure: true` and absent with `false`.
**Why this matters:** a stray HSTS breaks every developer's localhost.

## Test 3: `test_security_txt_is_served`
**What it verifies:** 200, `text/plain`, and all four fields.
**Why this matters:** RFC 9116 is what researchers actually check.

---

# Open Questions
1. *Verify in Task 1, not a decision:* does the production Caddy already set any of these headers? If so, remove them there. Two CSPs intersect and break pages.
2. *Verify in Task 1:* does `BrowserCanvas` load frames from another origin or as `blob:`? Adjust `img-src` from the report-only console output.

---

# Implementation Notes
