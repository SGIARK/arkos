# // Feature Spec: Vendor Frontend Assets (NMORGAN)

*Ready 07 · Wave 2 · needs ready 01 and 04. Serve React, Babel, supabase-js and the font from our own origin instead of unpkg and Google Fonts.*

---

**Sources**
- `frontend/index.html` (four `<script>` tags pinned with `integrity`; Google Fonts `<link>`s), `frontend/styles.css`, `frontend/legal/legal.css` (ready 01)
- `harness_module/headers.py` (`content_security_policy`, from ready 04)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-02

---

# Problem
Every page load sends the visitor's IP address and user agent to unpkg and Google Fonts, and the app runs whatever unpkg serves, guarded only by SRI hashes. The privacy pages have to disclose both CDNs, and the CSP has to allow them. Success: no page loads a third-party origin except the Supabase project, and the CSP drops both CDNs.

---

# Technical Background
`index.html` pins exact versions with SRI: `react@18.3.1`, `react-dom@18.3.1` (development builds), `@babel/standalone@7.29.0`, and `@supabase/supabase-js@2.112.4`. So each downloaded file can be verified against the hash already in the tree before commit. JetBrains Mono comes from Google Fonts CSS, which resolves to `fonts.gstatic.com` woff2 files and is SIL OFL-licensed, so redistribution is allowed with the license file. Ready 04's `content_security_policy` lists both CDNs until this lands.

---

# Proposed Approach
Put the exact files in `frontend/vendor/`, verified by hash; point `index.html` and `legal.css` at them, keeping `integrity` on the script tags; add `@font-face` rules; drop both CDNs from the CSP; update the cookies and subprocessor pages to remove the CDN rows.

**Not in scope:** switching to production React builds or prebuilding JSX. It's the same kind of file swap, but a behaviour change, and belongs in its own PR.

---

# Implementation Plan

## Task 1: Vendor and verify
**Problem:** the app depends on two CDNs at runtime.
**Done when:** `frontend/vendor/` holds the four scripts, each with a SHA-384 equal to its current `integrity` value, plus the JetBrains Mono woff2 weights used by `index.html` and `OFL.txt`; `index.html` and `legal.css` load only same-origin assets; `styles.css` has the `@font-face` rules; `?v=` is bumped.
**Touch point:** new `frontend/vendor/`; `frontend/index.html`; `frontend/styles.css`; `frontend/legal/legal.css`.
**Priority:** Medium | **Effort:** <4h | **Blockers:** ready 01 (`legal.css`)
**Out of scope:** the CSP.
**Acceptance test:** `test_vendored_assets_match_pinned_integrity`; in the browser pane, the network log shows no request leaving the origin except Supabase.

---

## Task 2: Tighten the CSP and the disclosures
**Problem:** the CSP and the privacy pages still name the CDNs.
**Done when:** `content_security_policy` no longer includes `unpkg.com`, `fonts.googleapis.com` or `fonts.gstatic.com`; `cookies.html` and `subprocessors.html` drop the CDN rows; `changelog.html` has an entry.
**Touch point:** `harness_module/headers.py`; `frontend/legal/cookies.html`, `frontend/legal/subprocessors.html`, `frontend/legal/changelog.html`.
**Priority:** Medium | **Effort:** <1h | **Blockers:** Task 1, ready 04
**Out of scope:** removing `'unsafe-eval'`.
**Acceptance test:** `test_csp_names_no_cdn`; the app signs in and runs a session with no CSP violations in the console.

---

# Tests

## Test 1: `test_vendored_assets_match_pinned_integrity`
**What it verifies:** each `frontend/vendor/*.js` hashes to the `integrity` value on its `<script>` tag.
**Why this matters:** vendoring must not quietly swap the code we audited.

## Test 2: `test_csp_names_no_cdn`
**What it verifies:** the policy string contains no `unpkg` or `fonts.g` host.
**Why this matters:** a leftover allowance keeps the door open that this spec closes.

---

# Open Questions
None.

---

# Implementation Notes
