# // Feature Spec: Host Our Own Scripts and Fonts (VBAKER)

*Ready 07 · Wave 2 · needs ready 01 and 04. Serve React, Babel, supabase-js and the font from arkos itself instead of unpkg and Google Fonts.*

---

**Sources**
- Pages: `frontend/index.html`, `frontend/styles.css`, `frontend/legal/legal.css` (from ready 01)
- Security policy: `harness_module/headers.py` (from ready 04)
- Privacy pages: `frontend/legal/cookies.html`, `frontend/legal/subprocessors.html` (from drafting 01)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
Every page load sends the visitor's IP address to unpkg and Google Fonts, and arkos runs whatever code unpkg serves. The privacy pages have to disclose both services, and the security policy has to allow them. Success: arkos loads nothing from other sites except its own Supabase project.

---

# Technical Background
- `index.html` already pins exact versions with integrity hashes (fingerprints of the expected file): react 18.3.1, react-dom 18.3.1, @babel/standalone 7.29.0 and supabase-js 2.112.4. A downloaded copy can be checked against the fingerprint already in the repo.
- The font, JetBrains Mono, has a free license (SIL OFL) that allows copying it, as long as the license file goes with it.

---

# Proposed Approach
Copy the four scripts and the font files into `frontend/vendor/`, check each against its fingerprint, and point the pages at the local copies. Then remove the two services from the security policy and from the privacy pages.

**Not in scope:** switching React to its production build (a separate change worth its own test).

---

# Implementation Plan

## Task 1: Copy and check the files
**Problem:** the app depends on two outside services every time it loads.
**Done when:** `frontend/vendor/` holds the four scripts, each matching its fingerprint, plus the font files and license; pages load nothing from unpkg or Google; `?v=` is bumped.
**Touch point:** new `frontend/vendor/`; `frontend/index.html`; `frontend/styles.css` (font rules); `frontend/legal/legal.css`.
**Priority:** Medium | **Effort:** <4h | **Blockers:** ready 01
**Out of scope:** the security policy.
**Acceptance test:** Test 1; in the browser, the network log shows no outside requests except Supabase.

---

## Task 2: Tighten the policy and the disclosures
**Problem:** the security policy and the privacy pages still name the two services.
**Done when:** the security policy no longer allows unpkg or Google Fonts; the cookies and subprocessor pages drop those rows; the legal changelog notes it.
**Touch point:** `harness_module/headers.py`; `frontend/legal/cookies.html`, `subprocessors.html`, `changelog.html`.
**Priority:** Medium | **Effort:** <1h | **Blockers:** Task 1; ready 04
**Out of scope:** further policy tightening.
**Acceptance test:** Test 2; the app signs in and runs a conversation with no policy warnings.

---

# Tests
No mocks needed.

| # | Test | Proves |
|---|---|---|
| 1 | `test_vendored_assets_match_pinned_integrity` | The local copies are exactly the files we vetted |
| 2 | `test_csp_names_no_cdn` | The policy no longer allows either outside service |

---

# Open Questions
None.

---

# Implementation Notes
