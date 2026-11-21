# // Feature Spec: Launch Gate Script (VBAKER)

*Ready 09 · Wave 3 · needs ready 01. One command that says whether arkos is allowed to launch.*

---

**Sources**
- Decisions: `docs/launch/DECISIONS.md`; placeholders: `docs/launch/LAUNCH_READINESS_PLAN.md` §0.2
- Settings: `config_module/config.yaml`, read with `config_module/loader.py`
- Existing operator script for style: `scripts/composio_server.py`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
Launch readiness is spread across nine specs, a decisions list and settings in outside dashboards. Nothing stops a launch with "{{OPERATOR_LEGAL_NAME}}" still in the Terms, insecure cookies, or an unanswered launch-blocking decision. Success: `python scripts/check_launch_ready.py` passes only when everything checkable in the repo is ready.

---

# Technical Background
- The script reads settings the same way the server does, so it checks what will actually run.
- It runs by hand, not on every commit: until launch day the repo is supposed to contain placeholders and open decisions, and a check that's always red gets ignored.
- Each entry in `DECISIONS.md` has a `Status:` line and a `Blocks launch:` line, which a simple line-by-line scan can read.

---

# Proposed Approach
The script runs these checks and prints one line each. Any failing "blocks" check means the launch isn't ready.

| Check | Blocks launch? |
|---|---|
| No `{{placeholder}}` left in the legal pages, help pages, landing page (ready 10) or `SECURITY.md` (names the file) | Yes |
| No legal page still marked "Draft" | Yes |
| The document versions in config match the versions on the pages | Yes |
| Login cookies are https-only (`auth.cookie_secure`) | Yes |
| The security contact isn't a placeholder | Yes |
| No decision in `DECISIONS.md` is still open (names each one) | Yes |
| The security policy is enforced, not report-only | Warning only |

**Not in scope:** dashboard settings (email sender, Google sign-in screen, spend caps). The launch checklist in drafting 02 covers those by hand.

---

# Implementation Plan

## Task 1: Checks and command
**Problem:** nothing blocks an unready launch.
**Done when:** each check in the table behaves as described against a test folder; the output is one line per check; the command fails if any blocking check fails, and passes otherwise.
**Touch point:** new `scripts/check_launch_ready.py`.
**Priority:** High | **Effort:** <4h | **Blockers:** ready 01 (settings and page versions)
**Out of scope:** fixing anything automatically.
**Acceptance test:** Tests 1–4.

---

# Tests
Each test builds a small fake repo folder, so results never depend on the real repo's state.

| # | Test | Proves |
|---|---|---|
| 1 | `test_launch_gate_fails_on_placeholder` | A leftover placeholder blocks launch and names the file |
| 2 | `test_open_launch_decision_blocks` | An open decision blocks launch; resolved ones don't |
| 3 | `test_launch_gate_passes_clean_tree` | A ready folder passes, so the gate isn't always red |
| 4 | `test_report_only_csp_only_warns` | Report-only security policy warns but doesn't block |

---

# Open Questions
None.

---

# Implementation Notes
