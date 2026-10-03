# // Feature Spec: Launch Gate Script (NMORGAN)

*Ready 10 · Wave 3 · blocked by ready 02. One command that says whether we're allowed to launch, including whether the decisions queue is clear. The operator documents it used to carry moved to drafting 02.*

---

**Sources**
- `docs/launch/DECISIONS.md` (entry format: `### D<n> — …`, `**Status:**`, `**Blocks launch:**`)
- `docs/launch/LAUNCH_READINESS_PLAN.md` §0.2 (placeholders), §9
- `config_module/config.yaml` (`auth.cookie_secure`, `legal:`, `security:`), `config_module/loader.py` (`cfg`)
- `scripts/composio_server.py` (the existing operator-script style)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
Launch readiness lives in ten specs, a decisions queue and console settings nobody can see from the repo. Nothing mechanical stops a deploy with `{{OPERATOR_LEGAL_NAME}}` still in the Terms, insecure cookies, or an unanswered launch-blocking decision. Success: `python scripts/check_launch_ready.py` exits 0 only when every repo-checkable condition holds and no `Blocks launch: yes` decision is still open.

---

# Technical Background
Config is read with `config_module.loader.cfg("dotted.key", default)`, so the script checks the config the server will actually run with. `scripts/composio_server.py` sets the style: a CLI with a docstring header, run by hand. The gate stays out of CI because the repo legitimately holds placeholders, draft banners and open decisions until launch day; a CI check would be red for weeks and get ignored.

`DECISIONS.md` entries are headed `### D<n> — <title>`, with bullet lines `**Status:** open|resolved` and `**Who:** … · **Blocks launch:** yes|no`. That's regular enough to parse with a line scan, no Markdown library needed.

---

# Proposed Approach
A list of named checks, each returning pass/fail plus a reason, printed as a table. Any failing blocking check exits 1; warnings never fail.

| Check | Blocking |
|---|---|
| no `{{…}}` in `frontend/legal/`, `frontend/help/`, `SECURITY.md` (names file and token) | yes |
| no page under `frontend/legal/` still says "Draft" | yes |
| `legal.*_version` equals each page's `arkos-doc-version` | yes |
| `auth.cookie_secure` is true | yes |
| `security.contact` holds no `{{…}}` | yes |
| no `DECISIONS.md` entry is `open` with `Blocks launch: yes` (names each one) | yes |
| `security.csp_enforce` is true | warn |

**Not in scope:** console state (Supabase SMTP, Google consent screen, spend caps). `LAUNCH_CHECKLIST.md` (drafting 02) covers those by hand.

---

# Implementation Plan

## Task 1: Checks and CLI
**Problem:** nothing mechanical blocks an unready launch.
**Done when:** each check in the table behaves as stated against a fixture tree; output is one line per check; exit code is 1 on any blocking failure and 0 otherwise.
**Touch point:** new `scripts/check_launch_ready.py`.
**Priority:** High | **Effort:** <4h | **Blockers:** ready 02 (`legal:` block and page metas)
**Out of scope:** auto-fixing.
**Acceptance test:** Tests 1–4.

```python
@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = True

def open_launch_blockers(decisions_md: str) -> list[str]: ...   # ["D1 — Who operates arkos", ...]
def run_checks(root: Path) -> list[Check]: ...
def main() -> int: ...
```

---

# Tests
New `tests/test_launch_gate.py`, running `run_checks` against a `tmp_path` fixture tree so it never depends on the real repo's state.

## Test 1: `test_launch_gate_fails_on_placeholder`
**What it verifies:** a legal page containing `{{OPERATOR_LEGAL_NAME}}` yields a failing blocking check that names the file.
**Why this matters:** the most likely launch-day mistake.

## Test 2: `test_open_launch_decision_blocks`
**What it verifies:** `open_launch_blockers` returns `D5` for an open `Blocks launch: yes` entry, and ignores resolved entries and `Blocks launch: no` entries.
**Why this matters:** the decisions queue is only a gate if something reads it.

## Test 3: `test_launch_gate_passes_clean_tree`
**What it verifies:** filled pages, matching versions, `cookie_secure: true`, a real contact and no open blockers give exit 0.
**Why this matters:** a gate that never passes gets bypassed.

## Test 4: `test_report_only_csp_only_warns`
**What it verifies:** `security.csp_enforce: false` produces a non-blocking check, and the exit code stays 0.
**Why this matters:** report-only CSP for one release is the plan, not a mistake.

---

# Open Questions
None.

---

# Implementation Notes
