# // Feature Spec: In-Product Disclosures (NMORGAN)

*Ready 08 · Wave 2 · blocked by ready 02. The warnings a user sees at the moment they matter, not buried in a policy.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §3 (R4, R5, R7), PR 11 copy table
- `frontend/app.jsx` (`App`, the `main` banner slot)
- `frontend/lookingglass.jsx` (`Composer`, `SessionDetail` and its autopilot button)
- `frontend/components.jsx` (`ApprovalCard`), `frontend/views.jsx` (`SettingsModal.connect`, `ComputerView`, `Login`)
- `harness_module/api.py` (`attention`), `harness_module/runner.py` (`_destructive`)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
The Terms will say the user is responsible for what they approve, but the approve button doesn't say so. Nothing near the composer says the agent can be wrong. Connecting Gmail is one click with no explanation that an assistant will act in the inbox. The upload zone doesn't warn off SSNs or export-controlled data. Success: each of the moments below carries one clear line, and the approval card knows when an action can't be undone.

---

# Technical Background
`GET /attention` builds each approval item from `approvals` joined to `sessions`, returning `tool_name` and `tool_args` for `call` rows. `ApprovalCard` renders `item.kind` as `call`, `plan` or `ask`. `runner._destructive(name)` already answers "is this on the always-ask list", stripping the registry's `mcp_` prefix. It's private, so it gets a public name rather than a second copy of the logic.

`SettingsModal.connect(row)` opens the Composio popup synchronously inside the click, and a comment there warns that any `await` first loses the user gesture. So the interstitial is a separate confirm click that then calls the existing `connect`.

The autopilot button is in `SessionDetail` (`{cancelled ? "resume" : "autopilot"}`). The unattended wall clock comes from `limits.unattended_wall_clock_s` in `/auth/config` (ready 02).

---

# Proposed Approach
Copy changes plus one backend field. The strings below are used verbatim; editing copy later is a text change, not a decision this spec waits on. Acknowledgments (beta banner, per-toolkit connect) live in `localStorage`, wrapped in try/catch, so a blocked store just shows the notice again. The beta banner doesn't come back on a Terms change, because the consent gate (ready 07) already handles that.

**Not in scope:** legal text (drafting 01); help pages (ready 09); a full accessibility audit (only controls touched here are fixed).

---

# Implementation Plan

## Task 1: Beta banner and composer line
**Problem:** no standing statement that this is a beta and that the agent can be wrong.
**Done when:** a dismissible banner, "arkos is in beta. It can make mistakes and the service may change. Keep copies of anything important. Learn more", shows until acknowledged (`arkos-beta-ack`) and then stays gone across reloads; every `Composer` shows "arkos can be wrong. It acts on your connected accounts only with your approval — check before you approve." muted, under the input.
**Touch point:** `frontend/app.jsx` (above the error banner in `main`); `frontend/lookingglass.jsx` (`Composer`); `frontend/styles.css`; `frontend/index.html` (`?v=`).
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** server-side acknowledgment.
**Acceptance test:** in the browser pane: dismiss → reload → gone; clear storage → back.

---

## Task 2: Approval card says who's acting
**Problem:** approving reads like confirming a suggestion, not authorizing an action as yourself.
**Done when:** every `call` card shows "Approving lets arkos do this as you, using your connected account. It may not be reversible.", and cards with `destructive: true` add "This action can't be undone from arkos." `/attention` items carry `destructive`.
**Touch point:** `harness_module/runner.py` (rename `_destructive` to `is_destructive`, updating callers); `harness_module/api.py` (`attention` adds `"destructive": runner.is_destructive(r["tool_name"]) if r["tool_name"] else False`); `frontend/components.jsx` (`ApprovalCard`).
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** changing what's gated.
**Acceptance test:** `test_attention_marks_destructive_calls`.

---

## Task 3: Connect interstitial
**Problem:** a user can hand the agent their inbox without being told what that means.
**Done when:** the first `connect` press for a toolkit shows, inline in its row: "Connecting {name} lets arkos read and act in your {name} account through Composio, our integration provider, whenever a session you start has {name} turned on. Actions that send, share, delete or merge always ask you first. You can disconnect anytime here." with a help link and Continue/Cancel. Continue calls `connect(row)` synchronously, so the popup isn't blocked. The acknowledgment is remembered per `row.server`.
**Touch point:** `frontend/views.jsx` (`SettingsModal`).
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** per-scope text. Managed configs fix scopes, and `list_connections` returns none.
**Acceptance test:** in the browser pane: first press → interstitial; Continue → popup opens; a later press → straight to the popup.

---

## Task 4: Upload warning, autopilot note, keyboard access
**Problem:** risky moments in Files and autopilot carry no guidance, and auth links aren't keyboard-reachable.
**Done when:** `ComputerView` shows "Don't upload passwords, government ID numbers, health records, other students' records, or export-controlled research data." linking `responsible-use.html`; the autopilot button has "Autopilot keeps working without you for up to N minutes, and still asks before anything on the always-ask list." (N from `limits.unattended_wall_clock_s`); `Login`'s `<span onClick>` controls (`auth-forgot`, `auth-link`) become `<button type="button">` with the same look.
**Touch point:** `frontend/views.jsx` (`ComputerView`, `Login`); `frontend/lookingglass.jsx` (`SessionDetail`); `frontend/styles.css`; `frontend/index.html` (`?v=`).
**Priority:** Medium | **Effort:** <4h | **Blockers:** ready 02 (`limits`)
**Out of scope:** blocking uploads by content.
**Acceptance test:** keyboard only: Tab reaches "forgot" and "make one", and Enter activates them; copy is visible at 375px.

---

# Tests

## Test 1: `test_attention_marks_destructive_calls`
**What it verifies:** a parked `call` for `GMAIL_SEND_EMAIL` (and its `mcp_`-prefixed form) is `destructive: true`; `GMAIL_FETCH_EMAILS` is `false`; `ask` rows are `false`.
**Why this matters:** the warning must follow the list the gate uses, not a copy of it.

Everything else is copy and UI, verified in the browser per each task's acceptance test.

---

# Open Questions
1. *Verify in Task 2, not a decision:* "as you" assumes each Composio grant is the user's own OAuth account. Confirm from Composio's managed-auth docs and record it in Implementation Notes; if it's wrong, the copy changes.

---

# Implementation Notes
