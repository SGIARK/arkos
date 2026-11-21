# // Feature Spec: In-Product Disclosures (VBAKER)

*Ready 06 · Wave 2 · needs ready 01. Short warnings shown at the moment they matter, not buried in a policy.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §3 (risks R4, R5, R7)
- Screens: `frontend/app.jsx`, `frontend/lookingglass.jsx`, `frontend/components.jsx`, `frontend/views.jsx`
- Server: `harness_module/api.py`, `harness_module/runner.py`

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
The Terms will say users are responsible for what they approve, but the approve button doesn't say so. Nothing near the chat box says the agent can be wrong. Connecting Gmail is one click, with no word that an assistant will now act in the inbox. The upload area doesn't warn against ID numbers or export-controlled research data. Success: each risky moment shows one clear line, and approval cards warn when an action can't be undone.

---

# Technical Background
- Approval cards get each pending action's tool name from the server's "what's waiting" list (`/attention`).
- The server already knows which tools always need a human ("always ask", such as sending email) through a private check in `runner.py`. It gets a public name instead of a second copy of the list.
- The "connect" button must open its popup within the click itself, or the browser blocks it. So the explainer is a separate click that then runs the existing connect.
- The autopilot time limit comes from public config (ready 01).

---

# Proposed Approach
These exact lines, used as written unless the owner edits them:

| Where | Text |
|---|---|
| Banner, until dismissed | "arkos is in beta. It can make mistakes and the service may change. Keep copies of anything important. Learn more" |
| Under every chat box | "arkos can be wrong. It acts on your connected accounts only with your approval — check before you approve." |
| Every approval card | "Approving lets arkos do this as you, using your connected account. It may not be reversible." |
| Irreversible approvals, added | "This action can't be undone from arkos." |
| First connect of an account | "Connecting {name} lets arkos read and act in your {name} account through Composio, our integration provider, whenever a session you start has {name} turned on. Actions that send, share, delete or merge always ask you first. You can disconnect anytime here." |
| Files view | "Don't upload passwords, government ID numbers, health records, other students' records, or export-controlled research data." |
| Autopilot button | "Autopilot keeps working without you for up to N minutes, and still asks before anything on the always-ask list." |

Dismissals are remembered in the browser under fixed `arkos-…` keys, each listed on the cookies page once drafting 01 Task 1 has merged, and the notice simply shows again if browser storage is blocked. The banner doesn't come back when the Terms change, because the agreement screen (ready 05) already handles that.

**Not in scope:** legal text (drafting 01); help pages (ready 08); a full accessibility review.

---

# Implementation Plan

## Task 1: Beta banner and chat-box line
**Problem:** nothing says this is a beta or that the agent can be wrong.
**Done when:** the banner shows until dismissed and stays gone after reloads; every chat box shows the line under it.
**Touch point:** `frontend/app.jsx` (banner); `frontend/lookingglass.jsx` (chat box); `frontend/styles.css`; `frontend/index.html` (bump `?v=`).
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** remembering dismissals on the server.
**Acceptance test:** in the browser: dismiss, reload, still gone; clear storage, back.

---

## Task 2: Approval cards say who's acting
**Problem:** approving reads like confirming a suggestion, not acting as yourself.
**Done when:** every action card shows the "as you" line, irreversible ones add the second line, and the "what's waiting" list marks each item irreversible or not.
**Touch point:** `harness_module/runner.py` (make the always-ask check public); `harness_module/api.py` (`/attention`); `frontend/components.jsx` (approval card).
**Priority:** High | **Effort:** <2h | **Blockers:** none
**Out of scope:** changing what needs approval.
**Acceptance test:** Test 1.

---

## Task 3: Explain before connecting
**Problem:** a user can hand the agent their inbox without being told what that means.
**Done when:** the first "connect" for each account shows the explainer inline; Continue opens the popup (not blocked); the explainer doesn't reappear for that account.
**Touch point:** `frontend/views.jsx` (Settings).
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** explaining individual permissions.
**Acceptance test:** in the browser: first connect shows it; Continue opens the popup; the next connect skips it.

---

## Task 4: Upload warning, autopilot note, keyboard access
**Problem:** the risky moments in Files and autopilot carry no guidance, and two sign-in links can't be reached by keyboard.
**Done when:** the Files view shows the upload warning, linking to the responsible-use page; the autopilot button shows its note with the configured minutes; "forgot" and "make one" on the sign-in screen are real buttons the keyboard can reach.
**Touch point:** `frontend/views.jsx` (Files view, sign-in); `frontend/lookingglass.jsx` (autopilot button); `frontend/styles.css`.
**Priority:** Medium | **Effort:** <4h | **Blockers:** ready 01 (time limit)
**Out of scope:** blocking uploads by content.
**Acceptance test:** keyboard-only check of sign-in; all text visible at phone width.

---

# Tests
Copy and layout are checked in the browser per task; one server test covers the new flag.

| # | Test | Proves |
|---|---|---|
| 1 | `test_attention_marks_destructive_calls` | Sending email is marked irreversible; reading email isn't |

---

# Open Questions
Checks for whoever builds this. None needs a decision.
1. Task 2: confirm from Composio's documentation that each connection acts as the user's own account. If it doesn't, the "as you" line changes.

---

# Implementation Notes
