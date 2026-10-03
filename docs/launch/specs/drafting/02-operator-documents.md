# // Feature Spec: Operator Documents (NMORGAN)

*Drafting 02 · can start now. The team's own rulebook: how arkos protects data, what to do in a breach, who may see user data, and what is kept. Finished once D2 and D9 are answered.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §4.3, §7.12, §8, §9; `docs/launch/DECISIONS.md` (D2, D7, D9, D19)
- Breach controls: `harness_module/api.py`, `harness_module/system_log.py`, `config_module/config.yaml`, `harness_module/blobs.py`, `CLAUDE.md`

**Status:** Drafting | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
Massachusetts law (201 CMR 17.00) requires a written information security program from anyone holding residents' personal information, and another law (M.G.L. c. 93H) says who must be told after a breach, and how fast. arkos has neither document, no written rule on who may read user data, and no record of what enforces each promise about keeping data. A breach today would be handled from memory. Success: five short documents in `docs/launch/internal/`, complete except for names and dates.

---

# Technical Background
The tools to contain a breach already exist:
- Changing `ARKOS_SESSION_SECRET` signs everyone out at once.
- Deleting a user's login sessions from the database ends them immediately.
- A conversation's "cancel" stops a run.
- Outside-service keys are environment settings, so replacing one is an edit and a restart: `SUPABASE_SECRET_KEY` (or the older `SUPABASE_SERVICE_KEY`), `E2B_API_KEY`, `COMPOSIO_API_KEY`, `OPENAI_API_KEY`, `SERPAPI_API_KEY`, `BROWSERLESS_URL`.

What enforces data-keeping today: logs are trimmed to 30 days at startup, and expired logins are swept regularly. Once ready 02 lands, so are the 30-day undo cleanup and account deletion.

---

# Proposed Approach
Markdown files next to the plan, not public pages. Each is short and practical: steps someone under pressure can follow, not policy prose. Names, phone numbers and dates are `{{placeholders}}`. The security program opens with a note on D2, since an MIT-run service might fall under MIT's own program instead.

**Not in scope:** public security text (drafting 01); filling in the names (owner and lawyer).

---

# Implementation Plan

## Task 1: Security program and breach runbook
**Problem:** there's no written security program and no breach plan.
**Done when:** `WISP.md` has a section for each element the regulation requires (listed in plan §7.12); `INCIDENT_RESPONSE.md` has severity levels, a containment checklist using the tools above, how to preserve evidence, who must be notified in Massachusetts (affected residents, the Attorney General, the consumer affairs office), notes for other states and the EU's 72-hour rule, and a template notice to users.
**Touch point:** new `docs/launch/internal/WISP.md`, `docs/launch/internal/INCIDENT_RESPONSE.md`.
**Priority:** High | **Effort:** 1 day | **Blockers:** none (D2 appears as a note)
**Out of scope:** practice drills.
**Acceptance test:** every item in plan §7.12 maps to a section, and every command and setting named in the checklist exists in the repo.

---

## Task 2: Data-access policy, retention record, launch checklist
**Problem:** there's no rule for staff access to user data, no map from data-keeping promises to the code that keeps them, and no tracked launch checklist.
**Done when:** `DATA_ACCESS_POLICY.md` follows D9's default (named people only, for listed reasons, every access logged, no personal copies, access removed when someone leaves); `RETENTION.md` has one row per line of plan §4.3 naming what enforces it (inactive accounts: ready 02 Part D, after launch); `LAUNCH_CHECKLIST.md` copies plan §9, adds the spend caps (D19) and "register a copyright agent with the US Copyright Office (recommended)", with who and when columns.
**Touch point:** new `docs/launch/internal/DATA_ACCESS_POLICY.md`, `RETENTION.md`, `LAUNCH_CHECKLIST.md`.
**Priority:** High | **Effort:** <4h | **Blockers:** none
**Out of scope:** tooling to audit access logs.
**Acceptance test:** everything `RETENTION.md` names either exists in the repo or is marked as planned in ready 02.

---

# Tests
None automated; these are documents, checked by the reviews in each task. Ready 09 blocks launch while D9 and D19 are open.

---

# Open Questions
None. D2, D7, D9 and D19 are tracked in `DECISIONS.md`.

---

# Implementation Notes
