# // Feature Spec: Operator Documents (NMORGAN)

*Drafting 02 · can start now. The written program behind "reasonable security": what we'd do in a breach, who may see user data, and what we keep. Drafted with placeholders; finalized once D2 and D9 are resolved.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §4.3, §7.12, §8, §9
- `docs/launch/DECISIONS.md` (D1, D2, D7, D9, D19)
- `harness_module/api.py` (`lifespan`, `_keys_and_sweep`), `harness_module/system_log.py` (`prune`)
- `.env.example` (the secret names to rotate), `config_module/config.yaml`

**Status:** Drafting | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
Massachusetts 201 CMR 17.00 requires a *written* information security program from anyone holding MA residents' personal information, and M.G.L. c. 93H sets breach-notice duties. arkos has neither document, no written rule on who may read user data, and no record of what enforces each retention promise. A breach today would be handled from memory. Success: five documents exist in `docs/launch/internal/`, each complete except for `{{…}}` names and dates and `data-decision` markers tied to `DECISIONS.md`.

---

# Technical Background
The containment levers already exist, and the runbook names them:
- Rotating `ARKOS_SESSION_SECRET` invalidates every cookie (every cookie carries `iss=arkos`, signed with that secret).
- `DELETE FROM auth_sessions` revokes sessions immediately, because `current_user` checks the row on every request.
- `POST /sessions/{id}/cancel` stops runs.
- Composio, e2b, SerpAPI, OpenAI and Supabase keys come from the environment (see `.env.example`), so rotation is an env change plus a restart.

Retention enforcement lives in code: `system_log.prune` (30 days, at boot), the auth-session sweep in `_keys_and_sweep`, and, once built, ready 06's `purge_deleted_files` and ready 05's `delete_account`.

---

# Proposed Approach
Markdown in the repo, beside the plan, not public pages. Each document is short and operational: steps a stressed person can follow, not policy prose. Names, phone numbers and dates are `{{…}}` placeholders. D2 decides whether MIT's own security program applies, so the WISP opens with a `data-decision="D2"` note.

**Not in scope:** public security text (drafting 01 Task 4), and completing the documents (owner and counsel).

---

# Implementation Plan

## Task 1: WISP and incident response
**Problem:** there's no written security program and no breach runbook.
**Done when:** `WISP.md` has a heading for each 201 CMR 17.00 element in §7.12 (responsible person, risk assessment, access control, encryption in transit, vendor oversight referencing §4.2, monitoring, training, disposal, annual review). `INCIDENT_RESPONSE.md` has severity levels, a containment checklist using the levers above, evidence preservation, the c. 93H notice steps (affected residents, the Attorney General, the Office of Consumer Affairs and Business Regulation), other-state and GDPR 72-hour notes, and a user notice template.
**Touch point:** new `docs/launch/internal/WISP.md`, `docs/launch/internal/INCIDENT_RESPONSE.md`.
**Priority:** High | **Effort:** 1 day | **Blockers:** none (D2 appears as a marker)
**Out of scope:** tabletop exercises.
**Acceptance test:** a reviewer can map each §7.12 bullet to a heading, and every command in the containment checklist exists in the repo or `.env.example`.

---

## Task 2: Access policy, retention record, launch checklist
**Problem:** there's no rule for insider access, no map from retention promises to the code that keeps them, and no tracked checklist.
**Done when:** `DATA_ACCESS_POLICY.md` implements D9's default (named operators, allowed reasons, logging each access, no personal copies, offboarding) with `{{…}}` names; `RETENTION.md` has one row per §4.3 line naming the enforcing function, endpoint or config key, or "not yet enforced — deferred 01" (the inactive-account sweep, post-launch); `LAUNCH_CHECKLIST.md` copies plan §9, adds the D19 spend caps and a "register a DMCA designated agent with the US Copyright Office (recommended, not blocking)" line, and has columns for who and when.
**Touch point:** new `docs/launch/internal/{DATA_ACCESS_POLICY,RETENTION,LAUNCH_CHECKLIST}.md`.
**Priority:** High | **Effort:** <4h | **Blockers:** none (rows for ready 05 and 06 name their planned functions)
**Out of scope:** audit tooling for access logs.
**Acceptance test:** every function or config key named in `RETENTION.md` exists in the repo, or is marked as belonging to an unmerged spec by its number.

---

# Tests
None automated. These are operator documents, checked by the reviews in each acceptance test. Ready 10 blocks launch while D9 and D19 are open.

---

# Open Questions
None. D2, D7, D9 and D19 are tracked in `DECISIONS.md`.

---

# Implementation Notes
