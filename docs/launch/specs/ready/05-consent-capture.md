# // Feature Spec: Consent Capture (NMORGAN)

*Ready 05 · Wave 2 · needs ready 01. Record that each user accepted the current Terms and Privacy Policy, and hold back new work until they have.*

---

**Sources**
- Plan: `docs/launch/LAUNCH_READINESS_PLAN.md` §5.3; `docs/launch/DECISIONS.md` (D4, D5)
- Server: `harness_module/api.py`; database: `db/migrations/`
- Screens: `frontend/app.jsx`, `frontend/views.jsx`, `frontend/api.jsx`
- Tests: `tests/test_api.py` (its sign-in helpers)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-10-03

---

# Problem
Nobody agrees to anything today. The Terms can't limit arkos's liability unless there's a record that the user accepted them. Google sign-in skips the sign-up form entirely, so a checkbox there isn't enough. And when a document changes, there's no way to ask existing users again. Success: everyone who creates work has accepted the current versions, and a new version sends everyone back through the agreement screen.

---

# Technical Background
- Every sign-in, email or Google, ends with the app asking "who am I" (`GET /auth/me`) once per page load. That's the right moment to report whether consent is needed.
- Every signed-in request goes through one shared check (`current_user`). Wrapping it in a second check is the smallest way to hold back work.
- Tests sign in through two helpers in `tests/test_api.py`. They must accept consent, or every existing test would start failing.

---

# Proposed Approach
1. A new table records each acceptance: user, document, version, time, and "confirmed 18 or older". No IP address or browser details.
2. "Who am I" reports whether consent is needed; a new endpoint records it.
3. Anything that creates work is refused until the user accepts. Reading, signing out, downloading data and deleting the account stay available, so someone who declines new terms can still leave with their data.
4. A full-screen agreement step after sign-in.

**Decided:** minimum age 18 (D5); sign-up is open to anyone, so age is the only eligibility check (D4). **Default:** consent records stay after account deletion as proof of agreement.

**Not in scope:** verifying age; emailing users about changes.

---

# Implementation Plan

## Task 1: The acceptances table
**Problem:** there's nowhere to store an acceptance.
**Done when:** the migration applies on new and existing databases, and accepting the same version twice keeps one record.
**Touch point:** one new migration in `db/migrations/` (next free number).
**Priority:** High | **Effort:** <1h | **Blockers:** none
**Out of scope:** backfilling existing users; they see the agreement screen.
**Acceptance test:** Test 1.

```sql
CREATE TABLE legal_acceptances (
    id            BIGSERIAL   PRIMARY KEY,
    user_id       UUID        NOT NULL,   -- no link to users: the record outlives the account
    document      TEXT        NOT NULL CHECK (document IN ('terms', 'privacy')),
    version       TEXT        NOT NULL,
    age_confirmed BOOLEAN     NOT NULL,
    accepted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, document, version)
);
```

---

## Task 2: Record and report consent
**Problem:** nothing knows whether a user is up to date, and nothing records acceptance.
**Done when:** "who am I" says whether consent is needed and for which versions; `POST /auth/consent` with the current versions and age confirmed records it; outdated versions or an unconfirmed age are refused.
**Touch point:** new `harness_module/consent.py`; `harness_module/api.py`.
**Priority:** High | **Effort:** <4h | **Blockers:** Task 1; ready 01 (document versions)
**Out of scope:** holding back work.
**Acceptance test:** Tests 2–3.

---

## Task 3: Hold back work until accepted
**Problem:** a client that skips the screen could still create work.
**Done when:** without consent, starting or messaging a conversation, uploading, creating a project or folder, connecting an account, answering an approval and resuming a run are all refused with `consent_required`. Reading, signing out, export and account delete still work.
**Touch point:** `harness_module/api.py` (one shared check on `create_session`, `post_message`, `upload_file`, `create_project`, `create_folder`, `connect_server`, `respond_to_approval`, `approve_session`, `resume_session`); `tests/test_api.py` (sign-in helpers accept consent).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2
**Out of scope:** caching the check; it's one quick lookup.
**Acceptance test:** Test 4.

---

## Task 4: The agreement screen
**Problem:** users have no way to accept.
**Done when:** a signed-in user who hasn't accepted sees the agreement screen instead of the app; Continue stays off until all three boxes are ticked (18 or older, Terms, Privacy); accepting opens the app; after a version change the screen returns, headed "We've updated our Terms / Privacy Policy", with a link to what changed; the sign-up form says "By creating an account you agree to the Terms and Privacy Policy".
**Touch point:** `frontend/views.jsx` (new `ConsentGate`, one line on sign-up); `frontend/app.jsx`; `frontend/api.jsx`; `frontend/index.html` (bump `?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2
**Out of scope:** a side-by-side diff of changes.
**Acceptance test:** in the browser: a new user sees the screen and gets in after accepting; change the Terms version in config and the screen returns.

---

# Tests
No outside services involved.

| # | Test | Proves |
|---|---|---|
| 1 | `test_acceptance_is_recorded_once_per_version` | Clicking twice doesn't create two records |
| 2 | `test_new_user_needs_consent` | A new user is asked; after accepting, they aren't |
| 3 | `test_stale_version_is_refused` | An old browser tab can't accept text the user never saw |
| 4 | `test_work_is_refused_without_consent` | The server, not the screen, holds back work |
| 5 | `test_version_bump_requires_consent_again` | A new version reaches existing users |

---

# Open Questions
Checks for whoever builds this. None needs a decision.
1. Task 3: an autopilot run that was paused before a Terms change should resume normally once the user accepts. Confirm it does.

---

# Implementation Notes
