# // Feature Spec: Consent Capture (NMORGAN)

*Ready 05 · Wave 2 · blocked by ready 01. Record that each user accepted the current Terms and Privacy Policy, and refuse new work until they have. It works the same whatever the documents end up saying.*

---

**Sources**
- `docs/launch/LAUNCH_READINESS_PLAN.md` §5.3; `docs/launch/DECISIONS.md` (D5, resolved)
- `harness_module/api.py` (`current_user`, `CurrentUser`, `auth_me`, `ApiError`, and the handlers in Task 3)
- `frontend/app.jsx` (`App`, `onError`), `frontend/views.jsx` (`Login`, `AuthAside`), `frontend/api.jsx` (`api.me`, `request`)
- `tests/test_api.py` (`_signed_in`, `_sign_in_as`, `_supabase_token`)

**Status:** Ready | **Author:** Vin Baker | **Last updated:** 2026-09-25

---

# Problem
Nobody agrees to anything. The Terms can't limit liability without a record that the user accepted them, and Google sign-in skips the sign-up form, so a checkbox there isn't a record. A document change has no way to reach existing users either. Success: every user who creates work has a stored acceptance of the current versions, and a version bump sends everyone back through the gate.

---

# Technical Background
Both sign-in paths end in `POST /auth/session`, then the app calls `GET /auth/me` once per load for every user, which makes it the right place to report consent state. `current_user` resolves the cookie with one `auth_sessions` lookup, and handlers take `user_id: str = CurrentUser`. A second dependency wrapping it is the smallest enforcement point. `ApiError(403, code, message)` renders `{code, message, retryable}`, and `App.onError` already special-cases `unauthenticated`. The test sign-in helpers `_signed_in` and `_sign_in_as` live in `tests/test_api.py`.

---

# Proposed Approach
1. One append-only table, `legal_acceptances`, one row per (user, document, version), with no IP or user agent.
2. `harness_module/consent.py`: current versions come from `legal:` config (ready 01); a user needs consent if either current version has no row.
3. `/auth/me` reports it, `POST /auth/consent` records it, and `ConsentedUser` refuses work-creating endpoints with `403 consent_required`.
4. A blocking `ConsentGate` after sign-in. Reads, sign-out, export and account delete stay open, so someone who refuses new terms can still leave with their data.

**Decided:** D5 → the age checkbox reads `legal.min_age` (18). D4 → no domain restriction; age self-attestation is the only eligibility check. **Default:** no foreign key on `legal_acceptances`, so records survive account deletion; deleting them too would be one migration.

**Not in scope:** age *verification*; emailing users about changes; a diff view of changes.

---

# Implementation Plan

## Task 1: Schema
**Problem:** nowhere to store an acceptance.
**Done when:** the migration applies on a fresh DB and one already migrated; a duplicate (user, document, version) insert is a no-op under `ON CONFLICT DO NOTHING`.
**Touch point:** new `db/migrations/NNNN_legal_acceptances.sql` (next free number at merge).
**Priority:** High | **Effort:** <1h | **Blockers:** none
**Out of scope:** backfill. Existing users go through the gate.
**Acceptance test:** `test_acceptance_is_recorded_once_per_version`.

```sql
CREATE TABLE legal_acceptances (
    id            BIGSERIAL   PRIMARY KEY,
    user_id       UUID        NOT NULL,   -- no FK: evidence outlives the account
    document      TEXT        NOT NULL CHECK (document IN ('terms', 'privacy')),
    version       TEXT        NOT NULL,
    age_confirmed BOOLEAN     NOT NULL,
    accepted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, document, version)
);
```

---

## Task 2: Consent module and endpoints
**Problem:** nothing decides whether a user is current, and nothing records acceptance.
**Done when:** `GET /auth/me` includes `consent: {required, terms_version, privacy_version}`; `POST /auth/consent` with current versions and `age_confirmed: true` writes two rows and returns 204; stale versions return `409 stale_version`; `age_confirmed: false` returns 422.
**Touch point:** new `harness_module/consent.py`; `harness_module/api.py` (`auth_me`, new `record_consent`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 1, ready 01
**Out of scope:** enforcement.
**Acceptance test:** `test_new_user_needs_consent`, `test_stale_version_is_refused`.

```python
def current_versions() -> dict[str, str]: ...
async def needs_consent(user_id: str) -> bool: ...
async def record(user_id: str, versions: dict[str, str], age_confirmed: bool) -> None: ...
```

---

## Task 3: Server-side enforcement
**Problem:** a client that skips the gate can still create work.
**Done when:** without current consent, `create_session`, `post_message`, `upload_file`, `create_project`, `create_folder`, `connect_server`, `respond_to_approval`, `approve_session` and `resume_session` return `403 consent_required`; every GET, `delete_auth_session`, and (when they exist) `export_me` and `delete_me` still work.
**Touch point:** `harness_module/api.py` (new `consented_user` plus `ConsentedUser`, swapped into those handlers); `tests/test_api.py` (`_sign_in_as` records consent, so existing tests keep their meaning).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2
**Out of scope:** caching. It's one indexed query, the same cost as `current_user`'s own lookup.
**Acceptance test:** `test_work_is_refused_without_consent`.

```python
async def consented_user(user_id: str = CurrentUser) -> str:
    if await consent.needs_consent(user_id):
        raise ApiError(403, "consent_required", "Accept the current Terms and Privacy Policy to continue.")
    return user_id
```

---

## Task 4: Consent gate UI
**Problem:** users have no way to accept.
**Done when:** a signed-in user with `consent.required` sees `ConsentGate` instead of the app; Continue stays disabled until the age, Terms and Privacy boxes are all ticked; after accepting, the app loads without a reload; a 403 `consent_required` anywhere re-reads `me()` and shows the gate; the sign-up form states "By creating an account you agree to the Terms and Privacy Policy" with links. When a user who accepted before sees a new version, the heading reads "We've updated our Terms / Privacy Policy" and links `changelog.html`.
**Touch point:** `frontend/views.jsx` (new `ConsentGate`, one line in `Login`); `frontend/app.jsx` (render branch, `onError`); `frontend/api.jsx` (new `api.consent`); `frontend/index.html` (`?v=`).
**Priority:** High | **Effort:** <4h | **Blockers:** Task 2
**Out of scope:** reviewing the copy with counsel (D17 covers that with the documents).
**Acceptance test:** in the browser pane: fresh user → gate → accept → app; bump `legal.terms_version` → reload → gate again.

---

# Tests

## Test 1: `test_acceptance_is_recorded_once_per_version`
**What it verifies:** recording the same versions twice leaves one row per document.
**Why this matters:** evidence mustn't depend on click count.

## Test 2: `test_new_user_needs_consent`
**What it verifies:** `/auth/me` reports `required: true` for a fresh user and `false` after `POST /auth/consent`.
**Why this matters:** the whole UI hangs on this switch.

## Test 3: `test_stale_version_is_refused`
**What it verifies:** posting last version's strings returns 409 and writes nothing.
**Why this matters:** a stale tab mustn't record acceptance of text the user never saw.

## Test 4: `test_work_is_refused_without_consent`
**What it verifies:** `POST /sessions` returns 403 `consent_required` before consent and 201 after; `GET /sessions` works either way.
**Why this matters:** the server, not the client, holds the record.

## Test 5: `test_version_bump_requires_consent_again`
**What it verifies:** with `legal.terms_version` monkeypatched forward, `needs_consent` is true again.
**Why this matters:** material changes have to reach existing users.

---

# Open Questions
1. *Verify in Task 3, not a decision:* does refusing `resume_session` strand an unattended run parked before a version bump? Check that the run stays `idle` and resumes cleanly after consent; record it in Implementation Notes.

---

# Implementation Notes
