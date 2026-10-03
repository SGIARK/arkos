# // Feature Spec: Inactive Account Sweep (NMORGAN)

*Deferred 01 · post-launch. Not needed to launch, so arkos keeps inactive accounts until their owners delete them. Written for later: 12 months of no sign-in, a 30-day notice, then deletion. It needs an email sender (D16) before it can run.*

---

**Sources**
- `docs/launch/DECISIONS.md` (D16)
- `harness_module/api.py` (`create_auth_session`, the `INSERT INTO users … ON CONFLICT` upsert; `lifespan`)
- `db/migrations/0000_migration_0.sql` (`users`)
- `harness_module/account.py` (`delete_account`, from ready 05); `harness_module/retention.py` (from ready 06)

**Status:** Deferred (post-launch) | **Author:** Vin Baker | **Last updated:** 2026-10-02

---

# Problem
Accounts nobody uses keep conversations, files, memory and live Composio grants forever. That's more data to protect and to lose in a breach. Success: an account with no sign-in for the configured window gets one notice email, and is deleted with `delete_account` if nobody signs in within 30 days.

---

# Technical Background
`users` has `created_at` but no last-activity column. `create_auth_session` upserts `users` on every sign-in, so stamping `last_sign_in_at` there costs nothing extra. It's per sign-in, not per request, because `current_user` runs on every call and must stay a single read. `delete_account` (ready 05) is idempotent and does the whole job. The retention module (ready 06) is the run-at-boot home for sweeps. arkos sends no email of its own today: Supabase Auth sends sign-up and reset mail, and nothing else exists. That's why D16 blocks the notice.

---

# Proposed Approach
Two columns, one sweep, run at boot:
- Stamp `last_sign_in_at` at sign-in.
- The sweep notices accounts past the window (sets `deletion_notice_at` and sends the email), then deletes accounts whose notice is older than 30 days with no later sign-in.
- Signing in clears the notice.
- The sweep is off while `retention.inactive_account_days` is null.

**Not in scope:** choosing an email provider (D16); any account without an email address (none exist, since both sign-in paths require one).

---

# Implementation Plan

## Task 1: Track last sign-in
**Problem:** there's no activity signal.
**Done when:** each `POST /auth/session` sets `users.last_sign_in_at = now()` and clears `deletion_notice_at`; existing rows are backfilled with `created_at`.
**Touch point:** new `db/migrations/NNNN_last_sign_in.sql`; `harness_module/api.py` (`create_auth_session` upsert).
**Priority:** Low | **Effort:** <2h | **Blockers:** none (safe to land early)
**Out of scope:** the sweep.
**Acceptance test:** `test_sign_in_stamps_and_clears_notice`.

```sql
ALTER TABLE users ADD COLUMN last_sign_in_at TIMESTAMPTZ;
ALTER TABLE users ADD COLUMN deletion_notice_at TIMESTAMPTZ;
UPDATE users SET last_sign_in_at = created_at WHERE last_sign_in_at IS NULL;
```

---

## Task 2: Notice and sweep
**Problem:** nothing acts on inactivity.
**Done when:** with `retention.inactive_account_days` set, the boot sweep sends one notice to each account past the window (and never a second), and calls `delete_account` for each account noticed more than 30 days ago with no sign-in since; with it null, the sweep does nothing.
**Touch point:** `harness_module/retention.py` (new `sweep_inactive_accounts`); `harness_module/api.py` (`lifespan`); `config_module/config.yaml` (`retention.inactive_account_days: null`, `retention.inactive_notice_days: 30`); the mail sender chosen in D16.
**Priority:** Low | **Effort:** 1 day | **Blockers:** D16 (an email sender), ready 05, ready 06
**Out of scope:** a reminder email.
**Acceptance test:** `test_sweep_notices_then_deletes`, `test_sweep_is_off_by_default`.

```python
async def sweep_inactive_accounts(now: datetime | None = None) -> tuple[int, int]:
    """(noticed, deleted). Off while retention.inactive_account_days is null."""
```

---

# Tests

## Test 1: `test_sign_in_stamps_and_clears_notice`
**What it verifies:** a sign-in sets `last_sign_in_at` and nulls `deletion_notice_at`.
**Why this matters:** coming back must cancel the countdown.

## Test 2: `test_sweep_notices_then_deletes`
**What it verifies:** with `now` injected, an inactive account gets exactly one notice; 31 days later `delete_account` is called once; an account that signed in after the notice isn't touched.
**Why this matters:** deleting without notice, or deleting someone who came back, is the failure.

## Test 3: `test_sweep_is_off_by_default`
**What it verifies:** with `inactive_account_days: null`, no notice is sent and nothing is deleted.
**Why this matters:** it ships dark, and the window is a config value to turn on later.

---

# Open Questions
None. When this is picked up after launch, the Privacy Policy retention section gains one sentence about inactive accounts.

---

# Implementation Notes
