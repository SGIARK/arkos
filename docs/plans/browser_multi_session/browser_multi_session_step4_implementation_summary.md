> **Feature plan:** [Step 1](./browser_multi_session_step1_solution_assessment.md) · [Step 2](./browser_multi_session_step2_feature_description.md) · [Step 3](./browser_multi_session_step3_development_plan.md) · [Step 4](./browser_multi_session_step4_implementation_summary.md)

Debug mode: disabled

Branch: `fix/browser-multi-session`. Planning commits: Step 1 `085c097`, Step 2 `c8b4af6`, Step 3 `1ddcce8`. The gate check found all three on the branch with only the plan files and the index.

## Verification setup

- Tools: a separate Python 3.11 venv with the pinned `requirements-dev.txt` (ruff 0.16.4, mypy 2.3.1), matching CI. The project `.venv` runs Python 3.14 and ruff 0.16.8, so it would not match CI.
- Database: a throwaway `postgres:15` container (`arkos-ci-pg`, port 55432) with `db/migrate.py` applied, matching the CI service.
- Environment: test-only values for `SUPABASE_JWT_SECRET`, `ARKOS_SESSION_SECRET` and the Supabase keys. `tests/conftest.py` loads the real `.env`, and its real JWT secret failed 181 tests before any change. CI has no `.env`.
- Baseline before Stage 1: ruff, format and mypy passed. The test suite had 884 passed and 0 failed.

## Stage 1 - Give a lease back before the session ends

- Changes:
  - `tool_module/envelope.py`: `ToolContext` gains `unlease: Callable[[str], Awaitable[None]] | None`.
  - `harness_module/runner.py`: new `_Sink._unlease(resource)`, wired into `tool_context()`. It releases the session's own row through `leases.release` and ignores the sandbox. It drops the key from `_leases` only after a successful release.
  - `tests/test_leases.py`: three tests. An early release goes to another session at once. A failed release keeps the session-end backup working. Releasing something never taken does nothing.
- Verification:
  - `ruff check .`: passed.
  - `ruff format --check .`: passed.
  - Scoped `mypy`: passed.
  - `pytest tests/ -q --timeout=120 -m "not integration"`: passed, 887 passed and 0 failed.
- Notes:
  - Found while writing it: `_release_leases` skips `release_all` when `_leases` is empty. Dropping the key before a failed release would have disarmed the backup, so the key is dropped only on success. A test pins this.

## Stage 2 - Claim one of N browser slots per user

- Changes:
  - `harness_module/leases.py`: new `slot_keys(resource, user_id, count)`, which returns `browser:<user>:0` to `:N-1`. A count below 1 becomes 1. The `key` docstring no longer says "one browser per user".
  - `harness_module/runner.py`: new `_claim_browser`. It tries each slot key through the existing atomic `leases.acquire`, inside the existing `_wait_for`. It remembers the key in a new `_slots` map, so `_unlease("browser")` frees exactly that key. `_release_leases` clears `_slots`. A full pool fails with "No browser was free: this account already runs N at once… safe to retry".
  - `config_module/config.yaml`: new `browser.max_concurrent_per_user: 3`.
  - `tests/test_leases.py`: six tests. Two sessions share the browser and a third is refused. A freed slot goes to the waiter. Five parallel claims at a cap of 2 grant exactly 2. Other users do not count. A cap of 0 acts as 1. The session-end release frees the slot.
  - `tests/test_browser.py`: one test pins that `browser_task` is not `readonly`, so the loop never runs two of them in one batch.
- Verification:
  - `ruff check .`: passed.
  - `ruff format --check .`: passed.
  - Scoped `mypy`: passed.
  - `pytest tests/ -q --timeout=120 -m "not integration"`: passed, 894 passed and 0 failed.
- Notes:
  - Correction to a Step 3 risk: the plan said the runner starts no parallel tool tasks. In fact `agent_module/loop.py` runs readonly calls in parallel. `browser_task` is not readonly, so it always runs alone, and the new test pins that.
  - A resumed session with an empty `_slots` map might take a second slot while an old row of its own survives. `release_all` at session end clears both, so the cost is one slot for the rest of that session, at most.
  - The browser tool still holds its slot until session end. Stage 3 releases it per call.

## Stage 3 - Free the browser slot on every exit

- Changes:
  - `tool_module/browser/tool.py`: `BrowserTask.call` claims the slot, then runs a new `_drive` method inside `try`/`finally`. The `finally` calls `ctx.unlease("browser")` after the run's own cleanup has closed the browser. `_drive` takes the checked `session_id` as a `str` instead of using an `assert`. The comment that said the browser "keeps its profile between calls" is gone.
  - `tests/test_browser.py`: the old "leased before it is driven" test became eight tests. They cover success, an unsuccessful run, a vendor exception, a missing `browser_use`, the time-budget stop, the hard backstop and cancellation, each claiming and freeing the slot once. One more checks that a refused call claims nothing. The unused `_done` helper is removed.
- Verification:
  - Mutation check: with the previous `tool.py`, 7 of the new tests failed. With the change, all 31 browser tests passed.
  - `ruff check .`: passed.
  - `ruff format --check .`: passed.
  - Scoped `mypy`: passed.
  - `pytest tests/ -q --timeout=120 -m "not integration"`: passed, 901 passed and 0 failed.
- Notes:
  - The result is built after the slot is freed. Building it only stores a blob and does not need the browser.

*— Claude Code · 2026-11-14 UTC · Model: claude-opus-5-5*
