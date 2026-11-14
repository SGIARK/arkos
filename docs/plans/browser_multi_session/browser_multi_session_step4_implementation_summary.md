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

## Stage 4 - Tell the truth in the docs

- Changes:
  - `tool_module/browser/tool.py`: the `browser_task` description no longer says "keeps the user's logins" or "acts as the signed-in user". It says each call starts a fresh, signed-out browser and that it can act on real sites. The approval advice is kept.
  - `config_module/config.yaml`: the `leases:` comment describes the browser pool and per-call slots.
  - `CLAUDE.md`: the browser paragraph names `cdp_url()` in `tool.py`, not the missing `endpoint.py`. It describes fresh browsers per call and the per-user slot cap.
  - `tests/test_browser.py`: one assertion message no longer says "signed-in user".
  - `docs/concepts/README.md` (new): entries for **Browser task** and **Resource lease**.
- Verification:
  - `git diff --check`: passed.
  - Every path, function and setting named in the changed docs and the concept entries exists: passed.
  - The tool description printed from `BrowserTask.spec` reads as intended: passed.
  - `ruff check .`, `ruff format --check .` and scoped `mypy`: passed.
  - `pytest tests/ -q --timeout=120 -m "not integration"`: passed, 901 passed and 0 failed.
- Notes:
  - Out of repo and not changed: `/home/service/arkos-project-map.md` still says the browser is "leased per user for the whole run".

## Stage 5 - End-to-end check and final review

- Changes: none to the product. A check script lives outside the repo.
- Verification:
  - **Parallel runs, real stack: passed.** A script ran two sessions of one user through the real runner (`_Sink.tool_context()`), the real `browser_task`, browser-use 0.12.9, the local Qwen3-8B model and the local Browserless. It used the throwaway test database.
    - Run 1, with the default 120s call limit: both calls started at 0.0s, and neither showed "waiting for the browser". Both were cut off at 120.1s by the registry (see the finding below). 0 browser lease rows were left, so the cancellation exit freed both slots for real.
    - Run 2, with the call limit raised to 330s for the check only: both calls started at 0.0s and overlapped for 53.8s. Neither waited, and both returned (53.8s and 77.0s). 0 browser lease rows were left.
    - Answer quality was weak, and that is the model, not this change. One run quoted body text as the heading, and the other judged itself unsuccessful.
  - **Through the HTTP API: not run.** The API server on port 1121 is the user's own process and still runs the pre-change code. Restarting it needs the user's go-ahead. Only the HTTP layer and the model's choice of tool are left unexercised.
  - **CI checks: passed.** `ruff check .`, `ruff format --check .`, scoped `mypy`, and `pytest -m "not integration"` (901 passed, 0 failed).
  - **Final diff review against `main`: passed.** Code changes stay within the five planned files. The new slot claim reuses `leases.acquire` and `_wait_for`. No unrelated files changed, and the untracked `.env.bak`, `.env.before-e2b` and `api.log` were left alone.
- Notes:
  - Before Stage 1, the local suite failed 181 tests because `tests/conftest.py` loads the real `.env` and its JWT secret. CI has no `.env`. With test secrets set, the baseline was fully green.

## Enhancement requests

- **Request:** check the Browserless SSPL licence for a hosted arkos, and pin the image version. **Disposition:** separate work. **Approval status:** not started. **Outcome:** pending.
- **Finding (Stage 5):** `tools.call_timeout_s` (120s) cuts off every tool call before the browser's own `browser.wall_clock_s` (240s) and `browser.hard_timeout_s` (300s). A slow run is cancelled from outside and loses its partial result. A call that first waits for a slot (up to `leases.wait_timeout_s`, 90s) has even less time. **Disposition:** separate work. It was there before this cycle and changes timeout policy for every tool, so it is outside the approved scope. **Approval status:** not started. **Outcome:** pending.

*— Claude Code · 2026-11-14 UTC · Model: claude-opus-5-5*
