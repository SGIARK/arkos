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

*— Claude Code · 2026-11-14 UTC · Model: claude-opus-5-5*
