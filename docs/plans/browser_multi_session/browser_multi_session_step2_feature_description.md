> **Feature plan:** [Step 1](./browser_multi_session_step1_solution_assessment.md) · [Step 2](./browser_multi_session_step2_feature_description.md) · [Step 3](./browser_multi_session_step3_development_plan.md) · [Step 4](./browser_multi_session_step4_implementation_summary.md)

Debug mode: disabled

Step 1 approved and committed in `085c097`. Chosen direction: Option A.

## Problem

Only one session per user can use `browser_task`, because that session holds the browser lease until it ends. Other sessions wait 90 seconds and then fail, although each run gets its own throwaway browser.

## User stories

- As a user with two sessions open, I want both to browse at once so that one slow task does not stall the other.
- As a user, I want a session that finished browsing to free its browser at once so that my other sessions are not blocked by its later work.
- As an operator, I want a cap on browsers per user so that one user cannot fill every Browserless slot.
- As a contributor, I want the docs to describe the browser truthfully so that nobody builds on logins that do not persist.

## Core requirements

- Sessions of one user run `browser_task` in parallel, up to a per-user cap.
- The cap is a config setting, `browser.max_concurrent_per_user`, defaulting to 3.
- A session holds a browser slot only while its `browser_task` call runs. Every exit frees the slot: success, failure, budget stop, backstop and cancellation.
- A call above the cap waits and says so, as other contended calls do today. If no slot frees up in time, it fails with a message that says it is safe to retry.
- The tool description, `config.yaml` and `CLAUDE.md` stop promising kept logins and stop describing a per-user lease held for a whole run. `CLAUDE.md` names `tool.py`, not the missing `endpoint.py`.

## Delivery scope

- Work type: application change, plus the documentation fixes above.
- Out of scope: persistent profiles (Option C), a backend change away from Browserless, and the licence check. These are recorded as separate work below.

## Completion boundary

- Normal entry: a model in any session calls `browser_task`.
- End-to-end outcome: two sessions of one user each finish a browser task, and their runs overlap in time.
- Recovery: if a process dies holding a slot, the slot frees itself when its expiry passes. A call over the cap gets a clear "busy, safe to retry" result.
- Release condition: the CI lint and test stages pass, and a manual check against the local Browserless container shows overlapping runs.

## Risks

- **A slot leaks on an unusual exit path.** Impact: a user loses browser capacity for up to 15 minutes. Earliest validation: tests that end a run each way and then check that the slot is free. Mitigation: free the slot in the tool's own cleanup, with the existing session-end release kept as a second safety net.
- **Two sessions take the last slot at the same moment.** Impact: the user exceeds the cap. Earliest validation: a test with two parallel claims at the cap. Mitigation: claim through the existing atomic lease insert.
- **Users relied on logins persisting.** Impact: low, because they never persisted. Earliest validation: the code review in this plan. Mitigation: the corrected tool description tells the model to sign in when needed.

## Shared component inventory

- `harness_module/leases.py` — the atomic, expiring lease. **Reuse:** a slot can be one of N lease keys per user, so no new table or migration is needed.
- `harness_module/runner.py` `_lease`, `_acquire`, `_wait_for` — the wait, the "waiting for…" status and the timeout. **Extend** with a way to give a lease back before the session ends.
- `tool_module/sandbox/manager.py` `claim_slot` — the sandbox's per-user cap. **Pattern only:** it uses its own table and is tied to e2b, so it is not reused directly.
- `tool_module/envelope.py` `ToolContext.lease` — how a tool asks for a resource. **Extend** with a matching release.
- `tool_module/browser/tool.py` `BrowserTask.call` — the caller. **Change** it to claim a slot and free it on every exit.

## User flow

1. Session A calls `browser_task` and takes a free slot.
2. Session B calls `browser_task` while A runs, takes a second slot and runs at the same time.
3. Each run ends, and its slot frees immediately.
4. A fourth parallel call by the same user, with a cap of 3, waits with a "waiting for the browser" status, then runs or fails as safe to retry.

## Success criteria

- Two sessions of one user complete overlapping browser runs locally, with no "waiting for the browser" status.
- With a cap of N, call N+1 waits, and it runs as soon as one slot frees.
- After every exit path, the user holds no browser slot (tests cover each path).
- `ruff check`, `ruff format --check`, the scoped `mypy` and the non-integration `pytest` run all pass.

## Concepts touched

No inventory exists in `docs/concepts/`. Proposed entries, to be written during implementation:

- **Browser task** (proposed): changed. It now holds a slot per call.
- **Resource lease** (proposed): changed. It gains early release and slot keys.

## Enhancement requests

- **Request:** check whether the Browserless SSPL licence fits a hosted arkos, and pin the image version. **Disposition:** separate work, out of this cycle's scope. **Approval status:** not started. **Outcome:** pending.

*— Claude Code · 2026-11-14 UTC · Model: claude-opus-5-5*
