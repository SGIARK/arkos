> **Feature plan:** [Step 1](./browser_multi_session_step1_solution_assessment.md) · [Step 2](./browser_multi_session_step2_feature_description.md) · [Step 3](./browser_multi_session_step3_development_plan.md) · [Step 4](./browser_multi_session_step4_implementation_summary.md)

Debug mode: disabled

Step 2 approved and committed in `c8b4af6`. Branch: `fix/browser-multi-session`.

## Completion Contract

- Normal entry: a model in any session calls `browser_task`.
- End-to-end outcome: two sessions of one user each finish a browser task, and their runs overlap in time.
- Required recovery: a slot frees on every exit of the call. A crashed process's slot frees itself when its expiry passes. A call over the cap waits, then fails as safe to retry.
- External verification: two overlapping runs against the local Browserless container, driven through the API.
- Release condition: the four CI checks pass locally (`ruff check .`, `ruff format --check .`, the scoped `mypy`, and the non-integration `pytest`), and the manual check passes.

## Key Risks

- **High risk:** a slot leaks on an unusual exit path. Impact: the user loses a browser for up to 15 minutes (`leases.ttl_s`). Early validation: Stage 3 tests end a run each way. Mitigation: release in the tool's `finally`, and keep the release at session end as a backup.
- Two sessions take the last slot at once. Impact: the user exceeds the cap. Early validation: a parallel-claim test in Stage 2. Mitigation: each slot is one row with an atomic insert in `resource_leases`.
- One session holds two slots. A turn runs its tool calls one after another (`runner.py` starts no parallel tool tasks), so this cannot happen today. Stage 2 adds a test that pins this behavior.
- Old `browser:<user_id>` rows from before the change. Impact: none, because nothing reads them, and they expire within 15 minutes. No migration is needed.

## Stage 1 — Give a lease back before the session ends

- Goal: a tool can release a lease it took, without ending the session.
- Dependencies: none.
- Expected changes: `ToolContext` gains `unlease: Callable[[str], Awaitable[None]] | None`. The runner gains `async def _unlease(self, resource: str) -> None`. It deletes the session's row through `leases.release` and drops the key from `self._leases`. `tool_context()` wires it in.
- Verification approach: new tests in `tests/test_leases.py` show that after an early release, a second session takes the resource at once, and that the session-end release still runs cleanly.
- Risks or open questions: a release that fails must not fail the tool call. Log it, and leave the expiry and the session-end backup to free the slot.
- Canonical components/API contracts touched: `tool_module/envelope.py` `ToolContext` (extended), `harness_module/runner.py` `tool_context`, `_leases` (extended), `harness_module/leases.py` `release` (reused).

## Stage 2 — Claim one of N browser slots per user

- Goal: up to N sessions of one user hold a browser at once.
- Dependencies: Stage 1.
- Expected changes: `leases.py` gains `def slot_keys(resource: str, user_id: str, count: int) -> list[str]`, which returns `browser:<user_id>:0` to `:N-1`. For `"browser"`, the runner tries those keys in order through the existing `_wait_for`, and takes the first free one. It remembers which key it holds, so `_unlease("browser")` frees exactly that key. A new setting, `browser.max_concurrent_per_user: 3`, goes in `config.yaml`, read with a default of 3.
- Verification approach: tests show three sessions hold slots together, a fourth waits and then times out with the "safe to retry" message, a freed slot goes to the waiter, and parallel claims at the cap grant exactly N. A test also pins that one turn runs tool calls one after another.
- Risks or open questions: a cap of 0 or less would block every call. Treat values below 1 as 1, and log a warning.
- Canonical components/API contracts touched: `harness_module/leases.py` (`key` reused, `slot_keys` new), `harness_module/runner.py` `_lease`, `_acquire`, `_wait_for` (reused), `config_module/config.yaml` `browser:` section. The sandbox pool's `claim_slot` is a pattern only; its table is tied to e2b.

## Stage 3 — Free the browser slot on every exit

- Goal: `browser_task` holds its slot only while it runs.
- Dependencies: Stage 2.
- Expected changes: `BrowserTask.call` calls `ctx.unlease("browser")` in its outer `finally`, after `run.close()`. That covers success, run failure, the time-budget stop, the hard timeout, a missing `browser_use` and cancellation. Early refusals happen before the claim, so they hold nothing.
- Verification approach: tests in `tests/test_browser.py` use the fake agent seam (`_agent_factory`) and check that the slot is released once on each exit path. The test at line 284 changes to expect a claim and then a release.
- Risks or open questions: on cancellation, the `finally` must still await the release. Test it with a cancelled task.
- Canonical components/API contracts touched: `tool_module/browser/tool.py` `BrowserTask.call`, `ToolContext.unlease` (from Stage 1).

## Stage 4 — Tell the truth in the docs

- Goal: the docs and the model's tool description match the new behavior.
- Dependencies: Stage 3.
- Expected changes: the `browser_task` description drops "keeps the user's logins" and says each run starts signed out. `config.yaml` (`leases:` and `browser:` comments) and `CLAUDE.md` describe per-call slots with a per-user cap. `CLAUDE.md` names `tool_module/browser/tool.py` instead of the missing `endpoint.py`. New `docs/concepts/` entries cover **Browser task** and **Resource lease**.
- Verification approach: `git diff --check`, and a check that every path named in the changed docs exists. Re-read the tool description in the manifest test output.
- Risks or open questions: none material.
- Canonical components/API contracts touched: `BrowserTask.spec`, `CLAUDE.md`, `config_module/config.yaml`, `docs/concepts/` (new).

## Stage 5 — End-to-end check and final review

- Goal: show overlapping runs for real, and pass CI locally.
- Dependencies: Stages 1–4.
- Expected changes: none planned. The Step 4 summary records the results.
- Verification approach: run the four CI checks with the `arkos` conda env and Postgres. Start the Browserless container, then start two sessions for one user through the API, each running a short `browser_task`. Show from the event streams that the runs overlap and neither shows "waiting for the browser". Then confirm that `resource_leases` holds no browser rows. Review the full diff for scope and duplication.
- Risks or open questions: signing in to the API may need a test user. If the manual check is blocked, record it as blocked, not passed.
- Canonical components/API contracts touched: none new.

## Concepts touched

- **Browser task** (proposed `docs/concepts/` entry): changed, created in Stage 4.
- **Resource lease** (proposed entry): changed, created in Stage 4.

## Enhancement requests

- **Request:** Browserless licence check and image version pin. **Disposition:** separate work. **Approval status:** not started. **Outcome:** pending.

*— Claude Code · 2026-11-14 UTC · Model: claude-opus-5-5*
