> **Feature plan:** [Step 1](./browser_multi_session_step1_solution_assessment.md) · [Step 2](./browser_multi_session_step2_feature_description.md) · [Step 3](./browser_multi_session_step3_development_plan.md) · [Step 4](./browser_multi_session_step4_implementation_summary.md)

Debug mode: disabled

## Original Query

"I want you to look into why browser use is currently not multi-session, meaning one session makes all the other sessions wait for it?"

"Can you please write Step 1 for fixing the browser use?"

## Understood Intent

Several sessions of one user should run `browser_task` at the same time. Today the second session waits up to 90 seconds and then fails.

## Problem

`browser_task` takes a per-user lease, `browser:<user_id>`, and the session keeps it until the session ends or parks, but no browser state survives between runs to justify it.

Evidence behind that sentence:

- `tool_module/browser/tool.py` claims the lease before each run, and `harness_module/leases.py` keys it per user ("one browser per user").
- `harness_module/runner.py` releases leases only at the end of a session or at a park, not after the tool call.
- Each run opens a fresh Browserless connection with no profile or reconnect setting, then closes it. The log shows `Cleared all owned data` after a run.
- Browserless already allows 10 browsers at once (`CONCURRENT` in `docker-compose.yml`).
- The tool description, `config.yaml` and `CLAUDE.md` promise a browser "that keeps the user's logins", and the code does not keep them.

## Concepts touched

No concept inventory exists yet in `docs/concepts/`.

- **Browser task** (proposed entry): changed by every option.
- **Resource lease** (proposed entry): changed by A and B, used unchanged by C.
- **Browser profile** (proposed entry): created by C only.

## Option A — Drop the browser lease and cap browsers per user

Remove the browser lease. Each run already gets its own throwaway browser, so sessions stop blocking each other. A per-user cap, modeled on the sandbox pool, keeps one user from taking every Browserless slot. Fix the text that promises kept logins.

- Pro: sessions run in parallel, which is the requested outcome.
- Pro: it matches what the code already does, so behavior inside one run does not change.
- Pro: it is a small change with no database migration. The existing lease table stays.
- Con: a user's browser still keeps no logins. The fix removes that promise but does not deliver it.
- Con: it needs a new per-user count of open browsers, and that count must recover from a crashed process.

## Option B — Keep the lease, but release it when the call ends

Keep the per-user lease, and release it in the tool's own cleanup rather than at session end. Sessions still take turns, but only for one run of up to 5 minutes each.

- Pro: it is the smallest change, in one or two files.
- Pro: it leaves room for a shared profile later.
- Con: sessions still wait for each other, so it does not meet the request.
- Con: a second waiting session can still time out after 90 seconds, because one run lasts up to 300 seconds.

## Option C — Add a persistent profile for each user

Give each user a stored Browserless profile, so logins really persist between runs. Two browsers cannot safely share one profile directory. Each session would therefore either take a short lease, as in B, or start from a copy of the profile.

- Pro: it delivers the logged-in browsing that the tool description promises.
- Con: it is much larger, with new storage, cleanup and Browserless settings that need verifying.
- Con: it stores user cookies, which raises security and retention questions.
- Con: the parallel copy variant loses logins made in parallel sessions, and the shared variant still waits.

## Recommendation

Choose **Option A** as this cycle.

It removes the wait and fixes the false promise, and it ships on its own as a vertical slice. Option C can be a separate follow-up if persistent logins become a requirement. Option B does not deliver the requested outcome, so it is only a stopgap.

Open question for Step 2: what per-user cap should apply? A default of 3 browsers per user would leave Browserless slots for other users.

*— Claude Code · 2026-11-14 UTC · Model: claude-opus-5-5*
