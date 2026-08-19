# ARKOS

Mid-redesign. The old architecture guidelines are gone: they mandated state
graphs, `StateOutput`, routers, and mem0, all of which this redesign deletes.
They are kept at `docs/deprecated/CLAUDE_pre_redesign.md` for history only.

**Start at `docs/single_loop_redesign_spec.md`.** It routes to everything else.

`docs/contracts.md` is law. The spec says what to build; contracts says whether
it is correct. If they disagree, contracts wins and the spec is the bug.

**Never read `docs/deprecated/`.** It is the architecture that was deleted. Do
not open it, cite it, or follow its task numbers. Stale pointers to it in
docstrings should be deleted, not followed.

**The old architecture is GONE as of 2026-08-13** (Tasks 7 and 8, pulled forward).
`state_module`, `memory_module` and `computer_module` no longer exist, and
neither do `app.py`, `task_runner.py` or `ArkModelNew.py`. If something tells you
to edit one of those, that instruction is stale — check `git log`, do not
recreate the file.

**The HTTP server is `harness_module/api.py`** (Task 4), run with
`uvicorn harness_module.api:app`. `harness_module/` is the control plane:
api · runner · store · workspace · leases · lifecycle · approvals · session_log ·
system_log · stream · hands · jwt_utils.

**Deleted code is still documentation.** When rebuilding something 8.10 removed,
read the deleted implementation in git history for operational facts contracts
does not carry — connection URLs, protocol choices, hard-won workarounds.
Contracts states the invariants; the old code knows the wiring. (Task 9 was
rebuilt "against contracts, not those files" and silently lost the Browserless
CDP connection, because no document said where the browser runs.)

**The browser is `tool_module/browser/`** (Task 9), and it runs in the
browserless container from `docker-compose.yml` — reached only over CDP at
`browser.cdp_url`, defaulting from `BROWSERLESS_URL`. It never launches a
browser in the harness process. The pre-redesign `browser_tool.py`,
`browser_actions.py`, `browser_stream.py` and `browser_routes.py` are deleted
(8.10); they are in `git log` and are still worth reading for wiring.

Where the live code is: `agent_module/loop.py` (the one loop),
`model_module/client.py` (the one model client), `tool_module/`
(envelope · registry · connections · smithery · tools/ · sandbox),
`db/pool.py` (asyncpg; the psycopg2 helpers are gone — do not add more).

Coding standards still need the rewrite Task 7 promised. Until then: ruff, type
hints on every signature, `async def` for anything that awaits, no blocking IO in
an async path, no `print()` in production paths.
