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

**There is no HTTP server right now.** Task 4 writes it. `base_module/` holds
only `jwt_utils.py` and `browser_routes.py`.

Where the live code is: `agent_module/loop.py` (the one loop),
`model_module/client.py` (the one model client), `tool_module/`
(envelope · registry · connections · smithery · tools/ · browser · sandbox),
`db/pool.py` (asyncpg; the psycopg2 helpers are gone — do not add more).

Coding standards still need the rewrite Task 7 promised. Until then: ruff, type
hints on every signature, `async def` for anything that awaits, no blocking IO in
an async path, no `print()` in production paths.
