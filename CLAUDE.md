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
(envelope · registry · connections · session_tools · smithery · tools/ ·
sandbox), `db/pool.py` (asyncpg; the psycopg2 helpers are gone — do not add
more).

**A session reaches only the MCP servers it was given** (11.4 + 11.5). The
toggles are `session_tools`, keyed by `mcp_url` and never by the `mcp_servers:`
config label. `registry.manifest` is the ONE builder of a turn's tool list and
it cannot exceed `llm.max_tools` whatever the toggles say — whole servers are
benched, most-recently-enabled first, and a benched server gets a `status` event
and a `system_events` row. **The system prompt is generated from the manifest
that shipped, never from the toggles**, which is why `_drive` builds the
manifest before it folds. Do not add a second path that assembles tool specs.

**A gated tool call PARKS the turn on itself** (11.7). `requires_approval` with
no grant leaves that call OPEN in the transcript, and the `approvals` row of
kind `call` carries the real `(tool_name, tool_args)` — consent binds to the
call, never to prose about it. Answering is `approve`/`decline`, and approving
runs that exact call once through normal dispatch, latched by `consumed_at`.
Never re-run a consumed-but-unclosed call: repair it as interrupted.

**The designs live under `designs/`** — checked-in copies of the Claude Design
project, one directory per canvas (`new-frontend/` is the 11.4 frame,
`planning-card/`, `filesystem_revamp/`, `settings-usage/`, `sign-up/`). Where a
design and `frontend/` disagree, the design wins and `frontend/` is amended, not
the other way round.

**An export is re-drafted in place, so check its date before trusting a
reading.** `designs/filesystem_revamp/` gained rename mid-build on 2026-08-20
and delete plus undo on 2026-08-21, each time overwriting the same file — a
card's "not this card" list can be overtaken by the canvas it was written
against. The exports were at the repo root until 2026-08-21 and are now only
under `designs/`; a path without that prefix is a stale pointer.

Coding standards still need the rewrite Task 7 promised. Until then: ruff, type
hints on every signature, `async def` for anything that awaits, no blocking IO in
an async path, no `print()` in production paths.
