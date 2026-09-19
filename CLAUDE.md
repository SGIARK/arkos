# arkos-core

An agent harness: one loop, one model client, native tool calling. AGPL-3.0;
see `NOTICE` for where this code came from.

**Coding standards are `.github/CLAUDE.md`.** Contribution rules, including the
CLA, are `.github/CONTRIBUTING.md`. `README.md` says how to run it.

**The sandbox is HEADLESS: a shell and a filesystem, and nothing a human
sees.** `tool_module/sandbox/` is an e2b box per session on the `base` template,
and it backs seven tools: `run_command`, `read_file`, `write_file`, `edit_file`,
`list_dir`, `grep`, `glob`. The box's disk is a CACHE OF THE STORE:
`workspace.materialize` copies the session's claimed folders in at the first
call that needs the box, and `workspace.flush` hashes what is on disk and
commits it back when the run ends. `~/store/<folder>/` is the only durable path
in the box; everything else dies with it.

**The VIRTUALIZED computer is a different thing, and it is not here.** The
desktop, the resident Chromium, the supervisor, the broker, takeover and the
frame stream were built after this cut and none of it exists in this tree. A box
here has no display and no one watches it. Do not reach for `box_agent/`; there
is none.

**The HTTP server is `harness_module/api.py`**, run with
`uvicorn harness_module.api:app`. `harness_module/` is the control plane:
api · runner · store · blobs · memory · workspace · leases · lifecycle ·
approvals · session_log · system_log · stream · hands · jwt_utils.

**The store is three files, one idea each.** `blobs.py` is content-addressed
bytes and the HTTP client that carries them, `store.py` is the TREE
(`files (user_id, path)`, folders derived from paths), `memory.py` is the user's
notes and curated core. Imports go ONE way, blobs to store to workspace, and
`store.py` re-exports the blob calls so a caller that reads a tree and then
wants bytes needs one import. `workspace.py` is the CLAIMS: which folders a
session may see. No bytes move through it.

**All MCP traffic flows through Composio.** `tool_module/composio_mcp.py` is the
only client. A "server" is a toolkit PREFIX in UPPER SNAKE (`GMAIL_*`,
`GOOGLEDRIVE_*`), and that prefix is the durable key: `user_connections` and
`session_tools` are keyed by it, never by the MCP url (derived per user, and
re-mintable) and never by the `mcp_servers:` config label. A Composio grant is
per TOOLKIT, so reading Gmail says nothing about Calendar. Web search does NOT
ride this wire: it is `web_search`, a local SerpAPI tool on an app-level key.

**A session reaches only the MCP servers it was given.** The toggles are
`session_tools`. `registry.manifest` is the ONE builder of a turn's tool list
and it cannot exceed `llm.max_tools` whatever the toggles say: whole servers are
benched, most-recently-enabled first, and a benched server gets a `status` event
and a `system_events` row. **The system prompt is generated from the manifest
that shipped, never from the toggles**, which is why `_drive` builds the
manifest before it folds. Do not add a second path that assembles tool specs.

**A gated tool call PARKS the turn on itself.** `requires_approval` with no
grant leaves that call OPEN in the transcript, and the `approvals` row of kind
`call` carries the real `(tool_name, tool_args)`: consent binds to the call,
never to prose about it. Answering is `approve`/`decline`, and approving runs
that exact call once through normal dispatch, latched by `consumed_at`. Never
re-run a consumed-but-unclosed call: repair it as interrupted.

**The browser is `browser_task`, and it runs in a Browserless container** over
CDP. `tool_module/browser/endpoint.py` resolves `cdp_url()` from
`browser.cdp_url` or `BROWSERLESS_URL`; an unset url is a refusal, not a
fallback, because the fallback would be a Chromium running model-chosen pages
beside the user's cookies and the store's secret key. The browser is leased per
user, so one session holds it for a whole run.

**Five identifiers carry side effects if you change them.** `_ISSUER` is `arkos`
(every cookie carries `iss=arkos`, and changing it signs everyone out),
`store.bucket` / `store.prefix` / `store.root` point at the `arkos` bucket
(changing them orphans every stored blob), and the cookie is `arkos_session`
signed with `ARKOS_SESSION_SECRET`. The fifth is the staging paths
`/tmp/arkos-*.tar` in `harness_module/workspace.py`, which materialize and flush
must agree on. They are ordinary identifiers, but they are the ones with side
effects: change any of them and say what it costs.

**CI is `.github/workflows/ci.yml`:** a lint stage (`ruff check .`,
`ruff format --check .`, and a scoped `mypy`) and a test stage
(`pytest tests/ -q --timeout=120 -m "not integration"`) against a `postgres:15`
service with the migrations applied. Run all of them yourself before claiming a
change is green; a red push is a slower way to learn the same thing. Integration
tests are deselected in CI and need real credentials.

Until the standards doc says otherwise: ruff, type hints on every signature,
`async def` for anything that awaits, no blocking IO in an async path, no
`print()` in production paths.
