# arkos-core

arkos-core is an agent harness: one loop, one model client, native tool calling.
A session runs a model against a tool manifest, streams its events to a web UI,
parks for human approval when a tool needs one, and keeps the user's files in a
content-addressed store.

It drives a real browser through a Browserless container, searches the web, and
reaches Gmail, GitHub, Linear, Outlook, Notion, Google Calendar and Google Drive
through Composio. It has no sandbox: nothing here runs shell commands or writes
files on a machine of its own.

Licensed under AGPL-3.0. See `LICENSE`, and `NOTICE` for where this code came
from.

## Where things are

| Path | What it is |
| --- | --- |
| `agent_module/loop.py` | the one loop |
| `model_module/client.py` | the one model client |
| `harness_module/` | control plane: api · runner · store · blobs · memory · workspace · leases · lifecycle · approvals · session_log · system_log · stream · hands · jwt_utils |
| `tool_module/` | envelope · registry · connections · session_tools · composio_mcp · tools/ · browser/ |
| `db/` | migrations and the asyncpg pool |
| `config_module/` | `config.yaml` and its loader |
| `frontend/` | the UI |

`.github/CONTRIBUTING.md` has the contribution rules, including the CLA every
pull request needs.

## Running it

1. Create your env file and set `DB_URL`:

   ```bash
   cp .env.example .env
   ```

   ```
   DB_URL=postgresql://postgres:<password>@db.<project-ref>.supabase.co:5432/postgres
   ```

   Percent-encode the password: a raw `@` reparses the DSN and everything after
   it silently becomes the host. `.env.example` carries the encoding table and
   the pooler DSN for IPv4-only networks.

   The server then refuses to start without two things, with no demo bypass.
   `ARKOS_SESSION_SECRET` signs the session cookie we issue. The second is SOME
   way to verify a Supabase token, and a project URL is enough: derived from the
   DSN above, which carries the project ref, or set as `SUPABASE_URL`, because
   current projects sign with a key published at the project's JWKS endpoint.
   `SUPABASE_JWT_SECRET` is only for a project still on the legacy shared
   secret; on a current one it stays empty. A non-Supabase Postgres has neither,
   and will not start until you set one.

2. Install dependencies and apply the migrations:

   ```bash
   pip install -r requirements.txt
   python db/migrate.py
   ```

   `db/migrate.py` applies `db/migrations/*.sql` in lexical order and records
   each one in `schema_migrations`, so re-running it is safe. Nothing applies
   them at startup: the API comes up happily against an unmigrated database and
   fails on the first request that touches a table.

   It prints the host and database it is about to touch, and REFUSES anything
   that is not local unless you say `--production`. `DB_URL` from the
   environment wins over `.env`, so a local apply is one variable:

   ```bash
   DB_URL=postgresql://test:test@localhost:5432/test python db/migrate.py
   python db/migrate.py --production   # the remote project named in .env
   ```

3. Start the API server on the port `app.public_url` names:

   ```bash
   python -m uvicorn harness_module.api:app --port 1121
   ```

   The port is not optional and `app.port` is not read by anything: uvicorn's
   own default is 8000. Every mutation is origin-checked against
   `app.public_url` (`http://localhost:1121` by default), so a UI served from
   any other port loads, reads fine, and gets 403 on every POST. Change both or
   neither.

4. Open `http://localhost:1121/app`.

The browser tool needs the browserless container, which is the only service in
`docker-compose.yml` this build uses:

```bash
docker compose up -d browserless
```

Name the service. A bare `docker compose up -d` fails: the `app` service builds
a `Dockerfile` that is not in this tree, and `sglang` and `tei` are optional GPU
services (see "Running at MIT" below).

Then set `BROWSERLESS_URL=ws://localhost:3000` in `.env`. Compose hands the
containerised `app` service `ws://browserless:3000`, which resolves to nothing
from your host, and an unset url makes `browser_task` refuse: deliberately,
because the fallback would be a Chromium running model-chosen pages beside your
cookies and the store's secret key.

## Running at MIT

Self-hosted: a local GPU serves the model, the browser runs in a container, and
there is no sandbox to provision.

1. `docker compose up -d sglang browserless` (sglang needs an NVIDIA GPU).
2. In `config_module/config.yaml` set `llm.base_url` to `http://localhost:30000/v1`
   (the `/v1` is required) and `llm.model_name` to the model that server was
   launched with: a tool-calling model with a matching `--tool-call-parser`, or
   turns come back as prose and no tool ever runs.
3. In `.env` set `BROWSERLESS_URL=ws://localhost:3000`, leave `OPENAI_API_KEY`
   empty (SGLang ignores it), and fill in `DB_URL`, the Supabase keys and
   `ARKOS_SESSION_SECRET`. `COMPOSIO_API_KEY` is optional.

## Tests and CI

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
mypy --follow-imports=silent --ignore-missing-imports --disable-error-code=arg-type tool_module/tools/ tool_module/browser/tool.py
DB_URL=postgresql://test:test@localhost:5432/test python db/migrate.py
DB_URL=postgresql://test:test@localhost:5432/test pytest tests/ -q --timeout=120 -m "not integration"
```

Those are the commands `.github/workflows/ci.yml` runs: a lint stage (ruff plus
the type check) and a test stage against a throwaway `postgres:15` service. Run
them yourself first; a red push is a slower way to learn the same thing.

The suite needs a database with the migrations applied. It FAILS rather than
skips when it finds a reachable database with no schema, because a suite that
skips itself green is worse than one that goes red. `tests/conftest.py` captures
`DB_URL` before it reads `.env` and throws the `.env` one away, because the
suite truncates tables and a developer's `.env` may point at a real project.
Integration tests are deselected in CI and need live credentials.
