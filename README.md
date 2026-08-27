# Buddy

> **PLACEHOLDER.** This README is a stub. The previous one described "ARK
> (Automated Resource Knowledgebase)" — a different product from what this repo
> now builds — and documented deployment to infrastructure this project no
> longer uses. It was replaced on 2026-08-25 rather than patched. Rewrite this
> file when the redesign settles.

Buddy is an agent harness: one loop, one model client, native tool calling.
The internal rename landed in 12.3.5: the predecessor's name is gone from the
tree, identifiers included.

## Where things are

Authoritative docs, in order:

1. **`CLAUDE.md`** — the working rules for this repo. Read first.
2. **`docs/single_loop_redesign_spec.md`** — the redesign spec; it routes to
   everything else.
3. **`docs/contracts.md`** — law. Where it and the spec disagree, contracts wins.

Code layout:

| Path | What it is |
| --- | --- |
| `agent_module/loop.py` | the one loop |
| `model_module/client.py` | the one model client |
| `harness_module/` | control plane — api · runner · store · blobs · memory · workspace · leases · lifecycle · approvals · session_log · system_log · stream · hands · jwt_utils |
| `tool_module/` | envelope · registry · connections · session_tools · composio_mcp · tools/ · sandbox · browser/ |
| `db/pool.py` | asyncpg pool |
| `config_module/` | `config.yaml` and its loader |
| `frontend/` | the UI |
| `designs/` | checked-in Claude Design canvases; these win over `frontend/` |

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
   `BUDDY_SESSION_SECRET` signs the session cookie we issue. The second is SOME
   way to verify a Supabase token, and a project URL is enough — derived from
   the DSN above, which carries the project ref, or set as `SUPABASE_URL` —
   because current projects sign with a key published at the project's JWKS
   endpoint. `SUPABASE_JWT_SECRET` is only for a project still on the legacy
   shared secret; on a current one it stays empty. A non-Supabase Postgres has
   neither, and will not start until you set one.

2. Install dependencies and apply the migrations:

   ```bash
   pip install -r requirements.txt
   python db/migrate.py
   ```

   `db/migrate.py` applies `db/migrations/*.sql` in lexical order and records
   each one in `schema_migrations`, so re-running it is safe. Nothing applies
   them at startup: the API comes up happily against an unmigrated database and
   fails on the first request that touches a table.

3. Start the API server on the port `app.public_url` names:

   ```bash
   python -m uvicorn harness_module.api:app --port 1121
   ```

   The port is not optional and `app.port` is not read by anything — uvicorn's
   own default is 8000. Every mutation is origin-checked against
   `app.public_url` (`http://localhost:1121` by default), so a UI served from
   any other port loads, reads fine, and gets 403 on every POST. Change both or
   neither.

4. Open `http://localhost:1121/app`.

The browser tools need the browserless container, which is the only service in
`docker-compose.yml` this redesign still uses:

```bash
docker compose up -d browserless
```

Name the service. A bare `docker compose up -d` fails: the `app` service builds
a `Dockerfile` that left the tree in the 2026-08-25 refactor, and `sglang` and
`tei` are pre-redesign GPU services no code reads any more — the model client
talks to whatever `llm.base_url` names, and nothing embeds.

Then set `BROWSERLESS_URL=ws://localhost:3000` in `.env`. Compose hands the
containerised `app` service `ws://browserless:3000`, which resolves to nothing
from your host, and an unset url makes `browser_task` refuse — deliberately,
because the fallback would be a Chromium running model-chosen pages beside your
cookies and the store's secret key.

## Tests and CI

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest tests/ -q --timeout=120 -m "not integration"
```

Those are the commands `.github/workflows/ci.yml` runs — a lint stage and a test
stage against a throwaway `postgres:15` service — on push to `main` and
`dev_refactor` and on pull requests to `main`. Run them yourself first; a red
push is a slower way to learn the same thing.

Database tests SKIP unless you name a database on the command line
(`DB_URL=... pytest ...`). `tests/conftest.py` captures `DB_URL` before it reads
`.env` and throws the `.env` one away, because the suite truncates tables and a
developer's `.env` points at the real project. Integration tests are deselected
in CI and need live credentials.

The workflow was deleted on 2026-08-25 and restored the same afternoon with only
those two stages. The deploy and monitor jobs did NOT come back and should not
be recreated from `git log`: they pushed to a container registry and a
university host that were never this project's infrastructure.
