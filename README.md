# Buddy

> **PLACEHOLDER.** This README is a stub. The previous one described "ARK
> (Automated Resource Knowledgebase)" — a different product from what this repo
> now builds — and documented deployment to infrastructure this project no
> longer uses. It was replaced on 2026-08-25 rather than patched. Rewrite this
> file when the redesign settles.

Buddy is an agent harness: one loop, one model client, native tool calling.
Formerly ARKOS; see `CLAUDE.md` for why the string `arkos` still appears in
load-bearing identifiers.

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
| `tool_module/` | envelope · registry · connections · session_tools · arcade · tools/ · sandbox · browser/ |
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
   DB_URL=postgresql://postgres:your-password@localhost:5432/postgres
   ```

   The server also refuses to start without `SUPABASE_JWT_SECRET` and
   `ARK_SESSION_SECRET` — without them tokens are unverifiable and sessions
   unsignable, and there is no demo bypass. See `.env.example`.

2. Start the API server:

   ```bash
   python -m uvicorn harness_module.api:app
   ```

3. Open `/app`, on `app.port` from `config_module/config.yaml`.

For the full stack, including the browserless container the browser tools
connect to over CDP:

```bash
docker compose up -d
```

## CI

There is none. `.github/workflows/` was deleted on 2026-08-25. Run `ruff` and
the test suite locally before calling a change green.
