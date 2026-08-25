# ARKOS

ARK (Automated Resource Knowledgebase) revolutionizes resource management via automation. Using advanced algorithms, it streamlines collection, organization, and access to resource data, facilitating efficient decision-making.

## Architecture

```
┌──────────────┐
│  arkos app   │  FastAPI server (port configurable)
│  /v1/chat/   │  OpenAI-compatible chat completions API
└──┬───┬───┬───┘
   │   │   │
   ▼   ▼   ▼
┌──────┐ ┌──────────┐ ┌──────────────────┐
│Postgres│ │  SGLang  │ │ Text Embeddings │
│memory  │ │  (LLM)   │ │ Inference (TEI)  │
│:5432   │ │  :30000  │ │ :4444            │
└────────┘ └──────────┘ └──────────────────┘
```

- **App** -- FastAPI agent that orchestrates state transitions, LLM calls, and tool usage
- **SGLang** -- serves Qwen 2.5-7B-Instruct for inference (requires GPU)
- **TEI** -- Hugging Face text embeddings for memory search, runs on port 4444 (requires GPU)
- **Postgres** -- stores conversation history and OAuth tokens (via Supabase)

## Languages and Dependencies

The entire codebase is in Python, except for a few shell scripts. Docker is needed for quick setup.

All dependencies are listed in [`requirements.txt`](requirements.txt). Development tools (linting, testing) are in [`requirements-dev.txt`](requirements-dev.txt).

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # dev only
```

## File Structure

* `harness_module/` -- control plane: lifecycle, session log, runner, api
* `agent_module/` -- the one agent loop (`run_turn`) and its event vocabulary
* `model_module/` -- the one model client (streaming + native tool calling)
* `tool_module/` -- all hands: envelope, registry, MCP via the Arcade gateway, browser, sandbox
* `config_module/` -- YAML configuration loader
* `db/` -- Postgres schema migration and the asyncpg pool
* `frontend/` -- web UI served at `/app`
* `tests/` -- pytest test suite

Mid-redesign: `state_module`, `memory_module` and `computer_module` were deleted
on 2026-08-13. `harness_module` was written in Task 4 and holds `api`, `runner`,
`lifecycle`, `session_log`, `stream`, `hands` and `jwt_utils`; unattended runs,
approvals and the browser leash are Tasks 5, 6 and 9. See
`docs/single_loop_redesign_spec.md`.

## CI/CD Pipeline

CI/CD is configured via GitHub Actions. All workflows live in `.github/workflows/`.

### CI (`ci.yml`) -- runs on every PR and push to `main`

| Job | What it does | Tools |
|-----|-------------|-------|
| **Lint** | Checks code style and formatting | `ruff check .` and `ruff format --check .` |
| **Test** | Runs unit tests with a Postgres service container | `pytest` with coverage |
| **Build & Push** | Builds the Docker image and pushes to GHCR | `docker buildx`, pushes to `ghcr.io/sgiark/arkos` |

On PRs, the build job builds without pushing (verification only). On pushes to `main`, the image is tagged with both the commit SHA and `latest` and pushed to GitHub Container Registry.

### Deploy (`deploy.yml`) -- runs on push to `main` *(currently disabled)*

Deploys the app as a Docker container on `ark.mit.edu`:
1. SSHes in as the `kshitij` user (has docker group access, no sudo needed)
2. Pulls the latest image from GHCR
3. Records the current image tag (for rollback)
4. Stops and removes the old container, starts a new one with `--network host` so it can reach SGLang/TEI/Postgres on localhost
5. Runs a health check (pings `/health`)
6. Runs a smoke test (sends a real chat request through the full stack)
7. Rolls back to the previous image automatically if any step fails

**Setup required on `ark.mit.edu` (one-time):**
1. Create `/home/kshitij/arkos/.env` with `DB_URL`, `ARCADE_API_KEY`, `HF_TOKEN`, etc.
2. Either make the `ghcr.io/sgiark/arkos` package public (GitHub > repo > Packages > package settings), or run once as kshitij:
   ```bash
   echo $GITHUB_TOKEN | docker login ghcr.io -u <your-gh-username> --password-stdin
   ```

**Setup required on GitHub (one-time):**
1. Add `SSH_PRIVATE_KEY` as a repo secret (Settings > Secrets and variables > Actions). This is the private key for the `kshitij` user on ark.mit.edu.
2. Ensure the corresponding public key is in `~/.ssh/authorized_keys` on the server.

### Monitor (`monitor.yml`) -- runs every 30 minutes *(currently disabled)*

Pings the `/health` endpoint and reports per-service status (SGLang, TEI, Postgres). GitHub sends an email notification if the check fails.

**Setup required:** Add `MONITOR_URL` as a GitHub repo secret (e.g., `http://ark.mit.edu:1112`).

### Health Check Endpoint

`GET /health` returns the status of all services:

```json
{
  "status": "ok",
  "services": {
    "sglang": "running",
    "tei": "running",
    "postgres": "running"
  },
  "port": 1112
}
```

Returns `200` if all services are running, `503` if any service is down (status becomes `"degraded"`).

### Running CI Checks Locally

```bash
ruff check .                              # linting
ruff format --check .                     # formatting
pytest tests/ -v -m "not integration"     # unit tests
```

## Deployment Environment: MIT SIPB Shared Server (ark.mit.edu)

ARK OS is deployed on a **shared server** where multiple team members work simultaneously. This means:
- **Port conflicts** can occur when multiple users run the same services
- The **LLM inference server (port 30000)** is shared among all users
- You should use **unique ports** for your API server instance

### Start Inference Engine (REQUIRED FIRST)

The LLM server MUST be running before starting any ARK OS applications.

#### Check if LLM Server is Already Running

Since this is a shared server, someone else may have already started it:

```bash
# Check if port 30000 is in use
lsof -i :30000

# Or verify it's responding
curl http://localhost:30000/v1/models
```

If you see output, the LLM server is already running -- you can skip starting it.

#### Starting the LLM Server (if not running)

Before starting, check with your team to avoid conflicts.

```bash
bash model_module/run.sh
```

This starts the SGLang server on port 30000 using Docker and GPU. Wait for "server started" messages (may take 1-2 minutes on first run).

```bash
bash model_module/run_tei.sh
```

This starts the Huggingface-TEI server on port 4444 using Docker and GPU.

### Setting .env Variables

You need to create a `.env` and set `DB_URL` before starting the application:

1. Copy example env file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env`:
   ```bash
   DB_URL=postgresql://postgres:your-password@localhost:5432/postgres
   ```

### Running the Application

1. **Start the API server**:
   ```bash
   python -m uvicorn harness_module.api:app
   ```
   It refuses to start without `SUPABASE_JWT_SECRET` and `ARK_SESSION_SECRET`
   (see `.env.example`): without them tokens are unverifiable and sessions
   unsignable, and there is no demo bypass.

2. **Access the UI**: Port-forward to `app.port` from `config_module/config.yaml` and navigate to `/app` in your browser.

### Docker Compose (Full Stack)

To run the entire stack (app + SGLang + TEI + Supabase) with Docker:

```bash
docker compose up -d
```

This requires an NVIDIA GPU with Docker GPU support configured.

## Contributors

| Name                  | Role           | GitHub username | Affiliation   |
| --------------------  | -------------- | --------------- | --------------|
| Nathaniel Morgan      | Project leader | nmorgan         | MIT           |
| Joshua Guo            | Frontend       | duck_master     | MIT           |
| Ilya Gulko            | Backend        | gulkily         | MIT           |
| Jack Luo              | Backend        | thejackluo      | Georgia Tech  |
| Bryce Roberts         | Backend        | BryceRoberts13  | MIT           |
| Angela Liu            | Backend        | angelaliu6      | MIT           |
| Ishaana Misra         | Backend        | ishaanam        | MIT           |
| Hudson Hilal          | Backend        | hhilal123       | MIT           |
| Calvin Baker          | Backend        | Calvinlb404     | MIT           |
| Kshitij Duraphe       | DevOps         | ksd3            | BU            |
