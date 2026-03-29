# ARKOS

ARK (Automated Resource Knowledgebase) revolutionizes resource management via automation. Using advanced algorithms, it streamlines collection, organization, and access to resource data, facilitating efficient decision-making.


## Languages and Dependencies

The entire codebase is in Python, except for a few shell scripts. Docker is needed for quick setup 

### Core Dependencies

* **`openai>=1.61.0`** - OpenAI Python SDK for standardizing inference engine communication and API compatibility
* **`pyyaml>=6.0.2`** - YAML parser for configuration files (state graphs, etc.)
* **`pydantic>=2.10.6`** - Data validation and schema definition using Python type annotations
* **`requests>=2.32.3`** - HTTP library for making API requests to external services and tools

### Web Framework

* **`fastapi>=0.115.0`** - Modern, fast web framework for building the API server with automatic OpenAPI documentation
* **`uvicorn>=0.32.0`** - ASGI server for running FastAPI applications

### Database & Memory

* **`psycopg2-binary>=2.9.11`** - PostgreSQL adapter for Python (binary distribution, no compilation required). Used for storing conversation context and long-term memory
* **`mem0ai`** - Memory management library for vector-based memory storage and retrieval using Supabase

### Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Note:** `psycopg2-binary` is used instead of `psycopg2` to avoid requiring PostgreSQL development libraries (`libpq-dev`) on the system. For production deployments, you may want to use `psycopg2` with proper system dependencies.

## File structure

(As of September 11, 2025.)


* `base_module/` for main interface
* `config_module/` for YAML configuration files
* `model_module/` for core LLM-inference logic
* `agent_module/` for agentic structure 
* `state_module/` for defining agent state graphs 
* `tool_module/` for MCP compatibility 
* `memory_module/` for long term memory and context management
* `state_module/`
* `tool_module/`
* `.gitignore`
* `README.md` (this very file)
* `requirements.txt` (Python dependencies)

## Instructions

### Deployment Environment: MIT SIPB Shared Server (ark.mit.edu)

**⚠️ IMPORTANT:** ARK OS is deployed on a **shared server** where multiple team members work simultaneously. This means:
- **Port conflicts** can occur when multiple users run the same services
- The **LLM inference server (port 30000)** is shared among all users
- You should use **unique ports** for your API server instance

### Start Inference Engine (REQUIRED FIRST!)

**⚠️ IMPORTANT:** The LLM server MUST be running before starting any ARK OS applications. Without it, you'll get connection errors.

#### Check if LLM Server is Already Running

Since this is a shared server, someone else may have already started it:

```bash
# Check if port 30000 is in use
lsof -i :30000

# Or verify it's responding
curl http://localhost:30000/v1/models
```

If you see output, the LLM server is already running - **you can skip starting it**.

#### Starting the LLM Server (if not running)

**Before starting:** Check with your team to avoid conflicts.

The project uses SGLang to run the Qwen 2.5-7B-Instruct model:

```bash
bash model_module/run.sh
```

This starts the SGLang server on port 30000 using Docker and GPU. Wait for "server started" messages (may take 1-2 minutes on first run).



```bash
bash model_module/run_tei.sh
```

This starts the Huggingface-TEI server on port 4444 using Docker and GPU. Wait for "server started" messages 

### Setting .env Variables 

You need to create a .env and set DB_URL before starting the application

1. ***Copy example env file**:
   ```bash
   cp .env.example .env 
   ```
2. ** Edit .env**:
   ```bash
   # Set DB URL
   DB_URL=postgresql://postgres:your-super-secret-and-long-postgres-password@localhost:54322/postgres
   ```
   
### Google Calendar Setup

To enable Google Calendar for a user:

1. Add the following to `.env` on the server (both credential files must come from the same Google Cloud project):
   ```
   GOOGLE_OAUTH_CREDENTIALS=/path/to/google_credentials.json
   GOOGLE_MCP_CREDENTIALS=/path/to/google_credentials_mcp.json
   ```
   The first is a web application credential, the second is a desktop/installed application credential.

2. Add the user's email as a Test User in Google Cloud Console under OAuth consent screen > Test Users.

3. Have the user make any calendar request. The agent will reply with an auth link:
   ```
   http://ark.mit.edu:1114/auth/connect?user_id=nmorgan@mit.edu
   ```

4. The user clicks the link, signs in with Google, and their token is saved. All future calendar requests work automatically.

### Running the Application

You need to run both the API server and the test interface:

1. **Start the API server** (in one terminal):
   ```bash
   python base_module/app.py
   ```
   This starts the FastAPI server on port 1111, providing the `/v1/chat/completions` endpoint.

2. **Run the test interface** (in another terminal):
   ```bash
   python base_module/main_interface.py
   ```
   This provides an interactive CLI to test the agent. Type your messages and press Enter. Type `exit` or `quit` to stop.

## Contributors + contact

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
