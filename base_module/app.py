import json
import os
import sys
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# Standard boilerplate for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_module.agent import Agent
from base_module.tasks import router as tasks_router
from base_module.users import router as users_router
from config_module.loader import config
from memory_module.memory import Memory
from model_module.ArkModelNew import AIMessage, ArkModelLink, SystemMessage, UserMessage
from state_module.core.state_handler import StateHandler
from tool_module.smithery import AuthRequiredError
from tool_module.tool_call import MCPToolManager

app = FastAPI(title="ArkOS Agent API", version="1.0.0")

# CORS so the frontend can talk to this API from file:// or another port during demos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users_router)
app.include_router(tasks_router)

# Serve the ark frontend at /app/ if the folder exists
_FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(_FRONTEND_DIR):
    app.mount("/app", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
    print(f"[ark] serving frontend from {_FRONTEND_DIR} at /app/")
else:
    print(f"[ark] no frontend folder at {_FRONTEND_DIR}; /app route disabled")


# Shared singletons (no per-user state)
flow = StateHandler(yaml_path=config.get("state.graph_path"))
llm = ArkModelLink(base_url=config.get("llm.base_url"), max_tokens=config.get("llm.max_tokens"))

# Per-user memory cache — keyed by user_id string.
# Memory.__init__ initialises mem0 (expensive), so we create once per user and reuse.
_memory_cache: dict[str, Memory] = {}

# Module-level prompt + tools, written by startup/refresh, stamped onto each per-request agent.
_system_prompt: str = ""
_available_tools: dict = {}


def _get_or_create_memory(user_id: str) -> Memory:
    if user_id not in _memory_cache:
        _memory_cache[user_id] = Memory(
            user_id=user_id,
            session_id=None,
            db_url=config.get("database.url"),
            use_long_term=config.get("memory.use_long_term", False),
        )
    return _memory_cache[user_id]


def _make_agent(user_id: str) -> Agent:
    """Create a fresh Agent for one request. Memory is cached per user; everything else is shared."""
    ag = Agent(
        agent_id=user_id,
        flow=flow,
        memory=_get_or_create_memory(user_id),
        llm=llm,
        tool_manager=tool_manager,
    )
    ag.system_prompt = _system_prompt
    ag.available_tools = _available_tools
    return ag


# MCP connectivity. Everything flows through Smithery Connect; per-user OAuth
# and credential storage happen on Smithery's side, not ours.
mcp_config = config.get("mcp_servers")
smithery_config = config.get("smithery") or {}
tool_manager = (
    MCPToolManager(mcp_config, smithery_config=smithery_config)
    if mcp_config and smithery_config.get("api_key")
    else None
)
if tool_manager is None:
    if not smithery_config.get("api_key"):
        print("[ark] SMITHERY_API_KEY missing; MCP tool manager disabled")
    elif not mcp_config:
        print("[ark] no mcp_servers configured; tool manager disabled")


def format_tools_for_system_prompt(tools_by_server: dict, deferred: list[dict] | None = None) -> str:
    """
    tools_by_server: {server_name: {tool_name: tool_spec}} from SmitheryManager.list_all_tools().
    deferred: optional list of {'service', 'name', 'setup_url'} for per-user
        OAuth services that exist in config but aren't connected for anyone
        yet. Surfacing them in the system prompt lets buddy tell the user
        'Linear exists, you just need to connect it' instead of falsely
        claiming no access.
    """
    lines: list[str] = []

    has_any = bool(tools_by_server) or bool(deferred)
    if not has_any:
        return "You do not currently have any external tools available."

    if tools_by_server:
        lines.append("You have access to the following tools. Only call tools listed here.")
        lines.append("")
        for server_name, server_tools in tools_by_server.items():
            if not server_tools:
                continue
            lines.append(f"# Service: {server_name}")
            for tool_name, tool in server_tools.items():
                if isinstance(tool, dict):
                    desc = tool.get("description") or ""
                else:
                    desc = getattr(tool, "description", "") or ""
                # One line per tool: name + short description only. Full schemas
                # bloat the system prompt past the model's context window.
                desc_short = desc.strip().splitlines()[0][:120] if desc.strip() else ""
                if desc_short:
                    lines.append(f"- {tool_name}: {desc_short}")
                else:
                    lines.append(f"- {tool_name}")
            lines.append("")

    if deferred:
        real_deferred = [svc for svc in deferred if svc.get("service") not in (tools_by_server or {})]
        if real_deferred:
            lines.append("The following services are configured but not yet connected for the current user.")
            lines.append("You cannot call their tools until the user completes the Smithery OAuth flow.")
            lines.append(
                "If the user asks for something that needs one of these, tell them it "
                "needs to be connected first and share the setup URL:"
            )
            lines.append("")
            for svc in real_deferred:
                nm = svc.get("name") or svc.get("service")
                url = svc.get("setup_url")
                if url:
                    lines.append(f"- {nm} (service id: {svc.get('service')}): connect via {url}")
                else:
                    lines.append(
                        f"- {nm} (service id: {svc.get('service')}): "
                        "needs connection; direct the user to the ark connections panel"
                    )
            lines.append("")

    return "\n".join(lines)


def _list_deferred_services() -> list[dict]:
    """Read config for per-user OAuth services (requires_auth: true) so we
    can advertise them in the system prompt even before anyone connects."""
    if not tool_manager:
        return []
    out: list[dict] = []
    for server_name, spec in (tool_manager.servers or {}).items():
        if not spec.get("requires_auth"):
            continue
        out.append(
            {
                "service": server_name,
                "name": spec.get("name") or server_name,
                "setup_url": None,  # real setup URL is minted on first per-user connect
            }
        )
    return out


@app.on_event("startup")
async def startup():
    """Initialize MCP servers and build the agent's system prompt with available tools."""
    global _system_prompt, _available_tools
    base_system_prompt = (config.get("app.system_prompt") or "").strip()

    if tool_manager:
        await tool_manager.initialize_servers()

        _available_tools = await tool_manager.list_all_tools()

        shared_tools = sum(len(tools) for tools in _available_tools.values())
        shared_servers = [s for s, tools in _available_tools.items() if tools]
        deferred = _list_deferred_services()

        print(
            f"[ark] MCP init: {len(shared_servers)} shared server(s) connected "
            f"({shared_tools} tool(s)); {len(deferred)} per-user service(s) deferred"
        )
        if shared_servers:
            print(f"[ark]   shared: {', '.join(shared_servers)}")
        if deferred:
            print(f"[ark]   deferred (needs user OAuth): {', '.join(s['service'] for s in deferred)}")

        tool_prompt = format_tools_for_system_prompt(_available_tools, deferred=deferred)
        _system_prompt = base_system_prompt + "\n\n" + tool_prompt if base_system_prompt else tool_prompt
    else:
        _system_prompt = base_system_prompt

    # Resume any subagent tasks that were in-flight before a restart.
    try:
        from base_module.task_runner import sweep_orphans

        resumed = await sweep_orphans()
        if resumed:
            print(f"[ark] resumed {resumed} orphan task(s) after restart")
    except Exception as e:
        print(f"[ark] task orphan sweep failed: {e}")


@app.get("/services")
async def list_services(request: Request):
    """
    Returns the connection state of every per-user (requires_auth) service
    for the calling user, plus every no-auth shared service. The frontend
    uses this to render a connections panel with Smithery setup links.
    """
    user_id = request.headers.get("X-User-ID") or config.get("memory.fallback_user_id")

    if not tool_manager:
        return JSONResponse(content={"user_id": user_id, "shared": [], "per_user": []})

    # Shared (no-auth) services are whatever initialize_servers() connected.
    shared = []
    for server_name in tool_manager._shared_tools or {}:
        shared.append(
            {
                "service": server_name,
                "name": (tool_manager.servers.get(server_name, {}) or {}).get("name", server_name),
                "connected": True,
            }
        )

    per_user = [{"service": svc, **info} for svc, info in tool_manager.get_user_service_status(user_id).items()]

    return JSONResponse(content={"user_id": user_id, "shared": shared, "per_user": per_user})


async def _refresh_system_prompt() -> None:
    """Rebuild _system_prompt from the current tool_manager state.
    Called after a connection change so buddy picks up the new tools."""
    global _system_prompt, _available_tools
    if not tool_manager:
        return
    _available_tools = await tool_manager.list_all_tools()
    deferred = _list_deferred_services()
    tool_prompt = format_tools_for_system_prompt(_available_tools, deferred=deferred)
    base = (config.get("app.system_prompt") or "").strip()
    _system_prompt = base + "\n\n" + tool_prompt if base else tool_prompt


def _callback_return_url(request: Request, service: str, user_id: str) -> str:
    """Build the URL Smithery should redirect back to after OAuth.

    We derive the scheme+host from the request itself so this works whether
    the app is served at localhost:1114, a reverse-proxied hostname, or
    anywhere else. The user_id is encoded into the URL so the callback can
    verify the connection for the right user without needing a session.
    """
    from urllib.parse import urlencode

    scheme = request.headers.get("X-Forwarded-Proto") or request.url.scheme
    host = request.headers.get("X-Forwarded-Host") or request.url.netloc
    qs = urlencode({"user_id": user_id})
    return f"{scheme}://{host}/oauth/callback/{service}?{qs}"


@app.post("/services/{service}/connect")
async def connect_service(service: str, request: Request):
    """
    Trigger (or re-trigger) the per-user Smithery OAuth flow for a given
    service. Returns the setup_url if Smithery needs the user to authorize,
    or status='connected' if the connection is already live.

    The setup_url returned here already has a Smithery-side returnUrl baked
    in pointing at /oauth/callback/{service}, so when the user finishes
    OAuth on Smithery's hosted page they get bounced back into the ark app.
    """
    user_id = request.headers.get("X-User-ID") or config.get("memory.fallback_user_id")

    if not tool_manager:
        return JSONResponse(
            content={"error": "tool manager disabled"},
            status_code=503,
        )
    if service not in (tool_manager.servers or {}):
        return JSONResponse(
            content={"error": f"unknown service '{service}'"},
            status_code=404,
        )

    import aiohttp  # local import; the rest of the file stays light

    return_url = _callback_return_url(request, service, user_id)

    async with aiohttp.ClientSession() as session:
        try:
            await tool_manager._ensure_user_server(session, user_id, service, return_url=return_url)
        except Exception as e:
            # AuthRequiredError is the happy path here: it carries setup_url
            setup_url = getattr(e, "setup_url", None)
            if setup_url:
                return JSONResponse(
                    content={
                        "service": service,
                        "status": getattr(e, "state", "auth_required"),
                        "setup_url": setup_url,
                        "return_url": return_url,
                    }
                )
            return JSONResponse(
                content={"service": service, "status": "error", "error": str(e)},
                status_code=500,
            )

    await _refresh_system_prompt()
    return JSONResponse(content={"service": service, "status": "connected"})


@app.post("/services/{service}/disconnect")
async def disconnect_service(service: str, request: Request):
    """Forget a per-user connection client-side. Smithery retains the token
    until the user revokes it there; we just stop tracking/surfacing it."""
    user_id = request.headers.get("X-User-ID") or config.get("memory.fallback_user_id")
    if not tool_manager:
        return JSONResponse(content={"error": "tool manager disabled"}, status_code=503)

    by_server = tool_manager._user_tools.get(user_id) or {}
    by_server.pop(service, None)
    tool_manager._pending.get(user_id, {}).pop(service, None)
    # rebuild the tool registry without the disconnected service's tools
    tool_manager._tool_registry = {
        tname: sname
        for tname, sname in tool_manager._tool_registry.items()
        if sname != service or sname in tool_manager._shared_tools
    }
    await _refresh_system_prompt()
    return JSONResponse(content={"service": service, "status": "disconnected"})


@app.get("/oauth/callback/{service}")
async def oauth_callback(service: str, request: Request):
    """
    Smithery redirects the user here after they finish the OAuth flow for
    `service`. We re-run upsert_connection so the manager caches the now-live
    connection and its tools, then serve a tiny HTML page that notifies the
    opener window (the ark app) and closes itself.
    """
    from fastapi.responses import HTMLResponse

    user_id = request.query_params.get("user_id") or config.get("memory.fallback_user_id")
    status = "connected"
    error_msg: str | None = None

    if not tool_manager:
        status = "error"
        error_msg = "tool manager disabled"
    elif service not in (tool_manager.servers or {}):
        status = "error"
        error_msg = f"unknown service '{service}'"
    else:
        import aiohttp

        async with aiohttp.ClientSession() as session:
            try:
                await tool_manager._ensure_user_server(session, user_id, service)
            except AuthRequiredError as e:
                # User bounced back but the connection isn't live yet. Leave
                # the setup URL in place so they can retry.
                status = "pending"
                error_msg = e.message
            except Exception as e:
                status = "error"
                error_msg = str(e)

        if status == "connected":
            await _refresh_system_prompt()

    import json as _json

    payload = _json.dumps({"service": service, "status": status, "error": error_msg})
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ark - {service} connected</title>
<style>
  body {{ font-family: ui-monospace, monospace; background:#0b0b0b; color:#e6e6e6;
         display:flex; align-items:center; justify-content:center; height:100vh; margin:0; }}
  .card {{ border:1px solid #333; padding:20px 24px; max-width:360px; text-align:center; }}
  .ok {{ color:#4ade80; }} .err {{ color:#f87171; }} .pending {{ color:#fbbf24; }}
  button {{ background:none; border:1px solid #666; color:inherit; padding:6px 12px;
            font-family:inherit; cursor:pointer; margin-top:12px; }}
</style></head><body>
<div class="card">
  <h3 class="{status}">{service}: {status}</h3>
  <p style="font-size:12px;color:#888;">{error_msg or "you can close this window."}</p>
  <button id="closeBtn">close</button>
</div>
<script>
  const payload = {payload};
  try {{
    if (window.opener) {{
      window.opener.postMessage({{ type: "ark-oauth-callback", payload }}, "*");
    }}
  }} catch (e) {{}}
  document.getElementById("closeBtn").addEventListener("click", () => window.close());
  if (payload.status === "connected") {{ setTimeout(() => window.close(), 800); }}
</script>
</body></html>"""

    return HTMLResponse(content=html)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and all dependencies."""
    import requests

    services = {}

    # Check SGLang (LLM inference)
    try:
        resp = requests.get("http://localhost:30000/v1/models", timeout=2)
        services["sglang"] = "running" if resp.status_code == 200 else "error"
    except Exception:
        services["sglang"] = "not_running"

    # Check TEI (text embeddings)
    try:
        resp = requests.get("http://localhost:4444/health", timeout=2)
        services["tei"] = "running" if resp.status_code == 200 else "error"
    except Exception:
        services["tei"] = "not_running"

    # Check Postgres
    try:
        import psycopg2

        conn = psycopg2.connect(config.get("database.url"), connect_timeout=2)
        conn.close()
        services["postgres"] = "running"
    except Exception:
        services["postgres"] = "not_running"

    # Overall status: degraded if any service is down, ok if all running
    all_running = all(s == "running" for s in services.values())
    overall = "ok" if all_running else "degraded"

    return JSONResponse(
        content={"status": overall, "services": services, "port": 1112},
        status_code=200 if all_running else 503,
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OAI-compatible endpoint wrapping the full ArkOS agent."""
    payload = await request.json()

    messages = payload.get("messages", [])
    model = payload.get("model", "ark-agent")
    stream = payload.get("stream", False)

    # Extract user_id from header or body for per-user tool auth
    user_id = request.headers.get("X-User-ID") or payload.get("user") or payload.get("user_id")

    # Lazily connect any per-user OAuth servers (e.g. Linear) that the user
    # has already authorized.  This populates tool_manager._user_tools so
    # the system prompt includes their tools instead of a "please connect" stub.
    if tool_manager and user_id:
        import contextlib

        import aiohttp as _aiohttp

        async with _aiohttp.ClientSession() as _sess:
            for svc_name, spec in tool_manager.servers.items():
                if not spec.get("requires_auth"):
                    continue
                # Skip if already loaded for this user
                if svc_name in (tool_manager._user_tools.get(user_id) or {}):
                    continue
                # auth_required is expected if user hasn't connected yet
                with contextlib.suppress(Exception):
                    await tool_manager._ensure_user_server(_sess, user_id, svc_name)
        await _refresh_system_prompt()

    effective_user_id = user_id or config.get("memory.fallback_user_id")
    req_agent = _make_agent(effective_user_id)

    context_msgs = []
    context_msgs.append(SystemMessage(content=_system_prompt))

    # Convert OAI messages into internal message objects
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            context_msgs.append(SystemMessage(content=content))
        elif role == "user":
            context_msgs.append(UserMessage(content=content))
        elif role == "assistant":
            context_msgs.append(AIMessage(content=content))

    # Handle streaming
    if stream:

        async def generate_stream():
            """Yield SSE chunks in OpenAI streaming format."""
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            async for chunk in req_agent.step_stream(context_msgs, user_id=effective_user_id):
                data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Send final chunk
            final_data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )

    # Non-streaming response
    agent_response = await req_agent.step(context_msgs, user_id=effective_user_id)
    final_msg = agent_response or AIMessage(content="(no response)")

    completion = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": final_msg.content},
                "finish_reason": "stop",
            }
        ],
    }

    return JSONResponse(content=completion)


if __name__ == "__main__":
    uvicorn.run(
        "base_module.app:app",
        host=config.get("app.host"),
        port=int(config.get("app.port")),
        reload=config.get("app.reload"),
    )
