from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import time
import uuid
import os
import sys
import json
import logging
import requests

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_module.loader import config
from agent_module.agent import Agent
from state_module.state_handler import StateHandler
from memory_module.memory import Memory
from model_module.ArkModelNew import ArkModelLink, UserMessage, SystemMessage, AIMessage
from tool_module.tool_call import MCPToolManager
from tool_module.token_store import UserTokenStore
from base_module.auth import router as auth_router


logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ArkOS Agent API", version="1.0.0")
app.include_router(auth_router)

flow = StateHandler(yaml_path=config.get("state.graph_path"))

memory = Memory(
    user_id=config.get("memory.user_id"),
    session_id=None,
    db_url=config.get("database.url"),
    use_long_term=config.get("memory.use_long_term", False),
)

llm = ArkModelLink(base_url=config.get("llm.base_url"), max_tokens=config.get("llm.max_tokens"))

token_store = UserTokenStore(config.get("database.url"))

mcp_config = config.get("mcp_servers")
tool_manager = MCPToolManager(mcp_config, token_store=token_store, base_url=config.get("app.base_url", "http://localhost:1112")) if mcp_config else None
agent = Agent(
    agent_id=config.get("memory.user_id"),
    flow=flow,
    memory=memory,
    llm=llm,
    tool_manager=tool_manager,
)


def format_tools_for_system_prompt(tools: dict) -> str:
    lines = [
        "You have access to the following tools.",
        "Use them when appropriate. Only call tools that are listed below.",
        "",
    ]
    for server_name, server_tools in tools.items():
        for tool_name, tool_spec in server_tools.items():
            lines.append(f"Tool name: {tool_name}")
            desc = tool_spec.get("description")
            if desc:
                lines.append(f"Description: {desc}")
            lines.append("")
    return "\n".join(lines)


@app.on_event("startup")
async def startup():
    base_system_prompt = (config.get("app.system_prompt") or "").strip()

    if tool_manager:
        logger.info("Initializing MCP servers...")
        try:
            await tool_manager.initialize_servers()
        except Exception as e:
            logger.error("ERROR during initialization: %s", e)

        logger.info("Initialized %d MCP servers", len(tool_manager.clients))
        agent.available_tools = await tool_manager.list_all_tools()
        logger.info("Available tools: %s", list(agent.available_tools.keys()))

        tool_prompt = format_tools_for_system_prompt(agent.available_tools)
        agent.system_prompt = (
            base_system_prompt + "\n\n" + tool_prompt
            if base_system_prompt
            else tool_prompt
        )
    else:
        agent.system_prompt = base_system_prompt


@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and dependencies."""
    llm_status = "unknown"
    try:
        response = requests.get("http://localhost:30000/v1/models", timeout=2)
        llm_status = "running" if response.status_code == 200 else "error"
    except:
        llm_status = "not_running"

    return JSONResponse(
        content={"status": "ok", "llm_server": llm_status, "port": config.get("app.port")}
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OAI-compatible endpoint wrapping the full ArkOS agent."""
    payload = await request.json()

    messages = payload.get("messages", [])
    model = payload.get("model", "ark-agent")
    stream = payload.get("stream", False)

    user_id = request.headers.get("X-User-ID") or payload.get("user") or payload.get("user_id")

    context_msgs = []
    context_msgs.append(SystemMessage(content=agent.system_prompt))

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            context_msgs.append(SystemMessage(content=content))
        elif role == "user":
            context_msgs.append(UserMessage(content=content))
        elif role == "assistant":
            context_msgs.append(AIMessage(content=content))

    if stream:
        async def generate_stream():
            chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            async for chunk in agent.step_stream(context_msgs, user_id=user_id):
                data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": chunk},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(data)}\n\n"

            final_data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    agent_response = await agent.step(context_msgs, user_id=user_id)
    final_msg = agent_response or AIMessage(content="(no response)")

    completion = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": final_msg.content},
            "finish_reason": "stop",
        }],
    }

    return JSONResponse(content=completion)


if __name__ == "__main__":
    uvicorn.run(
        "base_module.app:app",
        host=config.get("app.host"),
        port=int(config.get("app.port")),
        reload=config.get("app.reload"),
    )
