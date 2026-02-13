from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
import time
import uuid
import os
import sys
import json

# Standard boilerplate for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config_module.loader import config
from agent_module.agent import Agent
from state_module.state_handler import StateHandler
from memory_module.memory import Memory
from model_module.ArkModelNew import ArkModelLink, UserMessage, SystemMessage, AIMessage
from tool_module.tool_call import MCPToolManager
from tool_module.token_store import UserTokenStore
from base_module.auth import router as auth_router


app = FastAPI(title="ArkOS Agent API", version="1.0.0")
app.include_router(auth_router)


# Initialize the agent and dependencies once

flow = StateHandler(yaml_path=config.get("state.graph_path"))


memory = Memory(
    user_id=config.get("memory.user_id"),
    session_id=None,
    db_url=config.get("database.url"),
    use_long_term=config.get("memory.use_long_term", False),  # Disabled for speed
)

# Default system prompt for the agent

# ArkModelLink now uses AsyncOpenAI internally
llm = ArkModelLink(base_url=config.get("llm.base_url"), max_tokens=config.get("llm.max_tokens"))



# Token store for per-user MCP authentication
token_store = UserTokenStore(config.get("database.url"))

mcp_config = config.get("mcp_servers")
tool_manager = MCPToolManager(mcp_config, token_store=token_store) if mcp_config else None
agent = Agent(
    agent_id=config.get("memory.user_id"),
    flow=flow,
    memory=memory,
    llm=llm,
    tool_manager=tool_manager,
)

def format_tools_for_system_prompt(tools: dict) -> str:
    """
    tools: {tool_name: ToolDefinition}
    ToolDefinition is whatever MCPToolManager.list_all_tools() returns.
    """
    lines = []
    lines.append("You have access to the following tools.")
    lines.append("Use them when appropriate. Only call tools that are listed below.")
    lines.append("")

    for name, tool in tools.items():
        lines.append(f"Tool name: {name}")
        if getattr(tool, "description", None):
            lines.append(f"Description: {tool.description}")
        if getattr(tool, "input_schema", None):
            lines.append("Input schema:")
            lines.append(str(tool.input_schema))
        lines.append("")

    return "\n".join(lines)

@app.on_event("startup")
async def startup():
        
    base_system_prompt = (config.get("app.system_prompt") or "").strip()

    system_prompt=None

    if tool_manager:
        await tool_manager.initialize_servers()

        print(f"Initialized {len(tool_manager.clients)} MCP servers")
        # Cache tools on agent
        agent.available_tools = await tool_manager.list_all_tools()
        print(f"Initialized {len(tool_manager.clients)} MCP servers")
        print(f"Available tools: {list(agent.available_tools.keys())}")


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
    import requests

    llm_status = "unknown"
    try:
        response = requests.get("http://localhost:30000/v1/models", timeout=2)
        llm_status = "running" if response.status_code == 200 else "error"
    except:
        llm_status = "not_running"

    return JSONResponse(
        content={"status": "ok", "llm_server": llm_status, "port": 1112}
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

    context_msgs = []
    context_msgs.append(SystemMessage(content=agent.system_prompt))

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

            # Send final chunk
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

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
        )

    # Non-streaming response
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
