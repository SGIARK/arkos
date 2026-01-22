from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
import time
import uuid
import os
import sys

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
)

# Default system prompt for the agent

# ArkModelLink now uses AsyncOpenAI internally
llm = ArkModelLink(base_url=config.get("llm.base_url"))

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


@app.on_event("startup")
async def startup():
    if tool_manager:
        await tool_manager.initialize_servers()
        print(f"Initialized {len(tool_manager.clients)} MCP servers")
        # Cache tools on agent
        agent.available_tools = await tool_manager.list_all_tools()
        print(f"Initialized {len(tool_manager.clients)} MCP servers")
        print(f"Available tools: {list(agent.available_tools.keys())}")


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
        content={"status": "ok", "llm_server": llm_status, "port": 1111}
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OAI-compatible endpoint wrapping the full ArkOS agent."""
    # Awaiting request.json() is correct for FastAPI's async handling of the request body
    payload = await request.json()

    messages = payload.get("messages", [])
    model = payload.get("model", "ark-agent")
    response_format = payload.get("response_format")

    # Extract user_id from header or body for per-user tool auth
    user_id = request.headers.get("X-User-ID") or payload.get("user") or payload.get("user_id")

    context_msgs = []

    context_msgs.append(SystemMessage(content=config.get("app.system_prompt")))

    # Convert OAI messages into internal message objects
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        # Handling for tool calls, which is often crucial in OAI-compatible APIs
        # if role == "tool" and msg.get("tool_call_id"):
        #     context_msgs.append(ToolMessage(content=content, tool_call_id=msg["tool_call_id"]))
        # elif role == "assistant" and msg.get("tool_calls"):
        #     context_msgs.append(AIMessage(content=content, tool_calls=msg["tool_calls"]))
        # else:
        if role == "system":
            context_msgs.append(SystemMessage(content=content))
        elif role == "user":
            context_msgs.append(UserMessage(content=content))
        elif role == "assistant":
            # Assuming a simple assistant message here for brevity
            context_msgs.append(AIMessage(content=content))
        # Note: You may need to refine the message parsing logic to correctly handle
        # tool_calls and tool_messages if your agent uses them heavily.

    # *** THE CRITICAL CHANGE: AWAIT the agent's step method ***
    # This prevents the 'coroutine' object has no attribute 'content' error.
    agent_response = await agent.step(context_msgs, user_id=user_id)

    # Handle the case where the agent might return None (though it should return an AIMessage)
    final_msg = agent_response or AIMessage(content="(no response)")

    # Format as OpenAI chat completion response
    completion = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                # Now final_msg is guaranteed to be an AIMessage object (or placeholder)
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
