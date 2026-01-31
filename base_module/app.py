import os
import sys
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Standard boilerplate for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent_module.agent import Agent
from config_module.loader import config
from memory_module.memory import Memory
from model_module.ArkModelNew import AIMessage, ArkModelLink, Message, SystemMessage, UserMessage
from state_module.state_handler import StateHandler

app = FastAPI(title="ArkOS Agent API", version="1.0.0")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent and dependencies once

flow = StateHandler(yaml_path=config.get("state.graph_path"))


memory = Memory(
    user_id=config.get("memory.user_id"),
    session_id=None,
    db_url=config.get("database.url"),
)

# ArkModelLink uses AsyncOpenAI internally
llm = ArkModelLink(base_url=config.get("llm.base_url"))
agent = Agent(agent_id=config.get("memory.user_id"), flow=flow, memory=memory, llm=llm)


@app.get("/health")
async def health_check():
    """Health check endpoint to verify server and dependencies."""
    import httpx

    llm_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:30000/v1/models", timeout=2.0)
            llm_status = "running" if response.status_code == 200 else "error"
    except Exception:
        llm_status = "not_running"

    return JSONResponse(content={"status": "ok", "llm_server": llm_status, "port": config.get("app.port")})


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OAI-compatible endpoint wrapping the full ArkOS agent."""
    # Awaiting request.json() is correct for FastAPI's async handling of the request body
    payload = await request.json()

    messages = payload.get("messages", [])
    model = payload.get("model", "ark-agent")
    _ = payload.get("response_format")  # Reserved for future use

    context_msgs: list[Message] = []

    context_msgs.append(SystemMessage(content=config.get("app.system_prompt")))

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

    # *** THE CRITICAL CHANGE: AWAIT the agent's step method ***
    # This prevents the 'coroutine' object has no attribute 'content' error.
    agent_response = await agent.step(context_msgs)

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
