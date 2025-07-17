import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, Response # Added Response import

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, AsyncIterator, Union
import asyncio
import json
import time
import os
import sys # Added to support sys.path.append

# Assuming these are available from ark_model_refactor.py
# Adjust path as necessary based on your project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_module')))
from ArkModelRefactored import ArkModelLink, UserMessage, AIMessage, Message

# --- Pydantic Models for Request and Response (Simplified) ---
class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = "tgi" # Default to 'tgi' as our LLM
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7 # Default temperature to match LLM

# For non-streaming responses, mimic essential parts of OpenAI response
class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]

# --- FastAPI Application ---
app = FastAPI(title="Simple Chat Completions API")

# Initialize LLM. Base URL for TGI server. Can be overridden by environment variable.
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
llm_model = ArkModelLink(base_url=LLM_BASE_URL)

@app.post("/v1/chat/completions", response_model=None) # Changed response_model to None
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Creates a chat completion or streams chunks of a chat completion.
    """
    input_messages: List[Message] = []
    for msg in request.messages:
        # Convert incoming ChatMessage to internal Message types
        if msg.role == "user":
            input_messages.append(UserMessage(content=msg.content or ""))
        elif msg.role == "assistant":
            input_messages.append(AIMessage(content=msg.content or ""))
        elif msg.role == "system": # Treat system as user for simplicity
            input_messages.append(UserMessage(content=msg.content or ""))
        # Tool messages from the API request are not directly handled in this simplified flow.

    if request.stream:
        async def generate_stream():
            async for chunk in llm_model.astream_response(input_messages):
                if chunk.content:
                    # Corrected: Changed outer f-string delimiters to triple double quotes
                    yield f"""data: {json.dumps({
                        'id': f"chatcmpl-stream-{int(time.time() * 1000)}",
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': request.model,
                        'choices': [{
                            'index': 0,
                            'delta': {'content': chunk.content},
                            'finish_reason': None
                        }]
                    })}\n\n"""
            yield "data: [DONE]\n\n"
        return StreamingResponse(generate_stream(), media_type="text/event-stream")
    else:
        # Non-streaming response
        final_llm_response = llm_model.generate_response(input_messages)
        response_message = ChatMessage(role="assistant", content=final_llm_response.content or "")
        response_payload = ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=response_message)]
        )
        return JSONResponse(content=response_payload.model_dump(exclude_none=True))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
