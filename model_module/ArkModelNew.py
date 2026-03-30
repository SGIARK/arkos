from typing import Any, AsyncIterator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Import the asynchronous client
from openai import AsyncOpenAI


# --- Custom Message Classes ---
# These classes define the structure for different types of messages
class Message(BaseModel):
    """Base class for all messages."""

    content: str
    role: str


class SystemMessage(Message):
    """Represents a message to the system"""

    role: str = "system"


class UserMessage(Message):
    """Represents a message from the user."""

    role: str = "user"


class ToolMessage(Message):
    """Represents a message from a tool call"""

    role: str = "tool"
    tool_calls: Optional[dict] = None


class AIMessage(Message):
    """
    Represents a message from the AI.
    Can include tool calls if the AI decides to use tools.
    """

    role: str = "assistant"
    # content is now Optional[str] to handle cases where the AI's turn is solely a tool call.
    content: Optional[str] = None

    tool_calls: Optional[dict] = None


class ArkModelLink(BaseModel):
    """
    A custom chat model designed to interface with Hugging Face TGI
    servers that expose an OpenAI-compatible API, supporting tool calling.

    This version uses the AsyncOpenAI client for non-blocking I/O.
    """

    model_name: str = Field(default="tgi")
    base_url: str = Field(default="http://0.0.0.0:30000/v1")
    max_tokens: int = Field(default=4096)  # Increased from 1024 to handle complex structured outputs
    temperature: float = Field(default=0.7)

    # Use a property or method to initialize the client asynchronously if needed,
    # or just create it in the async method, as AsyncOpenAI handles the session.

    # We'll use a property for a lazy, non-async instantiation of the client wrapper.
    # The actual network calls will be awaited inside the method.
    @property
    def client(self) -> AsyncOpenAI:
        """Returns the configured AsyncOpenAI client."""
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key="-",  # Placeholder/Dummy API key
        )

    async def make_llm_call(
        self, messages: List[Message], json_schema: Optional, stream=False
    ) -> Union[Dict[str, Any], str]:
        """
        Makes an ASYNCHRONOUS call to the OpenAI-compatible LLM endpoint.

        Args:
            messages: A list of custom Message objects representing the conversation history.
            json_schema: An optional schema to expose to the LLM.

        Returns:
            The content of the LLM's text response (str) or a detailed dict if streaming.
        """

        # Convert custom Message objects into the format expected by the OpenAI API.

        openai_messages_payload = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                openai_messages_payload.append({"role": "user", "content": msg.content})

            elif isinstance(msg, SystemMessage):
                openai_messages_payload.append(
                    {"role": "system", "content": msg.content}
                )

            elif isinstance(msg, ToolMessage):
                # Note: ToolMessage in OpenAI API usually requires 'tool_call_id'
                # and 'name' if it's a ToolMessage response, but this format
                # (role='tool', content=...) is often used for simple outputs.
                openai_messages_payload.append({"role": "tool", "content": msg.content})

            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant"}
                # Always include 'content' key for assistant messages.
                msg_dict["content"] = msg.content if msg.content is not None else ""
                openai_messages_payload.append(msg_dict)
            else:
                print(type(msg))
                print(msg)
                raise ValueError("Unsupported Message Type ArkModel.py")

        try:
            if stream:
                # The stream logic for an async generator is complex and omitted here
                # but if implemented, it would use client.chat.completions.create(..., stream=True)
                raise NotImplementedError("Asynchronous streaming not yet implemented.")

            # Use the asynchronous client and AWAIT the call
            chat_completion = await self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages_payload,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format=json_schema,
            )

            # The result is now available after the await
            message_from_llm = chat_completion.choices[0].message.content

            return message_from_llm

        except Exception as e:
            # Handle exceptions during the asynchronous API call
            print(f"Error during async LLM call: {e}")
            return f"Error: An error occurred during async LLM call: {e}"

    async def generate_response(self, messages: List[Message], json_schema) -> str:
        """
        ASYNCHRONOUSLY Generates a response from the model.

        This method will be called by your async agent logic (e.g., Agent.call_llm).
        Returns the raw content (which may be a JSON string if schema was used).

        Args:
            messages: The list of messages to send to the model.

        Returns:
            The raw response content (string).
        """

        conversation_history = messages

        # *** AWAIT the asynchronous LLM call ***
        response_content = await self.make_llm_call(
            conversation_history, json_schema=json_schema
        )

        # The response is the string content (either regular text or a JSON string)
        return response_content

    def _format_messages(self, messages: List[Message]) -> List[Dict[str, str]]:
        """Convert Message objects to OpenAI format."""
        formatted = []
        for msg in messages:
            if isinstance(msg, (UserMessage, SystemMessage, ToolMessage)):
                formatted.append({"role": msg.role, "content": msg.content or ""})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content or ""})
        return formatted

    async def generate_stream(self, messages: List[Message]) -> AsyncIterator[str]:
        """Stream tokens as they're generated."""
        openai_messages = self._format_messages(messages)

        try:
            stream = await self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            print(f"Error during streaming: {e}")
            yield f"Error: {e}"
