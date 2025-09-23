import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from openai import OpenAI


# --- Custom Message Classes ---
# These classes define the structure for different types of messages
# in the conversation, replacing Langchain's BaseMessage, AIMessage, HumanMessage.
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
    """ Represents a message from a tool call"""
    role: str = "tool"

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
    """

    model_name: str = Field(default="tgi")
    base_url: str = Field(default="http://0.0.0.0:30000/v1")
    max_tokens: int = Field(default=1024)
    temperature: float = Field(default=0.7)
    # tools: Optional[List[CustomTool]] = Field(default_factory=list)

    # def _convert_tools_to_openai_format(self) -> Optional[List[Dict[str, Any]]]:
    #     """
    #     Converts the list of internal CustomTool objects into the
    #     list of OpenAI function schemas required by the API.
    #     """
    #     if not self.tools:
    #         return None
    #     return [tool.to_openai_function_schema() for tool in self.tools]

    # def _get_tool_by_name(self, name: str) -> Optional[CustomTool]:
    #     """
    #     Retrieves a CustomTool object from the internal list by its name.
    #     """
    #     return next((tool for tool in self.tools if tool.name == name), None)

    def make_llm_call(
        self, messages: List[Message], json_schema: Optional, stream=False
    ) -> Dict[str, Any]:
        """
        Makes a call to the OpenAI-compatible LLM endpoint.

        Args:
            messages: A list of custom Message objects representing the conversation history.
            json_schema: An optional schema to expose to the LLM.

        Returns:
            A dictionary containing:
            - 'schema_result': A dictionary containing then result of the schema
            - 'message': The content of the LLM's text response.
        """
        client = OpenAI(
            base_url=self.base_url,
            api_key="-",
        )

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
                openai_messages_payload.append(
                    {"role": "tool", "content": msg.content}
                )


            elif isinstance(msg, AIMessage):
                msg_dict = {"role": "assistant"}
                # Always include 'content' key for assistant messages.
                # If msg.content is None, set it to an empty string.
                msg_dict["content"] = msg.content if msg.content is not None else ""
                openai_messages_payload.append(msg_dict)
            else:
                print(type(msg))
                print(msg)
                raise ValueError("Unsupported Message Type ArkModel.py")

        try:

            # Call the OpenAI API chat completions endpoint.
            chat_completion = client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages_payload,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format=json_schema,
            )
            message_from_llm = chat_completion.choices[0].message.content

            return message_from_llm

        except Exception as e:
            print(f"Error during LLM call: {e}")
            return f"Error: An error occurred during LLM call: {e}"
        if stream:
            raise NotImplementedError

    def generate_response(self, messages: List[Message], json_schema) -> AIMessage:
        """
        Generates a response from the model

        Args:
            initial_messages: The initial list of messages to start the conversation.

        Returns:
            An AIMessage object containing the final response content
        """

        conversation_history = messages

        response = self.make_llm_call(conversation_history, json_schema=json_schema)

        # this can be a schema or a regular message response
        return response

    # def bind_tools(self, tools: List[CustomTool]) -> "ArkModelLink":
    #     """
    #     Adds a list of CustomTool objects to the model instance,
    #     making them available for the LLM to use.
    #     """
    #     self.tools.extend(tools)
    #     return self``


if __name__ == "__main__":
    print("Initializing ArkModelLink...")
    model = ArkModelLink(base_url="http://0.0.0.0:30000/v1")

    messages = [UserMessage(content="Give me a simple product listing.")]
    schema = {
        "type": "json_schema",
        "json_schema": {
            "type": "object",
            "properties": {
                "product_name": {"type": "string"},
                "price": {"type": "number"},
                "in_stock": {"type": "boolean"},
            },
            "required": ["product_name", "price", "in_stock"],
        },
    }
    schema = json.dumps(schema)
    result = model.make_llm_call(messages, json_schema=schema)

    print(result)

    # --- Define Dummy Tools ---
    # These are example tools that the LLM can decide to call.
    # class GetCurrentWeatherTool(CustomTool):
    #     name: str = "get_current_weather"
    #     description: str = "Get the current weather in a given location."
    #     args_schema: Dict[str, Any] = {
    #         "type": "object",
    #         "properties": {
    #             "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
    #             "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "fahrenheit"},
    #         },
    #         "required": ["location"],
    #     }

    #     def invoke(self, args: Dict[str, Any]) -> str:
    #         location = args.get("location")
    #         unit = args.get("unit", "fahrenheit")
    #         print(f"DEBUG: Invoking get_current_weather for '{location}' in unit '{unit}'")
    #         # Simulate an external API call to get weather data
    #         if "San Francisco, CA" == location:
    #             return f"The current weather in San Francisco, CA is 72 degrees {unit} and sunny."
    #         elif "New York, NY" == location:
    #             return f"The current weather in New York, NY is 60 degrees {unit} and cloudy with a chance of rain."
    #         else:
    #             return f"Weather data not available for '{location}'."

    # class GetCurrentTimeTool(CustomTool):
    #     name: str = "get_current_time"
    #     description: str = "Get the current time in a specified timezone."
    #     args_schema: Dict[str, Any] = {
    #         "type": "object",
    #         "properties": {
    #             "timezone": {"type": "string", "description": "The timezone to get the time for, e.g., 'America/Los_Angeles'"}
    #         },
    #         "required": ["timezone"]
    #     }

    #     def invoke(self, args: Dict[str, Any]) -> str:
    #         import datetime
    #         import pytz # Requires 'pytz' library: pip install pytz
    #         timezone_str = args.get("timezone", "UTC")
    #         try:
    #             tz = pytz.timezone(timezone_str)
    #             now = datetime.datetime.now(tz)
    #             return f"The current time in {timezone_str} is {now.strftime('%H:%M:%S')}."
    #         except pytz.UnknownTimeZoneError:
    #             return f"Error: Unknown timezone '{timezone_str}'."
    #         except Exception as e:
    #             return f"Error getting time: {e}"
