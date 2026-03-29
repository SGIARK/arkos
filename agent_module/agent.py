import os
import sys
import logging
from pydantic import create_model, Field
from typing import List, Tuple, Dict, Any
import json
from enum import Enum

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from state_module.state_handler import StateHandler

from model_module.ArkModelNew import ArkModelLink, AIMessage, SystemMessage
from memory_module.memory import Memory

logger = logging.getLogger(__name__)

MAX_ITER = 10


class Agent:
    def __init__(
        self,
        agent_id: str,
        flow: StateHandler,
        memory: Memory,
        llm: ArkModelLink,
        tool_manager=None,
    ):
        self.agent_id = agent_id
        self.flow = flow
        self.memory = memory
        self.llm = llm
        self.tool_manager = tool_manager
        self.current_state = self.flow.get_initial_state()
        self.system_prompt = None
        self.available_tools = {}
        self.current_user_id = None

    async def create_tool_option_class(self):
        """
        Returns a Pydantic model class with a single field 'tool_name',
        whose value must be one of the available tool IDs.
        """
        server_tool_map = await self.tool_manager.list_all_tools()

        enum_members = {}
        for server_name in server_tool_map:
            for tool_name in server_tool_map[server_name]:
                enum_members[tool_name] = tool_name

        ToolEnum = Enum("ToolEnum", enum_members)

        ToolOptionsModel = create_model(
            "ToolCall",
            tool_name=(
                ToolEnum,
                Field(description="The name of the tool to execute next"),
            ),
        )

        return ToolOptionsModel

    def create_next_state_class(self, options: List[Tuple[str, str]]):
        """
        options: list of tuples (next_state, description of state)
        Returns a Pydantic model class with a single field 'next_state',
        whose value must be one of the provided state names.
        """
        enum_dict = {state: state for state, desc in options}
        next_state_enum = Enum("NextStateEnum", enum_dict)
        next_state_model = create_model(
            "NextState",
            next_state=(
                next_state_enum,
                Field(..., description="The chosen next state"),
            ),
        )
        return next_state_model

    async def call_llm(self, context=None, json_schema=None):
        llm_response = await self.llm.generate_response(context, json_schema)
        return AIMessage(content=llm_response)

    async def choose_transition(self, transitions_dict, messages):
        transition_tuples = list(zip(transitions_dict["tt"], transitions_dict["td"]))

        tool_guidance = ""
        if "use_tool" in transitions_dict["tt"] and self.available_tools:
            tool_names = []
            for server_tools in self.available_tools.values():
                tool_names.extend(server_tools.keys())
            tool_guidance = f"""

Available tools: {', '.join(tool_names)}

CRITICAL: If the user's request requires a tool (e.g., calendar events, search, file operations), ALWAYS choose 'use_tool'.
- If the user says "try again", "check again", "look again", or similar retry phrases, ALWAYS choose 'use_tool'
- Tools handle authentication automatically - do NOT tell users to authenticate first
- Do NOT manually generate auth URLs - the tool system handles that
- Only choose 'agent_reply' if the request is purely conversational and needs NO external data"""

        prompt = f"""Given the conversation context and available state options {transition_tuples}, choose the most appropriate next state.{tool_guidance}

Do not use tool results to determine the next state."""

        NextStates = self.create_next_state_class(transition_tuples)
        json_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "class_options",
                "schema": NextStates.model_json_schema(),
            },
        }

        context_text = [SystemMessage(content=prompt)] + messages
        output = await self.call_llm(context=context_text, json_schema=json_schema)
        structured_output = json.loads(output.content)
        return structured_output["next_state"]

    def add_context(self, messages):
        assert isinstance(messages, list), "agent.py messages not a list"
        for message in messages:
            self.memory.add_memory(message)

    def get_context(self, turns=5, include_long_term=True):
        short_term_mem = self.memory.retrieve_short_memory(turns)
        if include_long_term:
            long_term_mem = self.memory.retrieve_long_memory(context=short_term_mem)
            if long_term_mem and long_term_mem.content.strip():
                return [long_term_mem] + short_term_mem
        return short_term_mem

    async def step(self, messages, user_id: str = None):
        self.current_user_id = user_id
        self.add_context(messages)

        messages_list = self.memory.retrieve_short_memory(5)
        entry = self.flow.get_entry_transitions()
        if len(entry["tt"]) > 1:
            first_state_name = await self.choose_transition(entry, messages_list)
        else:
            first_state_name = entry["tt"][0]
        self.current_state = self.flow.get_state(first_state_name)

        last_ai_message = None
        retry_count = 0

        while not self.current_state.is_terminal:
            if retry_count > MAX_ITER:
                logger.warning("Max iterations reached")
                break
            retry_count += 1

            context = self.get_context()
            update = await self.current_state.run(context, self)

            if update:
                self.add_context([update])
                if isinstance(update, AIMessage):
                    last_ai_message = update

            if self.current_state.is_terminal:
                break

            messages_list = self.memory.retrieve_short_memory(5)
            if self.current_state.check_transition_ready(messages_list):
                transition_dict = self.flow.get_transitions(self.current_state, messages_list)
                transition_names = transition_dict["tt"]

                if len(transition_names) == 1:
                    next_state_name = transition_names[0]
                else:
                    next_state_name = await self.choose_transition(transition_dict, messages_list)

                logger.info("Transition -> %s", next_state_name)
                self.current_state = self.flow.get_state(next_state_name)
            else:
                break

        self.current_state = self.flow.get_initial_state()
        return last_ai_message

    async def step_stream(self, messages, user_id: str = None):
        self.current_user_id = user_id
        self.add_context(messages)

        messages_list = self.memory.retrieve_short_memory(5)
        entry = self.flow.get_entry_transitions()
        if len(entry["tt"]) > 1:
            first_state_name = await self.choose_transition(entry, messages_list)
        else:
            first_state_name = entry["tt"][0]
        self.current_state = self.flow.get_state(first_state_name)

        retry_count = 0
        last_ai_message = None

        while not self.current_state.is_terminal:
            if retry_count > MAX_ITER:
                yield "\n[Max iterations reached]"
                break
            retry_count += 1

            context = self.get_context()

            try:
                update = await self.current_state.run(context, self)
            except Exception as e:
                logger.error("State error in %s: %s", self.current_state.name, e)
                update = AIMessage(content=f"I encountered an error: {str(e)[:200]}. Please try rephrasing your request.")
                self.current_state = self.flow.get_state("ask_user")
                yield update.content
                break

            if update and hasattr(update, 'content') and update.content:
                self.add_context([update])
                if isinstance(update, AIMessage):
                    last_ai_message = update

            if self.current_state.is_terminal:
                break

            messages_list = self.memory.retrieve_short_memory(5)
            if self.current_state.check_transition_ready(messages_list):
                transition_dict = self.flow.get_transitions(self.current_state, messages_list)
                transition_names = transition_dict["tt"]

                if len(transition_names) == 1:
                    next_state_name = transition_names[0]
                else:
                    next_state_name = await self.choose_transition(transition_dict, messages_list)

                logger.info("Transition -> %s", next_state_name)
                self.current_state = self.flow.get_state(next_state_name)
            else:
                break

        if last_ai_message and last_ai_message.content:
            for char in last_ai_message.content:
                yield char

        self.current_state = self.flow.get_initial_state()


if __name__ == "__main__":
    pass
