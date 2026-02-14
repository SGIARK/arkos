# agent.py

import os
import sys
import time
from pydantic import create_model, Field
from typing import List, Tuple, Dict, Any
import json
from enum import Enum


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from state_module.state_handler import StateHandler

# Assuming ArkModelLink.generate_response is actually ArkModelLink.agenerate_response
from model_module.ArkModelNew import ArkModelLink, AIMessage, SystemMessage
from memory_module.memory import Memory


MAX_ITER = 10


class Agent:
    """

    Default agent class
    """

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
        self.system_prompt=None

        self.startup_flag = True
        self.tools = []
        self.tool_names = []
        self.available_tools = {}
        self.current_user_id = None  # Set per-request for per-user tool auth

    # def bind_tool(self, tool):
    #
    #    self.tool.append(tool)

    # def find_downloaded_tool(self, embedding):
    #    tool = Tool.pull_tool_from_registry(embedding)
    #    tool_name = tool.tool
    #    self.bind_tool(tool)
    #    self.tool_names.append(tool_name)

    def fill_tool_args_class(self, tool_name: str, tool_args: Dict[str, Any]):
        """
        Returns a Pydantic object whose .model_dump() is:
          {"tool_name": <tool_name>, "tool_args": <tool_args>}
        """

        ToolCall = create_model(
            "ToolCall",
            tool_name=(str, Field(description="Tool name to execute")),
            tool_args=(
                Dict[str, Any],
                Field(default_factory=dict, description="Tool args"),
            ),
        )

        return ToolCall(tool_name=tool_name, tool_args=tool_args)

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

        # Dynamically build an Enum of allowed states
        enum_dict = {state: state for state, desc in options}

        # add desc into enum dict
        next_state_enum = Enum("NextStateEnum", enum_dict)

        # Build the model with a single constrained field
        next_state_model = create_model(
            "NextState",
            next_state=(
                next_state_enum,
                Field(..., description="The chosen next state"),
            ),
        )

        return next_state_model

    async def call_llm(self, context=None, json_schema=None):
        """
        Agent's interface with chat model
        input: messages (list), json_schema (json)

        output: AI Message
        """

        chat_model = self.llm

        llm_response = await chat_model.generate_response(context, json_schema)

        # else:
        #    messages = [SystemMessage(content=input)]
        #    llm_response = chat_model.generate_response(messages, json_schema)

        return AIMessage(content=llm_response)

    async def choose_transition(self, transitions_dict, messages):
        """
        Chooses subsequent transition in state graph
        """

        transition_tuples = list(zip(transitions_dict["tt"], transitions_dict["td"]))

        # Build tool info if use_tool is an option
        tool_guidance = ""
        if "use_tool" in transitions_dict["tt"] and self.available_tools:
            tool_names = []
            for server_tools in self.available_tools.values():
                tool_names.extend(server_tools.keys())
            tool_guidance = f"""

Available tools: {', '.join(tool_names)}

CRITICAL: If the user's request requires a tool (e.g., calendar events, search, file operations), ALWAYS choose 'use_tool'.
- Tools handle authentication automatically - do NOT tell users to authenticate first
- Do NOT manually generate auth URLs - the tool system handles that
- Only choose 'ask_user' if you need to clarify what the user wants, not for technical issues"""

        prompt = f"""Given the conversation context and available state options {transition_tuples}, choose the most appropriate next state.{tool_guidance}

Do not use tool results to determine the next state."""

        # creates pydantic class and a model dump
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

        next_state_name = structured_output["next_state"]

        return next_state_name

    def add_context(self, messages):
        """
        processes incoming messages for memory module
        """

        assert isinstance(messages, list), "agent.py messages not a list"

        for message in messages:
            self.memory.add_memory(message)

        return None

    def get_context(self, turns=5, include_long_term=True):
        """
        Wrap long term and short term into context window.
        output: list of messages
        """
        short_term_mem = self.memory.retrieve_short_memory(turns)

        if include_long_term:
            long_term_mem = self.memory.retrieve_long_memory(context=short_term_mem)
            # Only include if it has content
            if long_term_mem and long_term_mem.content.strip():
                return [long_term_mem] + short_term_mem

        return short_term_mem

    async def step(self, messages, user_id: str = None):
        """
        Runs the agent until reaching a terminal state or completion.
        Returns the last AIMessage produced.

        Parameters
        ----------
        messages : list
            List of messages to process
        user_id : str, optional
            User ID for per-user tool authentication
        """
        step_start = time.time()

        # Set current user for per-user tool auth
        self.current_user_id = user_id

        t0 = time.time()
        self.add_context(messages)
        print(f"[TIMING] add_context: {time.time() - t0:.3f}s")

        print("agent.py received message")

        last_ai_message = None
        retry_count = 0
        print("agent.py CURR STATE: ", self.current_state)
        print("agent.py IS TERMINAL?:", self.current_state.is_terminal)

        while not self.current_state.is_terminal:
            loop_start = time.time()
            print(f"Inner loop #{retry_count + 1}")

            if retry_count > MAX_ITER:
                print("MAX ITER REACHED")
                break
            retry_count += 1

            t0 = time.time()
            context = self.get_context()
            print(f"[TIMING] get_context: {time.time() - t0:.3f}s")

            t0 = time.time()
            update = await self.current_state.run(context, self)
            print(f"[TIMING] state.run: {time.time() - t0:.3f}s")
            print(f"[TIMING] loop total: {time.time() - loop_start:.3f}s")
            if update:
                # messages_list.append(update)
                update_list = [update]
                self.add_context(update_list)  # add update to memory

                if isinstance(update, AIMessage):
                    last_ai_message = update

            if self.current_state.is_terminal:
                print("REACHED TERMINAL")
                break

            messages_list = self.memory.retrieve_short_memory(5)
            if self.current_state.check_transition_ready(messages_list):
                transition_dict = self.flow.get_transitions(
                    self.current_state, messages_list
                )
                transition_names = transition_dict["tt"]

                if len(transition_names) == 1:
                    next_state_name = transition_names[0]
                else:
                    next_state_name = await self.choose_transition(
                        transition_dict, messages_list
                    )

                self.current_state = self.flow.get_state(next_state_name)
                print("agent.py CURR STATE: ", self.current_state)

            else:
                print("REACHED NO NEXT STATE")
                break  # No transition ready, exit gracefully

        print(f"[TIMING] step total: {time.time() - step_start:.3f}s")
        print("LAST_AI_MSG", last_ai_message)
        self.current_state = self.flow.get_state("agent_reply")
        return last_ai_message

    async def step_stream(self, messages, user_id: str = None):
        """
        Streaming version of step. Runs full state machine, streams output at state boundaries.

        Yields:
            str: Characters/chunks from each state's output
        """
        self.current_user_id = user_id
        self.add_context(messages)

        print("agent.py [STREAM] received message")
        print("agent.py [STREAM] CURR STATE:", self.current_state)
        print("agent.py [STREAM] IS TERMINAL?:", self.current_state.is_terminal)

        retry_count = 0
        last_ai_message = None  # Track the final message to stream

        while not self.current_state.is_terminal:
            print(f"agent.py [STREAM] Inner loop - State: {self.current_state.name}")

            if retry_count > MAX_ITER:
                print("agent.py [STREAM] MAX ITER REACHED")
                yield "\n[Max iterations reached]"
                break
            retry_count += 1

            context = self.get_context()

            # Run the state normally (same as non-streaming step)
            try:
                update = await self.current_state.run(context, self)
                print(f"agent.py [STREAM] State returned: {type(update).__name__}")
            except Exception as e:
                print(f"agent.py [STREAM] State error in {self.current_state.name}: {e}")
                # Return error message to user and transition to terminal
                update = AIMessage(content=f"I encountered an error: {str(e)[:200]}. Please try rephrasing your request.")
                # Force transition to terminal state to stop infinite loops
                self.current_state = self.flow.get_state("ask_user")
                yield update.content
                break

            # Add update to memory
            if update and hasattr(update, 'content') and update.content:
                self.add_context([update])

                # Track AIMessages but don't stream yet (wait for final one)
                if isinstance(update, AIMessage):
                    last_ai_message = update
                    print(f"agent.py [STREAM] AIMessage buffered ({len(update.content)} chars) - will stream at end")
                elif isinstance(update, SystemMessage):
                    print(f"agent.py [STREAM] SystemMessage added to memory ({len(update.content)} chars) - not streamed")

            # Check terminal
            if self.current_state.is_terminal:
                print("agent.py [STREAM] REACHED TERMINAL")
                break

            # Handle state transition (same logic as non-streaming step)
            messages_list = self.memory.retrieve_short_memory(5)
            if self.current_state.check_transition_ready(messages_list):
                transition_dict = self.flow.get_transitions(
                    self.current_state, messages_list
                )
                transition_names = transition_dict["tt"]
                print(f"agent.py [STREAM] Transitions: {transition_names}")

                # If we just presented a tool result, go directly to ask_user (don't loop)
                if (last_ai_message and
                    isinstance(update, AIMessage) and
                    messages_list and
                    len(messages_list) >= 2 and
                    isinstance(messages_list[-2], SystemMessage) and
                    "TOOL_RESULT" in messages_list[-2].content):
                    print(f"agent.py [STREAM] Just presented tool result -> forcing ask_user")
                    next_state_name = "ask_user"
                elif len(transition_names) == 1:
                    next_state_name = transition_names[0]
                else:
                    next_state_name = await self.choose_transition(
                        transition_dict, messages_list
                    )

                print(f"agent.py [STREAM] -> {next_state_name}")
                self.current_state = self.flow.get_state(next_state_name)
            else:
                print("agent.py [STREAM] No transition ready")
                break

        # Stream the final AIMessage to user
        if last_ai_message and last_ai_message.content:
            print(f"agent.py [STREAM] Streaming final response ({len(last_ai_message.content)} chars)")
            for char in last_ai_message.content:
                yield char
        else:
            print("agent.py [STREAM] No final message to stream")

        print("agent.py [STREAM] Complete")
        self.current_state = self.flow.get_state("agent_reply")


if __name__ == "__main__":
    pass
