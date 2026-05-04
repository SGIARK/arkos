# agent.py

import json
import os
import sys
import time
from enum import Enum
from typing import Any

from pydantic import Field, create_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from memory_module.memory import Memory

# Assuming ArkModelLink.generate_response is actually ArkModelLink.agenerate_response
from model_module.ArkModelNew import AIMessage, ArkModelLink, SystemMessage
from state_module.core.base_state import StateOutput
from state_module.core.state_handler import StateHandler

MAX_ITER = 10


class Agent:
    """Default agent class that orchestrates state transitions, LLM calls, and tool usage."""

    def __init__(
        self,
        agent_id: str,
        flow: StateHandler,
        memory: Memory,
        llm: ArkModelLink,
        tool_manager=None,
    ):
        """Initialize the agent with a state graph, memory backend, LLM, and optional tool manager."""
        self.agent_id = agent_id
        self.flow = flow
        self.memory = memory
        self.llm = llm
        self.tool_manager = tool_manager
        self.current_state = self.flow.get_initial_state()
        self.system_prompt = None

        self.startup_flag = True
        self.tools = []
        self.tool_names = []
        self.available_tools = {}
        self.current_user_id = None  # Set per-request for per-user tool auth
        self.last_state_output: StateOutput | None = None
        # Subagents override these. The executor graph uses them to iterate plan steps.
        self.task_id: str | None = None
        self.plan_steps: list[str] = []
        self.step_idx: int = 0
        self.max_iter: int = MAX_ITER

    # def bind_tool(self, tool):
    #
    #    self.tool.append(tool)

    # def find_downloaded_tool(self, embedding):
    #    tool = Tool.pull_tool_from_registry(embedding)
    #    tool_name = tool.tool
    #    self.bind_tool(tool)
    #    self.tool_names.append(tool_name)

    def fill_tool_args_class(self, tool_name: str, tool_args: dict[str, Any]):
        """
        Returns a Pydantic object whose .model_dump() is:
          {"tool_name": <tool_name>, "tool_args": <tool_args>}
        """

        ToolCall = create_model(
            "ToolCall",
            tool_name=(str, Field(description="Tool name to execute")),
            tool_args=(
                dict[str, Any],
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

    def create_next_state_class(self, options: list[tuple[str, str]]):
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

        transition_tuples = list(zip(transitions_dict["tt"], transitions_dict["td"], strict=False))
        prompt = (
            f"given the context of the conversation and the following state options "
            f"{transition_tuples} output the most reasonable next state. "
            f"do not use tool result to determine the next state"
        )

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

    async def add_context(self, messages):
        """
        processes incoming messages for memory module
        """

        assert isinstance(messages, list), "agent.py messages not a list"

        for message in messages:
            await self.memory.add_memory(message)

        return None

    async def get_context(self, turns=5, include_long_term=True):
        """
        Wrap long term and short term into context window.
        output: list of messages
        """
        short_term_mem = await self.memory.retrieve_short_memory(turns)

        if include_long_term:
            long_term_mem = await self.memory.retrieve_long_memory(context=short_term_mem)
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
        await self.add_context(messages)
        print(f"[TIMING] add_context: {time.time() - t0:.3f}s")

        print("agent.py received message")

        self.last_state_output = None
        retry_count = 0
        print("agent.py CURR STATE: ", self.current_state)
        print("agent.py IS TERMINAL?:", self.current_state.is_terminal)

        while True:
            loop_start = time.time()
            print(f"Inner loop #{retry_count + 1}")

            if retry_count > self.max_iter:
                print("MAX ITER REACHED")
                break
            retry_count += 1

            t0 = time.time()
            context = await self.get_context()
            print(f"[TIMING] get_context: {time.time() - t0:.3f}s")

            t0 = time.time()
            update = await self.current_state.run(context, self)
            print(f"[TIMING] state.run: {time.time() - t0:.3f}s")
            print(f"[TIMING] loop total: {time.time() - loop_start:.3f}s")
            if update:
                assert isinstance(update, StateOutput), "State's output was not instance StateOutput"
                self.last_state_output = update
                if update.content:
                    await self.add_context([AIMessage(content=update.content)])

            if self.current_state.is_terminal:
                print("REACHED TERMINAL")
                break

            messages_list = await self.memory.retrieve_short_memory(5)
            if self.current_state.check_transition_ready(messages_list):
                transition_dict = self.flow.get_transitions(self.current_state, messages_list)
                transition_names = transition_dict["tt"]

                # Deterministic override: a state can force the next transition
                # by putting "next_state" into StateOutput.structured_data. This is
                # how executor-style graphs bypass LLM-guided transition choice.
                forced = None
                if update and isinstance(update.structured_data, dict):
                    forced = update.structured_data.get("next_state")
                if forced and forced in transition_names:
                    next_state_name = forced
                elif len(transition_names) == 1:
                    next_state_name = transition_names[0]
                else:
                    next_state_name = await self.choose_transition(transition_dict, messages_list)

                self.current_state = self.flow.get_state(next_state_name)
                print("agent.py CURR STATE: ", self.current_state)

            else:
                print("REACHED NO NEXT STATE")
                break  # No transition ready, exit gracefully

        print(f"[TIMING] step total: {time.time() - step_start:.3f}s")
        print("LAST_STATE_OUTPUT", self.last_state_output)
        self.current_state = self.flow.get_initial_state()
        return self.last_state_output

    async def step_stream(self, messages, user_id: str = None):
        """
        Streaming version of step. Runs full state machine, streams output at state boundaries.

        Yields:
            str: Characters/chunks from each state's output
        """
        self.current_user_id = user_id
        await self.add_context(messages)

        print("agent.py [STREAM] received message")
        print("agent.py [STREAM] CURR STATE:", self.current_state)
        print("agent.py [STREAM] IS TERMINAL?:", self.current_state.is_terminal)

        retry_count = 0

        while True:
            print(f"agent.py [STREAM] Inner loop - State: {self.current_state.name}")

            if retry_count > self.max_iter:
                print("agent.py [STREAM] MAX ITER REACHED")
                yield "\n[Max iterations reached]"
                break
            retry_count += 1

            context = await self.get_context()

            # Run the state normally (same as non-streaming step)
            try:
                update = await self.current_state.run(context, self)
                print(f"agent.py [STREAM] State returned: {type(update).__name__}")
            except Exception as e:
                print(f"agent.py [STREAM] State error: {e}")
                update = StateOutput(
                    content=f"Error: {str(e)[:200]}",
                    completion_signal="error",
                    error_detail=str(e),
                )
                self.current_state = self.flow.get_state("agent_reply")

            if update:
                assert isinstance(update, StateOutput), "State's output was not instance StateOutput"
                self.last_state_output = update
                if update.content:
                    await self.add_context([AIMessage(content=update.content)])
                    print(f"agent.py [STREAM] Streaming {len(update.content)} chars")
                    for char in update.content:
                        yield char

            # Check terminal
            if self.current_state.is_terminal:
                print("agent.py [STREAM] REACHED TERMINAL")
                break

            # Handle state transition (same logic as non-streaming step)
            messages_list = await self.memory.retrieve_short_memory(5)
            if self.current_state.check_transition_ready(messages_list):
                transition_dict = self.flow.get_transitions(self.current_state, messages_list)
                transition_names = transition_dict["tt"]
                print(f"agent.py [STREAM] Transitions: {transition_names}")

                forced = None
                if update and isinstance(update.structured_data, dict):
                    forced = update.structured_data.get("next_state")
                if forced and forced in transition_names:
                    next_state_name = forced
                elif len(transition_names) == 1:
                    next_state_name = transition_names[0]
                else:
                    next_state_name = await self.choose_transition(transition_dict, messages_list)

                print(f"agent.py [STREAM] -> {next_state_name}")
                self.current_state = self.flow.get_state(next_state_name)

                # Separator between states (if continuing)
                if not self.current_state.is_terminal:
                    yield "\n\n"
            else:
                print("agent.py [STREAM] No transition ready")
                break

        print("agent.py [STREAM] Complete")
        self.current_state = self.flow.get_initial_state()


if __name__ == "__main__":
    pass
