# agent.py

import os
import sys
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
        self.current_state = self.flow.get_initial_state()

        self.startup_flag = True
        self.tools = []
        self.tool_names = []
        self.available_tools = {}

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
        prompt = f"""given the context of the conversation and the following state options {transition_tuples} output the most reasonable next state. 
                 do not use tool result to determine the next state"""

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

    def get_context(self, turns=5):
        """

        wrap long term and short term into context window
        output: list of messages

        """

        short_term_mem = self.memory.retrieve_short_memory(turns)

        long_term_mem = self.memory.retrieve_long_memory(context=short_term_mem)

        # output = {"relevant_memories": long_term_mem,
        #           "conversation_history": short_term_mem,
        # }

        output = [long_term_mem] + short_term_mem

        return output

    async def step(self, messages):
        """
        Runs the agent until reaching a terminal state or completion.
        Returns the last AIMessage produced.
        """

        # agent.context["messages"].extend(messages)

        ## process messages

        self.add_context(messages)

        print("agent.py recieved message")

        # messages_list = self.context.get("messages", [])
        # messages_list = self.memory.retrieve_memory()
        # if not self.current_state:
        #    print("GETTINT INITIAL")
        #    self.current_state = self.flow.get_initial_state()

        last_ai_message = None

        retry_count = 0
        print("agent.py CURR STATE: ", self.current_state)
        print("agent.py IS TERMINAL?:", self.current_state.is_terminal)

        while not self.current_state.is_terminal:
            print("Inner loop")
            ### DEBUGGING

            if retry_count > MAX_ITER:
                print("MAX ITER REACHED")
                break
            retry_count += 1

            ### DEBUGGING
            # print("MSGS_LIST", messages_list[-1])

            context = self.get_context()
            update = await self.current_state.run(context, self)
            print("inner_loop_update: ", update)
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
        print("LAST_AI_MSG", last_ai_message)
        self.current_state = self.flow.get_state("agent_reply")
        return last_ai_message


if __name__ == "__main__":
    pass
