# agent.py
from typing import List, Tuple
import json
from enum import Enum
from pydantic import create_model, Field
from state_module.state_handler import StateHandler
from model_module.ArkModelNew import ArkModelLink, AIMessage, SystemMessage
from tool_module.tool import Tool
from memory_module.memory import Memory



multiply_tool = {"url": "https://localhost:4040", "parameters": None}
add_tool = {"url": "https://localhost:3030", "parameters": None}
tool_registry = {"keyA": multiply_tool, "embedding_b": add_tool}
# TODO: add tool registry, navegable by directory


class Agent:
    def __init__(
        self, agent_id: str, flow: StateHandler, memory: Memory, llm: ArkModelLink
    ):
        self.agent_id = agent_id
        self.flow = flow
        self.memory = memory
        self.llm = llm
        self.current_state = self.flow.get_initial_state()
        self.context: Dict[str, Any] = {}

        self.tools = []
        self.tool_names = []

    def bind_tool(tool):

        self.tool.append(tool)

    def find_downloaded_tool(embedding):
        tool = Tool.pull_tool_from_registry(embedding)
        tool_name = tool.tool
        self.bind_tool(tool)
        self.tool_names.append(tool_name)
    def create_next_state_class(self, options: List[Tuple[str, str]]):
        """
        options: list of tuples (next_state, description of state)
        Returns a Pydantic model class with a single field 'next_state',
        whose value must be one of the provided state names.
        """

        # Dynamically build an Enum of allowed states
        enum_dict = {state: state for state, desc in options}
        
        # add desc into enum dict
        NextStateEnum = Enum("NextStateEnum", enum_dict)

        # Build the model with a single constrained field
        NextStateModel = create_model(
            "NextState",
            next_state=(NextStateEnum, Field(..., description="The chosen next state"))

        )

        return NextStateModel

    def call_llm(self, input=None, context=None, json_schema=None):
        """
        Agent's interface with chat model
        input: messages (list), json_schema (json)

        output: AI Message
        """

        chat_model = self.llm
        if context:

            llm_response = chat_model.generate_response(context, json_schema)

        else:
            messages = [SystemMessage(content=input)]
            llm_response = chat_model.generate_response(messages, json_schema)

        return AIMessage(content=llm_response)

    def choose_transition(self, transitions_dict, messages):

        prompt = "given the following state transitions, and the preceeding context. output the most reasonable next state. do not use tool result to determine the next state"
        transition_tuples = list(zip(transitions_dict["tt"], transitions_dict["td"]))

        # creates pydantic class and a model dump 
        NextStates = self.create_next_state_class(transition_tuples)
        json_schema = {
            "type": "json_schema",

            "json_schema": {
                "name":  "class_options", 
                "schema": NextStates.model_json_schema()
            }
        }


        context_text = [SystemMessage(content=prompt)] + messages
        output = self.call_llm(context=context_text, json_schema=json_schema)
        # print(output.content)
        structured_output = json.loads(output.content)

        # HANDLE ERROR GRACEFULL
        if "error" in output.content:
            raise ValueError("AGENT.PY FAILED LLM CALL")
        next_state_name  = structured_output["next_state"]

        return next_state_name

    def step(self, input: str):
        messages_list = self.context["messages"]
        # get first state from states
        if not self.current_state:
            self.current_state = flow.get_initial_state()

        
        while not self.current_state.is_terminal:

            # asumption run will always return a valid Messsage
            updates = self.current_state.run(self.context["messages"], self)
            if updates:
                messages_list.append(updates)

            if self.current_state.is_terminal:
                return  # or break the loop


            if self.current_state.check_transition_ready(self.context["messages"]):
                # NOTE: get_transitions will return list of tuples (state, state description)
                transition_dict = self.flow.get_transitions(self.current_state, messages_list)
                transition_names = transition_dict["tt"]

                num_transitions = len(transition_names)
                if num_transitions == 1: 
                    next_state_name = transition_names[0]
                    self.current_state = self.flow.get_state(next_state_name)
                else: 

                    next_state_name = self.choose_transition(transition_dict, self.context["messages"]) 

                    next_state  = self.flow.get_state(next_state_name)

                    self.current_state = next_state

            else:
                raise ValueError("State Loop Failed, No next State agent.py")

        return
