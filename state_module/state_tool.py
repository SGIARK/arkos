import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from model_module.ArkModelNew import SystemMessage

from state_module.state import State
from state_module.state_registry import register_state


@register_state
class StateTool(State):
    type = "tool"

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.is_terminal = False

    def check_transition_ready(self, context):
        return True



    def choose_tool(self, context, agent):
        """
        Chooses tool to use based on the context and server

        
        """

        prompt="based on the above user request, choose the tool which best satisfies the users request" 
        instructions =  context + [SystemMessage(content=prompt)] 

        tool_option_class = agent.create_tool_option_class()
        tool_name = agent.call_llm(context, tool_option_class)


        server_name = agent.tool_manager._tool_registry_[tool_name]

        tool_args = agent.tool_manager.list_tools[server_name][tool_name]


        fill_tool_args_class = agent.fill_tool_args_class(tool_name, tool_args)

        tool_call = agent.call_llm(context, fill_tool_args_class)



        return tool_call

    def execute_tool(self, tool_call, agent):
        """
        Parses and fills args for chosen tool for tool call execution


        """


        tool_name = tool_call['tool_name']
        tool_args= tool_call['tool_args']

        tool_result = agent.tool_manager.call_tool(tool_name=tool_name, arguments=tool_args)


    def run(self, context, agent=None):


        tool_arg_dict = self.choose_tool(context=context, agent=agent)

        tool_result = self.execute_tool(tool_call=tool_arg_dict, agent=agent)

        return SystemMessage(content=tool_result)

