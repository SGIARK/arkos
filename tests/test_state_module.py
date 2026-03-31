"""Tests for state module: State, StateUser, StateAI, StateTool, state_registry, StateHandler."""

import json
import pytest
import yaml
from unittest.mock import AsyncMock, MagicMock, patch

from state_module.state import State
from state_module.state_user import StateUser
from state_module.state_ai import StateAI, ReasonedOutput
from state_module.state_tool import StateTool
from state_module.state_registry import STATE_REGISTRY, register_state
from model_module.ArkModelNew import AIMessage, SystemMessage


# --- State base class ---


class TestState:
    def test_init(self):
        config = {"transition": {"next": ["state_b"]}}
        state = State("test_state", config)
        assert state.name == "test_state"
        assert state.is_terminal is False
        assert state.transition == {"next": ["state_b"]}

    def test_init_empty_transition(self):
        state = State("s", {})
        assert state.transition == {}

    def test_check_transition_ready_raises(self):
        state = State("s", {})
        with pytest.raises(NotImplementedError):
            state.check_transition_ready({})

    def test_run_raises(self):
        state = State("s", {})
        with pytest.raises(NotImplementedError):
            state.run({})


# --- StateUser ---


class TestStateUser:
    def test_init_is_terminal(self):
        su = StateUser("wait_for_user", {"transition": {}})
        assert su.is_terminal is True
        assert su.name == "wait_for_user"

    def test_type_attribute(self):
        assert StateUser.type == "user"

    def test_check_transition_ready_always_true(self):
        su = StateUser("u", {})
        assert su.check_transition_ready({}) is True
        assert su.check_transition_ready({"anything": "here"}) is True

    def test_run_returns_none(self):
        su = StateUser("u", {})
        assert su.run({}) is None


# --- ReasonedOutput ---


class TestReasonedOutput:
    def test_valid_output(self):
        data = ReasonedOutput(
            intent="answer question",
            approach=["think", "respond"],
            needs_clarification=False,
            clarifying_question=None,
            final="The answer is 42.",
        )
        assert data.intent == "answer question"
        assert len(data.approach) == 2
        assert data.final == "The answer is 42."

    def test_with_clarification(self):
        data = ReasonedOutput(
            intent="unclear request",
            approach=["analyze"],
            needs_clarification=True,
            clarifying_question="What do you mean?",
            final="I need more info.",
        )
        assert data.needs_clarification is True
        assert data.clarifying_question == "What do you mean?"

    def test_json_schema_generation(self):
        schema = ReasonedOutput.model_json_schema()
        assert "properties" in schema
        assert "intent" in schema["properties"]
        assert "final" in schema["properties"]


# --- StateAI ---


class TestStateAI:
    def test_init(self):
        sa = StateAI("reasoning", {"transition": {"next": ["user"]}})
        assert sa.name == "reasoning"
        assert sa.is_terminal is False

    def test_type_attribute(self):
        assert StateAI.type == "agent"

    def test_check_transition_ready(self):
        sa = StateAI("r", {})
        assert sa.check_transition_ready({}) is True

    @pytest.mark.asyncio
    async def test_run_with_valid_json(self):
        sa = StateAI("reasoning", {})
        reasoned = ReasonedOutput(
            intent="help user",
            approach=["step 1", "step 2"],
            needs_clarification=False,
            clarifying_question=None,
            final="Here is your answer.",
        )
        mock_agent = MagicMock()
        mock_agent.call_llm = AsyncMock(
            return_value=AIMessage(content=reasoned.model_dump_json())
        )

        result = await sa.run(
            [SystemMessage(content="system"), AIMessage(content="hi")],
            mock_agent,
        )

        assert isinstance(result, AIMessage)
        assert "Here is your answer." in result.content
        assert "step 1" in result.content

    @pytest.mark.asyncio
    async def test_run_with_invalid_json_falls_back(self):
        sa = StateAI("reasoning", {})
        mock_agent = MagicMock()
        mock_agent.call_llm = AsyncMock(
            return_value=AIMessage(content="not valid json at all")
        )

        result = await sa.run([], mock_agent)
        assert isinstance(result, AIMessage)
        assert result.content == "I encountered an issue processing that request. Could you rephrase it?"

    @pytest.mark.asyncio
    async def test_run_with_none_content(self):
        sa = StateAI("reasoning", {})
        mock_agent = MagicMock()
        mock_agent.call_llm = AsyncMock(return_value=AIMessage(content=None))

        result = await sa.run([], mock_agent)
        assert isinstance(result, AIMessage)
        assert (
            "issue" in result.content.lower() or "try again" in result.content.lower()
        )

    @pytest.mark.asyncio
    async def test_run_with_clarification(self):
        sa = StateAI("reasoning", {})
        reasoned = ReasonedOutput(
            intent="unclear",
            approach=["analyze"],
            needs_clarification=True,
            clarifying_question="Could you clarify?",
            final="I need more info.",
        )
        mock_agent = MagicMock()
        mock_agent.call_llm = AsyncMock(
            return_value=AIMessage(content=reasoned.model_dump_json())
        )

        result = await sa.run([], mock_agent)
        assert "Could you clarify?" in result.content


# --- StateTool ---


class TestStateTool:
    def test_init(self):
        st = StateTool("tool_use", {})
        assert st.is_terminal is False
        assert st.name == "tool_use"

    def test_type_attribute(self):
        assert StateTool.type == "tool"

    def test_check_transition_ready(self):
        st = StateTool("t", {})
        assert st.check_transition_ready({}) is True

    @pytest.mark.asyncio
    async def test_choose_tool(self):
        st = StateTool("t", {})

        # Mock agent with tool_manager
        mock_tool_class = MagicMock()
        mock_tool_class.model_json_schema.return_value = {
            "type": "object",
            "properties": {"tool_name": {"type": "string"}},
        }

        mock_agent = MagicMock()
        mock_agent.create_tool_option_class = AsyncMock(return_value=mock_tool_class)
        mock_agent.call_llm = AsyncMock(
            side_effect=[
                # First call: choose tool
                AIMessage(content=json.dumps({"tool_name": "search"})),
                # Second call: fill args
                AIMessage(content=json.dumps({"query": "test"})),
            ]
        )
        mock_agent.tool_manager = MagicMock()
        mock_agent.tool_manager._tool_registry = {"search": "brave-search"}
        mock_agent.tool_manager.list_all_tools = AsyncMock(
            return_value={
                "brave-search": {
                    "search": {
                        "name": "search",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                        },
                    }
                }
            }
        )

        result = await st.choose_tool(
            [SystemMessage(content="find something")], mock_agent
        )
        assert result["tool_name"] == "search"
        assert result["tool_args"] == {"query": "test"}

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        st = StateTool("t", {})
        mock_agent = MagicMock()
        mock_agent.current_user_id = "user1"
        mock_agent.tool_manager = MagicMock()
        mock_agent.tool_manager.call_tool = AsyncMock(return_value={"result": "ok"})

        result = await st.execute_tool(
            {"tool_name": "search", "tool_args": {"q": "test"}}, mock_agent
        )
        assert result == {"result": "ok"}
        mock_agent.tool_manager.call_tool.assert_called_once_with(
            tool_name="search", arguments={"q": "test"}, user_id="user1"
        )

    @pytest.mark.asyncio
    async def test_run_returns_system_message(self):
        st = StateTool("t", {})
        mock_agent = MagicMock()
        mock_agent.current_user_id = "user1"

        with (
            patch.object(st, "choose_tool", new_callable=AsyncMock) as mock_choose,
            patch.object(st, "execute_tool", new_callable=AsyncMock) as mock_exec,
        ):
            mock_choose.return_value = {"tool_name": "calc", "tool_args": {"x": 1}}
            mock_exec.return_value = "42"

            result = await st.run([], mock_agent)
            assert isinstance(result, SystemMessage)
            assert "42" in result.content


# --- state_registry ---


class TestStateRegistry:
    def test_register_state_adds_to_registry(self):
        # StateUser and StateAI already registered on import
        assert "user" in STATE_REGISTRY
        assert "agent" in STATE_REGISTRY
        assert "tool" in STATE_REGISTRY

    def test_register_state_returns_class(self):
        @register_state
        class DummyState(State):
            type = "dummy_test"

            def check_transition_ready(self, ctx):
                return True

            def run(self, ctx):
                return None

        assert STATE_REGISTRY["dummy_test"] is DummyState
        # Cleanup
        del STATE_REGISTRY["dummy_test"]

    def test_register_state_missing_type_raises(self):
        with pytest.raises((ValueError, AttributeError)):

            @register_state
            class BadState:
                pass


# --- StateHandler ---


class TestStateHandler:
    @pytest.fixture
    def state_graph_yaml(self, tmp_path):
        graph = {
            "initial": "agent_reply",
            "states": {
                "agent_reply": {
                    "type": "agent",
                    "transition": {"next": ["wait_for_user"]},
                },
                "wait_for_user": {
                    "type": "user",
                    "transition": {"next": ["agent_reply"]},
                },
            },
        }
        f = tmp_path / "graph.yaml"
        f.write_text(yaml.dump(graph))
        return str(f)

    def test_init_loads_states(self, state_graph_yaml):
        from state_module.state_handler import StateHandler

        handler = StateHandler(state_graph_yaml)
        assert "agent_reply" in handler.states
        assert "wait_for_user" in handler.states

    def test_get_initial_state(self, state_graph_yaml):
        from state_module.state_handler import StateHandler

        handler = StateHandler(state_graph_yaml)
        initial = handler.get_initial_state()
        assert initial.name == "agent_reply"
        assert isinstance(initial, StateAI)

    def test_get_state(self, state_graph_yaml):
        from state_module.state_handler import StateHandler

        handler = StateHandler(state_graph_yaml)
        user_state = handler.get_state("wait_for_user")
        assert isinstance(user_state, StateUser)
        assert user_state.is_terminal is True

    def test_unknown_state_type_raises(self, tmp_path):
        from state_module.state_handler import StateHandler

        graph = {
            "initial": "bad",
            "states": {"bad": {"type": "nonexistent_type"}},
        }
        f = tmp_path / "bad_graph.yaml"
        f.write_text(yaml.dump(graph))
        with pytest.raises(ValueError, match="Unknown state type"):
            StateHandler(str(f))
