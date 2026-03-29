"""
Unit tests for StateTool (tool selection and execution)
"""

import pytest
import json
from pydantic import create_model, Field
from enum import Enum
from unittest.mock import Mock, AsyncMock
from model_module.ArkModelNew import AIMessage, SystemMessage, UserMessage
from state_module.state_tool import StateTool


@pytest.fixture
def tool_enum():
    ToolEnum = Enum("ToolEnum", {"list-events": "list-events"})
    return create_model("ToolCall", tool_name=(ToolEnum, Field(description="Tool name")))


@pytest.fixture
def mock_agent():
    agent = Mock()
    agent.current_user_id = "test_user"
    agent.tool_manager = Mock()
    agent.tool_manager._tool_registry = {
        "list-events": "google-calendar",
        "list-calendars": "google-calendar"
    }
    agent.tool_manager.config = {}
    agent.tool_manager.list_all_tools = AsyncMock(return_value={
        "google-calendar": {
            "list-events": {
                "name": "list-events",
                "description": "List calendar events",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "account": {"type": "string"},
                        "calendarId": {"type": "string"},
                        "timeMin": {"type": "string"},
                        "timeMax": {"type": "string"}
                    },
                    "required": ["account"]
                }
            }
        }
    })
    # call_tool returns plain text (MCP unwrapping happens inside call_tool)
    agent.tool_manager.call_tool = AsyncMock(
        return_value='{"events": [{"summary": "Test Event"}]}'
    )
    return agent


@pytest.fixture
def mock_llm_valid():
    async def call_llm(context, json_schema=None):
        if json_schema:
            schema_name = json_schema.get('json_schema', {}).get('name', '')
            if schema_name == 'tool_choice':
                return AIMessage(content='{"tool_name": "list-events"}')
            elif schema_name == 'tool_args':
                return AIMessage(content=json.dumps({
                    "account": "normal",
                    "calendarId": "hhilal@mit.edu",
                    "timeMin": "2026-02-09T00:00:00",
                    "timeMax": "2026-02-09T23:59:59"
                }))
        return AIMessage(content='{}')
    return call_llm


@pytest.fixture
def mock_llm_malformed():
    async def call_llm(context, json_schema=None):
        return AIMessage(content='{tool_name: "list-events"}')
    return call_llm


@pytest.fixture
def mock_llm_truncated():
    async def call_llm(context, json_schema=None):
        schema_name = (json_schema or {}).get('json_schema', {}).get('name', '')
        if schema_name == 'tool_choice':
            return AIMessage(content='{"tool_name": "list-events"}')
        return AIMessage(content='{"account": "normal", "timeMin": "2026-02-09T00:')
    return call_llm


class TestToolSelection:
    """Tests for choose_tool method"""

    @pytest.mark.asyncio
    async def test_choose_tool_success(self, mock_agent, mock_llm_valid, tool_enum):
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)

        state = StateTool("use_tool", {"type": "tool"})
        result = await state.choose_tool([UserMessage(content="show my events")], mock_agent)

        assert result["tool_name"] == "list-events"
        assert "tool_args" in result

    @pytest.mark.asyncio
    async def test_choose_tool_with_long_context(self, mock_agent, mock_llm_valid, tool_enum):
        """Test that long context doesn't break tool selection"""
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)

        context = [UserMessage(content=f"message {i}") for i in range(9)] + [
            UserMessage(content="show my events")
        ]
        result = await StateTool("use_tool", {"type": "tool"}).choose_tool(context, mock_agent)

        assert result["tool_name"] == "list-events"

    @pytest.mark.asyncio
    async def test_choose_tool_malformed_json(self, mock_agent, mock_llm_malformed, tool_enum):
        mock_agent.call_llm = mock_llm_malformed
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)

        with pytest.raises(json.JSONDecodeError):
            await StateTool("use_tool", {"type": "tool"}).choose_tool(
                [UserMessage(content="show my events")], mock_agent
            )


class TestToolArgGeneration:
    """Tests for tool argument generation"""

    @pytest.mark.asyncio
    async def test_args_generation_success(self, mock_agent, mock_llm_valid, tool_enum):
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)

        result = await StateTool("use_tool", {"type": "tool"}).choose_tool(
            [UserMessage(content="show my events today from hhilal@mit.edu")], mock_agent
        )

        assert result["tool_args"]["account"] == "normal"
        assert result["tool_args"]["calendarId"] == "hhilal@mit.edu"
        assert "timeMin" in result["tool_args"]
        assert "timeMax" in result["tool_args"]

    @pytest.mark.asyncio
    async def test_args_with_config_hints(self, mock_agent, tool_enum):
        """Test that arg_hints from config appear in the prompt sent to the LLM"""
        captured_contexts = []

        async def capturing_llm(context, json_schema=None):
            captured_contexts.append(context)
            schema_name = (json_schema or {}).get('json_schema', {}).get('name', '')
            if schema_name == 'tool_choice':
                return AIMessage(content='{"tool_name": "list-events"}')
            return AIMessage(content='{"account": "normal", "calendarId": "primary"}')

        mock_agent.call_llm = capturing_llm
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)
        mock_agent.tool_manager.config = {
            "google-calendar": {"arg_hints": {"account": "normal", "calendarId": "primary"}}
        }

        await StateTool("use_tool", {"type": "tool"}).choose_tool(
            [UserMessage(content="list my events")], mock_agent
        )

        all_content = " ".join(
            msg.content for ctx in captured_contexts for msg in ctx
            if hasattr(msg, "content")
        )
        assert 'use "normal"' in all_content
        assert 'use "primary"' in all_content

    @pytest.mark.asyncio
    async def test_args_none_response_retry(self, mock_agent, tool_enum):
        """Test retry logic when args LLM call returns None"""
        args_call_count = [0]

        async def mock_call_llm_with_retry(context, json_schema=None):
            schema_name = (json_schema or {}).get('json_schema', {}).get('name', '')
            if schema_name == 'tool_choice':
                return AIMessage(content='{"tool_name": "list-events"}')
            args_call_count[0] += 1
            if args_call_count[0] == 1:
                return None
            return AIMessage(content='{"account": "normal"}')

        mock_agent.call_llm = mock_call_llm_with_retry
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)

        result = await StateTool("use_tool", {"type": "tool"}).choose_tool(
            [UserMessage(content="show my events")], mock_agent
        )

        assert result["tool_name"] == "list-events"
        assert args_call_count[0] >= 2

    @pytest.mark.asyncio
    async def test_args_truncated_fallback(self, mock_agent, mock_llm_truncated, tool_enum):
        """Test fallback to schema defaults when args JSON is truncated"""
        mock_agent.call_llm = mock_llm_truncated
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)

        result = await StateTool("use_tool", {"type": "tool"}).choose_tool(
            [UserMessage(content="show my events")], mock_agent
        )

        assert result["tool_name"] == "list-events"
        assert "tool_args" in result


class TestToolExecution:
    """Tests for full tool execution"""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_agent):
        result = await StateTool("use_tool", {"type": "tool"}).execute_tool(
            {"tool_name": "list-events", "tool_args": {"account": "normal", "calendarId": "hhilal@mit.edu"}},
            mock_agent
        )
        assert result is not None
        assert "events" in result

    @pytest.mark.asyncio
    async def test_run_full_flow(self, mock_agent, mock_llm_valid, tool_enum):
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(return_value=tool_enum)

        result = await StateTool("use_tool", {"type": "tool"}).run(
            [UserMessage(content="show my events")], mock_agent
        )

        assert isinstance(result, SystemMessage)
        assert "TOOL_RESULT" in result.content


class TestMCPTextExtraction:
    """Tests for MCPToolManager._extract_mcp_text"""

    def test_extracts_text_from_mcp_format(self):
        from tool_module.tool_call import MCPToolManager
        result = MCPToolManager._extract_mcp_text({
            "content": [{"type": "text", "text": '{"events": []}'}]
        })
        assert result == '{"events": []}'

    def test_falls_back_to_json_dumps_for_unknown_format(self):
        from tool_module.tool_call import MCPToolManager
        result = MCPToolManager._extract_mcp_text({"some": "data"})
        assert json.loads(result) == {"some": "data"}

    def test_falls_back_when_content_list_is_empty(self):
        from tool_module.tool_call import MCPToolManager
        result = MCPToolManager._extract_mcp_text({"content": []})
        assert json.loads(result) == {"content": []}

    def test_falls_back_when_no_text_key(self):
        from tool_module.tool_call import MCPToolManager
        result = MCPToolManager._extract_mcp_text({"content": [{"type": "image"}]})
        assert json.loads(result) == {"content": [{"type": "image"}]}
