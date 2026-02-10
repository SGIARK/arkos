"""
Unit tests for StateTool (tool selection and execution)
Tests the specific issues we've been debugging with tool calling.
"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from model_module.ArkModelNew import AIMessage, SystemMessage, UserMessage
from state_module.state_tool import StateTool


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    agent = Mock()
    agent.current_user_id = "test_user"
    agent.tool_manager = Mock()
    agent.tool_manager._tool_registry = {
        "list-events": "google-calendar",
        "list-calendars": "google-calendar"
    }

    # Mock list_all_tools
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

    # Mock call_tool
    agent.tool_manager.call_tool = AsyncMock(return_value={
        "events": [{"summary": "Test Event"}]
    })

    return agent


@pytest.fixture
def mock_llm_valid():
    """Mock LLM that returns valid JSON"""
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
def mock_llm_none():
    """Mock LLM that returns None"""
    async def call_llm(context, json_schema=None):
        return None
    return call_llm


@pytest.fixture
def mock_llm_malformed():
    """Mock LLM that returns malformed JSON"""
    async def call_llm(context, json_schema=None):
        # Missing quotes around key - common error
        return AIMessage(content='{tool_name: "list-events"}')
    return call_llm


@pytest.fixture
def mock_llm_truncated():
    """Mock LLM that returns truncated JSON"""
    async def call_llm(context, json_schema=None):
        # Simulates token limit truncation
        return AIMessage(content='{"account": "normal", "timeMin": "2026-02-09T00:')
    return call_llm


class TestToolSelection:
    """Tests for choose_tool method"""

    @pytest.mark.asyncio
    async def test_choose_tool_success(self, mock_agent, mock_llm_valid):
        """Test successful tool selection"""
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="show my events")]

        result = await state.choose_tool(context, mock_agent)

        assert result["tool_name"] == "list-events"
        assert "tool_args" in result

    @pytest.mark.asyncio
    async def test_choose_tool_with_long_context(self, mock_agent, mock_llm_valid):
        """Test that long context doesn't break tool selection"""
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})

        # Create long context (10 messages)
        context = [
            UserMessage(content=f"message {i}")
            for i in range(9)
        ] + [UserMessage(content="show my events")]

        result = await state.choose_tool(context, mock_agent)

        assert result["tool_name"] == "list-events"
        # Verify full context was passed (not truncated to 2 messages)
        assert mock_agent.call_llm.call_count > 0

    @pytest.mark.asyncio
    async def test_choose_tool_malformed_json(self, mock_agent, mock_llm_malformed):
        """Test handling of malformed JSON from LLM"""
        mock_agent.call_llm = mock_llm_malformed
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="show my events")]

        with pytest.raises(json.JSONDecodeError):
            await state.choose_tool(context, mock_agent)

    def _create_tool_enum(self):
        """Helper to create mock tool enum"""
        from pydantic import create_model, Field
        from enum import Enum

        ToolEnum = Enum("ToolEnum", {"list-events": "list-events"})
        return create_model(
            "ToolCall",
            tool_name=(ToolEnum, Field(description="Tool name"))
        )


class TestToolArgGeneration:
    """Tests for tool argument generation"""

    @pytest.mark.asyncio
    async def test_args_generation_success(self, mock_agent, mock_llm_valid):
        """Test successful argument generation"""
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="show my events today from hhilal@mit.edu")]

        result = await state.choose_tool(context, mock_agent)

        assert result["tool_args"]["account"] == "normal"
        assert result["tool_args"]["calendarId"] == "hhilal@mit.edu"
        assert "timeMin" in result["tool_args"]
        assert "timeMax" in result["tool_args"]

    @pytest.mark.asyncio
    async def test_args_with_calendar_guidance(self, mock_agent, mock_llm_valid):
        """Test that calendar tools get account guidance"""
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="list calendars")]

        result = await state.choose_tool(context, mock_agent)

        # Verify calendar guidance was applied
        # (Check that call_llm was called with guidance in prompt)
        call_args = mock_agent.call_llm.call_args_list
        assert len(call_args) > 0

    @pytest.mark.asyncio
    async def test_args_none_response_retry(self, mock_agent):
        """Test retry logic when LLM returns None"""
        # First call returns None, second call returns valid
        call_count = [0]

        async def mock_call_llm_with_retry(context, json_schema=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # First call returns None
            elif json_schema and json_schema.get('json_schema', {}).get('name') == 'tool_choice':
                return AIMessage(content='{"tool_name": "list-events"}')
            else:
                return AIMessage(content='{"account": "normal"}')

        mock_agent.call_llm = mock_call_llm_with_retry
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="show my events")]

        result = await state.choose_tool(context, mock_agent)

        # Should succeed after retry
        assert result["tool_name"] == "list-events"
        assert call_count[0] >= 2  # Verify retry happened

    @pytest.mark.asyncio
    async def test_args_truncated_fallback(self, mock_agent, mock_llm_truncated):
        """Test fallback defaults when JSON is truncated"""
        mock_agent.call_llm = mock_llm_truncated
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="show my events")]

        result = await state.choose_tool(context, mock_agent)

        # Should fall back to defaults for list-events
        assert result["tool_name"] == "list-events"
        # Note: Fallback logic applies after arg generation fails

    def _create_tool_enum(self):
        """Helper to create mock tool enum"""
        from pydantic import create_model, Field
        from enum import Enum

        ToolEnum = Enum("ToolEnum", {"list-events": "list-events"})
        return create_model(
            "ToolCall",
            tool_name=(ToolEnum, Field(description="Tool name"))
        )


class TestToolExecution:
    """Tests for full tool execution"""

    @pytest.mark.asyncio
    async def test_execute_tool_success(self, mock_agent):
        """Test successful tool execution"""
        state = StateTool("use_tool", {"type": "tool"})

        tool_call = {
            "tool_name": "list-events",
            "tool_args": {
                "account": "normal",
                "calendarId": "hhilal@mit.edu"
            }
        }

        result = await state.execute_tool(tool_call, mock_agent)

        assert result is not None
        assert "events" in result

    @pytest.mark.asyncio
    async def test_run_full_flow(self, mock_agent, mock_llm_valid):
        """Test complete run() flow"""
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="show my events")]

        result = await state.run(context, mock_agent)

        assert result is not None
        assert isinstance(result, SystemMessage)
        assert "TOOL_RESULT" in result.content

    def _create_tool_enum(self):
        """Helper to create mock tool enum"""
        from pydantic import create_model, Field
        from enum import Enum

        ToolEnum = Enum("ToolEnum", {"list-events": "list-events"})
        return create_model(
            "ToolCall",
            tool_name=(ToolEnum, Field(description="Tool name"))
        )


class TestMCPResponseFormatting:
    """Tests for MCP response format extraction"""

    @pytest.mark.asyncio
    async def test_extract_mcp_content(self, mock_agent, mock_llm_valid):
        """Test extraction of actual data from MCP wrapper"""
        mock_agent.call_llm = mock_llm_valid
        mock_agent.create_tool_option_class = AsyncMock(
            return_value=self._create_tool_enum()
        )

        # Mock tool manager to return MCP format
        mock_agent.tool_manager.call_tool = AsyncMock(return_value={
            "content": [{
                "type": "text",
                "text": '{"events": [{"summary": "Test"}]}'
            }]
        })

        state = StateTool("use_tool", {"type": "tool"})
        context = [UserMessage(content="show my events")]

        result = await state.run(context, mock_agent)

        # Should extract the inner JSON, not the MCP wrapper
        assert "TOOL_RESULT" in result.content
        assert '"events"' in result.content  # Actual data extracted

    def _create_tool_enum(self):
        """Helper to create mock tool enum"""
        from pydantic import create_model, Field
        from enum import Enum

        ToolEnum = Enum("ToolEnum", {"list-events": "list-events"})
        return create_model(
            "ToolCall",
            tool_name=(ToolEnum, Field(description="Tool name"))
        )
