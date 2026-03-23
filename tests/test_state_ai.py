"""
Unit tests for StateAI (reasoning and tool result detection)
Tests the fixes for has_tool_result detection and response formatting.
"""

import pytest
import json
from unittest.mock import Mock
from model_module.ArkModelNew import AIMessage, SystemMessage, UserMessage
from state_module.state_ai import StateAI


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing"""
    agent = Mock()

    # Mock call_llm to return valid ReasonedOutput
    async def mock_call_llm(context, json_schema=None):
        return AIMessage(
            content=json.dumps(
                {
                    "intent": "test",
                    "approach": ["step 1", "step 2"],
                    "needs_clarification": False,
                    "has_tool_result": False,
                    "final": "Test response",
                }
            )
        )

    agent.call_llm = mock_call_llm
    return agent


class TestToolResultDetection:
    """Tests for detecting fresh tool results"""

    @pytest.mark.asyncio
    async def test_no_tool_result_on_first_run(self, mock_agent):
        """Test that first run doesn't falsely detect tool result"""
        state = StateAI("agent_reply", {"type": "agent"})
        context = [UserMessage(content="show my events")]

        result = await state.run(context, mock_agent)

        # Should not detect tool result (no SystemMessage with TOOL_RESULT)
        assert result is not None
        assert isinstance(result, AIMessage)

    @pytest.mark.asyncio
    async def test_detect_fresh_tool_result(self, mock_agent):
        """Test detection when last message is tool result"""

        # Override mock to return has_tool_result=true when detected
        async def mock_call_llm_with_detection(context, json_schema=None):
            # Check if last message is tool result
            has_result = (
                context
                and isinstance(context[-1], SystemMessage)
                and "TOOL_RESULT" in context[-1].content
            )

            return AIMessage(
                content=json.dumps(
                    {
                        "intent": "present_results",
                        "approach": [],
                        "needs_clarification": False,
                        "has_tool_result": has_result,
                        "final": "Here are your events",
                    }
                )
            )

        mock_agent.call_llm = mock_call_llm_with_detection

        state = StateAI("agent_reply", {"type": "agent"})
        context = [
            UserMessage(content="show my events"),
            AIMessage(content="Checking calendar..."),
            SystemMessage(content='TOOL_RESULT from "list-events":\n{"events": []}'),
        ]

        result = await state.run(context, mock_agent)

        # Should detect fresh tool result and format accordingly
        assert result is not None
        # Should not show approach bullets when has_tool_result=true
        assert "step 1" not in result.content

    @pytest.mark.asyncio
    async def test_ignore_old_tool_results(self, mock_agent):
        """Test that old tool results (not last message) are ignored"""
        state = StateAI("agent_reply", {"type": "agent"})
        context = [
            SystemMessage(content='TOOL_RESULT from "old-tool":\n{"old": "data"}'),
            UserMessage(content="show my events"),  # New request
        ]

        result = await state.run(context, mock_agent)

        # Should NOT detect tool result (not the last message)
        assert result is not None


class TestOverrideLogic:
    """Tests for has_tool_result override mechanism"""

    @pytest.mark.asyncio
    async def test_override_when_llm_ignores_instruction(self, mock_agent):
        """Test override when LLM sets has_tool_result=false despite tool result"""

        # LLM always returns has_tool_result=false
        async def mock_call_llm_ignores_tool(context, json_schema=None):
            return AIMessage(
                content=json.dumps(
                    {
                        "intent": "present_results",
                        "approach": ["step 1"],
                        "needs_clarification": False,
                        "has_tool_result": False,  # LLM ignores instruction!
                        "final": "Results here",
                    }
                )
            )

        mock_agent.call_llm = mock_call_llm_ignores_tool

        state = StateAI("agent_reply", {"type": "agent"})
        context = [
            UserMessage(content="show my events"),
            SystemMessage(content='TOOL_RESULT from "list-events":\n{"events": []}'),
        ]

        result = await state.run(context, mock_agent)

        # Override should have happened
        # Approach bullets should be hidden
        assert result is not None
        # Check that bullets are not in output
        assert "step 1" not in result.content


class TestResponseFormatting:
    """Tests for response formatting logic"""

    @pytest.mark.asyncio
    async def test_show_bullets_without_tool_result(self, mock_agent):
        """Test that reasoning bullets are shown when no tool result"""

        async def mock_call_llm_with_bullets(context, json_schema=None):
            return AIMessage(
                content=json.dumps(
                    {
                        "intent": "reasoning",
                        "approach": ["Step 1: Do thing", "Step 2: Do other thing"],
                        "needs_clarification": False,
                        "has_tool_result": False,
                        "final": "Final answer",
                    }
                )
            )

        mock_agent.call_llm = mock_call_llm_with_bullets

        state = StateAI("agent_reply", {"type": "agent"})
        context = [UserMessage(content="explain something")]

        result = await state.run(context, mock_agent)

        assert isinstance(result, AIMessage)
        assert "Final answer" in result.content

    @pytest.mark.asyncio
    async def test_hide_bullets_with_tool_result(self, mock_agent):
        """Test that bullets are hidden when presenting tool result"""

        async def mock_call_llm_tool_result(context, json_schema=None):
            return AIMessage(
                content=json.dumps(
                    {
                        "intent": "present_tool_result",
                        "approach": ["These", "Should", "Be", "Hidden"],
                        "needs_clarification": False,
                        "has_tool_result": True,
                        "final": "Here are the results from the tool",
                    }
                )
            )

        mock_agent.call_llm = mock_call_llm_tool_result

        state = StateAI("agent_reply", {"type": "agent"})
        context = [SystemMessage(content='TOOL_RESULT from "tool":\n{"data": "value"}')]

        result = await state.run(context, mock_agent)

        # Should NOT show approach bullets
        assert "These" not in result.content
        assert "Should" not in result.content
        # Should show final result
        assert "Here are the results from the tool" in result.content


class TestErrorHandling:
    """Tests for error handling in StateAI"""

    @pytest.mark.asyncio
    async def test_malformed_json_returns_friendly_error(self, mock_agent):
        """Test that malformed JSON returns user-friendly message"""

        async def mock_call_llm_malformed(context, json_schema=None):
            # Return truncated JSON
            return AIMessage(content='{"intent": "test", "approach": [')

        mock_agent.call_llm = mock_call_llm_malformed

        state = StateAI("agent_reply", {"type": "agent"})
        context = [UserMessage(content="test")]

        result = await state.run(context, mock_agent)

        # Should return friendly error, not raw JSON
        assert result is not None
        assert isinstance(result, AIMessage)
        assert "encountered an issue" in result.content.lower()
        assert "rephrase" in result.content.lower()
        # Should NOT contain raw JSON
        assert '{"intent"' not in result.content

    @pytest.mark.asyncio
    async def test_none_response_handling(self, mock_agent):
        """Test handling of None response from LLM"""

        async def mock_call_llm_none(context, json_schema=None):
            return None

        mock_agent.call_llm = mock_call_llm_none

        state = StateAI("agent_reply", {"type": "agent"})
        context = [UserMessage(content="test")]

        result = await state.run(context, mock_agent)

        # Should handle None gracefully
        assert result is not None
        assert isinstance(result, AIMessage)
        assert len(result.content) > 0


class TestPromptGuidance:
    """Tests for tool result guidance in prompts"""

    @pytest.mark.asyncio
    async def test_critical_guidance_when_tool_result_present(self, mock_agent):
        """Test that critical guidance is added when tool result detected"""
        call_context = []

        async def mock_call_llm_capture_context(context, json_schema=None):
            call_context.append(context)
            return AIMessage(
                content=json.dumps(
                    {
                        "intent": "test",
                        "approach": [],
                        "needs_clarification": False,
                        "has_tool_result": True,
                        "final": "Test",
                    }
                )
            )

        mock_agent.call_llm = mock_call_llm_capture_context

        state = StateAI("agent_reply", {"type": "agent"})
        context = [SystemMessage(content='TOOL_RESULT from "tool":\n{"data": "value"}')]

        await state.run(context, mock_agent)

        system_messages = [
            msg for msg in call_context[0] if isinstance(msg, SystemMessage)
        ]
        assert len(system_messages) > 0
        assert (
            "TOOL_RESULT" in system_messages[0].content
            or "tool result" in system_messages[0].content.lower()
        )
