"""Tests for the ArkModel LLM interface with mocked responses."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from model_module.ArkModelNew import AIMessage, ArkModelLink, UserMessage


class TestArkModelLink:
    """Test cases for ArkModelLink with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_generation_response_mocked(self, mock_openai_client):
        """Test LLM generation with mocked client."""
        with patch("model_module.ArkModelNew.AsyncOpenAI", return_value=mock_openai_client):
            model = ArkModelLink(base_url="http://localhost:8080/v1")

            messages = [UserMessage(content="Say hello in Spanish.")]
            result = await model.make_llm_call(messages, json_schema=None)

            assert result is not None
            assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_generation_with_schema_mocked(self):
        """Test LLM generation with JSON schema using mocked client."""
        mock_client = AsyncMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[
            0
        ].message.content = '{"product_name": "Test Widget", "price": 9.99, "in_stock": true}'
        mock_completion.choices[0].message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)

        with patch("model_module.ArkModelNew.AsyncOpenAI", return_value=mock_client):
            model = ArkModelLink(base_url="http://localhost:8080/v1")

            messages = [UserMessage(content="Give me a product listing.")]
            schema = {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "product_name": {"type": "string"},
                        "price": {"type": "number"},
                        "in_stock": {"type": "boolean"},
                    },
                    "required": ["product_name", "price", "in_stock"],
                },
            }

            result = await model.make_llm_call(messages, json_schema=schema)
            assert result is not None

    @pytest.mark.asyncio
    async def test_message_formatting(self):
        """Test that messages are properly formatted."""
        user_msg = UserMessage(content="Test message")
        assert user_msg.role == "user"
        assert user_msg.content == "Test message"

        ai_msg = AIMessage(content="Response")
        assert ai_msg.role == "assistant"
        assert ai_msg.content == "Response"


@pytest.mark.integration
class TestArkModelLinkIntegration:
    """Integration tests that require actual LLM server.

    These are skipped in CI and only run locally with a running LLM server.
    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_real_generation(self):
        """Test actual LLM generation (requires running LLM server)."""
        pytest.skip("Integration test - requires running LLM server")
