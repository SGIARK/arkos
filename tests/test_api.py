"""Tests for FastAPI endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestHealthEndpoint:
    """Test cases for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_check_llm_running(self):
        """Test health check when LLM server is running."""
        from httpx import ASGITransport, AsyncClient

        from base_module.app import app

        # Mock the requests.get call inside health_check (imported locally in function)
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["llm_server"] == "running"

    @pytest.mark.asyncio
    async def test_health_check_llm_not_running(self):
        """Test health check when LLM server is not running."""
        from httpx import ASGITransport, AsyncClient

        from base_module.app import app

        with patch("requests.get") as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["llm_server"] == "not_running"


class TestChatCompletionsEndpoint:
    """Test cases for the chat completions endpoint."""

    @pytest.mark.asyncio
    async def test_chat_completions_basic(self, mock_openai_client, mock_memory):
        """Test basic chat completion request."""
        from httpx import ASGITransport, AsyncClient

        # Create a mock agent response
        mock_ai_message = MagicMock()
        mock_ai_message.content = "Hello! I'm ARK, how can I help you?"

        with patch("base_module.app.agent") as mock_agent:
            mock_agent.step = AsyncMock(return_value=mock_ai_message)

            from base_module.app import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "ark-agent",
                        "messages": [{"role": "user", "content": "Hello!"}],
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert "id" in data
            assert data["object"] == "chat.completion"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_completions_empty_messages(self):
        """Test chat completion with empty messages."""
        from httpx import ASGITransport, AsyncClient

        mock_ai_message = MagicMock()
        mock_ai_message.content = "How can I help you?"

        with patch("base_module.app.agent") as mock_agent:
            mock_agent.step = AsyncMock(return_value=mock_ai_message)

            from base_module.app import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={"model": "ark-agent", "messages": []},
                )

            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_chat_completions_system_message(self):
        """Test chat completion with system message."""
        from httpx import ASGITransport, AsyncClient

        mock_ai_message = MagicMock()
        mock_ai_message.content = "I understand the custom instructions."

        with patch("base_module.app.agent") as mock_agent:
            mock_agent.step = AsyncMock(return_value=mock_ai_message)

            from base_module.app import app

            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "ark-agent",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello!"},
                        ],
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["finish_reason"] == "stop"
