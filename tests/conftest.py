"""Shared pytest fixtures for ARKOS tests."""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set test environment variables before any module imports
os.environ.setdefault("DB_URL", "postgresql://postgres:postgres@localhost:5432/arkos_test")
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key")
os.environ.setdefault("HF_TOKEN", "test-token")


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing without actual LLM calls."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hola! Este es un mensaje de prueba."},
                "finish_reason": "stop",
            }
        ],
    }


@pytest.fixture
def mock_openai_client(mock_llm_response):
    """Mock AsyncOpenAI client."""
    mock_client = AsyncMock()
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = mock_llm_response["choices"][0]["message"][
        "content"
    ]
    mock_completion.choices[0].message.tool_calls = None
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    return mock_client


@pytest.fixture
def mock_memory():
    """Mock Memory class for testing without database."""
    mock = MagicMock()
    mock.add_memory = MagicMock(return_value=True)
    mock.retrieve_short_memory = MagicMock(return_value=[])
    mock.retrieve_long_memory = MagicMock(return_value=MagicMock(content=""))
    return mock


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client for tool tests."""
    mock = AsyncMock()
    mock.start = AsyncMock()
    mock.stop = AsyncMock()
    mock._initialized = True
    mock.list_tools = AsyncMock(
        return_value=[
            {"name": "test_tool", "description": "A test tool", "inputSchema": {}},
            {"name": "list_directory", "description": "List directory contents", "inputSchema": {}},
        ]
    )
    mock.call_tool = AsyncMock(return_value={"result": "success", "content": []})
    return mock


@pytest.fixture
def mock_mcp_manager(mock_mcp_client):
    """Mock MCP Tool Manager."""
    mock = AsyncMock()
    mock.initialize_servers = AsyncMock()
    mock.shutdown = AsyncMock()
    mock.list_all_tools = AsyncMock(
        return_value=[
            {"name": "test_tool", "description": "A test tool"},
            {"name": "list_directory", "description": "List directory contents"},
        ]
    )
    mock.call_tool = AsyncMock(return_value={"result": "success"})
    return mock


@pytest.fixture
def sample_config_yaml(tmp_path):
    """Create a temporary config YAML file for testing."""
    config_content = """
app:
  host: "0.0.0.0"
  port: 1112
  reload: false

llm:
  base_url: "http://localhost:30000/v1"

database:
  url: ${DB_URL}

memory:
  user_id: "test_user"
  short_context_turns: 5

state:
  graph_path: "state_module/state_graph.yaml"
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return str(config_file)
