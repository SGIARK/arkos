#!/usr/bin/env python3
"""
Manual test script to verify tool calling works without running full server.
Tests the specific issues we've been debugging.
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from model_module.ArkModelNew import AIMessage, SystemMessage, UserMessage
from state_module.state_tool import StateTool
from state_module.state_ai import StateAI


class MockAgent:
    """Mock agent for testing"""
    def __init__(self):
        self.tool_manager = MockToolManager()
        self.current_user_id = "test_user"
        self.call_count = 0

    async def call_llm(self, context, json_schema=None):
        """Mock LLM that returns valid JSON"""
        self.call_count += 1
        print(f"\n[MockAgent] LLM call #{self.call_count}")
        print(f"[MockAgent] Context length: {len(context)} messages")
        print(f"[MockAgent] Schema: {json_schema.get('json_schema', {}).get('name', 'None') if json_schema else 'None'}")

        # Determine what kind of response to return based on schema
        if json_schema:
            schema_name = json_schema.get('json_schema', {}).get('name', '')

            if schema_name == 'tool_choice':
                # Tool selection
                response = '{"tool_name": "list-events"}'
                print(f"[MockAgent] Returning tool choice: {response}")
                return AIMessage(content=response)

            elif schema_name == 'tool_args':
                # Tool arguments
                response = '''{
  "account": "normal",
  "calendarId": "hhilal@mit.edu",
  "timeMin": "2026-02-09T00:00:00",
  "timeMax": "2026-02-09T23:59:59"
}'''
                print(f"[MockAgent] Returning tool args: {response}")
                return AIMessage(content=response)

            elif schema_name == 'reasoned_output':
                # StateAI reasoning
                response = '''{
  "intent": "show_calendar_events",
  "approach": ["Use list-events tool", "Format results"],
  "needs_clarification": false,
  "has_tool_result": false,
  "final": "Checking your calendar..."
}'''
                print(f"[MockAgent] Returning reasoned output")
                return AIMessage(content=response)

        # Default response
        return AIMessage(content='{"response": "ok"}')

    async def create_tool_option_class(self):
        """Mock tool option class"""
        from pydantic import create_model, Field
        from enum import Enum

        ToolEnum = Enum("ToolEnum", {"list-events": "list-events", "list-calendars": "list-calendars"})
        ToolOptionsModel = create_model(
            "ToolCall",
            tool_name=(ToolEnum, Field(description="Tool name"))
        )
        return ToolOptionsModel


class MockToolManager:
    """Mock tool manager for testing"""
    def __init__(self):
        self._tool_registry = {
            "list-events": "google-calendar",
            "list-calendars": "google-calendar"
        }

    async def list_all_tools(self):
        return {
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
                },
                "list-calendars": {
                    "name": "list-calendars",
                    "description": "List available calendars",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "account": {"type": "string"}
                        },
                        "required": ["account"]
                    }
                }
            }
        }

    async def call_tool(self, tool_name, arguments, user_id=None):
        """Mock tool execution"""
        print(f"\n[MockToolManager] Executing {tool_name} with args: {arguments}")
        return {
            "events": [
                {
                    "summary": "Test Event",
                    "start": "2026-02-09T10:00:00",
                    "end": "2026-02-09T11:00:00"
                }
            ]
        }


async def test_tool_selection():
    """Test 1: Can we select a tool?"""
    print("\n" + "="*60)
    print("TEST 1: Tool Selection")
    print("="*60)

    agent = MockAgent()
    state = StateTool("use_tool", {"type": "tool"})

    context = [
        UserMessage(content="show my events today from hhilal@mit.edu calendar")
    ]

    try:
        tool_dict = await state.choose_tool(context, agent)
        print(f"\n✅ SUCCESS: Selected tool '{tool_dict['tool_name']}'")
        print(f"✅ SUCCESS: Generated args: {tool_dict['tool_args']}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_full_tool_execution():
    """Test 2: Can we execute a tool end-to-end?"""
    print("\n" + "="*60)
    print("TEST 2: Full Tool Execution")
    print("="*60)

    agent = MockAgent()
    state = StateTool("use_tool", {"type": "tool"})

    context = [
        UserMessage(content="show my events today from hhilal@mit.edu calendar")
    ]

    try:
        result = await state.run(context, agent)
        print(f"\n✅ SUCCESS: Tool executed")
        print(f"✅ Result type: {type(result).__name__}")
        print(f"✅ Result content (first 200 chars): {result.content[:200]}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_state_ai_reasoning():
    """Test 3: Can StateAI generate complete output?"""
    print("\n" + "="*60)
    print("TEST 3: StateAI Reasoning")
    print("="*60)

    agent = MockAgent()
    state = StateAI("agent_reply", {"type": "agent"})

    context = [
        UserMessage(content="show my events today from hhilal@mit.edu calendar")
    ]

    try:
        result = await state.run(context, agent)
        print(f"\n✅ SUCCESS: StateAI generated output")
        print(f"✅ Result type: {type(result).__name__}")
        print(f"✅ Result content: {result.content}")
        return True
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_context_length():
    """Test 4: Does context length affect results?"""
    print("\n" + "="*60)
    print("TEST 4: Context Length Impact")
    print("="*60)

    agent = MockAgent()
    state = StateTool("use_tool", {"type": "tool"})

    # Test with varying context lengths
    for context_size in [2, 5, 10]:
        print(f"\nTesting with {context_size} messages in context...")
        context = [
            UserMessage(content=f"message {i}")
            for i in range(context_size - 1)
        ] + [UserMessage(content="show my events today")]

        try:
            tool_dict = await state.choose_tool(context, agent)
            print(f"  ✅ Works with {context_size} messages")
        except Exception as e:
            print(f"  ❌ Fails with {context_size} messages: {e}")
            return False

    print(f"\n✅ SUCCESS: All context lengths work")
    return True


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("TOOL CALLING MANUAL TESTS")
    print("="*60)
    print("Testing fixes for tool arg generation issues...")

    results = []

    # Run tests
    results.append(("Tool Selection", await test_tool_selection()))
    results.append(("Full Tool Execution", await test_full_tool_execution()))
    results.append(("StateAI Reasoning", await test_state_ai_reasoning()))
    results.append(("Context Length", await test_context_length()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("The tool calling fix is working correctly.")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("Review the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
