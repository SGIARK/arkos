# Agent Module Interface

This document outlines the expected structure and responsibilities of an `Agent` class in the ARK system.

## Purpose

The `Agent` serves as the orchestrator of execution across tools, state transitions, memory interactions, and user communication. It holds a memory object, tracks current flow state, and is responsible for interpreting model decisions.

## Expected Structure

- **agent_module/**
  - `agent.py`: Main agent class definition.
  - `interface.md`: This file.
  - `utils.py`: Utility functions related to agent behavior.
  - `registry.py`: Optionally stores agent registry and agent-specific configurations.

## Class: Agent

### Responsibilities
- Load a flow (defined by state graph)
- Execute the state logic by coordinating with the StateRunner
- Handle tool calling, memory reads/writes, and voice I/O
- Maintain a session buffer (if using voice or live input)
- Maintain history and memory context

### Expected Methods

```python
class Agent:
    def __init__(self, agent_id: str, flow: FlowRunner, memory: Memory):
        pass

    def run(self):
        # Entry point for processing input and starting flow
        pass

    def step(self, user_input: str):
        # Single step of execution; may call state logic or tool
        pass

    def receive_result(self, result):
        # Accepts tool result, and routes it back into the state flow
        pass

    def serialize(self):
        # Serialize agent state to disk or memory
        pass
```

## Assumptions

- Agents are loosely coupled to flows and tools.
- An agent may persist across multiple conversations and resume context.
- Agents should not implement tool logic directly.