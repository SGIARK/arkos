# State Module Interface

This document outlines the interface and structure for defining and managing states in the ARK system.

## Purpose

States represent individual steps in a flow. Each state is responsible for determining:
- Whether a tool should be used
- What inputs are required
- When the state is complete

## Expected Structure

- **state_module/**
  - `state.py`: Main `State` class
  - `state_runner.py`: Manages transitions and flow control
  - `interface.md`: This file
  - `schemas/`: Folder containing YAML state definitions

## Class: State

### Responsibilities
- Represent a single logical step in an execution flow
- Declare inputs (parameters), tools, and transitions
- Handle progression logic, e.g. when a state is "done"
- Optionally request clarification or retry

### Expected Methods

```python
class State:
    def __init__(self, state_id: str, config: dict):
        pass

    def check_completion(self, memory: Memory) -> bool:
        # Returns True if all needed inputs are collected
        pass

    def get_missing_inputs(self, memory: Memory) -> List[str]:
        # Returns list of inputs still needed
        pass

    def should_call_tool(self) -> bool:
        # Determines whether tool should be invoked
        pass

    def on_tool_result(self, result: dict):
        # Handle result and store in memory
        pass
```

## Class: StateRunner

### Responsibilities
- Load a state graph from YAML
- Manage progression between states based on logic
- Interface with the Agent to signal state advancement

### Expected Methods

```python
class StateRunner:
    def __init__(self, graph: dict):
        pass

    def step(self, memory: Memory) -> str:
        # Progresses to next state if conditions are met
        pass

    def current_state(self) -> State:
        # Returns the current active state
        pass
```

## Assumptions

- States are serialized as YAML configs, imported dynamically.
- Transitions may be conditional or linear.