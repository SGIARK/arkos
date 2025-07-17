# Tool Interface Specification (MCP-Compatible)

## Overview

This document outlines the interface requirements and responsibilities for a Tool class that communicates via MCP (Model Context Protocol). Tools are modular, callable services that perform well-defined tasks. They are invoked by agents and participate in multi-turn workflows.

## Purpose

To encapsulate communication with external or internal tools that comply with the MCP 6.18.2025 update. The interface provides:

- Uniform request structure
- Schema-driven parameterization
- Session continuity through `session_state`
- Robust error handling

---

## Tool Class Interface

### Constructor

```python
Tool(name: str, endpoint: str, method: str = "POST", timeout: int = 10)

