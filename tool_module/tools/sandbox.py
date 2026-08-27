"""Registers the sandbox toolset for discovery: `registry.local_tools` scans only this package."""

from __future__ import annotations

from tool_module.sandbox.tools import TOOLS

__all__ = ["TOOLS"]
