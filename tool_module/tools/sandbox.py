"""Registers the sandbox toolset for discovery.

`registry.local_tools` scans this package, and the sandbox tools live next to
the e2b manager they drive. Importing them here puts them in the manifest
without moving them.
"""

from __future__ import annotations

from tool_module.sandbox.tools import TOOLS

__all__ = ["TOOLS"]
