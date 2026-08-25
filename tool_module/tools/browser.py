"""Registry entry for the browser tool.

`registry.local_tools()` discovers modules in this package; the implementation
lives in `tool_module/browser/` with the frame side-channel it needs. This is
the one line that puts `browser_task` in the manifest — the card it was rebuilt
under exists partly because the old version was complete and reachable from
nothing.
"""

from tool_module.browser.tool import TOOLS

__all__ = ["TOOLS"]
