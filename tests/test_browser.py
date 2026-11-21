"""The browser tools without a browser: what is offered, what is refused, and the frame feed.

`tests/test_browser_integration.py` drives a real browser through the acceptance test.
"""

from __future__ import annotations

import pytest

from tool_module import registry
from tool_module.browser import tool as browser_tool
from tool_module.browser.manager import BrowserManager
from tool_module.browser.stream import FrameBroker
from tool_module.envelope import ToolContext

pytestmark = pytest.mark.asyncio

BROWSER_TOOLS = {"browser_open", "browser_snapshot", "browser_click", "browser_type", "browser_press", "browser_back"}


async def test_the_browser_tools_replace_the_delegating_one():
    specs = {s.name for s in (await registry.manifest("user-1")).specs}
    assert specs >= BROWSER_TOOLS
    assert "browser_task" not in specs


async def test_a_browser_tool_refuses_when_no_browser_is_configured(monkeypatch):
    monkeypatch.setattr(browser_tool, "_cfg", lambda key, default=None: "")
    monkeypatch.delenv("BROWSERLESS_URL", raising=False)
    result = await registry.dispatch("browser_snapshot", {}, ToolContext(user_id="user-1", session_id="s"))
    assert not result.ok
    assert "not configured" in result.content


async def test_a_browser_tool_refuses_outside_a_session():
    result = await registry.dispatch("browser_snapshot", {}, ToolContext(user_id="user-1"))
    assert not result.ok
    assert "inside a session" in result.content


async def test_a_new_viewer_starts_from_the_latest_frame_of_that_users_browser():
    broker = FrameBroker()
    broker.publish("alice", "old")
    broker.publish("alice", "new")
    broker.publish("bob", "bobs")
    async with broker.subscribe("alice") as frames:
        assert frames.get_nowait() == "new"
        assert frames.empty()


async def test_a_person_cannot_act_on_a_browser_that_is_not_open():
    with pytest.raises(LookupError):
        await BrowserManager().act_for_person("alice", {"kind": "click", "x": 1, "y": 1})
