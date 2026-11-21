"""The browser tools against a real Browserless: the ticket's acceptance test, in code.

Deselected by default (`-m "not integration"`) and skipped unless BROWSERLESS_URL is set.
"""

from __future__ import annotations

import os
import uuid

import pytest

from tool_module import registry
from tool_module.browser.manager import manager
from tool_module.envelope import ToolContext

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("BROWSERLESS_URL"), reason="BROWSERLESS_URL is not set"),
]

PAGE = "data:text/html,<title>notes</title><input aria-label='note'><button>save</button>"


async def test_the_page_survives_turns_and_the_agent_sees_a_persons_change():
    user = str(uuid.uuid4())
    ctx = ToolContext(user_id=user, session_id=str(uuid.uuid4()))
    try:
        # Turn one: open a page and set visible state.
        await registry.dispatch("browser_open", {"url": PAGE}, ctx)
        typed = await registry.dispatch("browser_type", {"ref": 1, "text": "from the agent"}, ctx)
        assert typed.ok, typed.content
        first_browser = manager().get(user).browser

        # Turn two: the state is still there.
        seen = await registry.dispatch("browser_snapshot", {}, ctx)
        assert '= "from the agent"' in seen.content

        # A person changes the page between turns, through the same path the app uses.
        box = await manager().get(user).page.locator("input").bounding_box()
        center = {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2}
        assert await manager().act_for_person(user, {"kind": "click", **center}) is True
        assert await manager().act_for_person(user, {"kind": "key", "key": "Control+A"}) is False
        await manager().act_for_person(user, {"kind": "type", "text": "from the person"})

        # The agent's next look sees the person's change, in the same browser.
        seen = await registry.dispatch("browser_snapshot", {}, ctx)
        assert '= "from the person"' in seen.content
        assert manager().get(user).browser is first_browser
        assert manager().get(user).changed_by_human is False
    finally:
        await manager().close_all()
