"""The browser against the real library, not a fake of it.

`tests/test_browser.py` fakes the vendor to test the leash. That leaves exactly
one thing unproven, and it is the thing most likely to break: whether the calls
we make into `browser_use` are calls this version of `browser_use` has. A mock
encodes what we believe the API is, so believing it twice proves nothing.

Deselected by default (`-m "not integration"`); skipped outright unless
browser_use and a model key are both present. It costs a real browser, a real
model call and a minute.
"""

from __future__ import annotations

import os

import pytest

from tool_module import registry
from tool_module.envelope import ToolContext

browser_use = pytest.importorskip("browser_use", reason="the browser is not installed on this host")

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ARK_MODEL_KEY")),
        reason="a browser run needs a model key",
    ),
]

USER = "8f1d4a02-0000-4000-8000-000000000001"


async def test_the_signature_we_build_the_agent_with_is_one_this_version_has():
    """The cheap half: no browser, no model call, just the constructor's shape.

    If a version bump renamed the step callback, our own warning fires here and
    this fails — before a run in production goes quietly blind.
    """
    from tool_module.browser.tool import _accepted

    wanted = {"task": "t", "llm": object(), "register_new_step_callback": print}
    kept = _accepted(browser_use.Agent, wanted)

    assert "task" in kept, "browser_use.Agent no longer takes a task"
    assert "llm" in kept, "browser_use.Agent no longer takes an llm"
    assert "register_new_step_callback" in kept, (
        "browser_use renamed the step callback: progress events would silently stop"
    )


async def test_a_real_run_reports_progress_and_returns_an_envelope():
    """The expensive half: one small real task, end to end."""
    statuses: list[tuple[str, str | None]] = []
    ctx = ToolContext(
        user_id=USER,
        session_id="3f1d4a02-0000-4000-8000-0000000000ff",
        emit_status=lambda label, url=None: statuses.append((label, url)),
    )

    result = await registry.dispatch(
        "browser_task",
        {"task": "Open example.com and report the exact text of its heading.", "start_url": "https://example.com"},
        ctx,
    )

    assert result.ok, result.content
    assert "example" in result.content.lower()
    assert any(label.startswith("browsing · step") for label, _ in statuses), (
        "the step callback never fired, so a real run would be silent"
    )
    assert any(url for _, url in statuses), "the frame stream was never announced"


async def test_the_history_answers_the_questions_the_envelope_asks_it():
    """`ok` comes from the run's own verdict, so those accessors must exist."""
    from tool_module.browser.tool import _call

    agent = browser_use.Agent(task="say hello", llm=None)
    history = getattr(agent, "history", None)
    if history is None:
        pytest.skip("this version builds history only once a run starts")

    for accessor in ("final_result", "errors", "is_successful", "has_errors", "urls", "model_actions"):
        assert hasattr(history, accessor), f"browser_use history has no {accessor}"
        _call(history, accessor)  # must not raise
