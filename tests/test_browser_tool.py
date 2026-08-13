"""The browser automation tool in tool_module/browser_tool.py.

The browser-use Agent is mocked; what is pinned is the wiring around it.
"""

from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_browser_use(monkeypatch, run_side_effect=None, captured_cdp=None):
    """Install a fake `browser_use` matching the 0.12 surface.

    `run_side_effect` becomes the result of Agent.run(); a list pops one item per call.
    """
    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            self.cdp_url = cdp_url
            self.is_local = is_local
            self.closed = False
            if captured_cdp is not None:
                captured_cdp.append(cdp_url)

        async def close(self):
            self.closed = True

    class FakeHistory:
        def __init__(self, value):
            self._value = value

        def final_result(self):
            return self._value

    class FakeAgent:
        def __init__(self, task, llm, browser):
            self.task = task
            self.llm = llm
            self.browser = browser

        async def run(self, max_steps=None):
            value = run_side_effect.pop(0) if isinstance(run_side_effect, list) else run_side_effect
            if isinstance(value, Exception):
                raise value
            return FakeHistory(value)

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            self.model = model
            self.base_url = base_url
            self.api_key = api_key

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI

    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)


@pytest.mark.asyncio
async def test_browser_tool_returns_result(monkeypatch):
    from tool_module.browser_tool import run_browser_task

    captured = []
    _install_fake_browser_use(monkeypatch, run_side_effect="Example Domain", captured_cdp=captured)
    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_USE_STEALTH", "0")  # keep this test's URL assertion stable

    result = await run_browser_task("user_1", "go to example.com and return the title")

    assert result == "Example Domain"
    assert captured == ["ws://browserless:3000"]


@pytest.mark.asyncio
async def test_browser_tool_routes_llm_through_sglang(monkeypatch):
    """The Agent's LLM points at the in-cluster SGLang server, not the OpenAI API."""
    import tool_module.browser_tool as bt

    captured_llm = {}

    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            self.cdp_url = cdp_url

        async def close(self):
            pass

    class FakeHistory:
        def final_result(self):
            return "ok"

    class FakeAgent:
        def __init__(self, task, llm, browser):
            self.llm = llm

        async def run(self, max_steps=None):
            return FakeHistory()

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            captured_llm["model"] = model
            captured_llm["base_url"] = base_url
            captured_llm["api_key"] = api_key

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("SGLANG_URL", "http://sglang:30000")
    monkeypatch.delenv("BROWSER_USE_MODEL", raising=False)

    await bt.run_browser_task("user_1", "anything")

    assert captured_llm["base_url"] == "http://sglang:30000/v1"
    assert captured_llm["model"] == "tgi"


@pytest.mark.asyncio
async def test_browser_tool_user_isolation(monkeypatch):
    from tool_module.browser_tool import run_browser_task

    _install_fake_browser_use(monkeypatch, run_side_effect=["A", "B"])
    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")

    result_a, result_b = await asyncio.gather(
        run_browser_task("user_a", "task one"),
        run_browser_task("user_b", "task two"),
    )

    # The fake Agent pops from a shared list, so only the set is deterministic.
    assert {result_a, result_b} == {"A", "B"}


@pytest.mark.asyncio
async def test_browser_tool_missing_url_raises(monkeypatch):
    from tool_module.browser_tool import BrowserToolError, run_browser_task

    monkeypatch.delenv("BROWSERLESS_URL", raising=False)
    with pytest.raises(BrowserToolError, match="BROWSERLESS_URL"):
        await run_browser_task("user_1", "anything")


@pytest.mark.asyncio
async def test_browser_tool_agent_failure_surfaces(monkeypatch):
    from tool_module.browser_tool import BrowserToolError, run_browser_task

    _install_fake_browser_use(
        monkeypatch,
        run_side_effect=ConnectionRefusedError("CDP endpoint unreachable"),
    )
    monkeypatch.setenv("BROWSERLESS_URL", "ws://nope:3000")

    with pytest.raises(BrowserToolError, match="CDP endpoint unreachable"):
        await run_browser_task("user_1", "anything")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_browser_tool_calls_register_local_tool():
    from tool_module.browser_tool import register_browser_tool

    tm = MagicMock()
    tm.register_local_tool = MagicMock()

    register_browser_tool(tm)

    tm.register_local_tool.assert_called_once()
    kwargs = tm.register_local_tool.call_args.kwargs
    assert kwargs["name"] == "browser_task"
    assert "task" in kwargs["input_schema"]["properties"]
    assert kwargs["input_schema"]["required"] == ["task"]


def test_register_browser_tool_noop_on_none():
    from tool_module.browser_tool import register_browser_tool

    register_browser_tool(None)  # must not raise


@pytest.mark.asyncio
async def test_handler_routes_through_run_browser_task(monkeypatch):
    """The MCP-shaped handler unpacks `task` and returns content."""
    import tool_module.browser_tool as bt

    monkeypatch.setattr(bt, "run_browser_task", AsyncMock(return_value="hello"))
    out = await bt._handler({"task": "say hi"}, "user_1")
    assert out == {"content": [{"type": "text", "text": "hello"}]}
    bt.run_browser_task.assert_awaited_once_with("user_1", "say hi")


@pytest.mark.asyncio
async def test_handler_rejects_missing_task():
    from tool_module.browser_tool import BrowserToolError, _handler

    with pytest.raises(BrowserToolError, match="task"):
        await _handler({}, "user_1")


# ---------------------------------------------------------------------------
# Screencast wiring
# ---------------------------------------------------------------------------


class _FakeMethodNamespace:
    """Stand-in for cdp_use's chained `send.Page.x(...)` / `register.Page.y(...)` accessors."""

    def __init__(self, on_call):
        self._on_call = on_call

    def __getattr__(self, attr):
        return _FakeMethodNamespace(lambda *args, _path=attr, **kwargs: self._on_call(_path, args, kwargs))


class _FakeCDPClient:
    """Records both registered event handlers and sent commands."""

    def __init__(self):
        self.sent: list[tuple[str, tuple, dict]] = []
        self._handlers: dict[str, list] = {}
        self.send = _Sender(self.sent)
        self.register = _Registrar(self._handlers)

    def fire(self, method: str, event: dict, session_id=None):
        for handler in self._handlers.get(method, []):
            handler(event, session_id)


class _Sender:
    def __init__(self, sink):
        self._sink = sink

    def __getattr__(self, namespace):
        return _SenderNamespace(self._sink, namespace)


class _SenderNamespace:
    def __init__(self, sink, namespace):
        self._sink = sink
        self._namespace = namespace

    def __getattr__(self, method):
        async def call(params=None, session_id=None):
            self._sink.append((f"{self._namespace}.{method}", params or {}, session_id))

        return call


class _Registrar:
    def __init__(self, sink):
        self._sink = sink

    def __getattr__(self, namespace):
        return _RegistrarNamespace(self._sink, namespace)


class _RegistrarNamespace:
    def __init__(self, sink, namespace):
        self._sink = sink
        self._namespace = namespace

    def __getattr__(self, method):
        def register(handler):
            self._sink.setdefault(f"{self._namespace}.{method}", []).append(handler)

        return register


class _FakeCDPSession:
    def __init__(self, cdp_client, target_id="T1", session_id="S1"):
        self.cdp_client = cdp_client
        self.target_id = target_id
        self.session_id = session_id


class _FakeBrowserSession:
    """Minimal stand-in for browser-use 0.12 BrowserSession."""

    def __init__(self, cdp_client, *, target_id="T1", session_id="S1"):
        self.cdp_client = cdp_client
        self.agent_focus_target_id = target_id
        self._session = _FakeCDPSession(cdp_client, target_id=target_id, session_id=session_id)

    async def get_or_create_cdp_session(self, target_id=None, focus=False):
        return self._session


@pytest.mark.asyncio
async def test_browser_tool_starts_and_ends_screencast_session(monkeypatch):
    """One started/ended pair around agent.run(), with CDP frames forwarded to push_frame."""
    import tool_module.browser_tool as bt

    cdp_client = _FakeCDPClient()
    browser_session = _FakeBrowserSession(cdp_client)

    captured = []

    class FakeBroker:
        def start_session(self, user_id):
            captured.append(("start", user_id))

        def push_frame(self, user_id, jpeg_b64):
            captured.append(("frame", user_id, jpeg_b64))

        def end_session(self, user_id):
            captured.append(("end", user_id))

    monkeypatch.setattr(bt, "_stream_broker", FakeBroker())

    _install_fake_browser_use(monkeypatch, run_side_effect="done")
    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")

    real_browser_use = sys.modules["browser_use"]

    class FakeAgentWithFrame:
        def __init__(self, task, llm, browser, **_):
            self.browser = browser
            self.browser_session = browser_session

        async def run(self, max_steps=None):
            # Wait until the screencast handler has registered and Start was sent.
            for _ in range(40):
                if any(s[0] == "Page.startScreencast" for s in cdp_client.sent):
                    break
                await asyncio.sleep(0.01)
            cdp_client.fire(
                "Page.screencastFrame",
                {"data": "ZZZZ", "sessionId": 7},
                session_id="S1",
            )
            await asyncio.sleep(0.01)

            class _H:
                def final_result(self_inner):
                    return "done"

            return _H()

    real_browser_use.Agent = FakeAgentWithFrame

    result = await bt.run_browser_task("user_42", "do a thing")

    assert result == "done"
    assert ("start", "user_42") in captured
    assert ("end", "user_42") in captured
    assert ("frame", "user_42", "ZZZZ") in captured
    start_calls = [s for s in cdp_client.sent if s[0] == "Page.startScreencast"]
    assert start_calls, "startScreencast was never sent"
    assert start_calls[0][2] == "S1"  # session_id
    ack_calls = [s for s in cdp_client.sent if s[0] == "Page.screencastFrameAck"]
    assert any(s[1].get("sessionId") == 7 for s in ack_calls), "ack missing for fired frame"
    assert any(s[0] == "Page.stopScreencast" for s in cdp_client.sent)


@pytest.mark.asyncio
async def test_browser_tool_screencast_ignores_frames_from_other_sessions(monkeypatch):
    """The cdp_client is shared across targets, so other sessions' frames must not bleed in."""
    import tool_module.browser_tool as bt

    cdp_client = _FakeCDPClient()
    browser_session = _FakeBrowserSession(cdp_client, session_id="S1")

    frames = []

    class FakeBroker:
        def start_session(self, user_id):
            pass

        def push_frame(self, user_id, jpeg_b64):
            frames.append(jpeg_b64)

        def end_session(self, user_id):
            pass

    monkeypatch.setattr(bt, "_stream_broker", FakeBroker())
    _install_fake_browser_use(monkeypatch, run_side_effect="done")
    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")

    real_browser_use = sys.modules["browser_use"]

    class FakeAgent:
        def __init__(self, task, llm, browser, **_):
            self.browser_session = browser_session

        async def run(self, max_steps=None):
            for _ in range(40):
                if any(s[0] == "Page.startScreencast" for s in cdp_client.sent):
                    break
                await asyncio.sleep(0.01)
            cdp_client.fire("Page.screencastFrame", {"data": "MINE", "sessionId": 1}, session_id="S1")
            cdp_client.fire("Page.screencastFrame", {"data": "NOPE", "sessionId": 2}, session_id="OTHER")
            await asyncio.sleep(0.01)

            class _H:
                def final_result(self_inner):
                    return "done"

            return _H()

    real_browser_use.Agent = FakeAgent

    await bt.run_browser_task("user_1", "task")
    assert frames == ["MINE"]


@pytest.mark.asyncio
async def test_browser_tool_screencast_disabled_via_env(monkeypatch):
    """BROWSER_STREAM_ENABLED=0 should skip the broker entirely."""
    import tool_module.browser_tool as bt

    touched = []

    class TouchyBroker:
        def start_session(self, user_id):
            touched.append("start")

        def push_frame(self, user_id, jpeg_b64):
            touched.append("frame")

        def end_session(self, user_id):
            touched.append("end")

    monkeypatch.setattr(bt, "_stream_broker", TouchyBroker())
    _install_fake_browser_use(monkeypatch, run_side_effect="done")
    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    result = await bt.run_browser_task("user_1", "task")

    assert result == "done"
    assert touched == []


@pytest.mark.asyncio
async def test_browser_tool_screencast_failure_does_not_break_agent(monkeypatch):
    """With no focused target the screencast exits quietly and the agent still completes."""
    import tool_module.browser_tool as bt

    _install_fake_browser_use(monkeypatch, run_side_effect="ok")
    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")

    # Speed up the target-readiness poll loop.
    real_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await real_sleep(0)

    monkeypatch.setattr(bt.asyncio, "sleep", fast_sleep)

    real_browser_use = sys.modules["browser_use"]

    class NoTargetAgent:
        def __init__(self, task, llm, browser, **_):
            # agent_focus_target_id stays falsy, so the screencast bails.
            self.browser_session = SimpleNamespaceLike(agent_focus_target_id=None)

        async def run(self, max_steps=None):
            class _H:
                def final_result(self_inner):
                    return "ok"

            return _H()

    class SimpleNamespaceLike:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    real_browser_use.Agent = NoTargetAgent

    result = await bt.run_browser_task("user_1", "task")
    assert result == "ok"


@pytest.mark.asyncio
async def test_browser_tool_wall_clock_timeout(monkeypatch):
    """A runaway agent must be aborted after BROWSER_USE_MAX_SECONDS."""
    from tool_module.browser_tool import BrowserToolError

    # Install a fake browser_use whose Agent.run hangs forever.
    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            pass

        async def close(self):
            pass

    class FakeAgent:
        def __init__(self, task, llm, browser):
            pass

        async def run(self, max_steps=None):
            await asyncio.sleep(10)  # longer than the test timeout below
            return None

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            pass

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_USE_MAX_SECONDS", "0.1")
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    from tool_module.browser_tool import run_browser_task

    with pytest.raises(BrowserToolError, match="wall-clock"):
        await run_browser_task("user_1", "task")


@pytest.mark.asyncio
async def test_browser_tool_passes_max_steps_when_supported(monkeypatch):
    captured = {}

    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            pass

        async def close(self):
            pass

    class FakeHistory:
        def final_result(self):
            return "done"

    class FakeAgent:
        def __init__(self, task, llm, browser):
            pass

        async def run(self, max_steps=None):
            captured["max_steps"] = max_steps
            return FakeHistory()

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            pass

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_USE_MAX_STEPS", "7")
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    from tool_module.browser_tool import run_browser_task

    result = await run_browser_task("user_1", "task")
    assert result == "done"
    assert captured["max_steps"] == 7


@pytest.mark.asyncio
async def test_browser_tool_passes_agent_config_knobs(monkeypatch):
    """max_failures, max_actions_per_step and llm_timeout reach the Agent."""
    captured: dict[str, object] = {}

    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            pass

        async def close(self):
            pass

    class FakeHistory:
        def final_result(self):
            return "done"

    class FakeAgent:
        def __init__(self, task, llm, browser, max_failures=None, max_actions_per_step=None, llm_timeout=None):
            captured["max_failures"] = max_failures
            captured["max_actions_per_step"] = max_actions_per_step
            captured["llm_timeout"] = llm_timeout

        async def run(self, max_steps=None):
            return FakeHistory()

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            pass

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_USE_MAX_FAILURES", "5")
    monkeypatch.setenv("BROWSER_USE_MAX_ACTIONS_PER_STEP", "8")
    monkeypatch.setenv("BROWSER_USE_LLM_TIMEOUT", "30")
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    from tool_module.browser_tool import run_browser_task

    await run_browser_task("user_1", "task")

    assert captured["max_failures"] == 5
    assert captured["max_actions_per_step"] == 8
    assert captured["llm_timeout"] == 30


def test_augment_cdp_url_appends_stealth(monkeypatch):
    """Default behaviour: ?stealth=true is appended."""
    from tool_module.browser_tool import _augment_cdp_url

    monkeypatch.delenv("BROWSER_USE_STEALTH", raising=False)
    assert _augment_cdp_url("ws://browserless:3000") == "ws://browserless:3000?stealth=true"


def test_augment_cdp_url_preserves_existing_query(monkeypatch):
    from tool_module.browser_tool import _augment_cdp_url

    monkeypatch.delenv("BROWSER_USE_STEALTH", raising=False)
    out = _augment_cdp_url("ws://browserless:3000?token=secret")
    assert "token=secret" in out
    assert "stealth=true" in out


def test_augment_cdp_url_no_op_when_disabled(monkeypatch):
    from tool_module.browser_tool import _augment_cdp_url

    monkeypatch.setenv("BROWSER_USE_STEALTH", "0")
    assert _augment_cdp_url("ws://browserless:3000") == "ws://browserless:3000"


def test_parse_allowed_domains_handles_empty_and_csv():
    from tool_module.browser_tool import _parse_allowed_domains

    assert _parse_allowed_domains(None) == []
    assert _parse_allowed_domains("") == []
    assert _parse_allowed_domains("https://example.com") == ["https://example.com"]
    assert _parse_allowed_domains("https://a.com, https://*.b.com ,") == [
        "https://a.com",
        "https://*.b.com",
    ]


@pytest.mark.asyncio
async def test_browser_tool_passes_allowed_domains_to_browser(monkeypatch):
    captured: dict[str, object] = {}

    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True, allowed_domains=None):
            captured["allowed_domains"] = allowed_domains

        async def close(self):
            pass

    class FakeHistory:
        def final_result(self):
            return "ok"

    class FakeAgent:
        def __init__(self, task, llm, browser, **_):
            pass

        async def run(self, max_steps=None):
            return FakeHistory()

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            pass

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_USE_ALLOWED_DOMAINS", "https://example.com, https://*.wikipedia.org")
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    from tool_module.browser_tool import run_browser_task

    await run_browser_task("user_1", "task")
    assert captured["allowed_domains"] == ["https://example.com", "https://*.wikipedia.org"]


def test_augment_cdp_url_idempotent(monkeypatch):
    from tool_module.browser_tool import _augment_cdp_url

    monkeypatch.delenv("BROWSER_USE_STEALTH", raising=False)
    once = _augment_cdp_url("ws://browserless:3000")
    twice = _augment_cdp_url(once)
    assert twice.count("stealth=true") == 1


@pytest.mark.asyncio
async def test_browser_tool_defaults_vision_judge_thinking_off(monkeypatch):
    """For a text-only Qwen, vision/thinking/judge default OFF to save tokens."""
    captured: dict[str, object] = {}

    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            pass

        async def close(self):
            pass

    class FakeHistory:
        def final_result(self):
            return "ok"

    class FakeAgent:
        def __init__(self, task, llm, browser, use_vision=True, use_thinking=True, use_judge=True, **_):
            captured["use_vision"] = use_vision
            captured["use_thinking"] = use_thinking
            captured["use_judge"] = use_judge

        async def run(self, max_steps=None):
            return FakeHistory()

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            pass

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.delenv("BROWSER_USE_VISION", raising=False)
    monkeypatch.delenv("BROWSER_USE_THINKING", raising=False)
    monkeypatch.delenv("BROWSER_USE_USE_JUDGE", raising=False)
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    from tool_module.browser_tool import run_browser_task

    await run_browser_task("user_1", "task")

    assert captured["use_vision"] is False
    assert captured["use_thinking"] is False
    assert captured["use_judge"] is False


@pytest.mark.asyncio
async def test_browser_tool_vision_judge_thinking_envs_take_effect(monkeypatch):
    captured: dict[str, object] = {}

    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            pass

        async def close(self):
            pass

    class FakeHistory:
        def final_result(self):
            return "ok"

    class FakeAgent:
        def __init__(self, task, llm, browser, use_vision=False, use_thinking=False, use_judge=False, **_):
            captured["use_vision"] = use_vision
            captured["use_thinking"] = use_thinking
            captured["use_judge"] = use_judge

        async def run(self, max_steps=None):
            return FakeHistory()

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            pass

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_USE_VISION", "1")
    monkeypatch.setenv("BROWSER_USE_THINKING", "true")
    monkeypatch.setenv("BROWSER_USE_USE_JUDGE", "yes")
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    from tool_module.browser_tool import run_browser_task

    await run_browser_task("user_1", "task")

    assert captured["use_vision"] is True
    assert captured["use_thinking"] is True
    assert captured["use_judge"] is True


@pytest.mark.asyncio
async def test_browser_tool_silently_drops_unsupported_kwargs(monkeypatch):
    """An Agent that does not accept max_failures must not crash the call."""
    captured: dict[str, object] = {}

    fake_browser_use = types.ModuleType("browser_use")

    class FakeBrowser:
        def __init__(self, cdp_url=None, is_local=True):
            pass

        async def close(self):
            pass

    class FakeHistory:
        def final_result(self):
            return "fallback"

    class FakeAgent:
        # Old constructor: only the required three.
        def __init__(self, task, llm, browser):
            captured["constructed"] = True

        async def run(self, max_steps=None):
            return FakeHistory()

    class FakeChatOpenAI:
        def __init__(self, model, base_url=None, api_key=None):
            pass

    fake_browser_use.Agent = FakeAgent
    fake_browser_use.Browser = FakeBrowser
    fake_browser_use.ChatOpenAI = FakeChatOpenAI
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)

    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")
    monkeypatch.setenv("BROWSER_USE_MAX_FAILURES", "5")
    monkeypatch.setenv("BROWSER_STREAM_ENABLED", "0")

    from tool_module.browser_tool import run_browser_task

    result = await run_browser_task("user_1", "task")
    assert result == "fallback"
    assert captured["constructed"] is True
