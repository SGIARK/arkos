"""Custom Tools() registration in tool_module/browser_actions.py."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


def _install_minimal_browser_use(monkeypatch, action_registry: list) -> Any:
    """Install a fake browser_use exposing Tools and ActionResult.

    action_registry collects (description, handler) tuples on every @tools.action decoration.
    """
    fake = types.ModuleType("browser_use")

    class FakeActionResult:
        def __init__(self, extracted_content=None, include_in_memory=False):
            self.extracted_content = extracted_content
            self.include_in_memory = include_in_memory

    class FakeTools:
        def __init__(self):
            self.registered: list[tuple[str, Any]] = action_registry

        def action(self, description: str, param_model=None, **_kwargs):
            def decorator(fn):
                self.registered.append((description, fn))
                if param_model is not None:
                    fn._param_model = param_model
                return fn

            return decorator

    fake.Tools = FakeTools
    fake.ActionResult = FakeActionResult
    monkeypatch.setitem(sys.modules, "browser_use", fake)
    # Drop any cached browser_actions so the lazy import picks up the fake.
    sys.modules.pop("tool_module.browser_actions", None)
    return fake


def test_build_arkos_tools_returns_none_without_browser_use(monkeypatch):
    monkeypatch.setitem(sys.modules, "browser_use", None)  # make the import fail
    sys.modules.pop("tool_module.browser_actions", None)
    import importlib

    spec = importlib.util.find_spec("tool_module.browser_actions")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    monkeypatch.delitem(sys.modules, "browser_use", raising=False)
    # Block the re-import so build_arkos_tools sees an ImportError.
    import builtins as _b

    real_import = _b.__import__

    def block(name, *a, **kw):
        if name == "browser_use":
            raise ImportError("blocked for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(_b, "__import__", block)
    assert mod.build_arkos_tools() is None


def test_build_arkos_tools_registers_dismiss_overlay(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)

    from tool_module.browser_actions import build_arkos_tools

    tools = build_arkos_tools()
    assert tools is not None
    descriptions = [d for d, _ in registry]
    assert any("cookie" in d.lower() or "consent" in d.lower() for d in descriptions), (
        f"dismiss_overlay should mention cookie/consent in description; got {descriptions!r}"
    )
    handlers = [h for _, h in registry]
    assert any(h.__name__ == "dismiss_overlay" for h in handlers)


@pytest.mark.asyncio
async def test_dismiss_overlay_action_clicks_when_match(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)

    from tool_module.browser_actions import build_arkos_tools

    tools = build_arkos_tools()
    assert tools is not None
    handler = next(h for _, h in registry if h.__name__ == "dismiss_overlay")

    class FakeCDPClient:
        class send:
            class Runtime:
                @staticmethod
                async def evaluate(params=None, session_id=None):
                    assert "TERMS" in params["expression"]
                    return {
                        "result": {
                            "value": {
                                "clicked": True,
                                "label": "accept all",
                                "score": 100,
                                "considered": 12,
                            }
                        }
                    }

    class FakeCDPSession:
        cdp_client = FakeCDPClient
        session_id = "S1"

    class FakeBrowserSession:
        async def get_or_create_cdp_session(self, target_id=None, focus=False):
            return FakeCDPSession()

    result = await handler(FakeBrowserSession())
    assert result.include_in_memory is True
    assert "accept all" in result.extracted_content
    assert "score=100" in result.extracted_content


@pytest.mark.asyncio
async def test_dismiss_overlay_action_no_match_returns_idempotent(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)

    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "dismiss_overlay")

    class FakeCDPClient:
        class send:
            class Runtime:
                @staticmethod
                async def evaluate(params=None, session_id=None):
                    return {"result": {"value": {"clicked": False, "considered": 0}}}

    class FakeCDPSession:
        cdp_client = FakeCDPClient
        session_id = "S1"

    class FakeBrowserSession:
        async def get_or_create_cdp_session(self, target_id=None, focus=False):
            return FakeCDPSession()

    result = await handler(FakeBrowserSession())
    assert result.include_in_memory is False
    assert "No consent/cookie overlay" in result.extracted_content


@pytest.mark.asyncio
async def test_dismiss_overlay_action_swallows_cdp_failures(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)

    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "dismiss_overlay")

    class BrokenBrowserSession:
        async def get_or_create_cdp_session(self, target_id=None, focus=False):
            raise RuntimeError("CDP socket dead")

    result = await handler(BrokenBrowserSession())
    assert "page not ready" in result.extracted_content


def test_all_five_actions_register(monkeypatch):
    """build_arkos_tools swallows per-action exceptions, so the set is checked whole."""
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)
    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    names = {h.__name__ for _, h in registry}
    assert names == {
        "dismiss_overlay",
        "wait_for_element",
        "click_then_wait_for_url_change",
        "scroll_to_load_all",
        "extract_text_region",
    }, f"unexpected action set: {names!r}"


def _make_fake_cdp(value):
    """Build a fake browser_session whose CDP evaluate returns `value`."""

    class FakeCDPClient:
        class send:
            class Runtime:
                @staticmethod
                async def evaluate(params=None, session_id=None):
                    FakeCDPClient.send.Runtime.last_expression = params["expression"]
                    return {"result": {"value": value}}

    class FakeCDPSession:
        cdp_client = FakeCDPClient
        session_id = "S1"

    class FakeBrowserSession:
        async def get_or_create_cdp_session(self, target_id=None, focus=False):
            return FakeCDPSession()

    return FakeBrowserSession()


@pytest.mark.asyncio
async def test_wait_for_element_found(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)
    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "wait_for_element")
    Params = handler._param_model
    session = _make_fake_cdp({"found": True, "waited_ms": 320, "box": {"x": 10, "y": 20, "w": 100, "h": 40}})
    result = await handler(Params(selector="#submit"), session)
    assert "#submit" in result.extracted_content
    assert "320ms" in result.extracted_content
    assert result.include_in_memory is True


@pytest.mark.asyncio
async def test_wait_for_element_timeout(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)
    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "wait_for_element")
    Params = handler._param_model
    session = _make_fake_cdp({"found": False, "waited_ms": 5000})
    result = await handler(Params(selector=".nope", timeout_ms=5000), session)
    assert "not visible" in result.extracted_content
    assert result.include_in_memory is False


@pytest.mark.asyncio
async def test_click_then_wait_navigated(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)
    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "click_then_wait_for_url_change")
    Params = handler._param_model
    session = _make_fake_cdp(
        {
            "clicked": True,
            "navigated": True,
            "from_url": "https://a.com/login",
            "to_url": "https://a.com/dashboard",
            "waited_ms": 420,
        }
    )
    result = await handler(Params(selector="button[type=submit]"), session)
    assert "dashboard" in result.extracted_content
    assert result.include_in_memory is True


@pytest.mark.asyncio
async def test_scroll_to_load_all_reports_settled(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)
    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "scroll_to_load_all")
    Params = handler._param_model
    session = _make_fake_cdp({"scrolls": 7, "final_height": 18432})
    result = await handler(Params(), session)
    assert "Scrolled 7 times" in result.extracted_content
    assert "18432" in result.extracted_content


@pytest.mark.asyncio
async def test_extract_text_region_returns_cleaned_text(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)
    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "extract_text_region")
    Params = handler._param_model
    session = _make_fake_cdp({"found": True, "text": "Example Domain. This domain is for use in examples."})
    result = await handler(Params(selector="main"), session)
    assert result.extracted_content.startswith("Example Domain")
    assert result.include_in_memory is True


@pytest.mark.asyncio
async def test_extract_text_region_missing_selector(monkeypatch):
    registry: list = []
    _install_minimal_browser_use(monkeypatch, registry)
    from tool_module.browser_actions import build_arkos_tools

    build_arkos_tools()
    handler = next(h for _, h in registry if h.__name__ == "extract_text_region")
    Params = handler._param_model
    session = _make_fake_cdp({"found": False})
    result = await handler(Params(selector="#nothing-here"), session)
    assert "No element matched" in result.extracted_content
    assert result.include_in_memory is False
