"""The browser on its leash: progress, budget, envelope, and who may watch.

The vendor is faked here — a real browser_use run needs a browser, a model and
minutes. What is NOT faked is the leash itself, which is the whole of what this
card added: the fake agent is asked to stop and obeys or does not, and the tests
check what we do about it either way. `tests/test_browser_integration.py` is the
one that talks to the real library.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import pytest

from tool_module import registry
from tool_module.browser import tool as browser_tool
from tool_module.browser.stream import FrameBroker
from tool_module.envelope import ToolContext

pytestmark = pytest.mark.asyncio

USER = "8f1d4a02-0000-4000-8000-000000000001"


class FakeHistory:
    """What browser_use hands back: methods, mostly."""

    def __init__(self, answer="", errors=(), successful=True, urls=("https://example.com",), steps=3):
        self._answer = answer
        self._errors = list(errors)
        self._successful = successful
        self._urls = list(urls)
        self._steps = steps

    def final_result(self):
        return self._answer

    def errors(self):
        return self._errors

    def is_successful(self):
        return self._successful

    def has_errors(self):
        return bool(self._errors)

    def urls(self):
        return self._urls

    def model_actions(self):
        return [{"step": i} for i in range(self._steps)]


class FakeAgent:
    """An agent that calls the step callback, and stops when asked — or does not."""

    def __init__(self, run, steps=3, obeys_stop=True, step_delay=0.0, history=None, screenshot=None):
        self.run_state = run
        self.steps = steps
        self.obeys_stop = obeys_stop
        self.step_delay = step_delay
        self.history = history or FakeHistory(answer="done")
        self.stopped = False
        self.browser_session = _FakeSession(screenshot) if screenshot else None

    def stop(self):
        self.stopped = True

    async def run(self, max_steps=25):
        for _ in range(min(self.steps, max_steps)):
            if self.stopped and self.obeys_stop:
                break
            await self.run_state.on_step()
            if self.step_delay:
                await asyncio.sleep(self.step_delay)
        return self.history


class _FakeSession:
    def __init__(self, shot):
        self._shot = shot

    async def take_screenshot(self):
        return self._shot


def _ctx(**kw):
    statuses = kw.pop("statuses", None)
    return ToolContext(
        user_id=USER,
        session_id=kw.pop("session_id", "3f1d4a02-0000-4000-8000-0000000000aa"),
        emit_status=(lambda label, url=None: statuses.append((label, url))) if statuses is not None else None,
        **kw,
    )


def _with_agent(monkeypatch, **agent_kw):
    """Substitute the vendor with a fake, at the one seam built for it.

    The container url is faked too: every path below the guard assumes one, and
    the guard itself has its own test above.
    """
    made = {}
    monkeypatch.setattr(browser_tool, "cdp_url", lambda: "ws://browserless:3000", raising=False)

    def factory():
        def build(run):
            made["agent"] = FakeAgent(run, **agent_kw)
            return made["agent"]

        return build

    monkeypatch.setattr(browser_tool, "_agent_factory", factory)
    return made


# --- it is in the manifest, which is what makes it real -----------------------------


async def test_browser_task_is_in_the_manifest():
    """The old one was complete and reachable from nothing; that is the card's point."""
    specs = {s.name: s for s in await registry.manifest(USER)}

    assert "browser_task" in specs
    assert specs["browser_task"].readonly is False, "it acts on the web as the signed-in user"


# --- where the browser is -----------------------------------------------------------


async def test_no_cdp_url_is_a_loud_refusal_and_launches_nothing(monkeypatch):
    """A browser beside the harness's credentials is a different architecture.

    So an unset url refuses rather than falling back to a local Chromium: the
    process holding the user's cookies and the store's secret key does not get
    to run pages the model picked.
    """
    made = _with_agent(monkeypatch)
    monkeypatch.setattr(browser_tool, "cdp_url", lambda: "")

    result = await registry.dispatch("browser_task", {"task": "go and look"}, _ctx())

    assert result.ok is False
    assert result.retryable is False, "there is nothing the model can retry about a missing container"
    assert "browser.cdp_url" in result.content and "BROWSERLESS_URL" in result.content
    assert "agent" not in made, "a browser was built without a container to build it against"


async def test_the_container_url_carries_the_stealth_parameter():
    """Ported from the deleted implementation: Browserless reads it off the query
    string, and losing it is the difference between pages loading and bot walls."""
    assert browser_tool._augment_cdp_url("ws://browserless:3000") == "ws://browserless:3000?stealth=true"
    assert browser_tool._augment_cdp_url("ws://b:3000?token=x&stealth=true").endswith("stealth=true")


async def test_stealth_can_be_turned_off_by_the_environment(monkeypatch):
    monkeypatch.setenv("BROWSER_USE_STEALTH", "0")

    assert browser_tool._augment_cdp_url("ws://browserless:3000") == "ws://browserless:3000"


async def test_the_url_falls_back_to_the_environment(monkeypatch):
    """config.yaml keeps it literal, because an unset ${VAR} there fails config load
    for the whole app rather than for one tool."""
    monkeypatch.setattr(browser_tool, "_cfg", lambda key, default: "" if key == "browser.cdp_url" else default)
    monkeypatch.setenv("BROWSERLESS_URL", "ws://browserless:3000")

    assert browser_tool.cdp_url() == "ws://browserless:3000"


async def test_a_kwarg_the_vendor_hides_behind_kwargs_is_still_passed():
    """cdp_url is absent from some versions' signatures and swallowed by **kwargs
    in others; dropping it would silently launch a local browser."""

    def browser_without_it(is_local=False):
        return None

    kept = browser_tool._accepted(browser_without_it, {"cdp_url": "ws://x", "is_local": False}, keep={"cdp_url"})

    assert kept["cdp_url"] == "ws://x"


# --- progress is events, never silence ----------------------------------------------


async def test_every_step_reports_progress_to_the_session(monkeypatch):
    statuses = []
    _with_agent(monkeypatch, steps=3)

    result = await registry.dispatch("browser_task", {"task": "find the price"}, _ctx(statuses=statuses))

    assert result.ok
    labels = [label for label, _ in statuses]
    assert labels[0] == "using the browser…"
    steps = [label for label in labels if label.startswith("step ")]
    assert steps == ["step 1/25", "step 2/25", "step 3/25"], "the inner loop is not being read out"
    assert labels[-1] == "the browser is done"


async def test_the_frame_stream_is_announced_on_the_first_status(monkeypatch):
    """The UI mounts the pane from the event stream, so the url has to be in it."""
    statuses = []
    _with_agent(monkeypatch)
    session_id = "3f1d4a02-0000-4000-8000-0000000000bb"

    await registry.dispatch("browser_task", {"task": "look"}, _ctx(statuses=statuses, session_id=session_id))

    announced = [url for _, url in statuses if url]
    assert announced == [f"/sessions/{session_id}/browser/frames"]


# --- the budget is asked first, enforced second -------------------------------------


async def test_the_budget_stops_the_run_and_keeps_what_it_found(monkeypatch):
    """Asked, not killed: a stopped agent still returns the history it has."""
    made = _with_agent(monkeypatch, steps=50, step_delay=0.01, history=FakeHistory(answer="half an answer"))
    monkeypatch.setattr(browser_tool, "_cfg", lambda key, default: 0.02 if key == "browser.wall_clock_s" else default)

    result = await registry.dispatch("browser_task", {"task": "endless"}, _ctx())

    assert made["agent"].stopped, "the agent was never asked to stop"
    assert made["agent"].steps > 0
    assert result.ok
    assert "partial result" in result.content
    assert "half an answer" in result.content


async def test_an_agent_that_ignores_the_stop_is_cut_off_and_still_reports(monkeypatch):
    """The wait_for backstop, and the partial history it salvages."""
    _with_agent(
        monkeypatch,
        steps=500,
        step_delay=0.01,
        obeys_stop=False,
        history=FakeHistory(answer="whatever it managed"),
    )
    values = {"browser.wall_clock_s": 0.02, "browser.hard_timeout_s": 0.15}
    monkeypatch.setattr(browser_tool, "_cfg", lambda key, default: values.get(key, default))

    result = await registry.dispatch("browser_task", {"task": "ignores you"}, _ctx())

    assert result.ok
    assert "cut off" in result.content
    assert "whatever it managed" in result.content


# --- the result is an envelope, never a bare string ---------------------------------


async def test_a_failed_run_says_what_went_wrong(monkeypatch):
    """Failure and empty were the same string in the implementation this replaces."""
    _with_agent(
        monkeypatch,
        history=FakeHistory(answer="", errors=["captcha blocked the page"], successful=False),
    )

    result = await registry.dispatch("browser_task", {"task": "buy it"}, _ctx())

    assert result.ok is False
    assert "captcha blocked the page" in result.content
    assert result.retryable is True


async def test_an_empty_success_is_not_a_failure(monkeypatch):
    _with_agent(monkeypatch, history=FakeHistory(answer="", successful=True))

    result = await registry.dispatch("browser_task", {"task": "look around"}, _ctx())

    assert result.ok
    assert result.content.strip(), "an empty answer still has to say something"


async def test_the_whole_run_is_stored_behind_a_ref(monkeypatch):
    stored = {}

    async def store_blob(text):
        stored["body"] = text
        return "ref-1"

    _with_agent(monkeypatch, history=FakeHistory(answer="the price is 12", urls=["https://shop.example/item"]))

    result = await registry.dispatch("browser_task", {"task": "price"}, _ctx(store_blob=store_blob))

    assert result.ref == "ref-1"
    record = json.loads(stored["body"])
    assert record["task_answer"] == "the price is 12"
    assert record["urls"] == ["https://shop.example/item"]


# --- the leash on the resource ------------------------------------------------------


async def test_the_browser_is_leased_before_it_is_driven(monkeypatch):
    """Shared per user and stateful across calls, so it is leased (contracts)."""
    taken = []
    _with_agent(monkeypatch)

    await registry.dispatch("browser_task", {"task": "go"}, _ctx(lease=lambda name: taken.append(name) or _done()))

    assert taken == ["browser"]


def _done():
    async def nothing():
        return None

    return nothing()


async def test_a_task_with_no_goal_is_refused_before_anything_boots(monkeypatch):
    made = _with_agent(monkeypatch)

    result = await registry.dispatch("browser_task", {"task": "   "}, _ctx())

    assert result.ok is False
    assert "agent" not in made, "an empty task started a browser"


# --- dropped kwargs are loud --------------------------------------------------------


async def test_a_vendor_that_lost_our_callback_warns(caplog):
    """A version bump that renames the step callback takes every progress event
    with it, and that must never be quiet."""

    def agent_without_callbacks(task=None, llm=None):
        return None

    kept = browser_tool._accepted(
        agent_without_callbacks, {"task": "t", "llm": None, "register_new_step_callback": print}
    )

    assert "register_new_step_callback" not in kept
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert any("register_new_step_callback" in w for w in warnings), warnings


async def test_nothing_is_dropped_when_the_vendor_still_accepts_it():
    def agent_with_callbacks(task=None, llm=None, register_new_step_callback=None):
        return None

    kept = browser_tool._accepted(
        agent_with_callbacks, {"task": "t", "llm": None, "register_new_step_callback": print}
    )

    assert set(kept) == {"task", "llm", "register_new_step_callback"}


# --- frames are ephemeral, and keyed by the pair --------------------------------------


async def test_frames_reach_the_watcher_of_that_session_only():
    """Two of a user's sessions browsing at once clobbered each other before."""
    frames = FrameBroker()
    mine, theirs = "session-a", "session-b"

    async with frames.subscribe(USER, mine) as queue:
        frames.publish(USER, theirs, "not-for-me")
        frames.publish(USER, mine, "for-me")

        assert queue.get_nowait() == "for-me"
        assert queue.empty()


async def test_a_slow_watcher_sees_the_newest_frame_not_a_backlog():
    frames = FrameBroker(queue_size=2)

    async with frames.subscribe(USER, "s") as queue:
        for i in range(6):
            frames.publish(USER, "s", f"frame-{i}")

        held = [queue.get_nowait() for _ in range(queue.qsize())]

    assert held == ["frame-4", "frame-5"], "video is only worth watching live"


class _FakeCdpSession:
    """The half of browser_use's CDP session the screencast touches."""

    def __init__(self, owner):
        self.session_id = "target-1"
        self.cdp_client = _FakeCdpClient(owner)


class _FakeCdpClient:
    def __init__(self, owner):
        self.owner = owner
        self.register = _FakeRegister(owner)
        self.send = _FakeSend(owner)


class _FakeRegister:
    def __init__(self, owner):
        self.owner = owner

    def Page(self):  # pragma: no cover - attribute access only
        raise NotImplementedError

    @property
    def screencastFrame(self):
        raise NotImplementedError


class _FakeSend:
    def __init__(self, owner):
        self.owner = owner

    async def Page(self):  # pragma: no cover
        raise NotImplementedError


class _FakeCdpAgent:
    """An agent whose CDP surface records what the screencast asked it to do."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self._handler = None
        self.browser_session = self

    # what _wait_for_target reads
    agent_focus_target_id = "target-1"

    async def get_or_create_cdp_session(self, target_id=None, focus=False):
        return _Cdp(self)

    @property
    def cdp_client(self):
        return _Cdp(self).cdp_client

    def emit(self, event, frame_session="target-1"):
        if self._handler:
            self._handler(event, frame_session)


class _Cdp:
    def __init__(self, agent):
        self.agent = agent
        self.session_id = "target-1"
        self.cdp_client = self

    @property
    def register(self):
        return self

    @property
    def send(self):
        return self

    @property
    def Page(self):
        return self

    def screencastFrame(self, handler):
        self.agent._handler = handler

    async def startScreencast(self, params=None, session_id=None):
        self.agent.started = True

    async def stopScreencast(self, params=None, session_id=None):
        self.agent.stopped = True

    async def screencastFrameAck(self, params=None, session_id=None):
        return None


async def test_the_screencast_puts_frames_on_the_broker(monkeypatch):
    """The CDP path, not a screenshot poll: browser_use hands us frames and we
    forward them keyed by (user, session)."""
    frames = FrameBroker()
    monkeypatch.setattr(browser_tool, "broker", frames)
    session_id = "3f1d4a02-0000-4000-8000-0000000000cc"
    agent = _FakeCdpAgent()

    async with frames.subscribe(USER, session_id) as queue:
        cast = asyncio.create_task(browser_tool.run_screencast(agent, USER, session_id))
        await asyncio.sleep(0)
        # What Chromium sends on Page.screencastFrame.
        agent.emit({"data": "ZmFrZS1qcGVn", "sessionId": 7})
        await asyncio.sleep(0)

        assert queue.get_nowait() == "ZmFrZS1qcGVn"
        assert agent.started, "the screencast was never asked for"

        cast.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await cast
    assert agent.stopped, "the screencast outlived the run"


async def test_a_frame_for_another_target_is_ignored():
    """The cdp_client fires for every target the browser has open."""
    frames = FrameBroker()
    agent = _FakeCdpAgent()
    session_id = "3f1d4a02-0000-4000-8000-0000000000dd"

    async with frames.subscribe(USER, session_id) as queue:
        with_broker = asyncio.create_task(browser_tool.run_screencast(agent, USER, session_id))
        await asyncio.sleep(0)
        agent.emit({"data": "someone-elses-page"}, frame_session="a-different-target")
        await asyncio.sleep(0)

        assert queue.empty()

        with_broker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await with_broker
