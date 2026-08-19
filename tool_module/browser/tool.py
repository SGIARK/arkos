"""`browser_task`: hand a goal to the browser specialist, and stay in charge of it.

The leash has four strands, and each one exists because its absence was a bug in
the implementation this replaces:

- **Progress is events.** `browser_use`'s per-step callback becomes ordinary
  `status` events in the session log, so a three-minute browser call is never a
  silently frozen UI.
- **The result is an envelope.** `ok` comes from the run's own history, a
  failure carries what went wrong, and the full record is stored behind a `ref`.
  A bare string cannot distinguish "found nothing" from "crashed".
- **The budget is asked, then enforced.** The step callback stops the agent at
  the deadline so partial results survive; `wait_for` is the backstop for an
  agent that ignores being asked.
- **Nothing is dropped silently.** A `browser_use` version that no longer
  accepts one of our kwargs logs a WARNING, because progress callbacks quietly
  vanishing is exactly the opacity this card outlaws.

`browser_use` and its playwright are imported inside the call, so the process
starts, the manifest builds and every other tool works without them installed.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from config_module.loader import config
from tool_module.browser.stream import broker
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, ToolUnavailable, fail, ok

logger = logging.getLogger(__name__)


def _cfg(key: str, default: Any) -> Any:
    value = config.get(key)
    return default if value is None else value


def _frames_url(session_id: str) -> str:
    """Where the UI mounts the pane. Relative: the app and the API share an origin."""
    return f"/sessions/{session_id}/browser/frames"


class BrowserTask:
    spec = ToolSpec(
        name="browser_task",
        description=(
            "Do something on the web in a real browser that keeps the user's logins: "
            "research across pages, fill a form, work inside a web app. Give it one clear "
            "goal in a sentence or two, including what 'done' looks like and any specific "
            "site to start from — it plans its own steps. Prefer it over guessing a URL. "
            "It is slow (tens of seconds to minutes) and it acts as the signed-in user, so "
            "request_approval first for anything that sends, buys, publishes or deletes."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The goal, and what done looks like."},
                "start_url": {"type": "string", "description": "Optional page to begin at."},
            },
            "required": ["task"],
        },
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        task = str(args.get("task") or "").strip()
        if not task:
            return fail("invalid_args", "browser_task needs a `task` describing the goal.")
        if ctx.session_id is None:
            raise ToolUnavailable("invalid_args", "The browser is only available inside a session.", retryable=False)

        where = cdp_url()
        if not where:
            # Loud, and not retryable: there is nothing the model can do about a
            # container that is not running, and quietly launching a browser in
            # this process instead would put a page-executing Chromium beside
            # the harness's credentials.
            return fail(
                "upstream_error",
                "The browser container is not reachable: browser.cdp_url is unset (it defaults from "
                "BROWSERLESS_URL). Start the browserless service from docker-compose and set it. "
                "This tool never launches a browser on the harness host.",
                retryable=False,
            )

        # The browser is shared per user and keeps its profile between calls, so
        # it is leased rather than capped (contracts' resource table).
        if ctx.lease is not None:
            await ctx.lease("browser")

        # Announced before the first step, so the pane is mountable while the
        # run is still worth watching.
        if ctx.emit_status is not None:
            ctx.emit_status("using the browser…", _frames_url(ctx.session_id))

        budget = float(_cfg("browser.wall_clock_s", 300))
        backstop = float(_cfg("browser.hard_timeout_s", budget + 30))
        run = _Run(task=task, start_url=args.get("start_url"), ctx=ctx, budget=budget, url=where)

        try:
            async with asyncio.timeout(backstop):
                history = await run.execute()
        except TimeoutError:
            # The graceful stop was ignored. Whatever the run recorded before
            # the backstop fired is still worth reporting.
            logger.warning("browser run in session %s ignored its stop and hit the backstop", ctx.session_id)
            return await _envelope(run.partial(), ctx, stopped="hard")
        except ImportError as e:
            raise ToolUnavailable(
                "upstream_error", f"The browser is not installed on this host: {e}", retryable=False
            ) from e
        except Exception as e:  # noqa: BLE001 - the vendor raises its own types
            logger.exception("browser run failed in session %s", ctx.session_id)
            return fail("upstream_error", f"The browser run failed: {type(e).__name__}: {e}", retryable=True)
        finally:
            await run.close()
            if ctx.emit_status is not None:
                ctx.emit_status("the browser is done")

        return await _envelope(history, ctx, stopped="deadline" if run.out_of_time else None)


class _Run:
    """One browser run, and the leash held on it."""

    def __init__(self, task: str, start_url: str | None, ctx: ToolContext, budget: float, url: str = ""):
        self.task = task
        self.start_url = start_url
        self.ctx = ctx
        self.budget = budget
        self.cdp_url = url
        self.browser: Any = None
        self.started = time.monotonic()
        self.out_of_time = False
        self.steps = 0
        self._last_frame = 0.0
        self._agent: Any = None

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    async def on_step(self, *a: Any, **kw: Any) -> None:
        """Report a step, send a frame if anyone is watching, and enforce the budget.

        Asked, not killed: an agent told to stop finishes its step and returns
        the history it has, which is a partial answer. A cancelled coroutine
        returns nothing at all.
        """
        self.steps += 1
        if self.ctx.emit_status is not None:
            self.ctx.emit_status(f"browsing · step {self.steps}")
        await self.show()
        if self.elapsed() >= self.budget and not self.out_of_time:
            self.out_of_time = True
            logger.info("browser run in session %s reached its budget; asking it to stop", self.ctx.session_id)
            stop = getattr(self._agent, "stop", None)
            if callable(stop):
                result = stop()
                if inspect.isawaitable(result):
                    await result

    async def show(self) -> None:
        """Push one frame, if anyone is watching and it is time for another.

        Nothing is captured for an audience that is not there: a run nobody
        opened pays nothing for video. A vendor that will not give us a
        screenshot costs the pane, not the run.
        """
        user_id, session_id = self.ctx.user_id, self.ctx.session_id
        if session_id is None or not broker.watching(user_id, session_id):
            return
        interval = float(_cfg("browser.frame_interval_s", 1.0))
        now = time.monotonic()
        if now - self._last_frame < interval:
            return
        self._last_frame = now

        frame = await _screenshot(self._agent)
        if frame:
            broker.publish(user_id, session_id, frame)

    async def close(self) -> None:
        """Give the container's session back. A leaked session holds a slot in the pool."""
        if self.browser is None:
            return
        try:
            closing = self.browser.close()
            if inspect.isawaitable(closing):
                await closing
        except Exception:  # noqa: BLE001 - the run is over; a failed close is a log line
            logger.exception("could not close the browser session for %s", self.ctx.session_id)

    def partial(self) -> Any:
        """Whatever the agent recorded before the backstop fired."""
        return getattr(self._agent, "history", None)

    async def execute(self) -> Any:
        agent_factory = _agent_factory()
        self._agent = agent_factory(self)
        result = self._agent.run(max_steps=int(_cfg("browser.max_steps", 25)))
        return await result if inspect.isawaitable(result) else result


def _agent_factory() -> Any:
    """Return a callable that builds the vendor's agent for one run.

    Indirected so a test can substitute a fake without a browser, and so the
    vendor import happens at call time rather than at import time.
    """
    return _build_browser_use_agent


def cdp_url() -> str:
    """Where the browser is. It is a container, and it is never this process.

    The browser executes pages the model chose, and this process holds the
    user's session cookies, the store's secret key and the model's context. A
    browser launched beside them is a different architecture with a different
    blast radius, not a degraded mode — so an unset URL refuses rather than
    falling back to a local Chromium.
    """
    return str(_cfg("browser.cdp_url", "") or os.environ.get("BROWSERLESS_URL", "")).strip()


def _augment_cdp_url(url: str) -> str:
    """Append `stealth=true` to the CDP URL unless stealth is disabled.

    Ported verbatim from the implementation 8.10 deleted: Browserless reads it
    off the query string, and losing it is the difference between pages loading
    and pages serving a bot wall.
    """
    if os.environ.get("BROWSER_USE_STEALTH", "1") == "0":
        return url
    parts = urlparse(url)
    query = dict(parse_qsl(parts.query))
    if query.get("stealth") == "true":
        return url
    query["stealth"] = "true"
    return urlunparse(parts._replace(query=urlencode(query)))


def _build_browser_use_agent(run: _Run) -> Any:
    """Construct `browser_use`'s Agent against the Browserless container."""
    from browser_use import Agent, Browser, ChatOpenAI  # noqa: PLC0415 - lazy; see the module docstring

    wiring = {"cdp_url": _augment_cdp_url(run.cdp_url), "is_local": False}
    browser = Browser(**_accepted(Browser, wiring, keep={"cdp_url"}))
    run.browser = browser

    wanted: dict[str, Any] = {
        "task": run.task if not run.start_url else f"{run.task}\n\nStart at: {run.start_url}",
        "llm": ChatOpenAI(
            model=str(_cfg("browser.model", _cfg("llm.model_name", "gpt-4.1-mini"))),
            base_url=str(_cfg("llm.base_url", "")) or None,
            api_key=str(_cfg("llm.api_key", "")) or None,
        ),
        "browser": browser,
        "register_new_step_callback": run.on_step,
    }
    return Agent(**_accepted(Agent, wanted, keep={"task", "llm", "browser"}))


def _accepted(target: Any, kwargs: dict[str, Any], keep: set[str] | None = None) -> dict[str, Any]:
    """Keep the kwargs this version of the vendor accepts, and say what was dropped.

    A version bump that renames `register_new_step_callback` would otherwise
    take every progress event with it and look like a browser that simply went
    quiet — which is the failure this contract names by name.
    """
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # pragma: no cover - a C-implemented callable
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return kwargs

    # `keep` names what the call cannot be made without. `cdp_url` in particular
    # is swallowed by **kwargs on some versions and absent from the signature on
    # others, and dropping it would silently launch a local browser.
    required = keep or set()
    kept = {k: v for k, v in kwargs.items() if k in parameters or k in required}
    dropped = sorted(set(kwargs) - set(kept))
    if dropped:
        logger.warning(
            "browser_use %s does not accept %s; dropped. Progress or budget enforcement may be silently off.",
            getattr(target, "__name__", target),
            ", ".join(dropped),
        )
    return kept


async def _envelope(history: Any, ctx: ToolContext, stopped: str | None = None) -> ResultEnvelope:
    """Build the tool result from the run's own history.

    `ok` is the run's verdict, not ours, and a failure says what went wrong. The
    whole record goes behind a `ref` so the view stays small and the model can
    page the rest if it needs to.
    """
    if history is None:
        return fail("upstream_error", "The browser run produced no history to report.", retryable=True)

    answer = _call(history, "final_result") or ""
    errors = [str(e) for e in (_call(history, "errors") or []) if e]
    succeeded = _call(history, "is_successful")
    failed = bool(_call(history, "has_errors")) or bool(errors)
    urls = [str(u) for u in (_call(history, "urls") or []) if u]

    record = {
        "task_answer": answer,
        "urls": urls[-20:],
        "errors": errors,
        "steps": len(_call(history, "model_actions") or []),
        "stopped": stopped,
    }
    ref = None
    if ctx.store_blob is not None:
        ref = await ctx.store_blob(json.dumps(record, indent=2, default=str))

    if succeeded is False or (succeeded is None and failed):
        summary = "; ".join(errors[:3]) or "the run ended without reaching its goal"
        if stopped == "deadline":
            summary = f"stopped at the time budget after {record['steps']} step(s): {summary}"
        return fail("upstream_error", f"The browser did not finish the task: {summary}", retryable=True)

    body = answer or "The browser finished without a written answer."
    if stopped == "deadline":
        body = f"[stopped at the time budget; partial result]\n{body}"
    elif stopped == "hard":
        body = f"[the run overran its budget and was cut off; partial result]\n{body}"
    if urls:
        body = f"{body}\n\nLast page: {urls[-1]}"
    return ok(body, ref=ref)


async def _screenshot(agent: Any) -> str | None:
    """A base64 JPEG of what the browser is looking at, or None.

    The vendor has moved this between objects across versions, so the accessors
    are tried in order and a miss is reported once at WARNING rather than
    raising: losing the picture must never lose the run. Same reasoning as the
    dropped-kwarg warning above — silence is the thing being outlawed, not
    breakage.
    """
    global _screenshot_warned
    for owner_name, method_name in (
        ("browser_session", "take_screenshot"),
        ("browser_context", "take_screenshot"),
        ("browser", "take_screenshot"),
    ):
        owner = getattr(agent, owner_name, None)
        method = getattr(owner, method_name, None) if owner is not None else None
        if not callable(method):
            continue
        try:
            shot = method()
            if inspect.isawaitable(shot):
                shot = await shot
        except Exception:  # noqa: BLE001 - a failed picture is not a failed run
            logger.debug("browser screenshot via %s.%s raised", owner_name, method_name, exc_info=True)
            return None
        return shot if isinstance(shot, str) else None

    if not _screenshot_warned:
        _screenshot_warned = True
        logger.warning(
            "this browser_use version exposes no take_screenshot we know of; "
            "the frame pane will stay empty while runs still work"
        )
    return None


# Warned once per process: a version mismatch is one fact, not one per step.
_screenshot_warned = False


def _call(history: Any, name: str) -> Any:
    """Read one field from the vendor's history, whether it is a method or an attribute."""
    value = getattr(history, name, None)
    if callable(value):
        try:
            return value()
        except Exception:  # noqa: BLE001 - a history that cannot answer is not a crash
            logger.debug("browser history.%s() raised", name, exc_info=True)
            return None
    return value


TOOLS = [BrowserTask()]
