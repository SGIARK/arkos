"""One browser per user, kept open between turns so its pages, logins and form contents survive.

The browser runs in the Browserless container and is reached over CDP; this module holds the
connection. Playwright is imported on first use, so the app starts without it installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from config_module.loader import cfg as _cfg
from tool_module.browser.stream import broker

logger = logging.getLogger(__name__)

# The app's browser pane maps clicks onto this size, so it is fixed here rather than configured.
VIEWPORT = {"width": 1280, "height": 800}


@dataclass
class UserBrowser:
    """A user's open browser, the page everything acts on, and what has happened to it."""

    browser: Any
    page: Any
    last_used: float = field(default_factory=time.monotonic)
    # Set when a person acts on the page, cleared when the agent next looks at it.
    changed_by_human: bool = False


class BrowserManager:
    def __init__(self) -> None:
        self._playwright: Any = None
        self._browsers: dict[str, UserBrowser] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    def get(self, user_id: str) -> UserBrowser | None:
        """The user's browser if it is open, without opening one."""
        return self._browsers.get(user_id)

    async def open(self, user_id: str, cdp_url: str) -> UserBrowser:
        """The user's browser, connecting to Browserless if there is none or the old one dropped."""
        async with self._locks.setdefault(user_id, asyncio.Lock()):
            current = self._browsers.get(user_id)
            if current is not None and current.browser.is_connected() and not current.page.is_closed():
                current.last_used = time.monotonic()
                return current
            if current is not None:
                await self.close(user_id)
            opened = await self._connect(user_id, cdp_url)
            self._browsers[user_id] = opened
            return opened

    async def _connect(self, user_id: str, cdp_url: str) -> UserBrowser:
        if self._playwright is None:
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
        browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
        context = browser.contexts[0] if browser.contexts else await browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        await page.set_viewport_size(VIEWPORT)
        page.set_default_timeout(15_000)
        await _start_screencast(page, user_id)
        logger.info("opened a browser for user %s", user_id)
        return UserBrowser(browser=browser, page=page)

    async def act_for_person(self, user_id: str, action: dict[str, Any]) -> bool:
        """Replay a person's click, typing, key press or scroll on the user's page.

        Returns True when this is the person's first change since the agent last looked,
        which is when the agent should be told to look again.

        Raises:
            LookupError: the user has no open browser.
            ValueError: the action is not one of the four kinds.
        """
        opened = self._browsers.get(user_id)
        if opened is None:
            raise LookupError("no open browser")
        page, kind = opened.page, action.get("kind")
        if kind == "click":
            await page.mouse.click(float(action["x"]), float(action["y"]))
        elif kind == "type":
            await page.keyboard.type(str(action["text"]))
        elif kind == "key":
            await page.keyboard.press(str(action["key"]))
        elif kind == "scroll":
            await page.mouse.wheel(0, float(action["dy"]))
        else:
            raise ValueError(f"unknown action {kind!r}")
        first_change = not opened.changed_by_human
        opened.changed_by_human = True
        opened.last_used = time.monotonic()
        return first_change

    async def close(self, user_id: str) -> None:
        opened = self._browsers.pop(user_id, None)
        if opened is not None:
            with contextlib.suppress(Exception):
                await opened.browser.close()
            logger.info("closed the browser of user %s", user_id)

    async def close_idle(self) -> None:
        """Close every browser nobody has used for `browser.idle_s` seconds."""
        limit = float(_cfg("browser.idle_s", 1800))
        now = time.monotonic()
        for user_id, opened in list(self._browsers.items()):
            if now - opened.last_used > limit:
                await self.close(user_id)

    async def close_all(self) -> None:
        for user_id in list(self._browsers):
            await self.close(user_id)
        if self._playwright is not None:
            with contextlib.suppress(Exception):
                await self._playwright.stop()
            self._playwright = None


async def _start_screencast(page: Any, user_id: str) -> None:
    """Send the page's picture to the app's browser pane for as long as the page is open."""
    cdp = await page.context.new_cdp_session(page)

    def on_frame(params: dict[str, Any]) -> None:
        broker.publish(user_id, params["data"])
        # The browser sends the next frame only after this one is acknowledged.
        asyncio.ensure_future(cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]}))

    cdp.on("Page.screencastFrame", on_frame)
    size = {"maxWidth": VIEWPORT["width"], "maxHeight": VIEWPORT["height"]}
    await cdp.send("Page.startScreencast", {"format": "jpeg", "quality": 60, **size})


_manager: BrowserManager | None = None


def manager() -> BrowserManager:
    """The process-wide manager, so a user has one browser per process."""
    global _manager
    if _manager is None:
        _manager = BrowserManager()
    return _manager
