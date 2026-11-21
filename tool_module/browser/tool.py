"""The browser tools: the agent drives the user's own browser directly, one step at a time.

The browser stays open between turns (see `manager.py`), so a page the agent left, or a
person changed while the agent was paused, is still there for the next look.
"""

from __future__ import annotations

import os
from typing import Any

from config_module.loader import cfg as _cfg
from tool_module.browser.manager import manager
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, ToolUnavailable, fail, ok

# Numbers every visible thing a person could click or type into, so the agent can name it.
_LABEL_ELEMENTS = """() => {
  const selector = 'a[href], button, input, select, textarea, [role=button], [role=link], [contenteditable=true]';
  document.querySelectorAll('[data-arkos-ref]').forEach((el) => el.removeAttribute('data-arkos-ref'));
  const lines = [];
  for (const el of document.querySelectorAll(selector)) {
    const box = el.getBoundingClientRect();
    if (!box.width || !box.height) continue;
    const ref = lines.length + 1;
    el.setAttribute('data-arkos-ref', ref);
    const name = (el.getAttribute('aria-label') || el.innerText || el.placeholder || el.name || '').trim();
    const value = 'value' in el && el.value ? ` = "${el.value}"` : '';
    lines.push(`[${ref}] ${el.tagName.toLowerCase()} ${name.replace(/\\s+/g, ' ').slice(0, 80)}${value}`);
  }
  return lines;
}"""


def cdp_url() -> str:
    """Where the browser is: the Browserless container, never this process.

    An unset URL refuses rather than falling back to a local Chromium beside the harness's
    session cookies, store key and model context.
    """
    return str(_cfg("browser.cdp_url", "") or os.environ.get("BROWSERLESS_URL", "")).strip()


async def _page(ctx: ToolContext) -> Any:
    """The user's open page, holding the session's lease on the browser."""
    if ctx.session_id is None:
        raise ToolUnavailable("invalid_args", "The browser is only available inside a session.", retryable=False)
    where = cdp_url()
    if not where:
        raise ToolUnavailable(
            "upstream_error",
            "The browser is not configured: set browser.cdp_url or BROWSERLESS_URL.",
            retryable=False,
        )
    if ctx.lease is not None:
        await ctx.lease("browser")
    return (await manager().open(ctx.user_id, where)).page


async def _snapshot(ctx: ToolContext, page: Any) -> ResultEnvelope:
    """What the page shows now, as text: the model reads the page, it does not see it."""
    opened = manager().get(ctx.user_id)
    if opened is not None:
        opened.changed_by_human = False
    elements = await page.evaluate(_LABEL_ELEMENTS)
    content = await page.locator("body").aria_snapshot()
    return ok(
        f"URL: {page.url}\nTitle: {await page.title()}\n\n{content}\n\n"
        "Things you can click or type into:\n" + ("\n".join(elements) or "(none)")
    )


def _element(page: Any, ref: Any) -> Any:
    return page.locator(f'[data-arkos-ref="{int(ref)}"]')


async def _missing(page: Any, ref: Any) -> ResultEnvelope | None:
    """A refusal when the numbered element is gone, which happens after the page changes."""
    if await _element(page, ref).count() == 0:
        return fail("invalid_args", f"There is no element [{ref}] now. Take a browser_snapshot for fresh numbers.")
    return None


def _spec(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> ToolSpec:
    needed = list(properties) if required is None else required
    schema = {"type": "object", "properties": properties, "required": needed}
    return ToolSpec(name=name, description=description, input_schema=schema)


_REF = {"type": "integer", "description": "The element's number from the latest browser snapshot."}


class BrowserOpen:
    spec = _spec(
        "browser_open",
        "Go to a web address in your browser, then see the page. The browser keeps its tabs, logins "
        "and form contents between turns.",
        {"url": {"type": "string"}},
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        page = await _page(ctx)
        await page.goto(args["url"], wait_until="domcontentloaded")
        return await _snapshot(ctx, page)


class BrowserSnapshot:
    spec = _spec(
        "browser_snapshot",
        "See the page your browser is on now: its text, and a numbered list of what you can click or "
        "type into. Take one before acting on a page you saw earlier, because the person may have changed it.",
        {},
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        return await _snapshot(ctx, await _page(ctx))


class BrowserClick:
    spec = _spec("browser_click", "Click a numbered element, then see the page.", {"ref": _REF})

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        page = await _page(ctx)
        if refusal := await _missing(page, args["ref"]):
            return refusal
        await _element(page, args["ref"]).click()
        return await _snapshot(ctx, page)


class BrowserType:
    spec = _spec(
        "browser_type",
        "Replace the text in a numbered field, optionally pressing Enter after, then see the page.",
        {"ref": _REF, "text": {"type": "string"}, "submit": {"type": "boolean"}},
        required=["ref", "text"],
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        page = await _page(ctx)
        if refusal := await _missing(page, args["ref"]):
            return refusal
        await _element(page, args["ref"]).fill(args["text"])
        if args.get("submit"):
            await _element(page, args["ref"]).press("Enter")
        return await _snapshot(ctx, page)


class BrowserPress:
    spec = _spec(
        "browser_press",
        "Press a key, such as Enter, Escape, Tab or PageDown, then see the page.",
        {"key": {"type": "string"}},
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        page = await _page(ctx)
        await page.keyboard.press(args["key"])
        return await _snapshot(ctx, page)


class BrowserBack:
    spec = _spec("browser_back", "Go back one page, then see it.", {})

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        page = await _page(ctx)
        await page.go_back()
        return await _snapshot(ctx, page)


TOOLS = [BrowserOpen(), BrowserSnapshot(), BrowserClick(), BrowserType(), BrowserPress(), BrowserBack()]
