"""
Custom `browser_use.Tools` actions, each collapsing a multi-step LLM flow into
one deterministic call. browser_use is imported lazily so arkos boots without it;
`build_arkos_tools()` then returns None. Actions reuse the agent's CDP session,
since a second connection would land on a different browser instance.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# Scores every visible clickable by how well its text matches a consent verb
# and clicks the best one above threshold.
_DISMISS_OVERLAY_JS = r"""
(() => {
  const TERMS = [
    // English
    'accept all', 'accept cookies', 'accept', 'agree', 'i agree',
    'allow all', 'allow', 'got it', 'ok', 'continue', 'consent',
    'dismiss', 'close', 'reject all', 'decline', 'no thanks',
    // Common non-English
    'aceptar', 'aceptar todo', 'acepto',
    'einverstanden', 'akzeptieren', 'zustimmen',
    'accepter', 'accepter tout', "j'accepte",
    'concordo', 'aceitar',
  ];
  const visible = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && parseFloat(s.opacity) > 0.05;
  };
  const text = (el) => (el.innerText || el.textContent || el.getAttribute('aria-label') || el.value || '')
    .trim().toLowerCase().replace(/\s+/g, ' ');
  const score = (t) => {
    if (!t || t.length > 60) return 0;
    for (let i = 0; i < TERMS.length; i++) {
      const term = TERMS[i];
      if (t === term) return 100 - i;
      if (t.startsWith(term) || t.endsWith(term)) return 80 - i;
      if (t.includes(term)) return 50 - i;
    }
    return 0;
  };
  const sel = 'button, a, [role="button"], input[type="button"], input[type="submit"]';
  const elements = Array.from(document.querySelectorAll(sel)).filter(visible);
  let best = null;
  let bestScore = 0;
  for (const el of elements) {
    const s = score(text(el));
    if (s > bestScore) { best = el; bestScore = s; }
  }
  if (best && bestScore >= 50) {
    const label = text(best);
    best.click();
    return { clicked: true, label, score: bestScore, considered: elements.length };
  }
  return { clicked: false, considered: elements.length };
})()
"""


def _make_dismiss_overlay_action(tools_module: Any, action_result_cls: Any):
    """Closure factory so the decorator captures the real Tools instance."""

    @tools_module.action(
        description=(
            "Dismiss any cookie/consent/age-gate banner currently covering "
            "the page. Idempotent: a no-op if no banner is detected. Run "
            "this once on first load and again only if a new overlay "
            "appears mid-task."
        )
    )
    async def dismiss_overlay(browser_session: Any) -> Any:
        try:
            cdp_session = await browser_session.get_or_create_cdp_session(target_id=None, focus=False)
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={
                    "expression": _DISMISS_OVERLAY_JS,
                    "returnByValue": True,
                    "awaitPromise": False,
                },
                session_id=cdp_session.session_id,
            )
        except Exception as e:
            logger.info("dismiss_overlay: CDP evaluate failed: %s", e)
            return action_result_cls(
                extracted_content="dismiss_overlay: no overlay dismissed (page not ready)",
                include_in_memory=False,
            )

        value = ((result or {}).get("result") or {}).get("value") or {}
        if value.get("clicked"):
            return action_result_cls(
                extracted_content=(
                    f"Dismissed overlay via element labelled {value.get('label')!r} "
                    f"(score={value.get('score')}, considered={value.get('considered')})"
                ),
                include_in_memory=True,
            )
        return action_result_cls(
            extracted_content=(
                f"No consent/cookie overlay detected on this page "
                f"(considered {value.get('considered', 0)} clickable elements)."
            ),
            include_in_memory=False,
        )

    return dismiss_overlay


# --- wait_for_element -----------------------------------------------------

_WAIT_FOR_ELEMENT_JS = r"""
(async ({ selector, timeoutMs }) => {
  const visible = (el) => {
    if (!el) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) return false;
    const s = getComputedStyle(el);
    return s.visibility !== 'hidden' && s.display !== 'none' && parseFloat(s.opacity) > 0.05;
  };
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    const el = document.querySelector(selector);
    if (el && visible(el)) {
      const r = el.getBoundingClientRect();
      return { found: true, waited_ms: Math.round(performance.now() - start),
               box: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) } };
    }
    await new Promise(r => setTimeout(r, 100));
  }
  return { found: false, waited_ms: Math.round(performance.now() - start) };
})({ selector: __SEL__, timeoutMs: __TIMEOUT__ })
"""


def _make_wait_for_element_action(tools_module, action_result_cls):
    from pydantic import BaseModel, Field

    class WaitParams(BaseModel):
        selector: str = Field(..., description="CSS selector to wait for")
        timeout_ms: int = Field(5000, description="Max wait in milliseconds (default 5000)")

    @tools_module.action(
        description=(
            "Wait until a CSS selector renders AND is visible on the current "
            "page, up to a timeout. Use this on SPAs or any page where "
            "content streams in after navigation. Returns success and "
            "bounding box, or failure with how long it waited."
        ),
        param_model=WaitParams,
    )
    async def wait_for_element(params: WaitParams, browser_session: Any) -> Any:
        try:
            cdp_session = await browser_session.get_or_create_cdp_session(target_id=None, focus=False)
            expr = _WAIT_FOR_ELEMENT_JS.replace("__SEL__", _json_dump(params.selector)).replace(
                "__TIMEOUT__", str(int(params.timeout_ms))
            )
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True, "awaitPromise": True},
                session_id=cdp_session.session_id,
            )
        except Exception as e:
            logger.info("wait_for_element: CDP evaluate failed: %s", e)
            return action_result_cls(
                extracted_content=f"wait_for_element: CDP error ({e})",
                include_in_memory=False,
            )
        value = ((result or {}).get("result") or {}).get("value") or {}
        if value.get("found"):
            box = value.get("box", {})
            return action_result_cls(
                extracted_content=(
                    f"Selector {params.selector!r} visible after {value.get('waited_ms')}ms "
                    f"(box {box.get('w')}x{box.get('h')} at {box.get('x')},{box.get('y')})"
                ),
                include_in_memory=True,
            )
        return action_result_cls(
            extracted_content=(f"Selector {params.selector!r} not visible within {params.timeout_ms}ms"),
            include_in_memory=False,
        )

    return wait_for_element


# --- click_then_wait_for_url_change --------------------------------------

_CLICK_THEN_WAIT_JS = r"""
(async ({ selector, timeoutMs }) => {
  const startUrl = location.href;
  const el = document.querySelector(selector);
  if (!el) return { clicked: false, reason: 'selector_not_found' };
  el.click();
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    if (location.href !== startUrl) {
      return { clicked: true, navigated: true,
               from_url: startUrl, to_url: location.href,
               waited_ms: Math.round(performance.now() - start) };
    }
    await new Promise(r => setTimeout(r, 100));
  }
  return { clicked: true, navigated: false, from_url: startUrl, to_url: location.href,
           waited_ms: Math.round(performance.now() - start) };
})({ selector: __SEL__, timeoutMs: __TIMEOUT__ })
"""


def _make_click_then_wait_action(tools_module, action_result_cls):
    from pydantic import BaseModel, Field

    class ClickWaitParams(BaseModel):
        selector: str = Field(..., description="CSS selector of the element to click")
        timeout_ms: int = Field(8000, description="Max wait for navigation after click")

    @tools_module.action(
        description=(
            "Click a CSS selector and wait for the page's URL to change. "
            "Use this for submit buttons, 'next page' links, and login "
            "buttons, anywhere the next step depends on a navigation "
            "having actually happened. Returns whether navigation occurred."
        ),
        param_model=ClickWaitParams,
    )
    async def click_then_wait_for_url_change(params: ClickWaitParams, browser_session: Any) -> Any:
        try:
            cdp_session = await browser_session.get_or_create_cdp_session(target_id=None, focus=False)
            expr = _CLICK_THEN_WAIT_JS.replace("__SEL__", _json_dump(params.selector)).replace(
                "__TIMEOUT__", str(int(params.timeout_ms))
            )
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True, "awaitPromise": True},
                session_id=cdp_session.session_id,
            )
        except Exception as e:
            return action_result_cls(
                extracted_content=f"click_then_wait_for_url_change: CDP error ({e})",
                include_in_memory=False,
            )
        value = ((result or {}).get("result") or {}).get("value") or {}
        if not value.get("clicked"):
            return action_result_cls(
                extracted_content=f"Element {params.selector!r} not found",
                include_in_memory=False,
            )
        if value.get("navigated"):
            return action_result_cls(
                extracted_content=(
                    f"Clicked {params.selector!r}; URL changed from "
                    f"{value.get('from_url')} to {value.get('to_url')} in "
                    f"{value.get('waited_ms')}ms"
                ),
                include_in_memory=True,
            )
        return action_result_cls(
            extracted_content=(
                f"Clicked {params.selector!r} but URL did not change within "
                f"{params.timeout_ms}ms, still at {value.get('from_url')}"
            ),
            include_in_memory=True,
        )

    return click_then_wait_for_url_change


# --- scroll_to_load_all ---------------------------------------------------

_SCROLL_TO_LOAD_JS = r"""
(async ({ maxScrolls, settleMs }) => {
  let prevHeight = -1;
  let scrolls = 0;
  for (let i = 0; i < maxScrolls; i++) {
    window.scrollTo(0, document.documentElement.scrollHeight);
    await new Promise(r => setTimeout(r, settleMs));
    const h = document.documentElement.scrollHeight;
    scrolls++;
    if (h === prevHeight) break;
    prevHeight = h;
  }
  return { scrolls, final_height: prevHeight };
})({ maxScrolls: __MAX__, settleMs: __SETTLE__ })
"""


def _make_scroll_to_load_all_action(tools_module, action_result_cls):
    from pydantic import BaseModel, Field

    class ScrollParams(BaseModel):
        max_scrolls: int = Field(20, description="Hard cap on scroll iterations")
        settle_ms: int = Field(700, description="Pause between scrolls to let content load")

    @tools_module.action(
        description=(
            "Scroll the page to the bottom repeatedly until the document "
            "height stops growing (or max_scrolls is reached). Use for "
            "infinite-scroll feeds, 'load more' lists, and any vertical "
            "stream where you want the whole list materialised before "
            "extracting from it. Idempotent."
        ),
        param_model=ScrollParams,
    )
    async def scroll_to_load_all(params: ScrollParams, browser_session: Any) -> Any:
        try:
            cdp_session = await browser_session.get_or_create_cdp_session(target_id=None, focus=False)
            expr = _SCROLL_TO_LOAD_JS.replace("__MAX__", str(int(params.max_scrolls))).replace(
                "__SETTLE__", str(int(params.settle_ms))
            )
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True, "awaitPromise": True},
                session_id=cdp_session.session_id,
            )
        except Exception as e:
            return action_result_cls(
                extracted_content=f"scroll_to_load_all: CDP error ({e})",
                include_in_memory=False,
            )
        value = ((result or {}).get("result") or {}).get("value") or {}
        return action_result_cls(
            extracted_content=(
                f"Scrolled {value.get('scrolls', 0)} times; final document height {value.get('final_height', 0)}px"
            ),
            include_in_memory=True,
        )

    return scroll_to_load_all


# --- extract_text_region --------------------------------------------------

_EXTRACT_TEXT_JS = r"""
(({ selector, maxChars }) => {
  const el = document.querySelector(selector);
  if (!el) return { found: false };
  // Walk text nodes, skip script/style, collapse whitespace.
  const SKIP = new Set(['SCRIPT', 'STYLE', 'NOSCRIPT', 'IFRAME']);
  const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      let p = n.parentElement;
      while (p) {
        if (SKIP.has(p.tagName)) return NodeFilter.FILTER_REJECT;
        p = p.parentElement;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });
  const chunks = [];
  let n;
  let total = 0;
  while ((n = walker.nextNode())) {
    const t = (n.nodeValue || '').replace(/\s+/g, ' ').trim();
    if (!t) continue;
    chunks.push(t);
    total += t.length + 1;
    if (total >= maxChars) break;
  }
  return { found: true, text: chunks.join(' ').slice(0, maxChars) };
})({ selector: __SEL__, maxChars: __MAX__ })
"""


def _make_extract_text_region_action(tools_module, action_result_cls):
    from pydantic import BaseModel, Field

    class ExtractParams(BaseModel):
        selector: str = Field("body", description="CSS selector of the region to extract from")
        max_chars: int = Field(8000, description="Hard cap on returned text length")

    @tools_module.action(
        description=(
            "Extract clean plain text from a CSS-selected region of the "
            "current page (scripts/styles/iframes stripped, whitespace "
            "collapsed). Use this for 'summarize the article', 'get the "
            "comments', 'read the description' tasks. Default selector "
            "is 'body' for whole-page text."
        ),
        param_model=ExtractParams,
    )
    async def extract_text_region(params: ExtractParams, browser_session: Any) -> Any:
        try:
            cdp_session = await browser_session.get_or_create_cdp_session(target_id=None, focus=False)
            expr = _EXTRACT_TEXT_JS.replace("__SEL__", _json_dump(params.selector)).replace(
                "__MAX__", str(int(params.max_chars))
            )
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={"expression": expr, "returnByValue": True, "awaitPromise": False},
                session_id=cdp_session.session_id,
            )
        except Exception as e:
            return action_result_cls(
                extracted_content=f"extract_text_region: CDP error ({e})",
                include_in_memory=False,
            )
        value = ((result or {}).get("result") or {}).get("value") or {}
        if not value.get("found"):
            return action_result_cls(
                extracted_content=f"No element matched {params.selector!r}",
                include_in_memory=False,
            )
        text = value.get("text") or ""
        return action_result_cls(extracted_content=text, include_in_memory=True)

    return extract_text_region


def _json_dump(s: str) -> str:
    """JSON-escape a string for inline injection into a JS expression."""
    import json

    return json.dumps(s)


def build_arkos_tools() -> Any | None:
    """Build a `browser_use.Tools` carrying arkos's actions, or None if browser_use is missing."""
    try:
        import browser_use as bu
    except ImportError:
        return None

    tools_cls = getattr(bu, "Tools", None) or getattr(bu, "Controller", None)
    action_result_cls = getattr(bu, "ActionResult", None)
    if tools_cls is None or action_result_cls is None:
        logger.info("browser_actions: Tools/ActionResult not exported by browser_use; skipping registration")
        return None

    try:
        tools = tools_cls()
    except Exception:
        logger.exception("browser_actions: failed to construct Tools()")
        return None

    factories = (
        _make_dismiss_overlay_action,
        _make_wait_for_element_action,
        _make_click_then_wait_action,
        _make_scroll_to_load_all_action,
        _make_extract_text_region_action,
    )
    for factory in factories:
        try:
            factory(tools, action_result_cls)
        except Exception:
            logger.exception("browser_actions: failed to register action via %s", factory.__name__)

    return tools
