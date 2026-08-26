"""Web search: ours, native, over SerpAPI.

OURS, not a connector (11.10.2). There is no per-user OAuth and nothing for a
human to connect: the SerpAPI key is an app-level secret in `.env` and every
user's searches share our quota. So it is always in the manifest, counted in
`ours`, and it appears in no toggle and no settings row.

It used to ride the connector wire, because that was where its key was
configured. That was the only reason, and it cost a vendor in the path of a call
that has no user data in it at all — so 11.10.2 brought it home. The tool is
`readonly` and `auto_approve`: a search reads the web and changes nothing, and
gating it would put an approval card in front of the human every time the agent
looked something up.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import urlencode

import aiohttp

from config_module.loader import cfg as _cfg
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok

_ENDPOINT = "https://serpapi.com/search.json"
_TIMEOUT = aiohttp.ClientTimeout(total=30)
# Enough to answer from, short enough that ten of them do not eat the turn.
_MAX_RESULTS = 10
_SNIPPET_CHARS = 300


class WebSearch:
    spec = ToolSpec(
        name="web_search",
        description=(
            "Search the web and get back titles, links and snippets. Use it for anything you "
            "do not already know and cannot read from the project's files: current events, "
            "documentation, error messages, who or what something is. Prefer one specific "
            "query over several vague ones, and read the snippets before deciding to open a "
            "page with the browser."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."},
                "limit": {
                    "type": "integer",
                    "description": f"How many results, 1-{_MAX_RESULTS}. Default 5.",
                },
            },
            "required": ["query"],
        },
        readonly=True,
        requires_approval=False,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        query = (args.get("query") or "").strip()
        if not query:
            return fail("invalid_args", "There is nothing to search for; pass `query`.")

        key = str(_cfg("tools.serpapi_key", "") or os.environ.get("SERPAPI_API_KEY", "")).strip()
        if not key:
            # Not retryable and not the model's problem: a missing app-level
            # secret is an operator fact, and saying so plainly beats a
            # generic upstream error the model will retry into.
            return fail(
                "unavailable",
                "Web search is not configured on this server (no SerpAPI key).",
                retryable=False,
            )

        limit = args.get("limit")
        try:
            count = max(1, min(_MAX_RESULTS, int(limit))) if limit is not None else 5
        except (TypeError, ValueError):
            count = 5

        params = {"engine": "google", "q": query, "num": count, "api_key": key}
        url = f"{_ENDPOINT}?{urlencode(params)}"
        try:
            async with aiohttp.ClientSession(timeout=_TIMEOUT) as session, session.get(url) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    # The key is in the query string, so the url never goes in
                    # a message: SerpAPI echoes the request on some errors.
                    return fail("upstream_error", f"Search failed ({resp.status}).")
        except (aiohttp.ClientError, TimeoutError) as e:
            return fail("upstream_error", f"Search could not reach SerpAPI: {type(e).__name__}")

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return fail("upstream_error", "Search returned something that was not JSON.")

        if payload.get("error"):
            return fail("upstream_error", f"Search failed: {payload['error']}")

        return ok(_render(payload, count) or f"No results for {query!r}.")


def _render(payload: dict[str, Any], count: int) -> str:
    """Flatten SerpAPI's answer to the few fields a model can act on.

    An answer box or knowledge panel is put first when present: it is usually
    the whole answer, and burying it under ten blue links wastes the turn.
    """
    lines: list[str] = []

    box = payload.get("answer_box") or {}
    direct = box.get("answer") or box.get("snippet") or box.get("result")
    if direct:
        lines.append(f"Answer: {str(direct)[:_SNIPPET_CHARS]}")

    panel = payload.get("knowledge_graph") or {}
    if panel.get("description"):
        lines.append(f"{panel.get('title', 'About')}: {panel['description'][:_SNIPPET_CHARS]}")

    for i, hit in enumerate((payload.get("organic_results") or [])[:count], start=1):
        title = str(hit.get("title") or "").strip()
        link = str(hit.get("link") or "").strip()
        snippet = str(hit.get("snippet") or "").strip()[:_SNIPPET_CHARS]
        if not (title or link):
            continue
        lines.append(f"\n{i}. {title}\n   {link}" + (f"\n   {snippet}" if snippet else ""))

    return "\n".join(lines).strip()


TOOLS = [WebSearch()]
