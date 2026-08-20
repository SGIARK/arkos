"""The memory tools: what the agent carries between sessions.

Four hands over the store's memory region — save a fact, search for one, read
the curated core, rewrite it. The core is also injected into the system prompt
at fold time, so `read_memory` is for the full document when the injected copy
was capped, and for reading before a rewrite.

Every call is scoped to `ctx.user_id` in the SQL itself: memory is the user's,
and no argument here can widen that.
"""

from __future__ import annotations

import json
from typing import Any

from config_module.loader import cfg as _cfg
from harness_module import store
from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, ToolUnavailable, fail, ok

# How much of one hit is shown. A note is short; the core can be long, and a
# search result is a pointer to it, not a replacement for reading it.
_SNIPPET_CHARS = 600




def _user(ctx: ToolContext) -> str:
    """The user whose memory this is.

    Raises:
        ToolUnavailable: there is no user to key memory by, which is not a
            condition the model can do anything about.
    """
    if not ctx.user_id:
        raise ToolUnavailable("invalid_args", "Memory is per user, and this call has no user.", retryable=False)
    return ctx.user_id


def _read_core(ctx: ToolContext) -> dict[str, Any]:
    """Per-turn state: what `read_memory` last returned, which `update_memory` requires."""
    return ctx.scratch.setdefault("memory_read", {})


class SaveMemory:
    spec = ToolSpec(
        name="save_memory",
        description=(
            "Record one durable fact in your long-term memory, to be read in later sessions. "
            "Use it for what stays true: how the user wants things done, decisions and the "
            "reasoning behind them, who people are, where things live, standing constraints. "
            "Not for this session's narration, not for anything you can read out of the "
            "project's files, and never for a credential. One fact per call, written as a "
            "full sentence that will still make sense months from now with no other context."
        ),
        input_schema={
            "type": "object",
            "properties": {"text": {"type": "string", "description": "The fact, in one or two sentences."}},
            "required": ["text"],
        },
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        text = (args.get("text") or "").strip()
        if not text:
            return fail("invalid_args", "There is nothing to save; pass the fact as `text`.")

        limit = int(_cfg("memory.note_max_chars", 2000))
        if len(text) > limit:
            return fail(
                "invalid_args",
                f"A note is at most {limit} characters and this is {len(text)}. Save the fact, not the transcript.",
            )

        path = await store.append_note(_user(ctx), text)
        return ok(f"Saved to memory ({path}).")


class SearchMemory:
    spec = ToolSpec(
        name="search_memory",
        description=(
            "Search your long-term memory, across the curated core and every note you have "
            "saved. Full-text search: bare words, \"quoted phrases\", or. Reach for it when a "
            "request touches something the user may have told you before — a preference, a "
            "past decision, a name — rather than assuming you have never heard of it."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "description": "Default 10."},
            },
            "required": ["query"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        query = (args.get("query") or "").strip()
        if not query:
            return fail("invalid_args", "Pass what to search for as `query`.")

        try:
            limit = max(1, min(int(args.get("limit") or _cfg("memory.search_limit", 10)), 50))
        except (TypeError, ValueError):
            limit = int(_cfg("memory.search_limit", 10))

        hits = await store.search_memory(_user(ctx), query, limit)
        if not hits:
            return ok(f"Nothing in memory matches {query!r}.")

        return ok(
            json.dumps(
                [
                    {
                        "path": hit.path,
                        "is_core": hit.is_core,
                        "written_at": hit.written_at.isoformat(),
                        "text": hit.text[:_SNIPPET_CHARS],
                        "truncated": len(hit.text) > _SNIPPET_CHARS,
                    }
                    for hit in hits
                ],
                indent=2,
            )
        )


class ReadMemory:
    spec = ToolSpec(
        name="read_memory",
        description=(
            "Read your curated memory document (MEMORY.md) in full. A capped copy is already "
            "in your system prompt, so read it when you need the part that was cut, or before "
            "calling update_memory — which requires it, because a rewrite from the capped copy "
            "would silently throw the rest away."
        ),
        input_schema={"type": "object", "properties": {}},
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        core = await store.read_memory(_user(ctx))
        _read_core(ctx)["text"] = core
        if not core:
            return ok("Your memory document is empty. Notes you have saved are still searchable.")
        return ok(core)


class UpdateMemory:
    spec = ToolSpec(
        name="update_memory",
        description=(
            "Rewrite your curated memory document (MEMORY.md) whole. This is curation, not "
            "appending: fold what you have learned into the document, drop what has gone "
            "stale, and keep it short enough to be worth reading every session. Call "
            "read_memory first — you are replacing the entire document, so send the whole "
            "new text, not the part you changed. Use save_memory for a single new fact."
        ),
        input_schema={
            "type": "object",
            "properties": {"content": {"type": "string", "description": "The complete new document."}},
            "required": ["content"],
        },
    )

    async def validate(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        """Read before rewrite, for the same reason `edit_file` has it.

        The system prompt carries a capped copy of the core, so a model that has
        not read the document this turn may be holding a truncated one, and a
        whole-document write from that would delete the tail.
        """
        if "text" not in _read_core(ctx):
            return "Call read_memory before update_memory, so you rewrite the whole document and not the capped copy."
        return None

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        content = args.get("content")
        if content is None:
            return fail("invalid_args", "Pass the complete new document as `content`.")

        limit = int(_cfg("memory.core_max_chars", 20000))
        if len(content) > limit:
            return fail(
                "invalid_args",
                f"The memory document is at most {limit} characters and this is {len(content)}. "
                "Curate it down: what earns its place is what you would want to know cold.",
            )

        await store.update_memory(_user(ctx), content)
        # It has been rewritten, so what was read before is no longer the document.
        _read_core(ctx)["text"] = content
        return ok(f"Memory document rewritten ({len(content)} chars).")


TOOLS = [SaveMemory(), SearchMemory(), ReadMemory(), UpdateMemory()]
