"""The control tools: the ones with authority over the loop rather than the world."""

from __future__ import annotations

from typing import Any

from tool_module.envelope import ResultEnvelope, ToolContext, ToolSpec, fail, ok


class FinishTask:
    spec = ToolSpec(
        name="finish_task",
        description=(
            "Declare the task complete. Call this ONLY when the goal is actually met and "
            "you have verified it, not when you intend to finish. An unattended run cannot "
            "end any other way: text alone is not an exit. If you cannot finish, call this "
            "with a summary that says plainly what blocked you."
        ),
        input_schema={
            "type": "object",
            "properties": {"summary": {"type": "string", "description": "What was done, and what was verified."}},
            "required": ["summary"],
        },
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        return ok(args["summary"])


class Ask:
    spec = ToolSpec(
        name="ask",
        description=(
            "Ask the human a question and wait for the answer. Use when you are genuinely "
            "blocked on something only they know. The run parks until they reply, so do not "
            "use it for anything you could determine yourself."
        ),
        input_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        return ok(args["question"])


class RequestApproval:
    spec = ToolSpec(
        name="request_approval",
        description=(
            "Ask the human to approve an action before you take it. Use for anything "
            "irreversible or outward-facing: sending a message, spending money, deleting "
            "data. Describe exactly what you are about to do."
        ),
        input_schema={
            "type": "object",
            "properties": {"action": {"type": "string"}, "detail": {"type": "string"}},
            "required": ["action"],
        },
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        return ok(args["action"])


class TodoWrite:
    spec = ToolSpec(
        name="todo_write",
        description=(
            "Replace your todo list. Send the whole list every time; it is latest-wins, not "
            "a patch. Keep exactly one item in_progress."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Each item: {text, status: pending|in_progress|done}.",
                }
            },
            "required": ["items"],
        },
    )

    _STATUSES = {"pending", "in_progress", "done"}

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        items = args.get("items") or []
        for i, item in enumerate(items):
            if not isinstance(item, dict) or "text" not in item:
                return f"Item {i} needs a 'text' field."
            status = item.get("status", "pending")
            if status not in self._STATUSES:
                return f"Item {i} has status {status!r}; use pending, in_progress or done."
        if sum(1 for i in items if i.get("status") == "in_progress") > 1:
            return "Only one item may be in_progress."
        return None

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        items = args["items"]
        done = sum(1 for i in items if i.get("status") == "done")
        return ok(f"Todo list updated: {done}/{len(items)} done.")


class ReadResult:
    spec = ToolSpec(
        name="read_result",
        description=(
            "Page the full text of an earlier tool result that was too large to show inline. "
            "Pass the ref from that result."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "ref": {"type": "string"},
                "offset": {"type": "integer"},
                "limit": {"type": "integer"},
            },
            "required": ["ref"],
        },
        readonly=True,
    )

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        if ctx.read_blob is None:
            return fail("upstream_error", "Stored results are not available in this session.", retryable=False)
        text = await ctx.read_blob(args["ref"], args.get("offset", 0), args.get("limit", 2000))
        if text is None:
            return fail("not_found", f"No stored result for ref {args['ref']!r}.")
        return ok(text)


# The runner parks on these instead of continuing. Nothing parks on them yet.
PARK_TOOLS = frozenset({Ask.spec.name, RequestApproval.spec.name})

TOOLS = [FinishTask(), Ask(), RequestApproval(), TodoWrite(), ReadResult()]
