"""The control tools: the ones that act on the run itself, not on the world."""

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
        # Closes the call so the session can park. The human's answer arrives
        # later as a user message, not as this call's result.
        return ok(f"Asked: {args['question']}\nThe run is paused until a human answers.")


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
        return ok(f"Approval requested: {args['action']}\nThe run is paused until a human answers.")


class ProposePlan:
    """The one door into an unattended run.

    Two things funnel through this tool: the play button (which hands the model
    a `user{source: system}` instruction to draft one) and the model's own
    judgement that the transcript already specs the work. There is no third way
    to start unattended, because deciding is never the model's: it proposes, a
    human approves, and the approval is what flips the mode.

    The args ARE the plan. They are stored on the approvals row, and on approve
    the harness writes them to `plan.md` — so what the human read is what the
    run starts from, the same binding `call` rows give a gated tool call.
    """

    spec = ToolSpec(
        name="propose_plan",
        description=(
            "Propose a plan for an unattended run, and park until a human answers it. Call this "
            "whenever the work is big enough to run on its own, or the moment you are asked to "
            "draft one. The arguments are the plan the human reads and approves; approving writes "
            "them to plan.md and starts the run, so write them for them, not for you. "
            "An under-informed plan is still a plan: never refuse to call this because you do not "
            "know enough, and never ask your questions in prose instead. Put every open question "
            "in `missing` and leave the other fields honest — the card is the form they answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "One sentence: what this run is for."},
                "done_when": {
                    "type": "string",
                    "description": "How the human will know it finished. Observable, not aspirational.",
                },
                "steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The outline, in order. One line each.",
                },
                "inputs": {
                    "type": "array",
                    "description": "What you already have. Each item: {label, note}.",
                },
                "missing": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "The open questions this plan still needs answered, as questions. Empty "
                        "only when you genuinely have everything."
                    ),
                },
            },
            "required": ["goal", "done_when", "steps"],
        },
    )

    def validate(self, args: dict[str, Any], ctx: ToolContext) -> str | None:
        if not str(args.get("goal") or "").strip():
            return "A plan needs a goal: one sentence saying what this run is for."
        if not str(args.get("done_when") or "").strip():
            return "A plan needs done_when: how the human will know it finished."
        steps = args.get("steps")
        if not isinstance(steps, list) or not steps:
            return "steps must be a non-empty list of strings, in order."
        if any(not isinstance(s, str) or not s.strip() for s in steps):
            return "Every step must be a non-empty string."
        for field in ("missing",):
            value = args.get(field)
            if value is not None and (
                not isinstance(value, list) or any(not isinstance(v, str) for v in value)
            ):
                return f"{field} must be a list of strings."
        inputs = args.get("inputs")
        if inputs is not None:
            if not isinstance(inputs, list):
                return "inputs must be a list of {label, note} objects."
            for i, item in enumerate(inputs):
                if not isinstance(item, dict) or not str(item.get("label") or "").strip():
                    return f"inputs[{i}] needs a 'label'."
        return None

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope:
        # Closes the call so the session can park on it. The human's decision
        # arrives through /approvals/{id}/respond, never as this call's result.
        missing = [m for m in (args.get("missing") or []) if str(m).strip()]
        tail = (
            f" It names {len(missing)} open question(s), which the human answers on the card."
            if missing
            else ""
        )
        return ok(
            f"Plan proposed: {args['goal']}\n"
            f"The run is paused until a human approves it, asks for a change, or dismisses it.{tail}"
        )


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


# The runner parks the session after one of these returns, taking the park kind
# from this map. Each call is closed by its own result first: a transcript with
# an open tool_call cannot be folded back into messages.
PARK_KINDS: dict[str, str] = {
    Ask.spec.name: "ask",
    RequestApproval.spec.name: "approval",
    ProposePlan.spec.name: "plan",
}
PARK_TOOLS = frozenset(PARK_KINDS)

TOOLS = [FinishTask(), Ask(), RequestApproval(), ProposePlan(), TodoWrite(), ReadResult()]
