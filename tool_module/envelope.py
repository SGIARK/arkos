"""
One shape for every tool result.

Replaces three inconsistent conventions: state_tool.py raised, ComputerAgent
returned error strings, tools.py:dispatch returned a dict. `execute` never
raises except on cancellation, so a broken tool is model input rather than a
dead run.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

ErrorKind = Literal[
    "none",
    "invalid_args",
    "not_found",
    "auth_required",
    "timeout",
    "upstream_error",
    "interrupted",
]

# Retrying these may work; the rest are the caller's problem to fix.
_RETRYABLE: frozenset[str] = frozenset({"timeout", "upstream_error"})


@dataclass(slots=True)
class ResultEnvelope:
    """
    What every tool returns. `content` on a failure is written FOR the model:
    it is the only thing the model sees, so it has to say what to do next.
    """

    ok: bool
    content: str
    error_kind: ErrorKind = "none"
    retryable: bool = False
    ref: str | None = None


def ok(content: str, *, ref: str | None = None) -> ResultEnvelope:
    return ResultEnvelope(ok=True, content=content, ref=ref)


def fail(error_kind: ErrorKind, content: str) -> ResultEnvelope:
    return ResultEnvelope(
        ok=False,
        content=content,
        error_kind=error_kind,
        retryable=error_kind in _RETRYABLE,
    )


@dataclass(slots=True)
class ToolSpec:
    """
    One tool's contract. `readonly` drives concurrency in the loop, so a tool
    that mutates anything must not claim it.
    """

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    readonly: bool = False
    requires_approval: bool = False

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


@dataclass(slots=True)
class ToolContext:
    """
    Everything a tool may need that is not an argument.

    `emit_status` lets a long tool report progress without knowing what an event
    is. `store_blob` returns a ref for output too large to inline; a tool that
    does not call it gets truncated with no way to page the rest.
    """

    user_id: str
    session_id: str | None = None
    emit_status: Callable[[str], None] | None = None
    store_blob: Callable[[str], Awaitable[str]] | None = None
    read_blob: Callable[[str, int, int], Awaitable[str | None]] | None = None
    approve: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None


class Tool(Protocol):
    """What registry.py discovers. `validate` is optional."""

    spec: ToolSpec

    async def call(self, args: dict[str, Any], ctx: ToolContext) -> ResultEnvelope: ...


async def execute(
    name: str,
    args: dict[str, Any],
    ctx: ToolContext,
    *,
    lookup: Callable[[str], Tool | None],
    timeout_s: float = 120.0,
) -> ResultEnvelope:
    """
    Run one tool. Never raises except cancellation.

    Args:
        name: tool name, already stripped of any mcp_ prefix by the caller.
        args: validated-on-arrival arguments.
        ctx: per-call context.
        lookup: resolves a name to a Tool, or None.
        timeout_s: hard cap; a hung upstream must not hold a hop open.

    Returns:
        A ResultEnvelope, always.
    """
    tool = lookup(name)
    if tool is None:
        return fail("not_found", f"No tool named {name!r}.")

    problem = _check_schema(args, tool.spec.input_schema)
    if problem is not None:
        return fail("invalid_args", f"{problem} Call {name} again with corrected arguments.")

    if tool.spec.requires_approval:
        if ctx.approve is None:
            return fail("upstream_error", f"{name} needs approval but this session cannot ask for it.")
        if not await ctx.approve(name, args):
            return fail("upstream_error", f"The human declined {name}. Do not retry it; choose another approach.")

    validate = getattr(tool, "validate", None)
    if validate is not None:
        result = validate(args, ctx)
        if inspect.isawaitable(result):
            result = await result
        if result is not None:
            return fail("invalid_args", str(result))

    try:
        async with asyncio.timeout(timeout_s):
            return await tool.call(args, ctx)
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        return fail("timeout", f"{name} did not finish within {timeout_s:.0f}s. It may or may not have taken effect.")
    except Exception as e:
        logger.exception("tool %s raised", name)
        return fail("upstream_error", f"{name} failed: {type(e).__name__}: {e}")


def _check_schema(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """
    Required keys and declared types only.

    Deliberately not full JSON Schema: the model needs a short, actionable
    sentence, and a validator's error text is neither.
    """
    if not schema:
        return None

    properties = schema.get("properties") or {}
    missing = [k for k in schema.get("required") or [] if k not in args]
    if missing:
        return f"Missing required argument(s): {', '.join(sorted(missing))}."

    unknown = [k for k in args if k not in properties]
    if properties and unknown:
        return f"Unknown argument(s): {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(properties))}."

    for key, value in args.items():
        expected = (properties.get(key) or {}).get("type")
        if expected and not _type_ok(value, expected):
            return f"Argument {key!r} should be {expected}, got {type(value).__name__}."
    return None


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _type_ok(value: Any, expected: str) -> bool:
    python_type = _JSON_TYPES.get(expected)
    if python_type is None:
        return True
    # bool is an int subclass, and a bool where a number belongs is a real bug.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, python_type)
