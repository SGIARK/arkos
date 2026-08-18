"""
One shape for every tool result.

`execute` never raises except on cancellation, so a broken tool is model input
rather than a dead run.
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


class ToolUnavailable(Exception):
    """Raised by a tool that cannot run right now, carrying the kind to report.

    `execute` turns it into an envelope. Anything else a tool raises becomes an
    `upstream_error`.
    """

    def __init__(self, error_kind: str, message: str, *, retryable: bool = True):
        self.error_kind = error_kind
        self.message = message
        self.retryable = retryable
        super().__init__(message)


@dataclass(slots=True)
class ResultEnvelope:
    """What every tool returns; on a failure `content` is written for the model to act on."""

    ok: bool
    content: str
    error_kind: ErrorKind = "none"
    retryable: bool = False
    ref: str | None = None


def ok(content: str, *, ref: str | None = None) -> ResultEnvelope:
    return ResultEnvelope(ok=True, content=content, ref=ref)


def fail(error_kind: ErrorKind, content: str, *, retryable: bool | None = None) -> ResultEnvelope:
    """Build a failed envelope; `retryable` overrides the default for the kind."""
    return ResultEnvelope(
        ok=False,
        content=content,
        error_kind=error_kind,
        retryable=error_kind in _RETRYABLE if retryable is None else retryable,
    )


@dataclass(slots=True)
class ToolSpec:
    """One tool's contract; `readonly` drives loop concurrency, so a mutating tool must not claim it."""

    name: str
    description: str = ""
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
    """Everything a tool may need that is not an argument."""

    user_id: str
    session_id: str | None = None
    # `(label)` or `(label, url)`. The url is an ephemeral side-channel to mount
    # — the browser's frame stream — and is never stored for replay.
    emit_status: Callable[..., None] | None = None
    store_blob: Callable[[str], Awaitable[str]] | None = None
    read_blob: Callable[[str, int, int], Awaitable[str | None]] | None = None
    approve: Callable[[str, dict[str, Any]], Awaitable[bool]] | None = None
    # Claims the session's lease on a shared, stateful resource by name
    # ("sandbox", "browser"). Raises ToolUnavailable if the wait times out.
    lease: Callable[[str], Awaitable[None]] | None = None
    # Per-turn state shared between calls, keyed by the tool that owns it.
    # `edit_file` uses it to record which paths have been read.
    scratch: dict[str, Any] = field(default_factory=dict)


class Tool(Protocol):
    """What registry.py discovers; `validate` is optional."""

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
    Run one tool and always return a ResultEnvelope. Never raises except cancellation.

    `name` must already be stripped of any mcp_ prefix by the caller.
    """
    # All of it inside the try: lookup, schema check, approval and validate can
    # each raise, and any escaping breaks the promise above.
    try:
        tool = lookup(name)
        if tool is None:
            return fail("not_found", f"No tool named {name!r}.")

        problem = _check_schema(args, tool.spec.input_schema)
        if problem is not None:
            return fail("invalid_args", f"{problem} Call {name} again with corrected arguments.")

        if tool.spec.requires_approval:
            if ctx.approve is None:
                return fail(
                    "upstream_error",
                    f"{name} needs approval but this session cannot ask for it.",
                    retryable=False,
                )
            if not await _maybe_await(ctx.approve(name, args)):
                return fail(
                    "upstream_error",
                    f"The human declined {name}. Do not retry it; choose another approach.",
                    retryable=False,
                )

        validate = getattr(tool, "validate", None)
        if validate is not None:
            problem = await _maybe_await(validate(args, ctx))
            if problem is not None:
                return fail("invalid_args", f"{problem} Call {name} again with corrected arguments.")

        async with asyncio.timeout(timeout_s):
            result = await tool.call(args, ctx)
    except asyncio.CancelledError:
        raise
    except ToolUnavailable as e:
        return fail(e.error_kind, e.message, retryable=e.retryable)
    except TimeoutError:
        return fail("timeout", f"{name} did not finish within {timeout_s:.0f}s. It may or may not have taken effect.")
    except Exception as e:
        logger.exception("tool %s raised", name)
        return fail("upstream_error", f"{name} failed: {type(e).__name__}: {e}")

    if not isinstance(result, ResultEnvelope):
        logger.error("tool %s returned %s, not a ResultEnvelope", name, type(result).__name__)
        return fail("upstream_error", f"{name} returned a malformed result.", retryable=False)
    return result


async def _maybe_await(value: Any) -> Any:
    """Tolerate a sync callback where an async one is expected."""
    return await value if inspect.isawaitable(value) else value


def _check_schema(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    """Return one actionable sentence about `args`, or None. Required keys and declared types only."""
    if not schema:
        return None

    # Composed schemas (oneOf/anyOf/allOf/$ref) are not ours to interpret.
    if any(k in schema for k in ("oneOf", "anyOf", "allOf", "$ref")):
        return None

    properties = schema.get("properties") or {}
    required = schema.get("required") or []

    missing = [k for k in required if k not in args]
    if missing:
        return f"Missing required argument(s): {', '.join(sorted(missing))}."

    # Third-party schemas often require a key they never declare, so reject
    # unknown keys only when the schema is closed and self-consistent.
    closed = schema.get("additionalProperties") is not True and all(k in properties for k in required)
    if properties and closed:
        unknown = [k for k in args if k not in properties]
        if unknown:
            return f"Unknown argument(s): {', '.join(sorted(unknown))}. Allowed: {', '.join(sorted(properties))}."

    for key, value in args.items():
        expected = (properties.get(key) or {}).get("type")
        if expected and not _type_ok(value, expected):
            return f"Argument {key!r} should be {_describe(expected)}, got {type(value).__name__}."
    return None


def _describe(expected: Any) -> str:
    return " or ".join(expected) if isinstance(expected, list) else str(expected)


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _type_ok(value: Any, expected: Any) -> bool:
    # A union like ["string", "null"] is routine in third-party schemas.
    if isinstance(expected, list):
        return any(_type_ok(value, option) for option in expected)
    if expected == "null":
        return value is None
    python_type = _JSON_TYPES.get(expected)
    if python_type is None:
        return True
    # bool is an int subclass; a bool where a number belongs is a real bug.
    if expected in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, python_type)
