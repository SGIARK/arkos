"""
The model client. One client, one retry layer (D25, contracts `model_module`).

Replaces `ArkModelNew.py:ArkModelLink` and `computer_module/model.py:ToolCallingModel`,
which between them built a fresh `AsyncOpenAI` per property access, retried at two
nested levels, and had no streaming path that carried tool calls.

Public surface is one function:

    async for delta in generate(messages, tools=..., source=..., options=...): ...

`generate` raises `ModelError` and nothing else. `asyncio.CancelledError` passes
through untouched — cancellation is not a model failure.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)

from config_module.loader import config
from model_module.errors import ModelError

logger = logging.getLogger(__name__)

# Where the call came from. `background` is an unattended run with no human
# watching: it fails fast on overload rather than holding a GPU slot that an
# interactive turn could use (contracts: "background source: no overload retries").
Source = Literal["interactive", "background"]


# --- deltas -----------------------------------------------------------------
# The wire is one interleaved stream, so this is one union, not three iterators.


@dataclass(slots=True)
class TextDelta:
    """A fragment of the model's reply, for `content` events."""

    text: str


@dataclass(slots=True)
class ReasoningDelta:
    """
    A fragment of SGLang's `reasoning_content` (--reasoning-parser qwen3).

    Streamed, never dropped: Qwen3 reasons for hundreds of tokens before
    answering, so dropping it freezes the screen for the longest part of a turn.
    The FOLD is what discards it — it never replays into the message list.
    """

    text: str


@dataclass(slots=True)
class ToolCallDelta:
    """
    A fragment of one tool call. Keyed by `index`, because the model may
    interleave fragments of several calls and only `index` is present on every
    chunk — `id` and `name` arrive once, on the opening fragment.
    """

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(slots=True)
class Finish:
    """End of one completion. `usage` is None if the server did not report it."""

    reason: str | None
    usage: dict[str, int] | None = field(default=None)


Delta = TextDelta | ReasoningDelta | ToolCallDelta | Finish


# --- client -----------------------------------------------------------------

_client: AsyncOpenAI | None = None
_client_key: tuple[str, str] | None = None


def get_client() -> AsyncOpenAI:
    """
    The cached client. One per (base_url, api_key), built once.

    `max_retries=0` is deliberate and load-bearing: the SDK's own retry layer is
    the innermost of the three nested loops this redesign deletes. Retrying is
    `_attempts()`'s job, and only its job.
    """
    global _client, _client_key

    base_url = str(config.get("llm.base_url") or "")
    api_key = str(config.get("llm.api_key") or "-")
    key = (base_url, api_key)

    if _client is None or _client_key != key:
        _client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=float(config.get("llm.timeout_s") or 90),
            max_retries=0,
        )
        _client_key = key
    return _client


def reset_client() -> None:
    """Drop the cached client. For tests and for config reloads."""
    global _client, _client_key
    _client = None
    _client_key = None


# --- error classification ---------------------------------------------------

# Retrying these is pointless: the request itself is the problem.
_TERMINAL = (
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)


def _classify(exc: Exception, source: Source) -> ModelError:
    """Map an SDK exception to the one error type we raise."""
    if isinstance(exc, APITimeoutError):
        return ModelError(f"model request timed out: {exc}", retryable=True, kind="timeout", cause=exc)
    if isinstance(exc, APIConnectionError):
        return ModelError(f"cannot reach the model: {exc}", retryable=True, kind="connect", cause=exc)
    if isinstance(exc, RateLimitError):
        # The one place `source` changes behaviour: an unattended run waiting out
        # a queue is worth less than an interactive turn getting the slot.
        return ModelError(
            f"model overloaded: {exc}",
            retryable=source != "background",
            kind="rate_limit",
            cause=exc,
        )
    if isinstance(exc, _TERMINAL):
        kind = "auth" if isinstance(exc, (AuthenticationError, PermissionDeniedError)) else "bad_request"
        return ModelError(f"model rejected the request: {exc}", retryable=False, kind=kind, cause=exc)
    if isinstance(exc, InternalServerError):
        return ModelError(f"model server error: {exc}", retryable=True, kind="server_error", cause=exc)
    if isinstance(exc, APIStatusError):
        # Any 5xx we did not name above is still the server's fault.
        retryable = exc.status_code >= 500
        return ModelError(
            f"model returned {exc.status_code}: {exc}",
            retryable=retryable,
            kind="server_error" if retryable else "bad_request",
            cause=exc,
        )
    # Anything else (a broken stream, a malformed chunk) is transport-shaped.
    return ModelError(f"model call failed ({type(exc).__name__}): {exc}", retryable=True, kind="stream", cause=exc)


async def _backoff(attempt: int) -> None:
    """Exponential backoff with jitter. `attempt` is 1-based."""
    base = float(config.get("llm.retry_backoff_s") or 0.5)
    ceiling = float(config.get("llm.retry_backoff_max_s") or 8.0)
    delay = min(base * (2 ** (attempt - 1)), ceiling)
    await asyncio.sleep(delay * (0.5 + random.random() / 2))


# --- generate ---------------------------------------------------------------


async def generate(
    messages: Sequence[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    source: Source = "interactive",
    options: dict[str, Any] | None = None,
) -> AsyncIterator[Delta]:
    """
    One model turn, streamed.

    Args:
        messages: OpenAI-shape messages. Built by the fold, not by this module.
        tools: OpenAI-shape tool schemas, or None for a plain completion.
        source: `interactive` or `background`; see `Source`.
        options: Per-call model params, passed through. Written ONLY from config,
            never from model output. `chat_template_kwargs` is lifted into
            `extra_body` for SGLang (e.g. disabling the thinking scratchpad).

    Yields:
        `TextDelta` / `ReasoningDelta` / `ToolCallDelta` as they arrive, then
        exactly one `Finish`.

    Raises:
        ModelError: and nothing else. `asyncio.CancelledError` propagates.
    """
    max_attempts = int(config.get("llm.max_attempts") or 3)
    last: ModelError | None = None

    for attempt in range(1, max_attempts + 1):
        started = False
        try:
            async for delta in _stream_once(messages, tools, source, options):
                # Once the caller has seen a delta, this attempt is committed:
                # replaying it would duplicate text on screen and duplicate tool
                # calls in the transcript. A mid-stream failure is the LOOP's to
                # retry (as a whole hop), not ours.
                started = True
                yield delta
            return
        except ModelError as e:
            if started or not e.retryable or attempt == max_attempts:
                raise
            last = e
            logger.warning(
                "model attempt %d/%d failed (%s), retrying: %s",
                attempt,
                max_attempts,
                e.kind,
                e,
            )
            await _backoff(attempt)

    raise last  # unreachable: the loop either returns or raises


async def _stream_once(
    messages: Sequence[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    source: Source,
    options: dict[str, Any] | None,
) -> AsyncIterator[Delta]:
    """One attempt. Every failure leaves here as a ModelError."""
    opts = dict(options or {})
    extra_body: dict[str, Any] = {}
    if "chat_template_kwargs" in opts:
        extra_body["chat_template_kwargs"] = opts.pop("chat_template_kwargs")

    kwargs: dict[str, Any] = {
        "model": str(config.get("llm.model_name") or ""),
        "messages": list(messages),
        "max_tokens": int(config.get("llm.max_tokens") or 8192),
        "temperature": float(config.get("llm.temperature") or 0.7),
        "stream": True,
        # Without this the server sends no usage on a streamed call and every
        # budget downstream is blind.
        "stream_options": {"include_usage": True},
    }
    kwargs.update(opts)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = kwargs.get("tool_choice", "auto")
    if extra_body:
        kwargs["extra_body"] = extra_body

    try:
        stream = await get_client().chat.completions.create(**kwargs)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        raise _classify(e, source) from e

    finish_reason: str | None = None
    usage: dict[str, int] | None = None

    try:
        async for chunk in stream:
            # The usage chunk arrives last and carries no choices.
            if getattr(chunk, "usage", None):
                usage = chunk.usage.model_dump() if hasattr(chunk.usage, "model_dump") else dict(chunk.usage)
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason

            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            # reasoning_content is SGLang's, not in the OpenAI schema.
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield ReasoningDelta(text=reasoning)

            if delta.content:
                yield TextDelta(text=delta.content)

            for call in getattr(delta, "tool_calls", None) or []:
                fn = getattr(call, "function", None)
                yield ToolCallDelta(
                    index=call.index,
                    id=getattr(call, "id", None),
                    name=getattr(fn, "name", None) if fn else None,
                    arguments=(getattr(fn, "arguments", None) or "") if fn else "",
                )
    except asyncio.CancelledError:
        raise
    except ModelError:
        raise
    except Exception as e:
        raise _classify(e, source) from e

    yield Finish(reason=finish_reason, usage=usage)
