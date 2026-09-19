"""The model client: one cached client, one streamed `generate`, one retry layer.

Raises ModelError and nothing else; CancelledError propagates.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from collections.abc import AsyncIterator, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from typing import Any, Literal

from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
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

from config_module.loader import cfg as _cfg
from model_module.errors import ModelError

logger = logging.getLogger(__name__)

# `background` is an unattended run; `interactive` has a human waiting on it.
Source = Literal["interactive", "background"]


# --- deltas -----------------------------------------------------------------


@dataclass(slots=True)
class TextDelta:
    text: str


@dataclass(slots=True)
class ReasoningDelta:
    """SGLang reasoning_content; streamed like text, never folded into messages."""

    text: str


@dataclass(slots=True)
class ToolCallDelta:
    """One fragment. `id` and `name` arrive once; `index` is on every fragment."""

    index: int
    id: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass(slots=True)
class Finish:
    reason: str | None
    # SGLang sends *_tokens_details as None, so the values are not all ints.
    usage: dict[str, Any] | None = None


@dataclass(slots=True)
class RetryDelta:
    """The client is waiting to try again, and says so before it waits."""

    attempt: int
    of: int
    delay_s: float
    kind: str


Delta = TextDelta | ReasoningDelta | ToolCallDelta | Finish | RetryDelta


# --- client -----------------------------------------------------------------
#
# One client per RUNNING LOOP, not one per process: httpx binds its sockets to
# the loop that opened them, so a client cached in a module global outlives its
# loop and the next one finds a dead pool ("Event loop is closed").

_clients: dict[Any, tuple[tuple[str, str, float], AsyncOpenAI]] = {}


def get_client() -> AsyncOpenAI:
    """Return this loop's client; max_retries=0 because `generate` is the only retry layer."""
    base_url = str(_cfg("llm.base_url", ""))
    api_key = str(_cfg("llm.api_key", "-"))
    # Part of the cache key: the SDK reads it only at construction.
    timeout = float(_cfg("llm.timeout_s", 90))
    key = (base_url, api_key, timeout)

    try:
        loop: Any = asyncio.get_running_loop()
    except RuntimeError:
        # No running loop, so nothing to key a cache entry by: uncached client.
        return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    cached = _clients.get(loop)
    if cached is not None and cached[0] == key:
        return cached[1]
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)
    _clients[loop] = (key, client)
    # Closed loops need no close(): their pools died with them, only the entry remains.
    for stale in [other for other in _clients if other.is_closed()]:
        _clients.pop(stale, None)
    return client


def reset_client() -> None:
    """Drop every cached client, for tests and config reloads."""
    _clients.clear()


# --- error classification ---------------------------------------------------

_TERMINAL = (
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
)

# Context overflow arrives as an ordinary 400 worded differently per provider,
# so both the error code and the prose are checked.
_OVERFLOW_CODE = "context_length_exceeded"
_OVERFLOW_TEXT = re.compile(
    r"context[ _-]?length|maximum context|context window|too many tokens|reduce the length|"
    r"longer than the model|input is too long",
    re.IGNORECASE,
)


def _is_context_overflow(exc: Exception) -> bool:
    """Return True when a 400 is the request being too long rather than malformed."""
    if getattr(exc, "code", None) == _OVERFLOW_CODE:
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict) and body.get("error", {}).get("code") == _OVERFLOW_CODE:
        return True
    return bool(_OVERFLOW_TEXT.search(str(exc)))


def _classify(exc: Exception, source: Source) -> ModelError:
    """Map an SDK exception to the one error type we raise."""
    if isinstance(exc, APITimeoutError):
        return ModelError(f"model request timed out: {exc}", retryable=True, kind="timeout", cause=exc)
    if isinstance(exc, APIConnectionError):
        return ModelError(f"cannot reach the model: {exc}", retryable=True, kind="connect", cause=exc)
    if isinstance(exc, RateLimitError):
        # Retryable for every source, background included: an unattended run is
        # the one with nobody to mind a few seconds' wait.
        return ModelError(
            f"model overloaded: {exc}",
            retryable=True,
            kind="rate_limit",
            cause=exc,
            retry_after=_retry_after(exc),
        )
    if isinstance(exc, _TERMINAL):
        if isinstance(exc, BadRequestError) and _is_context_overflow(exc):
            # Its own kind: the loop recovers from this one by shrinking the view.
            return ModelError(
                f"the request exceeded the context window: {exc}", retryable=False, kind="context_overflow", cause=exc
            )
        kind = "auth" if isinstance(exc, (AuthenticationError, PermissionDeniedError)) else "bad_request"
        return ModelError(f"model rejected the request: {exc}", retryable=False, kind=kind, cause=exc)
    if isinstance(exc, InternalServerError):
        return ModelError(f"model server error: {exc}", retryable=True, kind="server_error", cause=exc)
    if isinstance(exc, APIStatusError):
        retryable = exc.status_code >= 500
        return ModelError(
            f"model returned {exc.status_code}: {exc}",
            retryable=retryable,
            kind="server_error" if retryable else "bad_request",
            cause=exc,
        )
    if isinstance(exc, (APIError, APIResponseValidationError)):
        # A body-level error after 200 OK; a retry reproduces it.
        return ModelError(f"model stream failed: {exc}", retryable=False, kind="stream", cause=exc)
    # Anything the SDK does not raise is a bug on this side.
    return ModelError(
        f"model call failed ({type(exc).__name__}): {exc}",
        retryable=False,
        kind="internal",
        cause=exc,
    )


def _retry_after(exc: Exception) -> float | None:
    """The provider's own "wait this long", in seconds, read from the `retry-after` header."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for name in ("retry-after-ms", "retry-after"):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        seconds = value / 1000 if name.endswith("-ms") else value
        # Capped: a provider asking for an hour is an outage, not a retry.
        return min(seconds, float(_cfg("llm.retry_backoff_max_s", 8.0)))
    return None


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """How long to wait before `attempt` + 1, in seconds. Exponential with jitter."""
    base = float(_cfg("llm.retry_backoff_s", 0.5))
    ceiling = float(_cfg("llm.retry_backoff_max_s", 8.0))
    delay = retry_after if retry_after is not None else min(base * (2 ** (attempt - 1)), ceiling)
    return delay * (0.5 + random.random() / 2)


# --- generate ---------------------------------------------------------------


async def generate(
    messages: Sequence[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    *,
    source: Source = "interactive",
    options: dict[str, Any] | None = None,
) -> AsyncIterator[Delta]:
    """One model turn, streamed.

    Yields deltas as they arrive, then exactly one `Finish`. Raises ModelError
    and nothing else; CancelledError propagates.
    """
    try:
        # max(1, ...) keeps the attempt range below non-empty.
        max_attempts = max(1, int(_cfg("llm.max_retries", 3)))
    except (TypeError, ValueError) as e:
        raise ModelError(f"bad llm.max_retries in config: {e}", retryable=False, kind="bad_request", cause=e) from e

    last: ModelError | None = None

    for attempt in range(1, max_attempts + 1):
        started = False
        try:
            # aclosing closes the HTTP stream as soon as the iterator is abandoned.
            async with aclosing(_stream_once(messages, tools, source, options)) as attempt_stream:
                async for delta in attempt_stream:
                    # Past the first delta the attempt is committed: a retry would
                    # duplicate deltas the caller has already seen.
                    started = True
                    yield delta
            return
        except ModelError as e:
            if started or not e.retryable or attempt == max_attempts:
                if attempt > 1:
                    e.attempts = attempt
                raise
            last = e
            delay = backoff_delay(attempt, e.retry_after)
            logger.warning(
                "model attempt %d/%d failed (%s), retrying in %.1fs: %s", attempt, max_attempts, e.kind, delay, e
            )
            yield RetryDelta(attempt=attempt, of=max_attempts, delay_s=delay, kind=e.kind)
            await asyncio.sleep(delay)

    assert last is not None
    raise last


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
        "model": str(_cfg("llm.model_name", "")),
        "messages": list(messages),
        "max_tokens": int(_cfg("llm.max_tokens", 8192)),
        "temperature": float(_cfg("llm.temperature", 0.7)),
        "stream": True,
        # Streamed calls report no usage without this.
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
    usage: dict[str, Any] | None = None

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
    except Exception as e:
        raise _classify(e, source) from e
    finally:
        # Frees the decode slot and the pooled connection.
        await stream.close()

    yield Finish(reason=finish_reason, usage=usage)
