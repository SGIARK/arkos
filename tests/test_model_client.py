"""
Task 1 conformance: the model client has ONE retry layer and it is bounded.

Pins `test_retry_budget_bounded` from contracts.md: <=3 attempts, bounded wall
clock, background source does not retry on overload.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest
from openai import APITimeoutError, BadRequestError, RateLimitError

from model_module import client as mc
from model_module.errors import ModelError

# --- doubles ----------------------------------------------------------------


def _chunk(*, content=None, reasoning=None, tool_calls=None, finish=None, usage=None):
    """One SGLang-shaped streaming chunk."""
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    choice = SimpleNamespace(delta=delta, finish_reason=finish)
    return SimpleNamespace(choices=[choice], usage=usage)


def _usage_chunk(prompt=10, completion=5):
    """The trailing usage-only chunk: no choices, which readers must tolerate."""
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": prompt, "completion_tokens": completion})
    return SimpleNamespace(choices=[], usage=usage)


def _tool_call(index, *, id=None, name=None, arguments=""):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


class _Stream:
    def __init__(self, chunks, fail_after=None, error=None):
        self._chunks = chunks
        self._fail_after = fail_after
        self._error = error

    def __aiter__(self):
        async def gen():
            for i, c in enumerate(self._chunks):
                if self._fail_after is not None and i == self._fail_after:
                    raise self._error
                yield c

        return gen()


class _FakeCompletions:
    """Counts attempts so the retry budget is observable."""

    def __init__(self, behaviour):
        self._behaviour = behaviour
        self.calls = 0
        self.kwargs = None

    async def create(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        result = self._behaviour(self.calls)
        # BaseException, not Exception: CancelledError is not an Exception subclass.
        if isinstance(result, BaseException):
            raise result
        return result


@pytest.fixture
def fake(monkeypatch):
    """Install a fake client; returns a helper that arms its behaviour."""

    def install(behaviour):
        completions = _FakeCompletions(behaviour)
        stub = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(mc, "get_client", lambda: stub)
        return completions

    # Keep the suite fast: the retry budget is what's under test, not the clock.
    monkeypatch.setattr(mc.config, "get", _fast_config(mc.config.get))
    return install


def _fast_config(original):
    overrides = {"llm.retry_backoff_s": 0.001, "llm.retry_backoff_max_s": 0.004}
    return lambda key, *a, **kw: overrides.get(key, original(key, *a, **kw))


async def _drain(source="interactive", tools=None, options=None):
    return [d async for d in mc.generate([{"role": "user", "content": "hi"}], tools, source=source, options=options)]


# --- the retry budget -------------------------------------------------------


@pytest.mark.asyncio
async def test_hung_endpoint_stops_at_three_attempts(fake):
    """A permanently timing-out endpoint costs 3 requests, not 99."""
    completions = fake(lambda n: APITimeoutError(request=None))

    started = time.monotonic()
    with pytest.raises(ModelError) as excinfo:
        await _drain()
    elapsed = time.monotonic() - started

    assert completions.calls == 3
    assert excinfo.value.kind == "timeout"
    assert excinfo.value.retryable is True
    # Bounded wall clock: three attempts of backoff, not an open-ended wait.
    assert elapsed < 5


@pytest.mark.asyncio
async def test_background_source_does_not_retry_overload(fake):
    """An unattended run yields the GPU slot instead of queueing for it."""
    completions = fake(lambda n: RateLimitError("busy", response=_response(429), body=None))

    with pytest.raises(ModelError) as excinfo:
        await _drain(source="background")

    assert completions.calls == 1
    assert excinfo.value.kind == "rate_limit"
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_interactive_source_does_retry_overload(fake):
    completions = fake(lambda n: RateLimitError("busy", response=_response(429), body=None))

    with pytest.raises(ModelError):
        await _drain(source="interactive")

    assert completions.calls == 3


@pytest.mark.asyncio
async def test_bad_request_fails_fast(fake):
    """Retrying a malformed request just spends the budget on the same 400."""
    completions = fake(lambda n: BadRequestError("nope", response=_response(400), body=None))

    with pytest.raises(ModelError) as excinfo:
        await _drain()

    assert completions.calls == 1
    assert excinfo.value.kind == "bad_request"
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_recovers_on_second_attempt(fake):
    def behaviour(n):
        if n == 1:
            return APITimeoutError(request=None)
        return _Stream([_chunk(content="hello", finish="stop")])

    completions = fake(behaviour)
    deltas = await _drain()

    assert completions.calls == 2
    assert [d.text for d in deltas if isinstance(d, mc.TextDelta)] == ["hello"]


@pytest.mark.asyncio
async def test_midstream_failure_is_not_retried(fake):
    """
    Once a delta has been handed to the caller the attempt is committed: a retry
    would duplicate text on screen and duplicate tool calls in the transcript.
    The loop retries the whole hop instead.
    """

    def behaviour(n):
        return _Stream(
            [_chunk(content="par"), _chunk(content="tial")],
            fail_after=1,
            error=APITimeoutError(request=None),
        )

    completions = fake(behaviour)
    seen = []
    with pytest.raises(ModelError):
        async for d in mc.generate([{"role": "user", "content": "hi"}]):
            seen.append(d)

    assert completions.calls == 1
    assert [d.text for d in seen] == ["par"]


# --- streaming shape --------------------------------------------------------


@pytest.mark.asyncio
async def test_yields_text_reasoning_and_tool_calls(fake):
    fake(
        lambda n: _Stream(
            [
                _chunk(reasoning="thinking..."),
                _chunk(content="the answer"),
                _chunk(tool_calls=[_tool_call(0, id="c1", name="run_command", arguments='{"cmd')]),
                _chunk(tool_calls=[_tool_call(0, arguments='":"ls"}')]),
                _chunk(finish="tool_calls"),
                _usage_chunk(),
            ]
        )
    )

    deltas = await _drain()

    assert isinstance(deltas[0], mc.ReasoningDelta) and deltas[0].text == "thinking..."
    assert isinstance(deltas[1], mc.TextDelta) and deltas[1].text == "the answer"

    calls = [d for d in deltas if isinstance(d, mc.ToolCallDelta)]
    assert [c.index for c in calls] == [0, 0]
    assert calls[0].name == "run_command"
    # Fragments are handed over as they arrive; assembly is the loop's job.
    assert "".join(c.arguments for c in calls) == '{"cmd":"ls"}'

    finish = deltas[-1]
    assert isinstance(finish, mc.Finish)
    assert finish.reason == "tool_calls"
    assert finish.usage == {"prompt_tokens": 10, "completion_tokens": 5}


@pytest.mark.asyncio
async def test_first_delta_arrives_before_the_stream_ends(fake):
    """Streaming is structural: no accumulate-then-replay step exists."""
    released = asyncio.Event()

    class _SlowStream:
        def __aiter__(self):
            async def gen():
                yield _chunk(content="first")
                await released.wait()
                yield _chunk(content="last", finish="stop")

            return gen()

    fake(lambda n: _SlowStream())

    agen = mc.generate([{"role": "user", "content": "hi"}])
    first = await asyncio.wait_for(agen.__anext__(), timeout=1)
    assert first.text == "first"  # arrived while the model is still generating

    released.set()
    await agen.aclose()


@pytest.mark.asyncio
async def test_options_and_tools_reach_the_wire(fake):
    completions = fake(lambda n: _Stream([_chunk(content="ok", finish="stop")]))
    tools = [{"type": "function", "function": {"name": "finish_task", "parameters": {}}}]

    await _drain(tools=tools, options={"temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False}})

    sent = completions.kwargs
    assert sent["tools"] == tools
    assert sent["tool_choice"] == "auto"
    assert sent["temperature"] == 0.2
    # chat_template_kwargs is SGLang-only, so it rides extra_body, not the top level.
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_no_tools_sends_no_tool_choice(fake):
    completions = fake(lambda n: _Stream([_chunk(content="ok", finish="stop")]))
    await _drain()
    assert "tools" not in completions.kwargs
    assert "tool_choice" not in completions.kwargs


@pytest.mark.asyncio
async def test_cancellation_is_not_a_model_error(fake):
    """Cancel must propagate; wrapping it as ModelError would make cancel retryable."""
    fake(lambda n: asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _drain()


# --- the cached client ------------------------------------------------------


def test_client_is_cached_and_has_no_sdk_retries(monkeypatch):
    mc.reset_client()
    first = mc.get_client()
    second = mc.get_client()

    assert first is second, "a new client per call was the old bug"
    assert first.max_retries == 0, "the SDK retry layer is one of the three nested loops being deleted"
    mc.reset_client()


def _response(status_code):
    """Minimal httpx-shaped response for constructing SDK errors."""
    import httpx

    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "http://test/v1/chat/completions"))
