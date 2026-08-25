"""The model client has one retry layer and it is bounded.

Pins contracts.md's retry budget: <=3 attempts, and background does not retry on overload.
"""

import asyncio
from types import SimpleNamespace

import pytest
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from model_module import client as mc
from model_module.errors import ModelError

# Captured before monkeypatching so the override lambdas can defer to the real
# loader. `_cfg` rather than `config.get`: the module-level alias is the seam now
# (11.7.5 gave the eleven copies one home), and patching it isolates this module
# instead of reconfiguring every other one through the shared singleton.
_real_cfg = mc._cfg


# --- doubles ----------------------------------------------------------------


def _chunk(*, content=None, reasoning=None, tool_calls=None, finish=None):
    """One SGLang-shaped streaming chunk."""
    delta = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta, finish_reason=finish)], usage=None)


def _usage_chunk(**counts):
    """The trailing usage-only chunk, which carries no choices."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(model_dump=lambda: counts))


def _tool_call(index, *, id=None, name=None, arguments=""):
    return SimpleNamespace(index=index, id=id, function=SimpleNamespace(name=name, arguments=arguments))


def _response(status_code):
    """Minimal httpx-shaped response for constructing SDK errors."""
    import httpx

    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "http://t/v1/chat/completions"))


class _Stream:
    def __init__(self, chunks, fail_after=None, error=None):
        self._chunks = chunks
        self._fail_after = fail_after
        self._error = error
        self.closed = False

    def __aiter__(self):
        async def gen():
            for i, c in enumerate(self._chunks):
                if self._fail_after is not None and i == self._fail_after:
                    raise self._error
                yield c

        return gen()

    async def close(self):
        self.closed = True


class _FakeCompletions:
    """Counts attempts, so the retry budget is observable."""

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
        monkeypatch.setattr(mc, "get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
        return completions

    # Keep the suite fast: the attempt count is what's under test, not the clock.
    overrides = {"llm.retry_backoff_s": 0.001, "llm.retry_backoff_max_s": 0.004}
    monkeypatch.setattr(mc, "_cfg", lambda k, d=None: overrides.get(k, _real_cfg(k, d)))
    return install


async def _drain(source="interactive", tools=None, options=None):
    return [d async for d in mc.generate([{"role": "user", "content": "hi"}], tools, source=source, options=options)]


# --- the retry budget -------------------------------------------------------


@pytest.mark.parametrize(
    "error,kind",
    [
        (APITimeoutError(request=None), "timeout"),
        (APIConnectionError(request=None), "connect"),
        (InternalServerError("boom", response=_response(500), body=None), "server_error"),
    ],
)
@pytest.mark.asyncio
async def test_retryable_failures_stop_at_three_attempts(fake, error, kind):
    """A permanently failing endpoint costs 3 requests, not 99."""
    completions = fake(lambda n: error)

    with pytest.raises(ModelError) as excinfo:
        await _drain()

    assert completions.calls == 3
    assert excinfo.value.kind == kind
    assert excinfo.value.retryable is True


@pytest.mark.parametrize("source,expected_calls", [("background", 1), ("interactive", 3)])
@pytest.mark.asyncio
async def test_only_interactive_waits_out_an_overload(fake, source, expected_calls):
    """An unattended run yields the GPU slot instead of queueing for it."""
    completions = fake(lambda n: RateLimitError("busy", response=_response(429), body=None))

    with pytest.raises(ModelError) as excinfo:
        await _drain(source=source)

    assert completions.calls == expected_calls
    assert excinfo.value.kind == "rate_limit"


@pytest.mark.parametrize(
    "error,kind",
    [
        (BadRequestError("nope", response=_response(400), body=None), "bad_request"),
        # In-band SSE error: 200 OK, then {"error": ...} in the body.
        (APIError("context length exceeded", request=None, body=None), "stream"),
    ],
)
@pytest.mark.asyncio
async def test_terminal_failures_fail_fast(fake, error, kind):
    completions = fake(lambda n: error)

    with pytest.raises(ModelError) as excinfo:
        await _drain()

    assert completions.calls == 1
    assert excinfo.value.kind == kind
    assert excinfo.value.retryable is False


@pytest.mark.asyncio
async def test_failure_before_the_first_chunk_still_retries(fake):
    """Nothing has been yielded yet, so the attempt is not committed."""

    def behaviour(n):
        if n == 1:
            return _Stream([_chunk(content="never seen")], fail_after=0, error=APITimeoutError(request=None))
        return _Stream([_chunk(content="second time", finish="stop")])

    completions = fake(behaviour)
    deltas = await _drain()

    assert completions.calls == 2
    assert [d.text for d in deltas if isinstance(d, mc.TextDelta)] == ["second time"]


@pytest.mark.asyncio
async def test_midstream_failure_is_not_retried(fake):
    """Once a delta has been handed over the attempt is committed; the loop retries the hop."""
    completions = fake(
        lambda n: _Stream(
            [_chunk(content="par"), _chunk(content="tial")],
            fail_after=1,
            error=APITimeoutError(request=None),
        )
    )

    seen = []
    with pytest.raises(ModelError):
        async for d in mc.generate([{"role": "user", "content": "hi"}]):
            seen.append(d)

    assert completions.calls == 1
    assert [d.text for d in seen] == ["par"]


@pytest.mark.asyncio
async def test_parsing_bug_is_not_retried(fake):
    """Retrying cannot fix our own code."""
    broken = SimpleNamespace(usage=None)  # no .choices -> AttributeError in the parse loop
    completions = fake(lambda n: _Stream([broken]))

    with pytest.raises(ModelError) as excinfo:
        await _drain()

    assert completions.calls == 1
    assert excinfo.value.kind == "internal"
    assert excinfo.value.retryable is False


# --- streaming shape --------------------------------------------------------


@pytest.mark.asyncio
async def test_yields_text_reasoning_tool_calls_and_usage(fake):
    fake(
        lambda n: _Stream(
            [
                _chunk(reasoning="thinking..."),
                _chunk(content="the answer"),
                _chunk(tool_calls=[_tool_call(0, id="c1", name="run_command", arguments='{"cmd')]),
                _chunk(tool_calls=[_tool_call(0, arguments='":"ls"}')]),
                _chunk(finish="tool_calls"),
                # SGLang sends *_tokens_details beside the counts, so usage is not dict[str, int].
                _usage_chunk(prompt_tokens=9, completion_tokens=142, prompt_tokens_details=None),
            ]
        )
    )

    deltas = await _drain()

    assert isinstance(deltas[0], mc.ReasoningDelta) and deltas[0].text == "thinking..."
    assert isinstance(deltas[1], mc.TextDelta) and deltas[1].text == "the answer"

    calls = [d for d in deltas if isinstance(d, mc.ToolCallDelta)]
    assert calls[0].name == "run_command"
    # Fragments are handed over as they arrive; assembly is the loop's job.
    assert "".join(c.arguments for c in calls) == '{"cmd":"ls"}'

    finish = deltas[-1]
    assert isinstance(finish, mc.Finish)
    assert finish.reason == "tool_calls"
    assert finish.usage["prompt_tokens_details"] is None


@pytest.mark.asyncio
async def test_concurrent_tool_calls_stay_separated_by_index(fake):
    """The model interleaves calls; only `index` is on every fragment."""
    fake(
        lambda n: _Stream(
            [
                _chunk(tool_calls=[_tool_call(0, id="a", name="grep", arguments='{"q"')]),
                _chunk(tool_calls=[_tool_call(1, id="b", name="glob", arguments='{"p"')]),
                _chunk(tool_calls=[_tool_call(0, arguments=':"x"}')]),
                _chunk(tool_calls=[_tool_call(1, arguments=':"y"}')]),
                _chunk(finish="tool_calls"),
            ]
        )
    )

    calls = [d for d in await _drain() if isinstance(d, mc.ToolCallDelta)]
    assert "".join(c.arguments for c in calls if c.index == 0) == '{"q":"x"}'
    assert "".join(c.arguments for c in calls if c.index == 1) == '{"p":"y"}'


@pytest.mark.asyncio
async def test_finish_arrives_even_with_no_usage_chunk(fake):
    fake(lambda n: _Stream([_chunk(content="hi", finish="stop")]))
    deltas = await _drain()
    assert isinstance(deltas[-1], mc.Finish) and deltas[-1].usage is None


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

        async def close(self):
            pass

    fake(lambda n: _SlowStream())

    agen = mc.generate([{"role": "user", "content": "hi"}])
    # Any accumulate-then-yield implementation hangs here instead.
    first = await asyncio.wait_for(agen.__anext__(), timeout=1)
    assert first.text == "first"

    released.set()
    await agen.aclose()


@pytest.mark.asyncio
async def test_stream_is_closed_when_the_consumer_abandons_it(fake):
    """A hop abandoned on budget must free the decode slot immediately."""
    stream = _Stream([_chunk(content="a"), _chunk(content="b"), _chunk(content="c", finish="stop")])
    fake(lambda n: stream)

    agen = mc.generate([{"role": "user", "content": "hi"}])
    await agen.__anext__()
    await agen.aclose()

    assert stream.closed, "the SGLang decode slot keeps generating until this closes"


@pytest.mark.asyncio
async def test_cancellation_is_not_a_model_error(fake):
    """Wrapping cancel as ModelError would make it look retryable."""
    fake(lambda n: asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await _drain()


# --- config is the source of truth ------------------------------------------


@pytest.mark.asyncio
async def test_options_and_tools_reach_the_wire(fake):
    completions = fake(lambda n: _Stream([_chunk(content="ok", finish="stop")]))
    tools = [{"type": "function", "function": {"name": "finish_task", "parameters": {}}}]

    await _drain(tools=tools, options={"temperature": 0.2, "chat_template_kwargs": {"enable_thinking": False}})

    sent = completions.kwargs
    assert sent["tools"] == tools and sent["tool_choice"] == "auto"
    assert sent["temperature"] == 0.2
    # chat_template_kwargs is SGLang-only, so it rides extra_body, not the top level.
    assert sent["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}

    await _drain()
    assert "tools" not in completions.kwargs and "tool_choice" not in completions.kwargs


@pytest.mark.asyncio
async def test_falsy_config_values_are_honoured(monkeypatch):
    """`temperature: 0` must reach the wire rather than being discarded as falsy."""
    completions = _FakeCompletions(lambda n: _Stream([_chunk(content="ok", finish="stop")]))
    monkeypatch.setattr(mc, "get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(mc, "_cfg", lambda k, d=None: 0 if k == "llm.temperature" else _real_cfg(k, d))

    await _drain()
    assert completions.kwargs["temperature"] == 0.0


@pytest.mark.parametrize("bad_value", [-1, "hot"])
@pytest.mark.asyncio
async def test_bad_max_retries_still_raises_model_error(monkeypatch, bad_value):
    """generate() raises ModelError and nothing else, whatever the config holds."""
    completions = _FakeCompletions(lambda n: APITimeoutError(request=None))
    monkeypatch.setattr(mc, "get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
    monkeypatch.setattr(mc, "_cfg", lambda k, d=None: bad_value if k == "llm.max_retries" else _real_cfg(k, d))

    with pytest.raises(ModelError):
        await _drain()
    assert completions.calls <= 1


def test_client_is_cached_with_the_configured_timeout(monkeypatch):
    """The bounded wall clock depends on the timeout actually being applied."""
    mc.reset_client()
    monkeypatch.setattr(mc, "_cfg", lambda k, d=None: 12.5 if k == "llm.timeout_s" else _real_cfg(k, d))

    first = mc.get_client()
    assert first is mc.get_client(), "a new client per call was the old bug"
    assert first.timeout == 12.5
    assert first.max_retries == 0, "the SDK retry layer is one of the three nested loops being deleted"

    # timeout is part of the cache key, so a config reload is not silently ignored.
    monkeypatch.setattr(mc, "_cfg", lambda k, d=None: 34.0 if k == "llm.timeout_s" else _real_cfg(k, d))
    assert mc.get_client().timeout == 34.0
    mc.reset_client()


# --- a request too long is not the same as a request that is wrong -------------


def test_a_context_length_400_is_classified_as_overflow():
    """Recoverable by shrinking the view; a malformed request is not."""
    from openai import BadRequestError

    too_long = BadRequestError.__new__(BadRequestError)
    Exception.__init__(too_long, "This model's maximum context length is 128000 tokens")
    too_long.body = {"error": {"code": "context_length_exceeded"}}
    too_long.code = "context_length_exceeded"

    error = mc._classify(too_long, "interactive")

    assert error.kind == "context_overflow"
    assert error.retryable is False


def test_an_ordinary_400_stays_a_bad_request():
    from openai import BadRequestError

    malformed = BadRequestError.__new__(BadRequestError)
    Exception.__init__(malformed, "Invalid value for 'tool_choice'")
    malformed.body = {"error": {"code": "invalid_value"}}
    malformed.code = "invalid_value"

    assert mc._classify(malformed, "interactive").kind == "bad_request"


def test_overflow_is_recognised_from_the_prose_alone():
    """SGLang and OpenAI word it differently, and neither is guaranteed to send a code."""
    from openai import BadRequestError

    for wording in (
        "This model's maximum context length is 8192 tokens, however you requested 9000",
        "Input is too long for requested model",
        "please reduce the length of the messages",
    ):
        exc = BadRequestError.__new__(BadRequestError)
        Exception.__init__(exc, wording)
        assert mc._classify(exc, "interactive").kind == "context_overflow", wording
