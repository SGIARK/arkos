"""execute() never raises except cancellation, so a broken tool is model input."""

import asyncio

import pytest

from tool_module import envelope as env

SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
    "required": ["path"],
}


class _Tool:
    def __init__(self, name="read_file", result=None, boom=None, delay=0.0, **spec_kw):
        self.spec = env.ToolSpec(name=name, description="d", input_schema=SCHEMA, **spec_kw)
        self._result = result or env.ok("contents")
        self._boom = boom
        self._delay = delay
        self.calls = []

    async def call(self, args, ctx):
        self.calls.append(args)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._boom:
            raise self._boom
        return self._result


def _lookup(*tools):
    by_name = {t.spec.name: t for t in tools}
    return lambda name: by_name.get(name)


def _ctx(**kw):
    return env.ToolContext(user_id="u1", **kw)


@pytest.mark.asyncio
async def test_unknown_tool_is_an_envelope_not_an_exception():
    result = await env.execute("nope", {}, _ctx(), lookup=_lookup(_Tool()))
    assert result.ok is False and result.error_kind == "not_found"


@pytest.mark.asyncio
async def test_a_raising_tool_is_reported_to_the_model():
    tool = _Tool(boom=RuntimeError("disk on fire"))
    result = await env.execute("read_file", {"path": "/x"}, _ctx(), lookup=_lookup(tool))

    assert result.ok is False and result.error_kind == "upstream_error"
    assert "disk on fire" in result.content


@pytest.mark.asyncio
async def test_a_hung_tool_is_capped():
    tool = _Tool(delay=5)
    result = await env.execute("read_file", {"path": "/x"}, _ctx(), lookup=_lookup(tool), timeout_s=0.05)

    assert result.error_kind == "timeout" and result.retryable is True
    assert "may or may not" in result.content


@pytest.mark.asyncio
async def test_cancellation_is_the_one_thing_that_propagates():
    tool = _Tool(boom=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await env.execute("read_file", {"path": "/x"}, _ctx(), lookup=_lookup(tool))


@pytest.mark.parametrize(
    "args,fragment",
    [
        ({}, "Missing required"),
        ({"path": "/x", "bogus": 1}, "Unknown argument"),
        ({"path": 42}, "should be string"),
        ({"path": "/x", "limit": True}, "should be integer"),
    ],
)
@pytest.mark.asyncio
async def test_bad_arguments_are_rejected_before_the_tool_runs(args, fragment):
    tool = _Tool()
    result = await env.execute("read_file", args, _ctx(), lookup=_lookup(tool))

    assert result.error_kind == "invalid_args" and fragment in result.content
    assert tool.calls == [], "the tool must not see arguments that failed validation"


@pytest.mark.asyncio
async def test_approval_is_checked_at_execute_time():
    tool = _Tool(name="send_email", requires_approval=True)
    declined = await env.execute("send_email", {"path": "/x"}, _ctx(approve=_deny), lookup=_lookup(tool), timeout_s=1)
    assert declined.ok is False and tool.calls == []

    allowed = await env.execute("send_email", {"path": "/x"}, _ctx(approve=_allow), lookup=_lookup(tool), timeout_s=1)
    assert allowed.ok is True and len(tool.calls) == 1


@pytest.mark.asyncio
async def test_a_tool_needing_approval_in_a_session_that_cannot_ask_does_not_run():
    tool = _Tool(name="send_email", requires_approval=True)
    result = await env.execute("send_email", {"path": "/x"}, _ctx(), lookup=_lookup(tool))
    assert result.ok is False and tool.calls == []


@pytest.mark.asyncio
async def test_validate_runs_before_call():
    """Preconditions like read-before-edit belong to the tool, not the loop."""
    tool = _Tool(name="edit_file")
    tool.validate = lambda args, ctx: "Read the file before editing it."

    result = await env.execute("edit_file", {"path": "/x"}, _ctx(), lookup=_lookup(tool))

    assert result.error_kind == "invalid_args" and "Read the file" in result.content
    assert tool.calls == []


@pytest.mark.asyncio
async def test_only_transport_failures_are_marked_retryable():
    assert env.fail("timeout", "x").retryable is True
    assert env.fail("upstream_error", "x").retryable is True
    assert env.fail("invalid_args", "x").retryable is False
    assert env.fail("auth_required", "x").retryable is False


async def _allow(name, args):
    return True


async def _deny(name, args):
    return False


@pytest.mark.asyncio
async def test_nothing_escapes_execute():
    """The one promise this function makes, on every path that can raise."""

    def boom_lookup(name):
        raise RuntimeError("registry exploded")

    result = await env.execute("x", {}, _ctx(), lookup=boom_lookup)
    assert result.error_kind == "upstream_error"

    async def boom_approve(name, args):
        raise RuntimeError("park write failed")

    tool = _Tool(name="send_email", requires_approval=True)
    result = await env.execute("send_email", {"path": "/x"}, _ctx(approve=boom_approve), lookup=_lookup(tool))
    assert result.error_kind == "upstream_error"

    tool = _Tool()
    tool.validate = lambda args, ctx: 1 / 0
    result = await env.execute("read_file", {"path": "/x"}, _ctx(), lookup=_lookup(tool))
    assert result.error_kind == "upstream_error"


@pytest.mark.asyncio
async def test_a_sync_approve_is_tolerated():
    tool = _Tool(name="send_email", requires_approval=True)
    result = await env.execute(
        "send_email", {"path": "/x"}, _ctx(approve=lambda n, a: True), lookup=_lookup(tool), timeout_s=1
    )
    assert result.ok is True


@pytest.mark.asyncio
async def test_a_tool_returning_junk_does_not_reach_the_loop():
    tool = _Tool(result={"not": "an envelope"})
    result = await env.execute("read_file", {"path": "/x"}, _ctx(), lookup=_lookup(tool))
    assert result.ok is False and "malformed" in result.content


@pytest.mark.asyncio
async def test_useless_failures_are_not_marked_retryable():
    tool = _Tool(name="send_email", requires_approval=True)
    declined = await env.execute("send_email", {"path": "/x"}, _ctx(approve=_deny), lookup=_lookup(tool))
    cannot_ask = await env.execute("send_email", {"path": "/x"}, _ctx(), lookup=_lookup(tool))

    assert declined.retryable is False
    assert cannot_ask.retryable is False


@pytest.mark.asyncio
async def test_third_party_schemas_do_not_become_uncallable():
    """A key marked required but never declared in properties is still callable."""

    class _Remote:
        spec = env.ToolSpec(
            name="remote",
            input_schema={"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a", "b"]},
        )

        async def call(self, args, ctx):
            return env.ok("fine")

    result = await env.execute("remote", {"a": "x", "b": "y"}, _ctx(), lookup=_lookup(_Remote()))
    assert result.ok is True


@pytest.mark.parametrize("schema_extra", [{"additionalProperties": True}, {"oneOf": [{}]}])
@pytest.mark.asyncio
async def test_open_schemas_skip_unknown_key_rejection(schema_extra):
    class _Remote:
        spec = env.ToolSpec(
            name="remote",
            input_schema={"type": "object", "properties": {"a": {"type": "string"}}, **schema_extra},
        )

        async def call(self, args, ctx):
            return env.ok("fine")

    result = await env.execute("remote", {"a": "x", "extra": 1}, _ctx(), lookup=_lookup(_Remote()))
    assert result.ok is True


@pytest.mark.asyncio
async def test_a_nullable_type_union_is_accepted():
    class _Remote:
        spec = env.ToolSpec(
            name="remote",
            input_schema={"type": "object", "properties": {"a": {"type": ["string", "null"]}}},
        )

        async def call(self, args, ctx):
            return env.ok("fine")

    assert (await env.execute("remote", {"a": None}, _ctx(), lookup=_lookup(_Remote()))).ok is True
    assert (await env.execute("remote", {"a": 5}, _ctx(), lookup=_lookup(_Remote()))).error_kind == "invalid_args"


# Invariant: execute() converts EVERY exception into an envelope, with exactly one
# exception, cancellation, which is BaseException and must reach the event loop.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boom",
    [
        RuntimeError("plain"),
        ValueError("bad value"),
        KeyError("missing"),
        TypeError("wrong type"),
        AttributeError("no attr"),
        TimeoutError("aiohttp total timeout"),
        OSError("connection reset"),
        MemoryError("out of memory"),
        RecursionError("too deep"),
        UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid"),
        ZeroDivisionError("division by zero"),
        StopIteration("exhausted"),
        StopAsyncIteration("exhausted"),
        NotImplementedError("todo"),
    ],
    ids=lambda e: type(e).__name__,
)
async def test_every_exception_kind_comes_back_as_an_envelope(boom):
    tool = _Tool(boom=boom)
    result = await env.execute("read_file", {"path": "p"}, _ctx(), lookup=lambda n: tool)

    assert isinstance(result, env.ResultEnvelope)
    assert result.ok is False and result.error_kind in ("upstream_error", "timeout")


@pytest.mark.asyncio
async def test_cancellation_is_the_one_thing_that_escapes():
    tool = _Tool(boom=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await env.execute("read_file", {"path": "p"}, _ctx(), lookup=lambda n: tool)
