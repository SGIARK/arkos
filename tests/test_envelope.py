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
    # The model is told the outcome is unknown, not that it failed cleanly.
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
