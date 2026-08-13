"""One manifest, and an MCP tool that cannot shadow one of ours."""

import asyncio

import pytest

from tool_module import registry as reg
from tool_module.envelope import ToolContext, ToolSpec, ok


def _ctx(**kw):
    return ToolContext(user_id="u1", **kw)


def test_the_control_tools_are_all_discovered():
    names = set(reg.local_tools())
    assert {"finish_task", "ask", "request_approval", "todo_write", "read_result"} <= names


def test_manifest_namespaces_mcp_tools():
    remote = [ToolSpec(name="send_email", description="d")]
    names = [s.name for s in reg.manifest(remote)]

    assert "mcp_send_email" in names
    assert "send_email" not in names


def test_a_remote_tool_cannot_shadow_one_of_ours():
    """A remote read_file must not displace the sandbox read_file."""
    remote = [ToolSpec(name="read_result", description="impostor")]
    specs = reg.manifest(remote)

    ours = [s for s in specs if s.name == "read_result"]
    assert len(ours) == 1 and ours[0].description != "impostor"
    assert "mcp_read_result" in [s.name for s in specs]


def test_manifest_names_are_unique():
    specs = reg.manifest([ToolSpec(name="a", description="d"), ToolSpec(name="b", description="d")])
    names = [s.name for s in specs]
    assert len(names) == len(set(names))


@pytest.mark.asyncio
async def test_dispatch_strips_the_prefix_before_calling_mcp():
    seen = {}

    async def mcp_call(bare, args, ctx):
        seen["name"] = bare
        return ok("sent")

    result = await reg.dispatch("mcp_send_email", {"to": "x"}, _ctx(), mcp_call=mcp_call)

    assert seen["name"] == "send_email"
    assert result.ok is True


@pytest.mark.asyncio
async def test_a_raising_mcp_transport_is_still_an_envelope():
    async def mcp_call(bare, args, ctx):
        raise ConnectionError("smithery down")

    result = await reg.dispatch("mcp_send_email", {}, _ctx(), mcp_call=mcp_call)

    assert result.ok is False and result.error_kind == "upstream_error"
    assert "smithery down" in result.content


@pytest.mark.asyncio
async def test_an_mcp_call_with_no_transport_is_not_found():
    result = await reg.dispatch("mcp_send_email", {}, _ctx())
    assert result.error_kind == "not_found"


@pytest.mark.asyncio
async def test_dispatch_runs_a_local_tool_through_the_envelope():
    result = await reg.dispatch("finish_task", {"summary": "did the thing"}, _ctx())
    assert result.ok is True and result.content == "did the thing"


@pytest.mark.asyncio
async def test_finish_task_requires_a_summary():
    """An unattended run ends on this, so it may not end on nothing."""
    result = await reg.dispatch("finish_task", {}, _ctx())
    assert result.error_kind == "invalid_args"


@pytest.mark.asyncio
async def test_todo_write_rejects_two_in_progress():
    args = {"items": [{"text": "a", "status": "in_progress"}, {"text": "b", "status": "in_progress"}]}
    result = await reg.dispatch("todo_write", args, _ctx())
    assert result.error_kind == "invalid_args" and "one item" in result.content


@pytest.mark.asyncio
async def test_read_result_without_a_blob_store_says_so():
    result = await reg.dispatch("read_result", {"ref": "r1"}, _ctx())
    assert result.ok is False and result.error_kind == "upstream_error"


@pytest.mark.asyncio
async def test_read_result_pages_a_stored_blob():
    async def read_blob(ref, offset, limit):
        return "full text" if ref == "r1" else None

    hit = await reg.dispatch("read_result", {"ref": "r1"}, _ctx(read_blob=read_blob))
    miss = await reg.dispatch("read_result", {"ref": "nope"}, _ctx(read_blob=read_blob))

    assert hit.content == "full text"
    assert miss.error_kind == "not_found"


# --- gaps found by review ---------------------------------------------------


def test_manifest_hands_out_copies_not_the_cache():
    """A per-session manifest must not be able to poison every other session."""
    first = reg.manifest()
    first[0].description = "vandalised"

    assert reg.manifest()[0].description != "vandalised"


def test_a_remote_tool_already_called_mcp_something_is_not_double_stripped():
    """Prefixing conditionally meant dispatching mcp_foo as foo, which the server lacks."""
    specs = reg.manifest([ToolSpec(name="mcp_foo", description="d")])
    assert "mcp_mcp_foo" in [s.name for s in specs]


def test_two_remote_names_that_collide_after_prefixing_do_not_both_survive():
    """Duplicate function names in one manifest are a 400 from most servers."""
    specs = reg.manifest([ToolSpec(name="x", description="a"), ToolSpec(name="mcp_x", description="b")])
    names = [s.name for s in specs]
    assert len(names) == len(set(names))


def test_a_remote_tool_cannot_take_one_of_our_names():
    specs = reg.manifest([ToolSpec(name="finish_task", description="impostor")])
    ours = [s for s in specs if s.name == "finish_task"]
    assert len(ours) == 1 and ours[0].description != "impostor"


@pytest.mark.asyncio
async def test_dispatch_routes_each_half_to_the_right_place():
    """The other half of the shadowing claim: where each name actually lands."""
    seen = []

    async def mcp_call(bare, args, ctx):
        seen.append(bare)
        return ok("remote")

    ours = await reg.dispatch("finish_task", {"summary": "s"}, _ctx(), mcp_call=mcp_call)
    theirs = await reg.dispatch("mcp_finish_task", {}, _ctx(), mcp_call=mcp_call)

    assert ours.content == "s" and seen == ["finish_task"]
    assert theirs.content == "remote"


@pytest.mark.asyncio
async def test_a_hung_mcp_call_is_capped():
    """The remote tools are the ones that actually hang."""

    async def hangs(bare, args, ctx):
        await asyncio.sleep(5)
        return ok("never")

    result = await reg.dispatch("mcp_slow", {}, _ctx(), mcp_call=hangs, timeout_s=0.05)
    assert result.error_kind == "timeout"


@pytest.mark.asyncio
async def test_bind_produces_the_two_argument_shape_run_turn_needs():
    async def mcp_call(bare, args, ctx):
        return ok("remote")

    dispatch = reg.bind(_ctx(), mcp_call=mcp_call)
    result = await dispatch("finish_task", {"summary": "done"})

    assert result.content == "done"


@pytest.mark.asyncio
async def test_the_bound_dispatch_satisfies_the_loop(monkeypatch):
    """The consumer this was built for must actually be able to take it."""
    from agent_module import loop as lp
    from model_module import client as mc

    def generate(messages, tools=None, **kw):
        async def gen():
            yield mc.ToolCallDelta(index=0, id="c1", name="finish_task", arguments='{"summary":"done"}')
            yield mc.Finish(reason="tool_calls")

        return gen()

    monkeypatch.setattr(lp.model_client, "generate", generate)

    dispatch = reg.bind(_ctx())
    events = [
        e
        async for e in lp.run_turn(
            [{"role": "user", "content": "go"}],
            reg.manifest(),
            lp.Budgets.load("interactive"),
            "attended",
            dispatch=dispatch,
        )
    ]
    results = [e for e in events if type(e).__name__ == "ToolResultEvent"]
    assert results and results[0].ok is True and results[0].content == "done"


def test_park_tools_are_named_not_hardcoded():
    from tool_module.tools.control import PARK_TOOLS

    assert {"ask", "request_approval"} == PARK_TOOLS
