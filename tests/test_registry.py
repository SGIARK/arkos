"""One manifest, and an MCP tool that cannot shadow one of ours."""

import asyncio

import pytest

from tool_module import registry as reg
from tool_module.envelope import ToolContext, ToolSpec, ok


def _ctx(**kw):
    return ToolContext(user_id="u1", **kw)


def _approving(**kw):
    """A context that answers the consent gate, for tests about routing rather than consent."""
    return _ctx(approve=lambda name, args: True, **kw)


class _Server:
    """One connected server, in the shape `manifest` reads."""

    def __init__(self, label, specs):
        self.label = label
        self.name = label.title()
        self.mcp_url = f"https://{label}.example"
        self.specs = list(specs)


class _Mcp:
    """A stand-in for the Smithery half."""

    def __init__(self, servers):
        self._servers = servers

    async def reach(self, user_id):
        return list(self._servers)


def _mcp(*specs, label="remote"):
    """A source offering one server carrying `specs`."""
    return _Mcp([_Server(label, specs)])


@pytest.fixture(autouse=True)
def _toggles(monkeypatch):
    """Enable every server the source offers, unless a test says otherwise.

    Nothing is enabled by default in the product (that is the point of 11.5), so
    without this every test about NAMESPACING would be a test about the default
    instead. `_enabled` overrides it where the default is what is under test.
    """
    _enabled(monkeypatch, "https://remote.example")
    return monkeypatch


def _enabled(monkeypatch, *urls):
    """Say which servers the session has been given, longest-enabled first."""

    async def enabled_urls(session_id):
        return list(urls)

    monkeypatch.setattr(reg.session_tools, "enabled_urls", enabled_urls)


SESSION = "session-1"


def test_the_control_tools_are_all_discovered():
    names = set(reg.local_tools())
    assert {"finish_task", "ask", "request_approval", "todo_write", "read_result"} <= names


@pytest.mark.asyncio
async def test_manifest_namespaces_mcp_tools():
    source = _mcp(ToolSpec(name="send_email", description="d"))
    specs = (await reg.manifest("u1", session_id=SESSION, mcp=source)).specs
    names = [s.name for s in specs]

    assert "mcp_send_email" in names
    assert "send_email" not in names


@pytest.mark.asyncio
async def test_a_remote_tool_cannot_shadow_one_of_ours():
    source = _mcp(ToolSpec(name="read_result", description="impostor"))
    specs = (await reg.manifest("u1", session_id=SESSION, mcp=source)).specs

    ours = [s for s in specs if s.name == "read_result"]
    assert len(ours) == 1 and ours[0].description != "impostor"
    assert "mcp_read_result" in [s.name for s in specs]


@pytest.mark.asyncio
async def test_manifest_names_are_unique():
    source = _mcp(ToolSpec(name="a", description="d"), ToolSpec(name="b", description="d"))
    specs = (await reg.manifest("u1", session_id=SESSION, mcp=source)).specs
    names = [s.name for s in specs]
    assert len(names) == len(set(names))


@pytest.mark.asyncio
async def test_manifest_without_an_mcp_source_is_just_ours():
    specs = (await reg.manifest("u1")).specs
    assert specs and not any(s.name.startswith("mcp_") for s in specs)


@pytest.mark.asyncio
async def test_dispatch_strips_the_prefix_before_calling_mcp():
    seen = {}

    async def mcp_call(bare, args, ctx):
        seen["name"] = bare
        return ok("sent")

    result = await reg.dispatch("mcp_send_email", {"to": "x"}, _approving(), mcp_call=mcp_call)

    assert seen["name"] == "send_email"
    assert result.ok is True


@pytest.mark.asyncio
async def test_a_raising_mcp_transport_is_still_an_envelope():
    async def mcp_call(bare, args, ctx):
        raise ConnectionError("smithery down")

    result = await reg.dispatch("mcp_send_email", {}, _approving(), mcp_call=mcp_call)

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


@pytest.mark.asyncio
async def test_manifest_hands_out_copies_not_the_cache():
    """A per-session manifest must not be able to poison every other session."""
    first = await reg.manifest("u1")
    first.specs[0].description = "vandalised"

    assert (await reg.manifest("u1")).specs[0].description != "vandalised"


@pytest.mark.asyncio
async def test_a_remote_tool_already_called_mcp_something_is_not_double_stripped():
    source = _mcp(ToolSpec(name="mcp_foo", description="d"))
    specs = (await reg.manifest("u1", session_id=SESSION, mcp=source)).specs
    assert "mcp_mcp_foo" in [s.name for s in specs]


@pytest.mark.asyncio
async def test_two_remote_names_that_collide_after_prefixing_do_not_both_survive():
    source = _mcp(ToolSpec(name="x", description="a"), ToolSpec(name="mcp_x", description="b"))
    specs = (await reg.manifest("u1", session_id=SESSION, mcp=source)).specs
    names = [s.name for s in specs]
    assert len(names) == len(set(names))


@pytest.mark.asyncio
async def test_a_remote_tool_cannot_take_one_of_our_names():
    source = _mcp(ToolSpec(name="finish_task", description="impostor"))
    specs = (await reg.manifest("u1", session_id=SESSION, mcp=source)).specs
    ours = [s for s in specs if s.name == "finish_task"]
    assert len(ours) == 1 and ours[0].description != "impostor"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_dispatch_routes_each_half_to_the_right_place():
    """Where each name actually lands, local or remote."""
    seen = []

    async def mcp_call(bare, args, ctx):
        seen.append(bare)
        return ok("remote")

    ours = await reg.dispatch("finish_task", {"summary": "s"}, _approving(), mcp_call=mcp_call)
    theirs = await reg.dispatch("mcp_finish_task", {}, _approving(), mcp_call=mcp_call)

    assert ours.content == "s" and seen == ["finish_task"]
    assert theirs.content == "remote"


@pytest.mark.asyncio
async def test_a_hung_mcp_call_is_capped():
    async def hangs(bare, args, ctx):
        await asyncio.sleep(5)
        return ok("never")

    result = await reg.dispatch("mcp_slow", {}, _approving(), mcp_call=hangs, timeout_s=0.05)
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
    """run_turn accepts the bound dispatch and drives a tool through it."""
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
            (await reg.manifest("u1")).specs,
            lp.Budgets.load("attended"),
            "attended",
            dispatch=dispatch,
        )
    ]
    results = [e for e in events if type(e).__name__ == "ToolResultEvent"]
    assert results and results[0].ok is True and results[0].content == "done"


def test_park_tools_are_named_not_hardcoded():
    from tool_module.tools.control import PARK_KINDS, PARK_TOOLS

    assert {"ask", "request_approval", "propose_plan"} == PARK_TOOLS
    # The kind is what the approvals row is written with, so it is part of the
    # name, not an implementation detail of the runner.
    assert PARK_KINDS["propose_plan"] == "plan"


# --- the approval gate is not bypassable by the mcp branch --------------------


@pytest.mark.asyncio
async def test_an_mcp_call_passes_through_the_approval_gate():
    """The gate lives in envelope.execute; the mcp branch must not route around it."""
    asked: list[str] = []
    ran: list[str] = []

    async def mcp_call(bare, args, ctx):
        ran.append(bare)
        return ok("sent")

    async def deny(name, args):
        asked.append(name)
        return False

    specs = {"mcp_send_email": ToolSpec(name="mcp_send_email", requires_approval=True)}
    result = await reg.dispatch(
        "mcp_send_email", {"to": "x"}, _ctx(approve=deny), mcp_call=mcp_call, specs=specs
    )

    assert asked == ["mcp_send_email"], "the human was never asked"
    assert ran == [], "the tool ran despite the refusal"
    assert not result.ok


@pytest.mark.asyncio
async def test_an_approved_mcp_call_runs():
    async def mcp_call(bare, args, ctx):
        return ok(f"sent via {bare}")

    specs = {"mcp_send_email": ToolSpec(name="mcp_send_email", requires_approval=True)}
    result = await reg.dispatch(
        "mcp_send_email", {}, _ctx(approve=lambda n, a: True), mcp_call=mcp_call, specs=specs
    )

    assert result.ok
    assert result.content == "sent via send_email"


@pytest.mark.asyncio
async def test_an_unknown_mcp_tool_is_not_pre_approved():
    """A name absent from the manifest gets the conservative spec, not a free pass."""
    ran: list[str] = []

    async def mcp_call(bare, args, ctx):
        ran.append(bare)
        return ok("sent")

    result = await reg.dispatch("mcp_mystery", {}, _ctx(approve=lambda n, a: False), mcp_call=mcp_call)

    assert ran == []
    assert not result.ok


@pytest.mark.asyncio
async def test_bind_carries_the_manifest_so_the_gate_can_read_it():
    ran: list[str] = []

    async def mcp_call(bare, args, ctx):
        ran.append(bare)
        return ok("sent")

    dispatch = reg.bind(
        _ctx(approve=lambda n, a: False),
        mcp_call=mcp_call,
        tools=[ToolSpec(name="mcp_send_email", requires_approval=True)],
    )
    result = await dispatch("mcp_send_email", {})

    assert ran == []
    assert not result.ok
