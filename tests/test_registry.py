"""One manifest, and an MCP tool that cannot shadow one of ours."""

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
