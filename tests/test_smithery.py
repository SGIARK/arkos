"""Stored connection ids, one ClientSession, TTL, and the request-count claims.

Runs against a real Postgres with migration 0 applied; each case uses its own user id.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from db import pool
from tests.dbgate import require_db
from tool_module import connections as conns
from tool_module.envelope import ToolContext
from tool_module.smithery import AuthRequiredError, Smithery, _render, _to_spec

pytestmark = pytest.mark.asyncio

LINEAR = "https://linear.run.tools"
BRAVE = "https://brave.run.tools"

SERVERS = {
    "linear": {"mcp_url": LINEAR, "requires_auth": True, "name": "Linear"},
    "brave-search": {"mcp_url": BRAVE, "requires_auth": False, "name": "Brave"},
}
SMITHERY_CFG = {"api_key": "k", "namespace": "arkos-test"}


class FakeClient:
    """Counts what goes over the wire."""

    def __init__(self, *, status="connected", tools=None, setup_url=None):
        self.status = status
        self.tools = tools if tools is not None else [{"name": "create_issue", "description": "d"}]
        self.setup_url = setup_url
        self.puts: list[str] = []
        self.rpcs: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.call_result = {"content": [{"type": "text", "text": "ok"}]}

    async def upsert(self, connection_id, mcp_url, **kw):
        self.puts.append(connection_id)
        status = self.status if isinstance(self.status, str) else self.status
        if status == "connected":
            return {"status": "connected"}
        return {"status": {"state": status, "setupUrl": self.setup_url}}

    async def jsonrpc(self, connection_id, method, params=None):
        self.rpcs.append((connection_id, method))
        if method == "tools/list":
            return {"tools": self.tools}
        return self.call_result

    async def delete(self, connection_id):
        self.deleted.append(connection_id)

    async def close(self):
        pass


def _user() -> str:
    return str(uuid.uuid4())


async def _seed_user(user_id: str) -> None:
    """user_connections has an FK to users."""
    await pool.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", uuid.UUID(user_id))


def _hands(client=None, **kw) -> Smithery:
    s = Smithery(SERVERS, SMITHERY_CFG, **kw)
    s.client = client or FakeClient()
    return s


@pytest_asyncio.fixture(autouse=True)
async def _db():
    """Skip the module unless the real schema is reachable."""
    await require_db()
    yield
    # Shared rows have no user to scope them, so they have to be swept.
    await pool.execute("DELETE FROM shared_connections")
    # Each test gets its own event loop, and a pool outliving its loop is dead.
    await pool.close()


# --- the id is minted once and stored ----------------------------------------


async def test_the_row_is_written_before_the_put():
    user_id = _user()
    await _seed_user(user_id)

    connection_id = await conns.claim(user_id, LINEAR)
    stored = await conns.load(user_id)

    assert stored[LINEAR].connection_id == connection_id
    assert stored[LINEAR].status == conns.PENDING


async def test_a_retry_after_a_crashed_connect_reuses_the_same_id():
    user_id = _user()
    await _seed_user(user_id)

    first = await conns.claim(user_id, LINEAR)
    second = await conns.claim(user_id, LINEAR)

    assert first == second


async def test_renaming_the_config_key_changes_no_row_and_prompts_no_reconnect():
    """Config keys are labels, not identities."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()

    before = _hands(client)
    await before.connect(user_id, "linear")
    puts_after_connect = len(client.puts)

    # Same url, different label.
    renamed = Smithery({"linear_mcp": dict(SERVERS["linear"])}, SMITHERY_CFG)
    renamed.client = client
    specs = await renamed.specs(user_id)

    rows = await conns.load(user_id)
    assert list(rows) == [LINEAR]
    assert len(client.puts) == puts_after_connect  # no reconnect
    assert [s.name for s in specs] == ["create_issue"]


async def test_the_id_is_not_derived_from_the_config_key():
    a = conns.mint_id(LINEAR)
    b = conns.mint_id(LINEAR)

    assert a != b
    assert "linear-run-tools" in a  # recognisable in Smithery's dashboard


# --- the two request-count claims --------------------------------------------


async def test_a_warm_per_user_tool_call_is_exactly_one_http_request():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    await hands.connect(user_id, "linear")
    await hands.specs(user_id)  # warm
    client.puts.clear()
    client.rpcs.clear()

    result = await hands.call("create_issue", {"title": "x"}, ToolContext(user_id=user_id))

    assert result.ok is True
    assert client.puts == []
    assert [m for _, m in client.rpcs] == ["tools/call"]


async def test_a_restart_costs_one_db_read_and_zero_smithery_puts():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()

    await _hands(client).connect(user_id, "linear")

    # A fresh process: new caches, same database.
    restarted = _hands(client)
    client.puts.clear()
    client.rpcs.clear()
    specs = await restarted.specs(user_id)

    assert [s.name for s in specs] == ["create_issue"]
    assert client.puts == []


async def test_initialize_shared_does_not_reconnect_what_is_already_connected():
    client = FakeClient()
    await _hands(client).initialize_shared()
    assert len(client.puts) == 1

    client.puts.clear()
    await _hands(client).initialize_shared()
    assert client.puts == []


# --- TTL ---------------------------------------------------------------------


async def test_a_stale_tool_list_is_revalidated_without_a_put():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)
    await hands.connect(user_id, "linear")

    await pool.execute(
        "UPDATE user_connections SET refreshed_at = now() - interval '2 days' WHERE user_id = $1",
        uuid.UUID(user_id),
    )
    client.tools = [{"name": "create_issue"}, {"name": "list_issues"}]
    client.puts.clear()
    client.rpcs.clear()

    specs = await _hands(client).specs(user_id)

    assert {s.name for s in specs} == {"create_issue", "list_issues"}
    assert client.puts == []
    assert [m for _, m in client.rpcs] == ["tools/list"]


async def test_a_fresh_tool_list_is_not_revalidated():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)
    await hands.connect(user_id, "linear")
    client.rpcs.clear()

    await hands.specs(user_id)

    assert client.rpcs == []


async def test_a_failed_revalidation_keeps_the_tools_we_had():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    await _hands(client).connect(user_id, "linear")
    await pool.execute(
        "UPDATE user_connections SET refreshed_at = now() - interval '2 days' WHERE user_id = $1",
        uuid.UUID(user_id),
    )

    from tool_module.smithery import SmitheryError

    class Flaky(FakeClient):
        async def jsonrpc(self, connection_id, method, params=None):
            raise SmitheryError("tools/list 503")

    specs = await _hands(Flaky()).specs(user_id)
    assert [s.name for s in specs] == ["create_issue"]


async def test_mcp_tools_require_approval_unless_config_waives_it():
    """The unknown resolves safely: unattended runs do not get silent write access."""
    user_id = _user()
    await _seed_user(user_id)
    await _hands(FakeClient()).connect(user_id, "linear")

    default = await _hands(FakeClient()).specs(user_id)
    assert [s.requires_approval for s in default] == [True]
    assert [s.readonly for s in default] == [False]

    whole_server = _hands(FakeClient())
    whole_server.servers["linear"]["auto_approve"] = True
    assert [s.requires_approval for s in await whole_server.specs(user_id)] == [False]

    by_name = _hands(FakeClient())
    by_name.servers["linear"]["auto_approve"] = ["create_issue"]
    assert [s.requires_approval for s in await by_name.specs(user_id)] == [False]

    other = _hands(FakeClient())
    other.servers["linear"]["auto_approve"] = ["something_else"]
    assert [s.requires_approval for s in await other.specs(user_id)] == [True]


async def test_a_revoked_grant_found_while_refreshing_is_recorded():
    """Revocation must surface within one TTL, not wait for someone to call a tool."""
    user_id = _user()
    await _seed_user(user_id)
    await _hands(FakeClient()).connect(user_id, "linear")
    await pool.execute(
        "UPDATE user_connections SET refreshed_at = now() - interval '2 days' WHERE user_id = $1",
        uuid.UUID(user_id),
    )

    class Revoked(FakeClient):
        async def jsonrpc(self, connection_id, method, params=None):
            raise AuthRequiredError(service="linear", setup_url="https://smithery/setup/linear")

    hands = _hands(Revoked())
    specs = await hands.specs(user_id)

    row = await pool.fetchrow(
        "SELECT status, tools_cache FROM user_connections WHERE user_id = $1",
        uuid.UUID(user_id),
    )
    assert row["status"] == "auth_required"
    # Dropped from the manifest: the model is not offered a tool it cannot use.
    assert specs == []
    # tools_cache survives, so a call held over from an earlier manifest still
    # resolves to this connection and earns auth_required rather than not_found.
    assert row["tools_cache"] is not None
    result = await hands.call("create_issue", {}, ToolContext(user_id=user_id))
    assert result.error_kind == "auth_required"
    assert "smithery/setup/linear" in result.content


async def test_stale_is_measured_from_refreshed_at():
    fresh = conns.Connection(LINEAR, "c1", conns.CONNECTED, [], datetime.now(UTC))
    old = conns.Connection(LINEAR, "c1", conns.CONNECTED, [], datetime.now(UTC) - timedelta(hours=2))
    never = conns.Connection(LINEAR, "c1", conns.CONNECTED, [], None)

    assert not fresh.stale(3600)
    assert old.stale(3600)
    assert never.stale(3600)


async def test_revalidating_re_arms_the_clock_in_memory_not_just_the_db():
    """The same instance must stop re-fetching once a revalidation succeeds."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)
    await hands.connect(user_id, "linear")
    await pool.execute(
        "UPDATE user_connections SET refreshed_at = now() - interval '2 days' WHERE user_id = $1",
        uuid.UUID(user_id),
    )
    hands._invalidate(user_id)  # force a reload of the now-stale row
    client.rpcs.clear()  # connect() did its own tools/list

    await hands.specs(user_id)
    lists_after_first = [m for _, m in client.rpcs].count("tools/list")
    await hands.specs(user_id)
    await hands.specs(user_id)

    assert lists_after_first == 1
    assert [m for _, m in client.rpcs].count("tools/list") == 1, "TTL never re-armed"


async def test_a_failed_revalidation_also_re_arms_so_a_dead_server_is_not_retried_forever():
    user_id = _user()
    await _seed_user(user_id)
    await _hands(FakeClient()).connect(user_id, "linear")
    await pool.execute(
        "UPDATE user_connections SET refreshed_at = now() - interval '2 days' WHERE user_id = $1",
        uuid.UUID(user_id),
    )

    from tool_module.smithery import SmitheryError

    class Down(FakeClient):
        async def jsonrpc(self, connection_id, method, params=None):
            self.rpcs.append((connection_id, method))
            raise SmitheryError("tools/list 503")

    down = Down()
    hands = _hands(down)
    await hands.specs(user_id)
    await hands.specs(user_id)

    assert [m for _, m in down.rpcs].count("tools/list") == 1


async def test_concurrent_loads_do_not_install_a_stale_snapshot():
    """A load in flight during an invalidation must not install its old snapshot."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    reader = asyncio.create_task(hands.specs(user_id))
    await asyncio.sleep(0)  # let the read get in flight
    await hands.connect(user_id, "linear")
    await reader

    assert [s.name for s in await hands.specs(user_id)] == ["create_issue"]


async def test_the_manifest_and_dispatch_agree_on_which_server_owns_a_name():
    """The schema the model sees and the connection the call lands on are the same."""
    user_id = _user()
    await _seed_user(user_id)
    servers = {
        "linear": {"mcp_url": LINEAR, "requires_auth": True, "name": "Linear"},
        "brave-search": {"mcp_url": BRAVE, "requires_auth": False, "name": "Brave"},
    }
    shared = FakeClient(tools=[{"name": "search", "description": "SHARED"}])
    hands = Smithery(servers, SMITHERY_CFG)
    hands.client = shared
    await hands.initialize_shared()

    mine = FakeClient(tools=[{"name": "search", "description": "MINE"}])
    hands.client = mine
    await hands.connect(user_id, "linear")

    specs = await hands.specs(user_id)
    first = next(s for s in specs if s.name == "search")
    route = (await hands._routes(user_id))["search"]

    assert first.description == "MINE"
    assert route.owner == user_id and route.conn.mcp_url == LINEAR


async def test_a_per_user_server_without_a_user_is_refused_not_written_as_shared():
    from tool_module.smithery import SmitheryError

    hands = _hands(FakeClient())
    with pytest.raises(SmitheryError, match="requires per-user auth"):
        await hands.connect(None, "linear")

    assert await conns.load(None) == {}


# --- auth ---------------------------------------------------------------------


async def test_a_never_connected_server_does_not_open_oauth_on_a_tool_call():
    """A call must never PUT, and with no tool list the name is simply not found."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient(status="auth_required", setup_url="https://smithery/auth/x")
    hands = _hands(client)

    with pytest.raises(AuthRequiredError):
        await hands.connect(user_id, "linear")

    client.puts.clear()
    result = await hands.call("create_issue", {}, ToolContext(user_id=user_id))

    assert result.ok is False and result.error_kind == "not_found"
    assert client.puts == []  # a tool call never PUTs


async def test_a_pending_connection_is_left_in_the_table_holding_its_id():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient(status="auth_required", setup_url="https://smithery/auth/x")

    with pytest.raises(AuthRequiredError):
        await _hands(client).connect(user_id, "linear")

    rows = await conns.load(user_id)
    assert rows[LINEAR].connection_id == client.puts[0]
    assert rows[LINEAR].status == "auth_required"


async def test_a_dead_grant_mid_call_marks_the_connection_and_stops_retrying():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)
    await hands.connect(user_id, "linear")

    class Revoked(FakeClient):
        async def jsonrpc(self, connection_id, method, params=None):
            raise AuthRequiredError("linear", setup_url="https://smithery/reauth")

    hands.client = Revoked()
    result = await hands.call("create_issue", {}, ToolContext(user_id=user_id))

    assert result.error_kind == "auth_required" and result.retryable is False
    rows = await conns.load(user_id)
    assert rows[LINEAR].status == "auth_required"
    # The names must survive the status change for the next call to resolve.
    assert [t["name"] for t in rows[LINEAR].tools] == ["create_issue"]


async def test_the_call_after_a_dead_grant_still_says_auth_required():
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands(FakeClient())
    await hands.connect(user_id, "linear")

    class Revoked(FakeClient):
        async def jsonrpc(self, connection_id, method, params=None):
            raise AuthRequiredError("linear", setup_url="https://smithery/reauth")

    hands.client = Revoked()
    first = await hands.call("create_issue", {}, ToolContext(user_id=user_id))
    second = await hands.call("create_issue", {}, ToolContext(user_id=user_id))

    assert first.error_kind == "auth_required"
    assert second.error_kind == "auth_required", "the model lost the setup URL on the retry"
    assert "https://smithery/reauth" in second.content


async def test_a_db_failure_recording_a_dead_grant_still_says_do_not_retry():
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands(FakeClient())
    await hands.connect(user_id, "linear")

    class Revoked(FakeClient):
        async def jsonrpc(self, connection_id, method, params=None):
            raise AuthRequiredError("linear", setup_url="https://smithery/reauth")

    hands.client = Revoked()

    async def boom(*a, **kw):
        raise RuntimeError("connection pool exhausted")

    import tool_module.smithery as sm

    original = sm.conns.set_status
    sm.conns.set_status = boom
    try:
        result = await hands.call("create_issue", {}, ToolContext(user_id=user_id))
    finally:
        sm.conns.set_status = original

    assert result.error_kind == "auth_required" and result.retryable is False


async def test_disconnect_revokes_at_smithery_and_drops_the_row():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)
    await hands.connect(user_id, "linear")

    await hands.disconnect(user_id, "linear")

    assert client.deleted == client.puts
    assert await conns.load(user_id) == {}


# --- isolation ----------------------------------------------------------------


async def test_one_users_connection_is_never_visible_to_another():
    a, b = _user(), _user()
    await _seed_user(a)
    await _seed_user(b)
    client = FakeClient()
    await _hands(client).connect(a, "linear")

    specs = await _hands(client).specs(b)

    assert [s.name for s in specs] == []


async def test_a_tool_another_user_connected_is_not_callable():
    a, b = _user(), _user()
    await _seed_user(a)
    await _seed_user(b)
    client = FakeClient()
    hands = _hands(client)
    await hands.connect(a, "linear")

    result = await hands.call("create_issue", {}, ToolContext(user_id=b))

    assert result.error_kind == "not_found"


async def test_shared_servers_are_reachable_by_everyone():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient(tools=[{"name": "brave_search"}])
    hands = _hands(client)
    await hands.initialize_shared()

    result = await hands.call("brave_search", {"q": "x"}, ToolContext(user_id=user_id))

    assert result.ok is True


# --- envelope shape -----------------------------------------------------------


async def test_an_mcp_error_result_becomes_a_failed_envelope():
    text, is_error = _render({"content": [{"type": "text", "text": "boom"}], "isError": True})
    assert is_error and text == "boom"


async def test_content_blocks_are_flattened_not_dropped():
    text, _ = _render({"content": [{"type": "text", "text": "a"}, {"type": "image", "data": "z"}]})
    assert "a" in text and "image" in text


async def test_remote_tools_are_never_marked_readonly():
    assert _to_spec({"name": "delete_everything", "description": "d"}).readonly is False


async def test_an_oversized_result_is_stored_and_referenced():
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    client.call_result = {"content": [{"type": "text", "text": "x" * 9000}]}
    hands = _hands(client)
    await hands.connect(user_id, "linear")

    stored = {}

    async def store_blob(text):
        stored["text"] = text
        return "b_1"

    result = await hands.call("create_issue", {}, ToolContext(user_id=user_id, store_blob=store_blob))

    assert result.ref == "b_1"
    assert stored["text"] == "x" * 9000
    assert len(result.content) < 9000 and "read_result" in result.content


# --- one ClientSession --------------------------------------------------------


async def test_the_client_reuses_one_session():
    from tool_module.smithery import SmitheryClient

    client = SmitheryClient(api_key="k", namespace="n")
    first = await client.session()
    second = await client.session()

    assert first is second
    await client.close()
    assert first.closed


async def test_a_dead_grant_survives_a_restart_as_auth_required_not_not_found():
    """Contracts: auth_required is actionable. A fresh process must still resolve the name."""
    user_id = _user()
    await _seed_user(user_id)
    await _hands(FakeClient()).connect(user_id, "linear")

    class Revoked(FakeClient):
        async def jsonrpc(self, connection_id, method, params=None):
            raise AuthRequiredError("linear", setup_url="https://smithery/reauth")

    live = _hands(Revoked())
    assert (await live.call("create_issue", {}, ToolContext(user_id=user_id))).error_kind == "auth_required"

    # A fresh process: no in-memory cache, no remembered setup URL, same rows.
    restarted = _hands(FakeClient())
    result = await restarted.call("create_issue", {}, ToolContext(user_id=user_id))

    assert result.error_kind == "auth_required", "rehydration lost the name and degraded to not_found"
    assert result.retryable is False
    assert "Linear" in result.content and "Settings" in result.content
    # The manifest still hides it, so the model is not invited to call it again.
    assert await restarted.specs(user_id) == []


async def test_reachable_tools_converge_to_non_empty_under_interleaved_invalidation():
    """Invariant: however invalidate and refill interleave, a connected server ends up visible."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()
    hands = _hands(client)

    for delay in (0, 0.001, 0.005):
        hands._invalidate(user_id)
        readers = [asyncio.create_task(hands.specs(user_id)) for _ in range(4)]
        await asyncio.sleep(delay)
        hands._invalidate(user_id)
        await hands.connect(user_id, "linear")
        await asyncio.gather(*readers)

        assert [s.name for s in await hands.specs(user_id)] == ["create_issue"], f"blanked at delay={delay}"


async def test_two_concurrent_misses_do_not_both_query_the_database():
    user_id = _user()
    await _seed_user(user_id)
    hands = _hands(FakeClient())
    await hands.connect(user_id, "linear")
    hands._invalidate(user_id)

    calls = []
    original = conns.load

    async def counted(owner):
        calls.append(owner)
        return await original(owner)

    import tool_module.smithery as sm

    sm.conns.load = counted
    try:
        await asyncio.gather(hands._load(user_id), hands._load(user_id), hands._load(user_id))
    finally:
        sm.conns.load = original

    assert calls.count(user_id) == 1
