"""
The Smithery half: stored connection ids (D24), one ClientSession, TTL, and the
two request-count claims Task 3 is measured by.

Postgres is real here — migration 0 is applied and these are the tables the
redesign ships. Every case works in its own user id, so they do not collide.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio

from db import pool
from tool_module import connections as conns
from tool_module.envelope import ToolContext
from tool_module.smithery import AuthRequiredError, Smithery, _render, _to_specs

pytestmark = pytest.mark.asyncio

LINEAR = "https://linear.run.tools"
BRAVE = "https://brave.run.tools"

SERVERS = {
    "linear": {"mcp_url": LINEAR, "requires_auth": True, "name": "Linear"},
    "brave-search": {"mcp_url": BRAVE, "requires_auth": False, "name": "Brave"},
}
SMITHERY_CFG = {"api_key": "k", "namespace": "arkos-test"}


class FakeClient:
    """Counts what actually goes over the wire, which is the whole point here."""

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
    """user_connections has an FK to users; the id is a Supabase auth sub."""
    await pool.execute("INSERT INTO users (id) VALUES ($1) ON CONFLICT DO NOTHING", uuid.UUID(user_id))


def _hands(client=None, **kw) -> Smithery:
    s = Smithery(SERVERS, SMITHERY_CFG, **kw)
    s.client = client or FakeClient()
    return s


@pytest_asyncio.fixture(autouse=True)
async def _db():
    """
    These assert against the real schema, so they need the real database.

    Skipping beats faking it: the claims here are about what Postgres does with
    a composite primary key and an upsert, which a mock cannot tell us.
    """
    try:
        await pool.fetchval("SELECT 1")
    except Exception as e:  # noqa: BLE001 - any connection failure means skip
        await pool.close()
        pytest.skip(f"needs the arkos database (migration 0 applied): {e}")
    yield
    # Shared rows have no user to scope them, so they have to be swept.
    await pool.execute("DELETE FROM shared_connections")
    # Each test gets its own event loop, and a pool outliving its loop is dead.
    await pool.close()


# --- D24: the id is minted once and stored -----------------------------------


async def test_the_row_is_written_before_the_put():
    """A crash between mint and PUT must leave the id behind, not strand it."""
    user_id = _user()
    await _seed_user(user_id)

    connection_id = await conns.claim(user_id, LINEAR)
    stored = await conns.load(user_id)

    assert stored[LINEAR].connection_id == connection_id
    assert stored[LINEAR].status == conns.PENDING


async def test_a_retry_after_a_crashed_connect_reuses_the_same_id():
    """Minting a second id would strand the first, still holding the OAuth grant."""
    user_id = _user()
    await _seed_user(user_id)

    first = await conns.claim(user_id, LINEAR)
    second = await conns.claim(user_id, LINEAR)

    assert first == second


async def test_renaming_the_config_key_changes_no_row_and_prompts_no_reconnect():
    """The Task 3 acceptance test for D24: config keys are labels, not keys."""
    user_id = _user()
    await _seed_user(user_id)
    client = FakeClient()

    before = _hands(client)
    await before.connect(user_id, "linear")
    puts_after_connect = len(client.puts)

    # Same url, different label. This is the rename that used to disconnect everyone.
    renamed = Smithery({"linear_mcp": dict(SERVERS["linear"])}, SMITHERY_CFG)
    renamed.client = client
    specs = await renamed.specs(user_id)

    rows = await conns.load(user_id)
    assert list(rows) == [LINEAR]
    assert len(client.puts) == puts_after_connect  # no reconnect
    assert [s.name for s in specs] == ["create_issue"]


async def test_the_id_is_not_derived_from_the_config_key():
    """There is no formula, so two urls never collide and a label never appears."""
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
    """A refresh blip must not take away tools the user connected."""
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


async def test_stale_is_measured_from_refreshed_at():
    fresh = conns.Connection(LINEAR, "c1", conns.CONNECTED, [], datetime.now(UTC))
    old = conns.Connection(LINEAR, "c1", conns.CONNECTED, [], datetime.now(UTC) - timedelta(hours=2))
    never = conns.Connection(LINEAR, "c1", conns.CONNECTED, [], None)

    assert not fresh.stale(3600)
    assert old.stale(3600)
    assert never.stale(3600)


# --- auth ---------------------------------------------------------------------


async def test_an_unconnected_server_fails_the_call_instead_of_opening_oauth():
    """The model cannot complete a browser redirect, so it must be told to stop."""
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
    """A remote server does not tell us; guessing wrong runs writes in parallel."""
    specs = _to_specs([{"name": "delete_everything", "description": "d"}])
    assert specs[0].readonly is False


async def test_an_oversized_result_is_stored_and_referenced():
    """Without this the tail is lost: `ref` only ever comes from the envelope."""
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
