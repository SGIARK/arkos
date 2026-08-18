"""The blob backend against a real Supabase Storage bucket.

Marked `integration`: it needs SUPABASE_SECRET_KEY and a private bucket named by
STORE_BUCKET; the project URL is derived from DB_URL. Run with
`pytest -m integration`.

The mock-transport tests in test_store.py pin the HTTP contract. What only this
can show is that the bucket exists, the key is accepted, and a blob survives a
round trip.
"""

from __future__ import annotations

import os
import uuid

import pytest

from harness_module import store

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not (store.project_url() and store.secret_key()),
        reason="the Supabase project URL or secret key is not configured",
    ),
]


@pytest.fixture
def backend():
    made = store.SupabaseBlobs(
        store.project_url(),
        store.secret_key(),
        os.environ.get("STORE_BUCKET") or "arkos",
    )
    yield made


async def test_a_blob_round_trips_through_the_bucket(backend):
    content = f"arkos store probe {uuid.uuid4()}".encode()
    content_hash = store.sha256(content)

    await backend.put(content_hash, content)

    assert await backend.get(content_hash) == content
    assert await backend.missing([content_hash]) == set()
    await backend.close()


async def test_uploading_the_same_blob_twice_is_accepted(backend):
    content = f"arkos idempotence probe {uuid.uuid4()}".encode()
    content_hash = store.sha256(content)

    await backend.put(content_hash, content)
    await backend.put(content_hash, content)

    assert await backend.get(content_hash) == content
    await backend.close()


async def test_a_hash_that_was_never_written_is_missing(backend):
    never = store.sha256(f"never written {uuid.uuid4()}".encode())

    assert await backend.get(never) is None
    assert await backend.missing([never]) == {never}
    await backend.close()
