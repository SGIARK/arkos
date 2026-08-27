"""The blob backend against a real Supabase Storage bucket, named by STORE_BUCKET."""

from __future__ import annotations

import contextlib
import uuid

import pytest
import pytest_asyncio

from harness_module import blobs, store

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.integration,
    pytest.mark.skipif(
        not (blobs.project_url() and blobs.secret_key()),
        reason="the Supabase project URL or secret key is not configured",
    ),
]


@pytest_asyncio.fixture
async def backend():
    """A live backend that removes whatever the test wrote."""
    made = store.SupabaseBlobs(blobs.project_url(), blobs.secret_key(), blobs.bucket())
    written: list[str] = []
    made.written = written
    yield made
    for content_hash in written:
        with contextlib.suppress(Exception):
            client = made._client_or_new()
            await client.delete(made._url(content_hash), headers=made._headers)
    await made.close()


async def test_a_blob_round_trips_through_the_bucket(backend):
    content = f"buddy store probe {uuid.uuid4()}".encode()
    content_hash = store.sha256(content)

    await backend.put(content_hash, content)
    backend.written.append(content_hash)

    assert await backend.get(content_hash) == content
    assert await backend.missing([content_hash]) == set()


async def test_uploading_the_same_blob_twice_is_accepted(backend):
    content = f"buddy idempotence probe {uuid.uuid4()}".encode()
    content_hash = store.sha256(content)

    await backend.put(content_hash, content)
    await backend.put(content_hash, content)
    backend.written.append(content_hash)

    assert await backend.get(content_hash) == content


async def test_a_hash_that_was_never_written_is_missing(backend):
    never = store.sha256(f"never written {uuid.uuid4()}".encode())

    assert await backend.get(never) is None
    assert await backend.missing([never]) == {never}
