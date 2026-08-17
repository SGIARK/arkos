"""SSE route for the live browser screencast viewer.

Streams the user's `started`, `frame` and `ended` events from the in-process broker.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from tool_module.browser_stream import broker

router = APIRouter()
logger = logging.getLogger(__name__)

# Proxies and EventSource drop idle connections after ~30s without bytes.
_KEEPALIVE_SECONDS = 15.0


def _resolve_user_id(request: Request) -> str | None:
    """Read the caller's claimed id.

    Trusts the client, which is a listed contract violation, so `api.py` does
    not mount this router. The replacement is a cookie-authed
    `GET /sessions/{id}/browser/frames` keyed by (user, session).
    """
    return request.headers.get("X-User-ID") or request.query_params.get("user_id")


@router.get("/v1/browser/stream")
async def browser_stream(request: Request) -> StreamingResponse:
    user_id = _resolve_user_id(request)
    logger.info("browser_stream: subscribe user_id=%s", user_id)
    return StreamingResponse(_event_stream(user_id), media_type="text/event-stream")


async def _event_stream(user_id: str):
    """Yield SSE frames for `user_id` until the client disconnects."""
    sub_iter = broker.subscribe(user_id).__aiter__()
    try:
        while True:
            try:
                event = await asyncio.wait_for(sub_iter.__anext__(), timeout=_KEEPALIVE_SECONDS)
            except TimeoutError:
                yield ": keepalive\n\n"
                continue
            except StopAsyncIteration:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except asyncio.CancelledError:
        # Starlette cancels this generator when the client disconnects.
        logger.info("browser_stream: unsubscribe user_id=%s", user_id)
        raise
