"""HTTP routes for the live browser screencast viewer.

A single SSE endpoint that streams the current user's screencast events
out of the in-process frame broker. Events:
  {"type": "started"}                  a new session began
  {"type": "frame", "jpeg_b64": "..."} one screencast frame
  {"type": "ended"}                    the active session finished
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from config_module.loader import config
from tool_module.browser_stream import broker

router = APIRouter()
logger = logging.getLogger(__name__)

# Cadence of the SSE keep-alive comment line. Without periodic bytes some
# proxies/load balancers and even some browsers' EventSource drop idle
# connections after ~30s; the comment is invisible to the consumer.
_KEEPALIVE_SECONDS = 15.0


def _resolve_user_id(request: Request) -> str:
    return (
        request.headers.get("X-User-ID") or request.query_params.get("user_id") or config.get("memory.fallback_user_id")
    )


@router.get("/v1/browser/stream")
async def browser_stream(request: Request) -> StreamingResponse:
    user_id = _resolve_user_id(request)
    logger.info("browser_stream: subscribe user_id=%s", user_id)
    return StreamingResponse(_event_stream(user_id), media_type="text/event-stream")


async def _event_stream(user_id: str):
    """Yield SSE frames for `user_id`.

    Relies on FastAPI/Starlette to cancel this generator when the client
    disconnects — no manual is_disconnected() polling, which we observed
    spuriously firing under some uvicorn versions and killing the stream
    before the first real event. Sends an SSE comment line every
    _KEEPALIVE_SECONDS so intermediaries don't time the connection out.
    """
    sub_iter = broker.subscribe(user_id).__aiter__()
    try:
        while True:
            try:
                event = await asyncio.wait_for(sub_iter.__anext__(), timeout=_KEEPALIVE_SECONDS)
            except TimeoutError:
                # SSE comment line — keeps the connection warm; consumers ignore it.
                yield ": keepalive\n\n"
                continue
            except StopAsyncIteration:
                break
            yield f"data: {json.dumps(event)}\n\n"
    except asyncio.CancelledError:
        # Client disconnected; let the generator close cleanly.
        logger.info("browser_stream: unsubscribe user_id=%s", user_id)
        raise
