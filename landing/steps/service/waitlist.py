"""
Public-facing waitlist intake for the ark landing page.

This is the ONLY thing on the public internet that talks to Postgres, and it is
deliberately tiny and hostile-input-hardened:
  * Uses a LOW-PRIVILEGE role (waitlist_writer) that can only INSERT into
    public.waitlist. It cannot read the list or touch any other table.
  * No read-back anywhere: returns 204 on success AND on duplicate, so there is
    no email-enumeration signal.
  * Honeypot field + per-IP rate limit + strict email validation to blunt bots.
  * CORS locked to the landing origin; body size capped by the model.

NEVER put a Supabase anon/service key in the browser: the waitlist table shares
a database with sensitive ARKOS data. Keep the key server-side, here.
"""
from __future__ import annotations

import asyncio
import os
import re
import time
from collections import defaultdict, deque

import asyncpg
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

DB_URL = os.environ["WAITLIST_DB_URL"]                       # waitlist_writer creds
ORIGIN = os.environ.get("WAITLIST_ORIGIN", "https://ark.mit.edu")
RATE_MAX = int(os.environ.get("WAITLIST_RATE_MAX", "5"))     # requests per window
RATE_WINDOW = int(os.environ.get("WAITLIST_RATE_WINDOW", "60"))  # seconds

# Sanity gate, not full RFC 5322. Rejects the obvious junk without a heavy dep.
_EMAIL = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")

app = FastAPI(title="ark waitlist", docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ORIGIN],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

_pool: asyncpg.Pool | None = None
_hits: dict[str, deque] = defaultdict(deque)
_lock = asyncio.Lock()


class Signup(BaseModel):
    email: str = Field(max_length=254)
    # Honeypot: hidden in the form; real users never fill it, bots do.
    company_website: str | None = Field(default=None, max_length=200)


@app.on_event("startup")
async def _start() -> None:
    global _pool
    _pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=4)


@app.on_event("shutdown")
async def _stop() -> None:
    if _pool:
        await _pool.close()


def _client_ip(r: Request) -> str:
    # Caddy sets X-Forwarded-For; trust only the last hop (our own proxy).
    xff = r.headers.get("x-forwarded-for", "")
    return xff.split(",")[-1].strip() if xff else (r.client.host if r.client else "?")


async def _rate_ok(ip: str) -> bool:
    now = time.time()
    async with _lock:
        q = _hits[ip]
        while q and q[0] <= now - RATE_WINDOW:
            q.popleft()
        if len(q) >= RATE_MAX:
            return False
        q.append(now)
        return True


@app.post("/landing/api/waitlist")
async def waitlist(sig: Signup, request: Request) -> Response:
    if not await _rate_ok(_client_ip(request)):
        return Response(status_code=429)
    if sig.company_website:                       # honeypot tripped: accept + drop
        return Response(status_code=204)
    email = sig.email.strip().lower()
    if not _EMAIL.match(email):
        return Response(status_code=400)
    # Idempotent: no duplicate error, no enumeration.
    await _pool.execute(
        "INSERT INTO public.waitlist (email, source, user_agent) "
        "VALUES ($1, 'landing', $2) ON CONFLICT (email) DO NOTHING",
        email,
        (request.headers.get("user-agent") or "")[:300],
    )
    return Response(status_code=204)


@app.get("/landing/api/health")
async def health() -> dict:
    return {"ok": True}
