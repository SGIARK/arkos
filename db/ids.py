"""Coercing an id to a UUID, in one place.

Every table in this schema is keyed by UUID and every caller holds a string, so
this coercion was written ten times across the harness and the tool layer — in
three quietly different flavors. Two of them were identical four-liners; the
third swallowed a bad id and returned None. Which one you got depended on which
module you happened to be in.

Two functions, because there are exactly two honest answers to "this is not a
UUID": raise, or record nothing. `api.py` keeps its own thin wrapper on top of
`as_uuid`, because a bad id arriving over HTTP is a 404 with a noun in it, and
that is a routing decision rather than a parsing one.
"""

from __future__ import annotations

import uuid
from typing import Any

__all__ = ["as_uuid", "as_uuid_or_none"]


def as_uuid(value: Any) -> uuid.UUID:
    """Return `value` as a UUID, passing a UUID straight through.

    Raises:
        ValueError: it is not one. The caller decides what that means — a 404
            over HTTP, a refused write, a programming error.
    """
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"not a UUID: {value!r}") from e


def as_uuid_or_none(value: Any) -> uuid.UUID | None:
    """Return `value` as a UUID, or None if it is absent or malformed.

    For the one caller that means it: `system_log` attaches a session id to a
    diagnostic, and a lost diagnostic is better than an exception raised from
    the logging path of a run that is trying to report something else.
    """
    if value is None:
        return None
    try:
        return as_uuid(value)
    except ValueError:
        return None
