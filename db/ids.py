"""Coercing an id to a UUID, in one place."""

from __future__ import annotations

import uuid
from typing import Any

__all__ = ["as_uuid", "as_uuid_or_none"]


def as_uuid(value: Any) -> uuid.UUID:
    """Return `value` as a UUID, passing a UUID straight through.

    Raises:
        ValueError: `value` is not a UUID.
    """
    if isinstance(value, uuid.UUID):
        return value
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError) as e:
        raise ValueError(f"not a UUID: {value!r}") from e


def as_uuid_or_none(value: Any) -> uuid.UUID | None:
    """Return `value` as a UUID, or None if it is absent or malformed."""
    if value is None:
        return None
    try:
        return as_uuid(value)
    except ValueError:
        return None
