from __future__ import annotations

import json

from edmo.domain.events.event import DomainEvent


def encode_event(event: DomainEvent) -> bytes:
    """Serialize a DomainEvent into a stable JSON byte string (UTF-8)."""
    return json.dumps(event.to_dict(), separators=(",", ":"), sort_keys=True).encode("utf-8")


def decode_event(payload: bytes) -> DomainEvent:
    """Parse a UTF-8 JSON byte string into a DomainEvent.

    Raises ``ValueError`` when the payload cannot be parsed or is missing
    required fields.
    """
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("payload is not valid UTF-8") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("payload is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("payload JSON must be an object")
    return DomainEvent.from_dict(data)
