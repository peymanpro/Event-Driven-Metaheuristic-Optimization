import pytest

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.event_bus.serialization import decode_event, encode_event


def test_roundtrip() -> None:
    e = DomainEvent(event_type="X", aggregate_id="p1", payload={"k": 1})
    data = encode_event(e)
    e2 = decode_event(data)
    assert e2.event_type == e.event_type
    assert e2.aggregate_id == e.aggregate_id
    assert e2.event_id == e.event_id
    assert dict(e2.payload) == dict(e.payload)


def test_decode_invalid_utf8() -> None:
    with pytest.raises(ValueError):
        decode_event(b"\xff\xfe\xfd")


def test_decode_invalid_json() -> None:
    with pytest.raises(ValueError):
        decode_event(b"not-json")


def test_decode_non_object_json() -> None:
    with pytest.raises(ValueError):
        decode_event(b"[1, 2, 3]")
