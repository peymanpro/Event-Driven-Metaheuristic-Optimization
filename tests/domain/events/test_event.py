from datetime import UTC, datetime

import pytest

from edmo.domain.events.event import DomainEvent


def test_event_defaults() -> None:
    e = DomainEvent(event_type="JobAdded", aggregate_id="p1")
    assert e.version == 1
    assert e.timestamp.tzinfo is not None
    assert e.payload == {}


def test_event_rejects_empty_type() -> None:
    with pytest.raises(ValueError):
        DomainEvent(event_type="", aggregate_id="p1")


def test_event_rejects_empty_aggregate() -> None:
    with pytest.raises(ValueError):
        DomainEvent(event_type="JobAdded", aggregate_id="")


def test_event_rejects_bad_version() -> None:
    with pytest.raises(ValueError):
        DomainEvent(event_type="JobAdded", aggregate_id="p1", version=0)


def test_event_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError):
        DomainEvent(
            event_type="JobAdded",
            aggregate_id="p1",
            timestamp=datetime(2026, 1, 1),
        )


def test_event_payload_is_immutable() -> None:
    e = DomainEvent(event_type="JobAdded", aggregate_id="p1", payload={"k": 1})
    with pytest.raises(TypeError):
        e.payload["k"] = 2


def test_event_roundtrip_dict() -> None:
    e = DomainEvent(
        event_type="JobAdded",
        aggregate_id="p1",
        payload={"job_id": "j1"},
        version=3,
    )
    d = e.to_dict()
    e2 = DomainEvent.from_dict(d)
    assert e2.event_type == e.event_type
    assert e2.aggregate_id == e.aggregate_id
    assert e2.version == e.version
    assert e2.event_id == e.event_id
    assert dict(e2.payload) == dict(e.payload)


def test_event_from_dict_missing_fields() -> None:
    with pytest.raises(ValueError):
        DomainEvent.from_dict({"event_type": "JobAdded"})


def test_event_accepts_timezone_aware() -> None:
    ts = datetime(2026, 1, 1, tzinfo=UTC)
    e = DomainEvent(event_type="JobAdded", aggregate_id="p1", timestamp=ts)
    assert e.timestamp == ts
