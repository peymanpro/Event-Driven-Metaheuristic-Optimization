import pytest

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.event_bus.in_memory import InMemoryTransport


def test_publish_and_drain() -> None:
    t = InMemoryTransport()
    received: list[DomainEvent] = []
    t.subscribe("topic", received.append)
    t.publish("topic", DomainEvent(event_type="X", aggregate_id="p1"))
    assert t.drain("topic") == 1
    assert len(received) == 1


def test_multiple_subscribers_fan_out() -> None:
    t = InMemoryTransport()
    a: list[DomainEvent] = []
    b: list[DomainEvent] = []
    t.subscribe("topic", a.append)
    t.subscribe("topic", b.append)
    t.publish("topic", DomainEvent(event_type="X", aggregate_id="p1"))
    assert t.drain("topic") == 2
    assert len(a) == 1
    assert len(b) == 1


def test_duplicate_event_dropped_per_subscriber() -> None:
    t = InMemoryTransport()
    received: list[DomainEvent] = []
    t.subscribe("topic", received.append)
    e = DomainEvent(event_type="X", aggregate_id="p1")
    t.publish("topic", e)
    t.publish("topic", e)
    assert t.drain("topic") == 1
    assert len(received) == 1


def test_publish_to_unknown_topic_is_noop() -> None:
    t = InMemoryTransport()
    t.publish("nobody-cares", DomainEvent(event_type="X", aggregate_id="p1"))
    assert t.drain("nobody-cares") == 0


def test_close_prevents_publish() -> None:
    t = InMemoryTransport()
    t.close()
    with pytest.raises(RuntimeError):
        t.publish("topic", DomainEvent(event_type="X", aggregate_id="p1"))


def test_subscribe_rejects_empty_topic() -> None:
    t = InMemoryTransport()
    with pytest.raises(ValueError):
        t.subscribe("", lambda e: None)


def test_publish_rejects_empty_topic() -> None:
    t = InMemoryTransport()
    with pytest.raises(ValueError):
        t.publish("", DomainEvent(event_type="X", aggregate_id="p1"))
