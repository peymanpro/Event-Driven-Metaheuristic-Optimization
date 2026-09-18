import pytest

from edmo.domain.events.bus import InMemoryEventBus
from edmo.domain.events.event import DomainEvent


def test_subscribe_and_publish() -> None:
    bus = InMemoryEventBus()
    received: list[DomainEvent] = []
    bus.subscribe("JobAdded", received.append)
    n = bus.publish(DomainEvent(event_type="JobAdded", aggregate_id="p1"))
    assert n == 1
    assert len(received) == 1


def test_publish_ignores_other_types() -> None:
    bus = InMemoryEventBus()
    received: list[DomainEvent] = []
    bus.subscribe("JobAdded", received.append)
    n = bus.publish(DomainEvent(event_type="JobRemoved", aggregate_id="p1"))
    assert n == 0
    assert received == []


def test_multiple_handlers_run_in_order() -> None:
    bus = InMemoryEventBus()
    order: list[int] = []
    bus.subscribe("X", lambda e: order.append(1))
    bus.subscribe("X", lambda e: order.append(2))
    bus.publish(DomainEvent(event_type="X", aggregate_id="p1"))
    assert order == [1, 2]


def test_unsubscribe() -> None:
    bus = InMemoryEventBus()
    received: list[DomainEvent] = []
    handler = received.append
    bus.subscribe("X", handler)
    bus.unsubscribe("X", handler)
    n = bus.publish(DomainEvent(event_type="X", aggregate_id="p1"))
    assert n == 0


def test_duplicate_event_is_dropped() -> None:
    bus = InMemoryEventBus()
    received: list[DomainEvent] = []
    bus.subscribe("X", received.append)
    e = DomainEvent(event_type="X", aggregate_id="p1")
    assert bus.publish(e) == 1
    assert bus.publish(e) == 0
    assert len(received) == 1


def test_subscribe_rejects_empty_type() -> None:
    bus = InMemoryEventBus()
    with pytest.raises(ValueError):
        bus.subscribe("", lambda e: None)
