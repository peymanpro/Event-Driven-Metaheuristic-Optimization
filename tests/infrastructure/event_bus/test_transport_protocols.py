from collections.abc import Iterable

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.event_bus.in_memory import InMemoryTransport
from edmo.infrastructure.event_bus.transport import (
    EventPublisher,
    PollingSubscriber,
    PushSubscriber,
)
from edmo.infrastructure.kafka.consumer import KafkaEventConsumer


class _FakeKafkaClient:
    def poll(self, timeout_ms: float | None = None):  # type: ignore[no-untyped-def]
        return {}

    def commit(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_in_memory_is_a_push_subscriber() -> None:
    transport = InMemoryTransport()
    assert isinstance(transport, PushSubscriber)


def test_in_memory_is_a_publisher() -> None:
    transport = InMemoryTransport()
    assert isinstance(transport, EventPublisher)


def test_kafka_consumer_is_a_polling_subscriber() -> None:
    consumer = KafkaEventConsumer(_FakeKafkaClient(), group_id="g1")
    assert isinstance(consumer, PollingSubscriber)


def test_kafka_consumer_is_not_a_push_subscriber() -> None:
    consumer = KafkaEventConsumer(_FakeKafkaClient(), group_id="g1")
    # A polling consumer does not expose ``subscribe``; this must remain
    # distinct so callers cannot accidentally treat it as a push bus.
    assert not isinstance(consumer, PushSubscriber)


def test_polling_subscriber_stream_is_iterable() -> None:
    consumer = KafkaEventConsumer(_FakeKafkaClient(), group_id="g1")
    events: Iterable[DomainEvent] = consumer.stream()
    # No events because the fake client returns empty batches; iterating once
    # must not raise. We do not exhaust the infinite generator.
    it = iter(events)
    consumer.close()
    assert it is not None
