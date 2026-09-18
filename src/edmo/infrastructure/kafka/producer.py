from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.event_bus.serialization import encode_event
from edmo.infrastructure.kafka.errors import ProducerError
from edmo.infrastructure.kafka.retry import RetryPolicy


class KafkaProducerLike(Protocol):
    """Minimal structural type for a kafka producer client.

    Matches the subset of ``kafka.KafkaProducer`` we actually use, so tests
    can substitute a fake without pulling in the real library.
    """

    def send(self, topic: str, value: bytes, key: bytes | None = ...) -> Any: ...

    def flush(self, timeout: float | None = ...) -> None: ...

    def close(self, timeout: float | None = ...) -> None: ...


def _key_for(event: DomainEvent) -> bytes:
    return event.aggregate_id.encode("utf-8")


class KafkaEventPublisher:
    """Publish :class:`DomainEvent` to Kafka topics.

    The event's ``aggregate_id`` is used as the message key so that events
    sharing an aggregate stay ordered within a partition.
    """

    def __init__(
        self,
        producer: KafkaProducerLike,
        *,
        retry_policy: RetryPolicy | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._producer = producer
        self._retry = retry_policy or RetryPolicy()
        self._clock = clock
        self._closed = False

    def publish(self, topic: str, event: DomainEvent) -> None:
        if not topic:
            raise ValueError("topic must be non-empty")
        if self._closed:
            raise ProducerError("publisher is closed")
        payload = encode_event(event)
        key = _key_for(event)

        def op() -> None:
            future = self._producer.send(topic, value=payload, key=key)
            # Some fakes return None; only wait when a future-like is given.
            if future is not None and hasattr(future, "get"):
                future.get(timeout=10.0)

        try:
            self._retry.run(op)
        except Exception as exc:
            raise ProducerError(
                f"failed to publish event {event.event_id} to {topic!r}"
            ) from exc

    def flush(self, timeout: float | None = None) -> None:
        self._producer.flush(timeout)

    def close(self, timeout: float | None = None) -> None:
        if self._closed:
            return
        self._producer.flush(timeout)
        self._producer.close(timeout)
        self._closed = True
