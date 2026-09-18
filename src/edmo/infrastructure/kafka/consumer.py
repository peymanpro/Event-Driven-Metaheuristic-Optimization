from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.event_bus.serialization import decode_event
from edmo.infrastructure.kafka.errors import ConsumerError


class KafkaMessageLike(Protocol):
    value: bytes | None


class KafkaConsumerLike(Protocol):
    def poll(
        self, timeout_ms: float | None = ...
    ) -> dict[Any, list[KafkaMessageLike]]: ...

    def commit(self) -> None: ...

    def close(self) -> None: ...


class KafkaEventConsumer:
    """Consume Kafka messages into :class:`DomainEvent` with idempotency.

    ``group_id`` is required so consumer offsets are tracked per logical
    consumer. Duplicate ``event_id`` values per group are ignored, providing
    at-least-once to effectively-once delivery for downstream handlers.
    """

    def __init__(
        self,
        consumer: KafkaConsumerLike,
        group_id: str,
        *,
        auto_commit: bool = True,
    ) -> None:
        if not group_id:
            raise ValueError("group_id must be non-empty")
        self._consumer = consumer
        self._group_id = group_id
        self._auto_commit = auto_commit
        self._seen: set[str] = set()
        self._closed = False

    @property
    def group_id(self) -> str:
        return self._group_id

    def _decode(self, message: KafkaMessageLike) -> DomainEvent | None:
        if message.value is None:
            return None
        try:
            event = decode_event(message.value)
        except ValueError as exc:
            raise ConsumerError("failed to decode kafka message") from exc
        key = str(event.event_id)
        if key in self._seen:
            return None
        self._seen.add(key)
        return event

    def poll(self, timeout_ms: float = 0.0) -> DomainEvent | None:
        if self._closed:
            raise ConsumerError("consumer is closed")
        batches = self._consumer.poll(timeout_ms)
        if not batches:
            return None
        result: DomainEvent | None = None
        for messages in batches.values():
            for message in messages:
                event = self._decode(message)
                if event is not None:
                    result = event
                    break
            if result is not None:
                break
        if self._auto_commit:
            self._consumer.commit()
        return result

    def stream(self) -> Iterable[DomainEvent]:
        while not self._closed:
            event = self.poll(timeout_ms=100.0)
            if event is not None:
                yield event

    def close(self) -> None:
        if self._closed:
            return
        self._consumer.close()
        self._closed = True


def run_consumer(
    consumer: KafkaEventConsumer,
    handler: Callable[[DomainEvent], None],
    *,
    max_events: int | None = None,
) -> int:
    """Consume events and invoke ``handler`` for each unique event.

    Returns the number of handler invocations. ``max_events`` bounds the loop
    so callers (and tests) can stop deterministically.
    """
    count = 0
    for event in consumer.stream():
        handler(event)
        count += 1
        if max_events is not None and count >= max_events:
            break
    return count
