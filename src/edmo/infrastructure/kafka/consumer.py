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
    """Consume Kafka messages into :class:`DomainEvent`.

    ``group_id`` is required so consumer offsets are tracked per logical
    consumer.

    Delivery semantics:

    - Kafka itself is at-least-once: a message may be redelivered after a
      crash, rebalance, or an uncommitted offset.
    - ``poll`` returns the next decoded event without filtering. It does not
      mark the event as processed; that is the caller's responsibility.
    - The caller must call :meth:`mark_processed` only after the handler has
      returned successfully. This is the contract that makes retry work:
      a handler that raises leaves the event unprocessed, so a redelivery
      with the same ``event_id`` will be returned again.
    - Once an event is marked processed, later redeliveries of the same
      ``event_id`` are filtered *for this consumer instance*. That filter is
      in-process and not durable: restarting the process forgets it.
    - This consumer does not claim exactly-once delivery. Handlers that must
      be exactly-once across restarts must persist their own dedup state.
    """

    def __init__(
        self,
        consumer: KafkaConsumerLike,
        group_id: str,
        *,
        auto_commit: bool = False,
    ) -> None:
        if not group_id:
            raise ValueError("group_id must be non-empty")
        self._consumer = consumer
        self._group_id = group_id
        self._auto_commit = auto_commit
        self._processed: set[str] = set()
        self._closed = False

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _decode(self, message: KafkaMessageLike) -> DomainEvent | None:
        if message.value is None:
            return None
        try:
            event = decode_event(message.value)
        except ValueError as exc:
            raise ConsumerError("failed to decode kafka message") from exc
        return event

    def poll(self, timeout_ms: float = 0.0) -> DomainEvent | None:
        """Return the next not-yet-processed event, or ``None``.

        Events whose ``event_id`` has already been successfully processed by
        this consumer instance are skipped. The returned event is not yet
        considered processed; call :meth:`mark_processed` after the handler
        succeeds.
        """
        if self._closed:
            raise ConsumerError("consumer is closed")
        batches = self._consumer.poll(timeout_ms)
        if not batches:
            return None
        for messages in batches.values():
            for message in messages:
                event = self._decode(message)
                if event is None:
                    continue
                if str(event.event_id) in self._processed:
                    continue
                return event
        return None

    def mark_processed(self, event: DomainEvent) -> None:
        """Record that ``event`` was handled successfully.

        Call only after the handler has returned without raising. Subsequent
        polls that redeliver the same ``event_id`` are filtered. When
        ``auto_commit`` is enabled, the underlying consumer is committed at
        this point, which is the only correct time to commit under
        at-least-once semantics.
        """
        key = str(event.event_id)
        if key in self._processed:
            return
        self._processed.add(key)
        if self._auto_commit:
            self._consumer.commit()

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

    An event is marked as processed only after ``handler`` returns without
    raising. If the handler raises, the exception propagates and the event is
    NOT marked, so a subsequent redelivery is returned again, matching
    at-least-once semantics. Once handled successfully, duplicate redeliveries
    of the same ``event_id`` are filtered for the lifetime of this consumer
    instance.

    Returns the number of successful handler invocations.
    """
    count = 0
    while not consumer.is_closed:
        event = consumer.poll(timeout_ms=100.0)
        if event is None:
            continue
        handler(event)  # if this raises, event is not marked
        consumer.mark_processed(event)
        count += 1
        if max_events is not None and count >= max_events:
            break
    return count
