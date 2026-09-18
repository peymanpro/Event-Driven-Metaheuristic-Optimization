from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Protocol, runtime_checkable

from edmo.domain.events.event import DomainEvent


@runtime_checkable
class EventPublisher(Protocol):
    """Publishes domain events to a named topic.

    Implementations must preserve ordering per ``aggregate_id`` at least on a
    best-effort basis, and must not silently drop events.
    """

    def publish(self, topic: str, event: DomainEvent) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class PollingSubscriber(Protocol):
    """Pull-based subscriber matching the Kafka consumer model.

    Used by consumers that fetch events on demand, identified by a
    ``group_id`` so multiple logical consumers read the same topic
    independently. ``poll`` returns the next event or ``None`` within the
    timeout. ``stream`` yields events until the consumer is closed.
    """

    @property
    def group_id(self) -> str: ...

    def poll(self, timeout: float = 0.0) -> DomainEvent | None: ...

    def stream(self) -> Iterable[DomainEvent]: ...

    def close(self) -> None: ...


@runtime_checkable
class PushSubscriber(Protocol):
    """Callback-based subscriber matching the in-memory bus model.

    Used by transports that push events to registered handlers, e.g.
    :class:`~edmo.infrastructure.event_bus.in_memory.InMemoryTransport`.
    """

    def subscribe(self, topic: str, handler: Callable[[DomainEvent], None]) -> None: ...

    def close(self) -> None: ...
