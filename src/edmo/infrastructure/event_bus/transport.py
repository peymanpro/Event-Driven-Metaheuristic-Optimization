from __future__ import annotations

from collections.abc import Callable
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
class EventSubscriber(Protocol):
    """Consumes domain events from a named topic.

    A subscriber is identified by a ``group_id`` so multiple logical consumers
    can read the same topic independently. ``poll`` returns the next event or
    ``None`` when the topic is empty within the poll timeout.
    """

    def poll(self, timeout: float = 0.0) -> DomainEvent | None: ...

    def subscribe(self, topic: str, handler: Callable[[DomainEvent], None]) -> None: ...

    def close(self) -> None: ...
