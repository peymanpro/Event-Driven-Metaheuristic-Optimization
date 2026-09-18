from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from edmo.domain.events.event import DomainEvent

Handler = Callable[[DomainEvent], None]


@dataclass(slots=True)
class InMemoryEventBus:
    """Simple synchronous event bus.

    Handlers are keyed by exact ``event_type``. The bus is deterministic:
    handlers run in registration order.

    Duplicate ``event_id`` values are dropped per subscription, but only
    within the lifetime of this bus instance. This is an in-process filter,
    not durable exactly-once delivery: recreating the bus resets the filter,
    so callers that need cross-restart dedup must persist their own state.
    """

    _handlers: dict[str, list[Handler]] = field(default_factory=dict)
    _seen: dict[str, set[str]] = field(default_factory=dict)

    def subscribe(self, event_type: str, handler: Handler) -> None:
        if not event_type:
            raise ValueError("event_type must be non-empty")
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        handlers = self._handlers.get(event_type)
        if handlers is None:
            return
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            self._handlers.pop(event_type, None)

    def publish(self, event: DomainEvent) -> int:
        """Deliver ``event`` to subscribers.

        Returns the number of handlers invoked. Duplicate events (same
        ``event_id``) are silently ignored to guarantee at-most-once delivery
        per subscription.
        """
        key = event.event_type
        seen = self._seen.setdefault(key, set())
        event_key = str(event.event_id)
        if event_key in seen:
            return 0
        seen.add(event_key)
        invoked = 0
        for handler in list(self._handlers.get(key, [])):
            handler(event)
            invoked += 1
        return invoked
