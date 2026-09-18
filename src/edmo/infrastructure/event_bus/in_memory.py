from __future__ import annotations

from collections import deque
from collections.abc import Callable
from threading import Lock

from edmo.domain.events.event import DomainEvent


class InMemoryTransport:
    """Process-local transport used for tests and single-process pipelines.

    The transport is thread-safe. Each topic has a FIFO queue. Multiple
    subscribers can read the same topic; each subscriber has its own queue
    view, so events are fanned out rather than load-balanced. Idempotency at
    the transport level is per (topic, subscriber, event_id).
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._topics: dict[str, deque[DomainEvent]] = {}
        self._subscribers: dict[str, list[_Subscriber]] = {}
        self._closed = False

    def publish(self, topic: str, event: DomainEvent) -> None:
        if not topic:
            raise ValueError("topic must be non-empty")
        if self._closed:
            raise RuntimeError("transport is closed")
        with self._lock:
            for sub in self._subscribers.get(topic, []):
                sub.enqueue(event)

    def poll(self, timeout: float = 0.0) -> DomainEvent | None:
        raise NotImplementedError("use subscribe() on InMemoryTransport")

    def subscribe(self, topic: str, handler: Callable[[DomainEvent], None]) -> None:
        if not topic:
            raise ValueError("topic must be non-empty")
        with self._lock:
            self._subscribers.setdefault(topic, []).append(_Subscriber(handler))

    def drain(self, topic: str) -> int:
        """Deliver all queued events for every subscriber on ``topic``.

        Returns the number of handler invocations performed. Duplicate events
        per subscriber are dropped.
        """
        with self._lock:
            subs = list(self._subscribers.get(topic, []))
        total = 0
        for sub in subs:
            total += sub.drain()
        return total

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._subscribers.clear()


class _Subscriber:
    def __init__(self, handler: Callable[[DomainEvent], None]) -> None:
        self._handler = handler
        self._queue: deque[DomainEvent] = deque()
        self._seen: set[str] = set()

    def enqueue(self, event: DomainEvent) -> None:
        key = str(event.event_id)
        if key in self._seen:
            return
        self._seen.add(key)
        self._queue.append(event)

    def drain(self) -> int:
        n = 0
        while self._queue:
            self._handler(self._queue.popleft())
            n += 1
        return n
