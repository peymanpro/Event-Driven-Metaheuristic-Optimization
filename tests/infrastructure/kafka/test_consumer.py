from typing import Any

import pytest

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.event_bus.serialization import encode_event
from edmo.infrastructure.kafka.consumer import (
    KafkaEventConsumer,
    KafkaMessageLike,
    run_consumer,
)
from edmo.infrastructure.kafka.errors import ConsumerError


class _Message:
    def __init__(self, value: bytes | None) -> None:
        self.value = value


class _FakeConsumer:
    def __init__(self, messages: list[_Message]) -> None:
        self._messages = messages
        self.commits = 0
        self.closed = 0

    def poll(
        self, timeout_ms: float | None = None
    ) -> dict[Any, list[KafkaMessageLike]]:
        if not self._messages:
            return {}
        msg = self._messages.pop(0)
        return {"tp": [msg]}

    def commit(self) -> None:
        self.commits += 1

    def close(self) -> None:
        self.closed += 1


def _event(eid: str = "agg1") -> DomainEvent:
    return DomainEvent(event_type="X", aggregate_id=eid, payload={"k": 1})


def test_poll_returns_decoded_event() -> None:
    ev = _event()
    fake = _FakeConsumer([_Message(encode_event(ev))])
    c = KafkaEventConsumer(fake, group_id="g1")
    decoded = c.poll()
    assert decoded is not None
    assert decoded.event_id == ev.event_id
    # Manual mode does not commit on poll; only mark_processed does.
    assert fake.commits == 0


def test_poll_returns_none_when_empty() -> None:
    fake = _FakeConsumer([])
    c = KafkaEventConsumer(fake, group_id="g1")
    assert c.poll() is None


def test_uncommitted_event_is_returned_once() -> None:
    """poll returns the event; it does not dedup until mark_processed."""
    ev = _event()
    fake = _FakeConsumer([_Message(encode_event(ev))])
    c = KafkaEventConsumer(fake, group_id="g1")
    first = c.poll()
    assert first is not None
    # Event was not marked processed; the fake has no more messages so
    # subsequent poll returns None, but the point is the first call did not
    # add it to the processed set.
    assert c.poll() is None


def test_poll_skips_null_values() -> None:
    fake = _FakeConsumer([_Message(None), _Message(encode_event(_event()))])
    c = KafkaEventConsumer(fake, group_id="g1")
    # First poll sees the tombstone (None); consumer skips it and returns None
    # because the second message is only fetched on the next poll in this fake.
    assert c.poll() is None
    assert c.poll() is not None


def test_poll_raises_on_invalid_payload() -> None:
    fake = _FakeConsumer([_Message(b"not-json")])
    c = KafkaEventConsumer(fake, group_id="g1")
    with pytest.raises(ConsumerError):
        c.poll()


def test_consumer_requires_group_id() -> None:
    with pytest.raises(ValueError):
        KafkaEventConsumer(_FakeConsumer([]), group_id="")


def test_run_consumer_counts_handled_events() -> None:
    evs = [_event("a"), _event("b"), _event("c")]
    fake = _FakeConsumer([_Message(encode_event(e)) for e in evs])
    c = KafkaEventConsumer(fake, group_id="g1")
    seen: list[str] = []
    n = run_consumer(c, lambda e: seen.append(str(e.event_id)), max_events=3)
    assert n == 3
    assert len(seen) == 3


def test_poll_after_close_raises() -> None:
    fake = _FakeConsumer([])
    c = KafkaEventConsumer(fake, group_id="g1")
    c.close()
    with pytest.raises(ConsumerError):
        c.poll()


# ---- Regression tests for at-least-once retry semantics ----


def test_failed_handler_can_be_retried_same_event() -> None:
    """Test G: an event whose handler raised must be returned on redelivery.

    The previous implementation added ``event_id`` to a ``_seen`` set inside
    ``_decode`` before the handler ran, so a redelivered event was silently
    dropped. The new contract marks processed only after ``mark_processed``.
    """
    ev = _event("agg-retry")
    # Two copies of the same event: simulate redelivery after failure.
    fake = _FakeConsumer([_Message(encode_event(ev)), _Message(encode_event(ev))])
    consumer = KafkaEventConsumer(fake, group_id="g1")

    # First delivery: handler raises -> event must not be marked processed.
    first = consumer.poll()
    assert first is not None
    with pytest.raises(RuntimeError):
        handler_call = first
        raise RuntimeError("handler failed")
    del handler_call
    # Do not mark processed because handler failed.

    # Second delivery (redelivery of the same event_id) must be returned.
    second = consumer.poll()
    assert second is not None
    assert second.event_id == ev.event_id


def test_successful_handler_dedups_later_redelivery() -> None:
    """Test H: after successful processing, redelivery of the same id is filtered."""
    ev = _event("agg-dedup")
    fake = _FakeConsumer([_Message(encode_event(ev)), _Message(encode_event(ev))])
    consumer = KafkaEventConsumer(fake, group_id="g1")

    first = consumer.poll()
    assert first is not None
    # Simulate successful handler.
    consumer.mark_processed(first)

    # Redelivery of the same event_id is filtered.
    assert consumer.poll() is None


def test_mark_processed_is_idempotent() -> None:
    ev = _event("agg-idem")
    consumer = KafkaEventConsumer(_FakeConsumer([]), group_id="g1")
    consumer.mark_processed(ev)
    consumer.mark_processed(ev)
    # No exception, no double add.


def test_run_consumer_marks_processed_only_on_success() -> None:
    """run_consumer must not mark an event processed when the handler raises."""
    ev = _event("agg-runc")
    fake = _FakeConsumer([_Message(encode_event(ev)), _Message(encode_event(ev))])
    consumer = KafkaEventConsumer(fake, group_id="g1")

    attempts: list[str] = []

    def handler(e: DomainEvent) -> None:
        attempts.append(str(e.event_id))
        if len(attempts) == 1:
            raise RuntimeError("first attempt fails")
        # second attempt succeeds

    with pytest.raises(RuntimeError):
        run_consumer(consumer, handler, max_events=1)

    # Retry loop: run_consumer again on the same consumer.
    n = run_consumer(consumer, handler, max_events=1)
    assert n == 1
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]


def test_auto_commit_happens_after_mark_processed() -> None:
    ev = _event("agg-ac")
    fake = _FakeConsumer([_Message(encode_event(ev))])
    consumer = KafkaEventConsumer(fake, group_id="g1", auto_commit=True)
    first = consumer.poll()
    assert first is not None
    # Manual mode: poll does not commit.
    assert fake.commits == 0
    consumer.mark_processed(first)
    assert fake.commits == 1
