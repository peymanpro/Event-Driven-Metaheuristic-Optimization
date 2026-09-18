import pytest

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.event_bus.serialization import encode_event
from edmo.infrastructure.kafka.consumer import KafkaEventConsumer, run_consumer
from edmo.infrastructure.kafka.errors import ConsumerError


class _Message:
    def __init__(self, value: bytes | None) -> None:
        self.value = value


class _FakeConsumer:
    def __init__(self, messages: list[_Message]) -> None:
        self._messages = messages
        self.commits = 0
        self.closed = 0

    def poll(self, timeout_ms=None):  # type: ignore[no-untyped-def]
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
    assert fake.commits == 1


def test_poll_returns_none_when_empty() -> None:
    fake = _FakeConsumer([])
    c = KafkaEventConsumer(fake, group_id="g1")
    assert c.poll() is None


def test_poll_drops_duplicate_events() -> None:
    ev = _event()
    fake = _FakeConsumer([_Message(encode_event(ev)), _Message(encode_event(ev))])
    c = KafkaEventConsumer(fake, group_id="g1")
    assert c.poll() is not None
    assert c.poll() is None


def test_poll_skips_null_values() -> None:
    fake = _FakeConsumer([_Message(None), _Message(encode_event(_event()))])
    c = KafkaEventConsumer(fake, group_id="g1")
    # First poll sees the tombstone (None) and returns None.
    assert c.poll() is None
    # Second poll sees the real event.
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
