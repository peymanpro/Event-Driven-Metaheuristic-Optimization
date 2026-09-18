import pytest

from edmo.domain.events.event import DomainEvent
from edmo.infrastructure.kafka.errors import ProducerError
from edmo.infrastructure.kafka.producer import KafkaEventPublisher
from edmo.infrastructure.kafka.retry import RetryPolicy


class _FakeFuture:
    def __init__(self) -> None:
        self.waited = 0

    def get(self, timeout: float | None = None) -> None:
        self.waited += 1


class _FakeProducer:
    def __init__(self, fail_times: int = 0) -> None:
        self.sent: list[tuple[str, bytes, bytes | None]] = []
        self.flushed = 0
        self.closed = 0
        self._fail_times = fail_times
        self._attempts = 0
        self.future = _FakeFuture()

    def send(self, topic, value, key=None):  # type: ignore[no-untyped-def]
        self._attempts += 1
        if self._attempts <= self._fail_times:
            raise RuntimeError("boom")
        self.sent.append((topic, value, key))
        return self.future

    def flush(self, timeout=None):  # type: ignore[no-untyped-def]
        self.flushed += 1

    def close(self, timeout=None):  # type: ignore[no-untyped-def]
        self.closed += 1


def _event() -> DomainEvent:
    return DomainEvent(event_type="X", aggregate_id="agg1", payload={"k": 1})


def test_publish_sends_with_aggregate_key() -> None:
    fake = _FakeProducer()
    pub = KafkaEventPublisher(fake, retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None))
    pub.publish("topic", _event())
    assert len(fake.sent) == 1
    topic, _, key = fake.sent[0]
    assert topic == "topic"
    assert key == b"agg1"
    assert fake.future.waited == 1


def test_publish_retries_on_failure() -> None:
    fake = _FakeProducer(fail_times=1)
    pub = KafkaEventPublisher(
        fake,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0, sleep=lambda _: None),
    )
    pub.publish("topic", _event())
    assert len(fake.sent) == 1


def test_publish_wraps_persistent_failure() -> None:
    fake = _FakeProducer(fail_times=99)
    pub = KafkaEventPublisher(
        fake,
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.0, sleep=lambda _: None),
    )
    with pytest.raises(ProducerError):
        pub.publish("topic", _event())


def test_publish_rejects_empty_topic() -> None:
    pub = KafkaEventPublisher(
        _FakeProducer(),
        retry_policy=RetryPolicy(max_attempts=1, sleep=lambda _: None),
    )
    with pytest.raises(ValueError):
        pub.publish("", _event())


def test_close_flushes_and_is_idempotent() -> None:
    fake = _FakeProducer()
    pub = KafkaEventPublisher(fake)
    pub.close()
    pub.close()
    assert fake.flushed == 1
    assert fake.closed == 1


def test_publish_after_close_raises() -> None:
    fake = _FakeProducer()
    pub = KafkaEventPublisher(fake)
    pub.close()
    with pytest.raises(ProducerError):
        pub.publish("topic", _event())
