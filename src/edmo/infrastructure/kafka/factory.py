from __future__ import annotations

from typing import Any

from edmo.infrastructure.kafka.config import KafkaConfig
from edmo.infrastructure.kafka.consumer import KafkaEventConsumer
from edmo.infrastructure.kafka.errors import ConsumerError, ProducerError
from edmo.infrastructure.kafka.producer import KafkaEventPublisher
from edmo.infrastructure.kafka.retry import RetryPolicy


def _import_kafka() -> Any:
    """Import the kafka package lazily.

    ``kafka-python`` is an optional dependency. Installing it via
    ``pip install -e ".[kafka]"`` enables real broker connectivity. Tests and
    CI run without it by injecting fake clients.
    """
    try:
        import kafka  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "kafka-python is not installed. Install with "
            "'pip install -e \".[kafka]\"' to use the Kafka adapter."
        ) from exc
    return kafka


def create_publisher(
    config: KafkaConfig,
    *,
    retry_policy: RetryPolicy | None = None,
) -> KafkaEventPublisher:
    """Construct a :class:`KafkaEventPublisher` from a :class:`KafkaConfig`."""
    kafka = _import_kafka()
    try:
        producer = kafka.KafkaProducer(
            bootstrap_servers=config.bootstrap_servers,
            client_id=config.client_id,
            request_timeout_ms=config.request_timeout_ms,
            value_serializer=None,
        )
    except Exception as exc:
        raise ProducerError("failed to create kafka producer") from exc
    return KafkaEventPublisher(producer, retry_policy=retry_policy)


def create_consumer(
    config: KafkaConfig,
    topics: list[str],
    *,
    auto_commit: bool = False,
) -> KafkaEventConsumer:
    """Construct a :class:`KafkaEventConsumer` from a :class:`KafkaConfig`."""
    if not topics:
        raise ValueError("topics must not be empty")
    kafka = _import_kafka()
    try:
        client = kafka.KafkaConsumer(
            *topics,
            bootstrap_servers=config.bootstrap_servers,
            client_id=config.client_id,
            group_id=config.group_id,
            auto_offset_reset=config.auto_offset_reset,
            enable_auto_commit=config.enable_auto_commit,
            request_timeout_ms=config.request_timeout_ms,
        )
    except Exception as exc:
        raise ConsumerError("failed to create kafka consumer") from exc
    return KafkaEventConsumer(client, group_id=config.group_id, auto_commit=auto_commit)
