import pytest

from edmo.infrastructure.kafka.config import KafkaConfig
from edmo.infrastructure.kafka.factory import create_consumer, create_publisher


def _cfg() -> KafkaConfig:
    return KafkaConfig(bootstrap_servers="localhost:9092")


def test_create_publisher_requires_kafka_installed() -> None:
    # kafka-python is an optional dependency, not installed in the dev env.
    with pytest.raises(RuntimeError):
        create_publisher(_cfg())


def test_create_consumer_requires_kafka_installed() -> None:
    with pytest.raises(RuntimeError):
        create_consumer(_cfg(), ["topic"])


def test_create_consumer_rejects_empty_topics() -> None:
    with pytest.raises(ValueError):
        create_consumer(_cfg(), [])
