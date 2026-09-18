import pytest

from edmo.infrastructure.kafka.config import KafkaConfig


def test_defaults() -> None:
    cfg = KafkaConfig(bootstrap_servers="localhost:9092")
    assert cfg.group_id == "edmo-optimizer"
    assert cfg.auto_offset_reset == "earliest"
    assert cfg.enable_auto_commit is False


def test_rejects_empty_servers() -> None:
    with pytest.raises(ValueError):
        KafkaConfig(bootstrap_servers="")


def test_rejects_empty_client_id() -> None:
    with pytest.raises(ValueError):
        KafkaConfig(bootstrap_servers="localhost:9092", client_id="")


def test_rejects_empty_group_id() -> None:
    with pytest.raises(ValueError):
        KafkaConfig(bootstrap_servers="localhost:9092", group_id="")


def test_rejects_bad_offset_reset() -> None:
    with pytest.raises(ValueError):
        KafkaConfig(bootstrap_servers="localhost:9092", auto_offset_reset="middle")


def test_rejects_bad_timeout() -> None:
    with pytest.raises(ValueError):
        KafkaConfig(bootstrap_servers="localhost:9092", request_timeout_ms=0)
