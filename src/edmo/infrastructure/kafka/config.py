from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KafkaConfig:
    """Configuration for connecting to a Kafka cluster.

    ``bootstrap_servers`` is a comma-separated list of ``host:port``. All
    other fields map directly to the underlying client options.
    """

    bootstrap_servers: str
    client_id: str = "edmo"
    group_id: str = "edmo-optimizer"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    request_timeout_ms: int = 10_000

    def __post_init__(self) -> None:
        if not self.bootstrap_servers:
            raise ValueError("bootstrap_servers must be non-empty")
        if not self.client_id:
            raise ValueError("client_id must be non-empty")
        if not self.group_id:
            raise ValueError("group_id must be non-empty")
        if self.auto_offset_reset not in {"earliest", "latest", "error"}:
            raise ValueError(
                "auto_offset_reset must be one of 'earliest', 'latest', 'error'"
            )
        if self.request_timeout_ms <= 0:
            raise ValueError("request_timeout_ms must be positive")
