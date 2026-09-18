from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Envelope for every domain event in the system.

    The envelope is transport-agnostic. Kafka or any other broker treats it
    as an opaque payload. ``event_id`` is a UUID generated on construction if
    not supplied, and ``timestamp`` defaults to the current UTC time.
    """

    event_type: str
    aggregate_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    version: int = 1

    def __post_init__(self) -> None:
        if not self.event_type:
            raise ValueError("DomainEvent.event_type must be non-empty")
        if not self.aggregate_id:
            raise ValueError("DomainEvent.aggregate_id must be non-empty")
        if self.version < 1:
            raise ValueError("DomainEvent.version must be >= 1")
        if self.timestamp.tzinfo is None:
            raise ValueError("DomainEvent.timestamp must be timezone-aware")
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": str(self.event_id),
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "aggregate_id": self.aggregate_id,
            "version": self.version,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainEvent:
        required = {"event_id", "event_type", "timestamp", "aggregate_id", "version"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"missing event fields: {sorted(missing)}")
        return cls(
            event_type=str(data["event_type"]),
            aggregate_id=str(data["aggregate_id"]),
            payload=dict(data.get("payload", {})),
            event_id=UUID(str(data["event_id"])),
            timestamp=datetime.fromisoformat(str(data["timestamp"])),
            version=int(data["version"]),
        )
