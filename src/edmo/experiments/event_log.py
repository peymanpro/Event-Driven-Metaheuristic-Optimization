from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from edmo.domain.events.event import DomainEvent


class EventLog:
    """Append-only JSONL event log.

    Each line is a single serialized :class:`DomainEvent`. The log is used
    to persist events that also travel over Kafka, so that offline analysis
    does not depend on a running broker.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: DomainEvent) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.to_dict(), sort_keys=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write("\n")

    def read(self) -> Iterator[DomainEvent]:
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                data = json.loads(stripped)
                yield DomainEvent.from_dict(data)

    def count(self) -> int:
        return sum(1 for _ in self.read())
