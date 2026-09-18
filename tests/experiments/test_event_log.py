from pathlib import Path

from edmo.domain.events.event import DomainEvent
from edmo.experiments.event_log import EventLog


def test_append_and_read(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    e1 = DomainEvent(event_type="JobAdded", aggregate_id="p1", payload={"j": "j1"})
    e2 = DomainEvent(event_type="JobRemoved", aggregate_id="p1", payload={"j": "j1"})
    log.append(e1)
    log.append(e2)
    events = list(log.read())
    assert len(events) == 2
    assert events[0].event_id == e1.event_id
    assert events[1].event_id == e2.event_id


def test_read_missing_file(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "missing.jsonl")
    assert list(log.read()) == []


def test_count(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.jsonl")
    for i in range(3):
        log.append(DomainEvent(event_type="X", aggregate_id=f"a{i}"))
    assert log.count() == 3


def test_path_property(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    log = EventLog(p)
    assert log.path == p
