import pytest

from edmo.application.change_codec import decode_change, encode_change
from edmo.domain.change import (
    AddJob,
    AddResource,
    ChangeDeadline,
    ChangeResourceCapacity,
    ChangeResourceSpeed,
    ChangeWeight,
    ProblemChange,
    RemoveJob,
    RemoveResource,
)
from edmo.domain.jobs import Job
from edmo.domain.resources import Resource


def _roundtrip(change: ProblemChange) -> ProblemChange:
    return decode_change(encode_change(change))


def test_roundtrip_add_job() -> None:
    j = Job(
        id="j1",
        processing_time=2.0,
        demand={"cpu": 1.5},
        release_time=0.5,
        deadline=10.0,
        weight=2.0,
        predecessors=frozenset({"j0"}),
    )
    change = AddJob(j)
    decoded = _roundtrip(change)
    assert isinstance(decoded, AddJob)
    assert decoded.job.id == "j1"
    assert decoded.job.processing_time == 2.0
    assert dict(decoded.job.demand) == {"cpu": 1.5}
    assert decoded.job.predecessors == frozenset({"j0"})


def test_roundtrip_remove_job() -> None:
    decoded = _roundtrip(RemoveJob("j9"))
    assert isinstance(decoded, RemoveJob)
    assert decoded.job_id == "j9"


def test_roundtrip_change_deadline() -> None:
    decoded = _roundtrip(ChangeDeadline("j1", 12.5))
    assert isinstance(decoded, ChangeDeadline)
    assert decoded.deadline == 12.5


def test_roundtrip_change_deadline_none() -> None:
    decoded = _roundtrip(ChangeDeadline("j1", None))
    assert isinstance(decoded, ChangeDeadline)
    assert decoded.deadline is None


def test_roundtrip_change_weight() -> None:
    decoded = _roundtrip(ChangeWeight("j1", 3.0))
    assert isinstance(decoded, ChangeWeight)
    assert decoded.weight == 3.0


def test_roundtrip_add_resource() -> None:
    r = Resource(id="r9", capacity={"cpu": 4.0}, speed=1.2)
    decoded = _roundtrip(AddResource(r))
    assert isinstance(decoded, AddResource)
    assert decoded.resource.id == "r9"
    assert dict(decoded.resource.capacity) == {"cpu": 4.0}
    assert decoded.resource.speed == 1.2


def test_roundtrip_remove_resource() -> None:
    decoded = _roundtrip(RemoveResource("r1"))
    assert isinstance(decoded, RemoveResource)
    assert decoded.resource_id == "r1"


def test_roundtrip_change_resource_capacity() -> None:
    decoded = _roundtrip(ChangeResourceCapacity("r1", {"cpu": 8.0}))
    assert isinstance(decoded, ChangeResourceCapacity)
    assert decoded.capacity == {"cpu": 8.0}


def test_roundtrip_change_resource_speed() -> None:
    decoded = _roundtrip(ChangeResourceSpeed("r1", 2.5))
    assert isinstance(decoded, ChangeResourceSpeed)
    assert decoded.speed == 2.5


def test_decode_unknown_kind() -> None:
    with pytest.raises(ValueError):
        decode_change({"kind": "NoSuch"})
