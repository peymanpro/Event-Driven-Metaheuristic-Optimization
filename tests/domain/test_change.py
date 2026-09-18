import pytest

from edmo.domain.change import (
    AddJob,
    AddResource,
    ChangeDeadline,
    ChangeResourceCapacity,
    ChangeResourceSpeed,
    ChangeWeight,
    RemoveJob,
    RemoveResource,
    apply_change,
    apply_changes,
)
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0", capacity={"cpu": 4.0}, speed=1.0),),
        jobs=(
            Job(id="j1", processing_time=1.0, deadline=10.0, weight=1.0),
            Job(id="j2", processing_time=2.0, deadline=10.0, weight=1.0),
        ),
    )


def test_apply_change_does_not_mutate_original() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeWeight("j1", 5.0))
    assert p.job_by_id("j1").weight == 1.0
    assert p2.job_by_id("j1").weight == 5.0


def test_version_increments_per_change() -> None:
    p = _problem()
    assert p.version == 1
    p2 = apply_change(p, ChangeWeight("j1", 2.0))
    assert p2.version == 2
    p3 = apply_change(p2, ChangeDeadline("j1", 20.0))
    assert p3.version == 3


def test_add_job() -> None:
    p = _problem()
    p2 = apply_change(p, AddJob(Job(id="j3", processing_time=1.0)))
    assert {j.id for j in p2.jobs} == {"j1", "j2", "j3"}


def test_add_job_duplicate_rejected() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(p, AddJob(Job(id="j1", processing_time=1.0)))


def test_add_job_unknown_predecessor_rejected() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(
            p,
            AddJob(Job(id="j3", processing_time=1.0, predecessors=frozenset({"ghost"}))),
        )


def test_remove_job() -> None:
    p = _problem()
    p2 = apply_change(p, RemoveJob("j2"))
    assert {j.id for j in p2.jobs} == {"j1"}


def test_remove_job_unknown_rejected() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(p, RemoveJob("ghost"))


def test_remove_job_with_dependent_rejected() -> None:
    p = Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(
            Job(id="j1", processing_time=1.0),
            Job(id="j2", processing_time=1.0, predecessors=frozenset({"j1"})),
        ),
    )
    with pytest.raises(ValueError):
        apply_change(p, RemoveJob("j1"))


def test_change_deadline() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeDeadline("j1", 5.0))
    assert p2.job_by_id("j1").deadline == 5.0


def test_change_deadline_to_none() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeDeadline("j1", None))
    assert p2.job_by_id("j1").deadline is None


def test_change_deadline_below_release_rejected() -> None:
    p = Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(Job(id="j1", processing_time=1.0, release_time=5.0),),
    )
    with pytest.raises(ValueError):
        apply_change(p, ChangeDeadline("j1", 1.0))


def test_change_weight() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeWeight("j1", 3.5))
    assert p2.job_by_id("j1").weight == 3.5


def test_change_weight_invalid() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(p, ChangeWeight("j1", 0.0))


def test_add_resource() -> None:
    p = _problem()
    p2 = apply_change(p, AddResource(Resource(id="r1")))
    assert {r.id for r in p2.resources} == {"r0", "r1"}


def test_add_resource_duplicate_rejected() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(p, AddResource(Resource(id="r0")))


def test_remove_resource() -> None:
    p = _problem()
    p2 = apply_change(p, AddResource(Resource(id="r1")))
    p3 = apply_change(p2, RemoveResource("r0"))
    assert {r.id for r in p3.resources} == {"r1"}


def test_remove_resource_unknown_rejected() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(p, RemoveResource("ghost"))


def test_change_resource_capacity() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeResourceCapacity("r0", {"cpu": 8.0}))
    assert p2.resource_by_id("r0").capacity["cpu"] == 8.0


def test_change_resource_capacity_negative_rejected() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(p, ChangeResourceCapacity("r0", {"cpu": -1.0}))


def test_change_resource_speed() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeResourceSpeed("r0", 2.0))
    assert p2.resource_by_id("r0").speed == 2.0


def test_change_resource_speed_invalid() -> None:
    p = _problem()
    with pytest.raises(ValueError):
        apply_change(p, ChangeResourceSpeed("r0", 0.0))


def test_apply_changes_sequence() -> None:
    p = _problem()
    p2 = apply_changes(
        p,
        [
            AddJob(Job(id="j3", processing_time=1.0)),
            ChangeWeight("j3", 2.0),
            ChangeDeadline("j3", 15.0),
        ],
    )
    assert p2.version == 4
    assert p2.job_by_id("j3").weight == 2.0
    assert p2.job_by_id("j3").deadline == 15.0
