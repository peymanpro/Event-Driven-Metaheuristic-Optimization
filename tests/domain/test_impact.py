from edmo.domain.change import (
    AddJob,
    AddResource,
    ChangeDeadline,
    ChangeResourceCapacity,
    ChangeResourceSpeed,
    ChangeWeight,
    RemoveJob,
    apply_change,
)
from edmo.domain.impact import analyze_change
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(
            Resource(id="r0", capacity={"cpu": 4.0}, speed=1.0),
            Resource(id="r1", capacity={"cpu": 4.0}, speed=1.0),
        ),
        jobs=(
            Job(id="j1", processing_time=1.0, deadline=10.0, weight=1.0),
            Job(id="j2", processing_time=2.0, deadline=10.0, weight=1.0),
        ),
    )


def test_impact_no_change() -> None:
    p = _problem()
    impact = analyze_change(p, p)
    assert impact.jobs_added == ()
    assert impact.jobs_removed == ()
    assert not impact.is_structural


def test_impact_added_job_is_structural() -> None:
    p = _problem()
    p2 = apply_change(p, AddJob(Job(id="j3", processing_time=1.0)))
    impact = analyze_change(p, p2)
    assert impact.jobs_added == ("j3",)
    assert impact.is_structural
    assert impact.job_count_delta == 1


def test_impact_removed_job_is_structural() -> None:
    p = _problem()
    p2 = apply_change(p, RemoveJob("j2"))
    impact = analyze_change(p, p2)
    assert impact.jobs_removed == ("j2",)
    assert impact.is_structural
    assert impact.job_count_delta == -1


def test_impact_param_change_not_structural() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeDeadline("j1", 20.0))
    impact = analyze_change(p, p2)
    assert impact.jobs_with_changed_deadline == ("j1",)
    assert not impact.is_structural


def test_impact_weight_change() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeWeight("j2", 9.0))
    impact = analyze_change(p, p2)
    assert impact.jobs_with_changed_weight == ("j2",)


def test_impact_added_resource_is_structural() -> None:
    p = _problem()
    p2 = apply_change(p, AddResource(Resource(id="r2")))
    impact = analyze_change(p, p2)
    assert impact.resources_added == ("r2",)
    assert impact.resource_count_delta == 1


def test_impact_capacity_and_speed_changes() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeResourceCapacity("r0", {"cpu": 8.0}))
    p2 = apply_change(p2, ChangeResourceSpeed("r1", 2.0))
    impact = analyze_change(p, p2)
    assert impact.resources_with_changed_capacity == ("r0",)
    assert impact.resources_with_changed_speed == ("r1",)


def test_impact_versions_recorded() -> None:
    p = _problem()
    p2 = apply_change(p, ChangeWeight("j1", 2.0))
    impact = analyze_change(p, p2)
    assert impact.version_before == p.version
    assert impact.version_after == p2.version == p.version + 1
