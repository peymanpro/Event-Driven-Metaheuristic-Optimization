import pytest

from edmo.domain.jobs import Job


def test_job_defaults() -> None:
    j = Job(id="j1", processing_time=5.0)
    assert j.id == "j1"
    assert j.processing_time == 5.0
    assert dict(j.demand) == {}
    assert j.release_time == 0.0
    assert j.deadline is None
    assert j.weight == 1.0
    assert j.predecessors == frozenset()


def test_job_valid_full() -> None:
    j = Job(
        id="j1",
        processing_time=5.0,
        demand={"cpu": 2.0},
        release_time=1.0,
        deadline=10.0,
        weight=3.0,
        predecessors=frozenset({"j0"}),
    )
    assert j.demand["cpu"] == 2.0
    assert j.deadline == 10.0
    assert j.predecessors == frozenset({"j0"})


def test_job_demand_is_immutable() -> None:
    j = Job(id="j1", processing_time=1.0, demand={"cpu": 1.0})
    with pytest.raises(TypeError):
        j.demand["cpu"] = 2.0  # type: ignore[index]


def test_job_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        Job(id="", processing_time=1.0)


def test_job_rejects_non_positive_processing_time() -> None:
    with pytest.raises(ValueError):
        Job(id="j1", processing_time=0.0)
    with pytest.raises(ValueError):
        Job(id="j1", processing_time=-1.0)


def test_job_rejects_negative_release_time() -> None:
    with pytest.raises(ValueError):
        Job(id="j1", processing_time=1.0, release_time=-1.0)


def test_job_rejects_deadline_before_release() -> None:
    with pytest.raises(ValueError):
        Job(id="j1", processing_time=1.0, release_time=5.0, deadline=3.0)


def test_job_rejects_non_positive_weight() -> None:
    with pytest.raises(ValueError):
        Job(id="j1", processing_time=1.0, weight=0.0)


def test_job_rejects_negative_demand() -> None:
    with pytest.raises(ValueError):
        Job(id="j1", processing_time=1.0, demand={"cpu": -1.0})
