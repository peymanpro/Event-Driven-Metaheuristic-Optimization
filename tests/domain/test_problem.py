import pytest

from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _resources() -> tuple[Resource, ...]:
    return (Resource(id="r1"), Resource(id="r2"))


def _jobs() -> tuple[Job, ...]:
    return (Job(id="j1", processing_time=1.0), Job(id="j2", processing_time=2.0))


def test_problem_valid() -> None:
    p = Problem(id="p1", resources=_resources(), jobs=_jobs())
    assert p.id == "p1"
    assert p.version == 1
    assert len(p.resources) == 2
    assert len(p.jobs) == 2


def test_problem_metadata_is_immutable() -> None:
    p = Problem(id="p1", resources=_resources(), jobs=_jobs(), metadata={"k": "v"})
    with pytest.raises(TypeError):
        p.metadata["k"] = "w"  # type: ignore[index]


def test_problem_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        Problem(id="", resources=_resources(), jobs=_jobs())


def test_problem_rejects_empty_resources() -> None:
    with pytest.raises(ValueError):
        Problem(id="p1", resources=(), jobs=_jobs())


def test_problem_rejects_empty_jobs() -> None:
    with pytest.raises(ValueError):
        Problem(id="p1", resources=_resources(), jobs=())


def test_problem_rejects_duplicate_resource_ids() -> None:
    with pytest.raises(ValueError):
        Problem(id="p1", resources=(Resource(id="r1"), Resource(id="r1")), jobs=_jobs())


def test_problem_rejects_duplicate_job_ids() -> None:
    with pytest.raises(ValueError):
        Problem(
            id="p1",
            resources=_resources(),
            jobs=(Job(id="j1", processing_time=1.0), Job(id="j1", processing_time=2.0)),
        )


def test_problem_rejects_unknown_predecessor() -> None:
    jobs = (
        Job(id="j1", processing_time=1.0, predecessors=frozenset({"missing"})),
    )
    with pytest.raises(ValueError):
        Problem(id="p1", resources=_resources(), jobs=jobs)


def test_problem_rejects_invalid_version() -> None:
    with pytest.raises(ValueError):
        Problem(id="p1", resources=_resources(), jobs=_jobs(), version=0)


def test_problem_lookup_helpers() -> None:
    p = Problem(id="p1", resources=_resources(), jobs=_jobs())
    assert p.resource_by_id("r1").id == "r1"
    assert p.job_by_id("j2").id == "j2"
    with pytest.raises(KeyError):
        p.resource_by_id("nope")
    with pytest.raises(KeyError):
        p.job_by_id("nope")
