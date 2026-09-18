import pytest

from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.domain.solution import Assignment, Solution
from edmo.infrastructure.workers.job import FitnessJob, FitnessResult


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(Job(id="j1", processing_time=1.0),),
    )


def _solution() -> Solution:
    return Solution(assignments={"j1": Assignment(resource_id="r0", start_time=0.0)})


def test_job_valid() -> None:
    j = FitnessJob(job_id="a", problem=_problem(), solution=_solution())
    assert j.job_id == "a"


def test_job_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        FitnessJob(job_id="", problem=_problem(), solution=_solution())


def test_result_succeeded_flag() -> None:
    r = FitnessResult(job_id="a", objective=1.0, penalty=0.0, total=1.0, feasible=True)
    assert r.succeeded
    r2 = FitnessResult(
        job_id="a",
        objective=float("inf"),
        penalty=float("inf"),
        total=float("inf"),
        feasible=False,
        error="boom",
    )
    assert not r2.succeeded
