import pytest

from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.domain.solution import Assignment, Solution
from edmo.infrastructure.workers.job import FitnessJob
from edmo.infrastructure.workers.local import (
    WorkerConfig,
    evaluate_job,
    submit_all,
)


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(Job(id="j1", processing_time=1.0),),
    )


def _job(name: str) -> FitnessJob:
    return FitnessJob(
        job_id=name,
        problem=_problem(),
        solution=Solution(assignments={"j1": Assignment(resource_id="r0", start_time=0.0)}),
    )


def test_config_defaults() -> None:
    cfg = WorkerConfig()
    assert cfg.kind == "process"
    assert cfg.max_workers == 4
    assert cfg.max_retries == 0


def test_config_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        WorkerConfig(max_workers=0)
    with pytest.raises(ValueError):
        WorkerConfig(timeout_per_job=0.0)
    with pytest.raises(ValueError):
        WorkerConfig(max_retries=-1)


def test_evaluate_job_success() -> None:
    r = evaluate_job(_job("a"))
    assert r.succeeded
    assert r.job_id == "a"
    assert r.total == r.objective + r.penalty


def test_submit_all_empty() -> None:
    assert submit_all([]) == []


def test_submit_all_threads_preserves_order() -> None:
    jobs = [_job(f"j{i}") for i in range(5)]
    results = submit_all(jobs, WorkerConfig(kind="thread", max_workers=2))
    assert [r.job_id for r in results] == [j.job_id for j in jobs]
    assert all(r.succeeded for r in results)


def test_submit_all_processes() -> None:
    jobs = [_job(f"j{i}") for i in range(3)]
    results = submit_all(jobs, WorkerConfig(kind="process", max_workers=2))
    assert all(r.succeeded for r in results)


def test_submit_all_captures_evaluation_error() -> None:
    from edmo.domain.problem import Problem

    bad = FitnessJob(job_id="bad", problem=object.__new__(Problem), solution=Solution())
    results = submit_all([bad], WorkerConfig(kind="thread", max_workers=1))
    assert len(results) == 1
    assert not results[0].succeeded
    assert results[0].error is not None
