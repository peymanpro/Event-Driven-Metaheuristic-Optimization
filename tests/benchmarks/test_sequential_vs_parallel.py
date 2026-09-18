from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.benchmarks.sequential_vs_parallel import (
    SequentialParallelResult,
    compare_sequential_vs_parallel,
    results_match_by_job_id,
)
from edmo.domain.problem import Problem
from edmo.domain.solution import Assignment, Solution
from edmo.infrastructure.workers.local import WorkerConfig


def _problem() -> Problem:
    return random_problem(RandomProblemSpec(n_resources=2, n_jobs=5, seed=0))


def _solutions(problem: Problem) -> list[Solution]:
    out: list[Solution] = []
    for k in range(8):
        assignments = {
            j.id: Assignment(
                resource_id=problem.resources[k % len(problem.resources)].id,
                start_time=float(k),
            )
            for j in problem.jobs
        }
        out.append(Solution(assignments=assignments))
    return out


def test_compare_returns_result() -> None:
    p = _problem()
    result = compare_sequential_vs_parallel(
        p,
        _solutions(p),
        parallel_config=WorkerConfig(kind="thread", max_workers=2),
    )
    assert isinstance(result, SequentialParallelResult)
    assert result.sequential.job_count == 8
    assert result.parallel.job_count == 8


def test_sequential_and_parallel_agree() -> None:
    p = _problem()
    result = compare_sequential_vs_parallel(
        p,
        _solutions(p),
        parallel_config=WorkerConfig(kind="process", max_workers=2),
    )
    assert results_match_by_job_id(result.sequential, result.parallel)


def test_speedup_positive() -> None:
    p = _problem()
    result = compare_sequential_vs_parallel(
        p,
        _solutions(p),
        parallel_config=WorkerConfig(kind="process", max_workers=2),
    )
    assert result.speedup > 0.0


def test_empty_workload() -> None:
    p = _problem()
    result = compare_sequential_vs_parallel(p, [], parallel_config=WorkerConfig(kind="thread"))
    assert result.sequential.job_count == 0
    assert result.parallel.job_count == 0
