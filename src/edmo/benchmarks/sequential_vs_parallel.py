from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from edmo.domain.evaluation import PenaltyWeights
from edmo.domain.problem import Problem
from edmo.domain.solution import Solution
from edmo.infrastructure.workers.job import FitnessJob, FitnessResult
from edmo.infrastructure.workers.local import WorkerConfig, evaluate_job, submit_all


@dataclass(frozen=True, slots=True)
class TimingResult:
    label: str
    job_count: int
    wall_time_seconds: float
    throughput_per_second: float
    success_count: int
    failure_count: int
    results: tuple[FitnessResult, ...]


@dataclass(frozen=True, slots=True)
class SequentialParallelResult:
    sequential: TimingResult
    parallel: TimingResult

    @property
    def speedup(self) -> float:
        """Wall-time speedup of parallel over sequential (>1 means parallel wins)."""
        if self.parallel.wall_time_seconds <= 0.0:
            return 0.0
        return self.sequential.wall_time_seconds / self.parallel.wall_time_seconds


def _timing(label: str, results: list[FitnessResult], wall: float) -> TimingResult:
    success = sum(1 for r in results if r.succeeded)
    failure = len(results) - success
    throughput = len(results) / wall if wall > 0.0 else 0.0
    return TimingResult(
        label=label,
        job_count=len(results),
        wall_time_seconds=wall,
        throughput_per_second=throughput,
        success_count=success,
        failure_count=failure,
        results=tuple(results),
    )


def _make_jobs(
    problem: Problem,
    solutions: list[Solution],
    weights: PenaltyWeights | None,
) -> list[FitnessJob]:
    return [
        FitnessJob(job_id=f"job-{i}", problem=problem, solution=s, weights=weights)
        for i, s in enumerate(solutions)
    ]


def compare_sequential_vs_parallel(
    problem: Problem,
    solutions: list[Solution],
    weights: PenaltyWeights | None = None,
    parallel_config: WorkerConfig | None = None,
) -> SequentialParallelResult:
    """Time the same fitness workload sequentially and in parallel.

    Results are compared by job_id so the benchmark can assert equivalence
    between the two execution modes. The benchmark is wall-time based and
    therefore inherently noisy; it is intended as a reproducible sanity
    check, not as a microbenchmark guarantee.
    """
    cfg = parallel_config or WorkerConfig(kind="process", max_workers=4)
    jobs = _make_jobs(problem, solutions, weights)

    t0 = perf_counter()
    seq_results = [evaluate_job(job) for job in jobs]
    seq_wall = perf_counter() - t0

    t0 = perf_counter()
    par_results = submit_all(jobs, cfg)
    par_wall = perf_counter() - t0

    return SequentialParallelResult(
        sequential=_timing("sequential", seq_results, seq_wall),
        parallel=_timing("parallel", par_results, par_wall),
    )


def results_match_by_job_id(
    sequential: TimingResult,
    parallel: TimingResult,
    *,
    atol: float = 1.0e-9,
) -> bool:
    """Return True when both runs produced identical totals per job_id."""
    seq = {r.job_id: r for r in sequential.results}
    par = {r.job_id: r for r in parallel.results}
    if seq.keys() != par.keys():
        return False
    for job_id, s in seq.items():
        p = par[job_id]
        if s.succeeded != p.succeeded:
            return False
        if not s.succeeded:
            continue
        if abs(s.total - p.total) > atol:
            return False
    return True
