from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from edmo.domain.evaluation import evaluate
from edmo.infrastructure.workers.errors import JobExecutionError
from edmo.infrastructure.workers.job import FitnessJob, FitnessResult

ExecutorKind = Literal["thread", "process"]


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    """Configuration for the local fitness executor.

    ``kind`` selects the executor backend. ``process`` offers true CPU
    parallelism at the cost of pickling; ``thread`` is safer for debugging
    but is limited by the GIL for pure-Python fitness functions.

    ``max_workers`` bounds concurrency. ``timeout_per_job`` is a soft
    per-job timeout enforced by the caller (``submit_all``), not by the
    executor itself.
    """

    kind: ExecutorKind = "process"
    max_workers: int = 4
    timeout_per_job: float | None = None
    max_retries: int = 0

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.timeout_per_job is not None and self.timeout_per_job <= 0.0:
            raise ValueError("timeout_per_job must be positive when set")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")


def evaluate_job(job: FitnessJob) -> FitnessResult:
    """Pure evaluation of a single job.

    Kept as a module-level function so it can be pickled by the process
    executor. Any exception is captured and returned as an ``error`` field
    rather than raised, so one bad job cannot take down a batch.
    """
    try:
        if job.weights is None:
            evaluation = evaluate(job.problem, job.solution)
        else:
            evaluation = evaluate(job.problem, job.solution, job.weights)
    except Exception as exc:
        return FitnessResult(
            job_id=job.job_id,
            objective=float("inf"),
            penalty=float("inf"),
            total=float("inf"),
            feasible=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    return FitnessResult(
        job_id=job.job_id,
        objective=evaluation.objective,
        penalty=evaluation.penalty,
        total=evaluation.total,
        feasible=evaluation.feasible,
    )


def _make_executor(
    kind: ExecutorKind,
    max_workers: int,
) -> ProcessPoolExecutor | ThreadPoolExecutor:
    if kind == "process":
        return ProcessPoolExecutor(max_workers=max_workers)
    if kind == "thread":
        return ThreadPoolExecutor(max_workers=max_workers)
    raise ValueError(f"unsupported executor kind: {kind!r}")


def submit_all(
    jobs: list[FitnessJob],
    config: WorkerConfig | None = None,
) -> list[FitnessResult]:
    """Evaluate ``jobs`` in parallel and return results in submission order.

    On timeout or per-job exceptions, the corresponding entry in the result
    list carries ``error`` set and ``total == inf``. Jobs are retried up to
    ``config.max_retries`` additional times when they fail.
    """
    cfg = config or WorkerConfig()
    if not jobs:
        return []

    results: list[FitnessResult | None] = [None] * len(jobs)
    pending: list[int] = list(range(len(jobs)))
    attempt = 0

    with _make_executor(cfg.kind, cfg.max_workers) as executor:
        while pending and attempt <= cfg.max_retries:
            futures: dict[Future[FitnessResult], int] = {
                executor.submit(evaluate_job, jobs[i]): i for i in pending
            }
            next_pending: list[int] = []
            for future, idx in futures.items():
                try:
                    result = future.result(timeout=cfg.timeout_per_job)
                except TimeoutError:
                    next_pending.append(idx)
                    continue
                except Exception as exc:
                    result = FitnessResult(
                        job_id=jobs[idx].job_id,
                        objective=float("inf"),
                        penalty=float("inf"),
                        total=float("inf"),
                        feasible=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                if result.succeeded:
                    results[idx] = result
                else:
                    next_pending.append(idx)
            pending = next_pending
            attempt += 1

    for idx in pending:
        results[idx] = FitnessResult(
            job_id=jobs[idx].job_id,
            objective=float("inf"),
            penalty=float("inf"),
            total=float("inf"),
            feasible=False,
            error=str(JobExecutionError(f"job {jobs[idx].job_id!r} failed after retries")),
        )

    final: list[FitnessResult] = []
    for idx, r in enumerate(results):
        if r is None:
            raise AssertionError(f"unexpected None result for job index {idx}")
        final.append(r)
    return final
