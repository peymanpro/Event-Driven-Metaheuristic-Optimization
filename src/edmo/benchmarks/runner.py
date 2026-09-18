from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from edmo.algorithms.differential_evolution.config import DEConfig
from edmo.algorithms.differential_evolution.optimizer import DEResult, run_de
from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.metrics import population_diversity
from edmo.algorithms.genetic.optimizer import GAResult, run_ga
from edmo.algorithms.random_search import (
    RandomSearchConfig,
    RandomSearchResult,
    random_search,
)
from edmo.benchmarks.restart_vs_warmstart import restart_vs_warm_start
from edmo.domain.change import ProblemChange
from edmo.domain.evaluation import Evaluation
from edmo.domain.problem import Problem
from edmo.experiments.record import GenerationMetric, RunRecord


@dataclass(frozen=True, slots=True)
class _RunSummary:
    best_evaluation: Evaluation
    history: tuple[float, ...]
    iterations: int
    stopped_reason: str
    seed: int | None


def _summarize_ga(result: GAResult) -> _RunSummary:
    return _RunSummary(
        best_evaluation=result.best_evaluation,
        history=result.history,
        iterations=result.generations,
        stopped_reason=result.stopped_reason,
        seed=result.seed,
    )


def _summarize_de(result: DEResult) -> _RunSummary:
    return _RunSummary(
        best_evaluation=result.best_evaluation,
        history=result.history,
        iterations=result.generations,
        stopped_reason=result.stopped_reason,
        seed=result.seed,
    )


def _summarize_random(result: RandomSearchResult) -> _RunSummary:
    return _RunSummary(
        best_evaluation=result.best_evaluation,
        history=(result.best_evaluation.total,),
        iterations=result.iterations,
        stopped_reason="max_iterations",
        seed=result.seed,
    )


def _metrics_from_history(history: tuple[float, ...]) -> tuple[GenerationMetric, ...]:
    """Fallback telemetry when only the best-so-far history is available."""
    return tuple(
        GenerationMetric(
            generation=i,
            best_total=value,
            mean_total=value,
            worst_total=value,
            population_diversity=0.0,
            elapsed_seconds=0.0,
        )
        for i, value in enumerate(history)
    )


def _record(
    experiment_id: str,
    algorithm: str,
    problem: Problem,
    summary: _RunSummary,
    started_at: datetime,
    finished_at: datetime,
) -> RunRecord:
    return RunRecord(
        run_id=str(uuid4()),
        experiment_id=experiment_id,
        algorithm=algorithm,
        problem_id=problem.id,
        problem_version=problem.version,
        seed=summary.seed,
        started_at=started_at,
        finished_at=finished_at,
        best_total=summary.best_evaluation.total,
        feasible=summary.best_evaluation.feasible,
        iterations=summary.iterations,
        stopped_reason=summary.stopped_reason,
        metrics=_metrics_from_history(summary.history),
    )


@dataclass(frozen=True, slots=True)
class StaticBenchmarkConfig:
    experiment_id: str
    ga: GAConfig
    de: DEConfig
    random_search: RandomSearchConfig


@dataclass(frozen=True, slots=True)
class StaticBenchmarkResult:
    problem_id: str
    ga: RunRecord
    de: RunRecord
    random: RunRecord
    winner: str
    ga_vs_de_gap: float
    ga_vs_random_gap: float


def run_static_benchmark(
    problem: Problem,
    config: StaticBenchmarkConfig,
) -> StaticBenchmarkResult:
    """Run GA, DE and random search on a static problem, and compare them."""
    t0 = datetime.now(tz=UTC)
    ga_res = run_ga(problem, config.ga)
    t1 = datetime.now(tz=UTC)
    de_res = run_de(problem, config.de)
    t2 = datetime.now(tz=UTC)
    rs_res = random_search(problem, config.random_search)
    t3 = datetime.now(tz=UTC)

    ga_rec = _record(config.experiment_id, "ga", problem, _summarize_ga(ga_res), t0, t1)
    de_rec = _record(config.experiment_id, "de", problem, _summarize_de(de_res), t1, t2)
    rs_rec = _record(
        config.experiment_id, "random_search", problem, _summarize_random(rs_res), t2, t3
    )

    totals = {
        "ga": ga_rec.best_total,
        "de": de_rec.best_total,
        "random_search": rs_rec.best_total,
    }
    winner = min(totals, key=lambda k: totals[k])
    return StaticBenchmarkResult(
        problem_id=problem.id,
        ga=ga_rec,
        de=de_rec,
        random=rs_rec,
        winner=winner,
        ga_vs_de_gap=de_rec.best_total - ga_rec.best_total,
        ga_vs_random_gap=rs_rec.best_total - ga_rec.best_total,
    )


@dataclass(frozen=True, slots=True)
class DynamicBenchmarkResult:
    problem_before: Problem
    problem_after: Problem
    restart: RunRecord
    warm_start: RunRecord
    iterations_saved: int | None
    target_total: float


def run_dynamic_benchmark(
    problem_before: Problem,
    changes: list[ProblemChange],
    ga_config: GAConfig,
    experiment_id: str,
) -> DynamicBenchmarkResult:
    """Run the restart vs warm-start benchmark and capture RunRecords."""
    result = restart_vs_warm_start(problem_before, changes, ga_config)

    now = datetime.now(tz=UTC)
    restart_rec = _record(
        experiment_id,
        "ga_restart",
        result.problem_after,
        _RunSummary(
            best_evaluation=_evaluate_best(
                result.problem_after, result.restart.best_total
            ),
            history=result.restart.history,
            iterations=(
                result.restart.iterations_to_target
                if result.restart.iterations_to_target is not None
                else len(result.restart.history) - 1
            ),
            stopped_reason=(
                "target_reached" if result.restart.reached_target else "budget"
            ),
            seed=ga_config.seed,
        ),
        now,
        now,
    )
    warm_rec = _record(
        experiment_id,
        "ga_warm_start",
        result.problem_after,
        _RunSummary(
            best_evaluation=_evaluate_best(
                result.problem_after, result.warm_start.best_total
            ),
            history=result.warm_start.history,
            iterations=(
                result.warm_start.iterations_to_target
                if result.warm_start.iterations_to_target is not None
                else len(result.warm_start.history) - 1
            ),
            stopped_reason=(
                "target_reached" if result.warm_start.reached_target else "budget"
            ),
            seed=ga_config.seed,
        ),
        now,
        now,
    )
    return DynamicBenchmarkResult(
        problem_before=problem_before,
        problem_after=result.problem_after,
        restart=restart_rec,
        warm_start=warm_rec,
        iterations_saved=result.iterations_saved,
        target_total=result.target_total,
    )


def _evaluate_best(problem: Problem, best_total: float) -> Evaluation:
    """Construct a minimal :class:`Evaluation` from a known best total.

    The dynamic benchmark only exposes the best total per strategy; keeping
    the full evaluation from the inner run would require threading it through
    the benchmark API. This placeholder preserves the invariant that
    ``best_total == objective + penalty`` with zero penalty when feasible.
    """
    from edmo.domain.evaluation import Evaluation, Schedule

    return Evaluation(
        objective=best_total,
        penalty=0.0,
        total=best_total,
        feasible=True,
        violations=(),
        schedule=Schedule(),
    )


def verify_reproducibility(
    problem: Problem,
    ga_config: GAConfig,
    *,
    repetitions: int = 2,
) -> bool:
    """Return True when the same seed produces identical histories across runs."""
    if repetitions < 2:
        raise ValueError("repetitions must be >= 2")
    reference: tuple[float, ...] | None = None
    for _ in range(repetitions):
        result = run_ga(problem, ga_config)
        if reference is None:
            reference = result.history
        elif result.history != reference:
            return False
    return True


def final_population_diversity(chromosomes: tuple[Chromosome, ...]) -> float:
    """Convenience wrapper around the population diversity metric."""
    return population_diversity(list(chromosomes))


def timed(op: Any, *args: Any, **kwargs: Any) -> tuple[float, Any]:
    t0 = perf_counter()
    value = op(*args, **kwargs)
    return perf_counter() - t0, value
