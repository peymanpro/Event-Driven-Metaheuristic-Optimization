from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import mean, median

from edmo.algorithms.genetic.config import GAConfig
from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.benchmarks.restart_vs_warmstart import (
    RecoveryMetrics,
    RestartWarmStartResult,
    restart_vs_warm_start,
)
from edmo.domain.change import ChangeWeight, ProblemChange
from edmo.domain.problem import Problem

ChangeFactory = Callable[[Problem], ProblemChange]
GAConfigFactory = Callable[[int], GAConfig]


@dataclass(frozen=True, slots=True)
class StrategyAggregate:
    """Aggregated statistics for one strategy across multiple seeds.

    ``iterations_to_target`` statistics only include seeds where the strategy
    actually reached the target. ``success_rate`` is the fraction of seeds
    where the strategy reached the target. ``paired_mean_iterations_saved``
    (on the parent result) is computed only on seeds where both strategies
    reached the target.
    """

    label: str
    runs: int
    success_count: int
    success_rate: float
    median_iterations_to_target: float | None
    mean_iterations_to_target: float | None
    stdev_iterations_to_target: float | None
    best_iterations_to_target: int | None
    worst_iterations_to_target: int | None
    mean_initial_total: float
    mean_best_total: float
    mean_final_diversity: float
    feasible_fraction: float


@dataclass(frozen=True, slots=True)
class RepeatedRecoveryResult:
    seeds: tuple[int, ...]
    change_descriptions: tuple[str, ...]
    restart: StrategyAggregate
    warm_start: StrategyAggregate
    paired_iterations_saved: tuple[int, ...]
    paired_mean_iterations_saved: float | None
    paired_median_iterations_saved: float | None


def _describe_change(change: ProblemChange) -> str:
    return type(change).__name__


def _stdev(values: Sequence[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return float(var ** 0.5)


def _aggregate(label: str, metrics: Sequence[RecoveryMetrics]) -> StrategyAggregate:
    iterations_int = [
        int(m.iterations_to_target)
        for m in metrics
        if m.iterations_to_target is not None
    ]
    success = sum(1 for m in metrics if m.reached_target)
    n = len(metrics)
    feasible = sum(1 for m in metrics if m.best_evaluation.feasible)
    stdev: float | None = (
        _stdev([float(i) for i in iterations_int]) if iterations_int else None
    )
    return StrategyAggregate(
        label=label,
        runs=n,
        success_count=success,
        success_rate=success / n if n else 0.0,
        median_iterations_to_target=(
            float(median(iterations_int)) if iterations_int else None
        ),
        mean_iterations_to_target=(
            float(mean(iterations_int)) if iterations_int else None
        ),
        stdev_iterations_to_target=stdev,
        best_iterations_to_target=min(iterations_int) if iterations_int else None,
        worst_iterations_to_target=max(iterations_int) if iterations_int else None,
        mean_initial_total=mean([m.initial_total for m in metrics]) if n else 0.0,
        mean_best_total=mean([m.best_total for m in metrics]) if n else 0.0,
        mean_final_diversity=(
            mean([m.final_diversity for m in metrics]) if n else 0.0
        ),
        feasible_fraction=feasible / n if n else 0.0,
    )


def _run_single_seed(
    seed: int,
    spec_kwargs: dict[str, int | float],
    change_factory: ChangeFactory,
    ga_config_factory: GAConfigFactory,
) -> RestartWarmStartResult:
    problem = random_problem(
        RandomProblemSpec(
            seed=seed,
            n_resources=int(spec_kwargs.get("n_resources", 3)),
            n_jobs=int(spec_kwargs.get("n_jobs", 8)),
            max_processing_time=float(spec_kwargs.get("max_processing_time", 5.0)),
            max_release_time=float(spec_kwargs.get("max_release_time", 2.0)),
            max_deadline_slack=float(spec_kwargs.get("max_deadline_slack", 10.0)),
        )
    )
    change = change_factory(problem)
    cfg = ga_config_factory(seed)
    return restart_vs_warm_start(problem, [change], cfg)


def _default_ga_config_factory(seed: int) -> GAConfig:
    return GAConfig(
        population_size=15,
        generations=15,
        crossover_rate=0.9,
        mutation_rate=0.15,
        tournament_size=3,
        elite_count=2,
        seed=seed,
    )


def _default_change_factory(problem: Problem) -> ProblemChange:
    if problem.jobs:
        return ChangeWeight(problem.jobs[0].id, 2.0)
    raise ValueError("problem must have at least one job")


def repeated_recovery_benchmark(
    seeds: Sequence[int],
    *,
    spec_kwargs: dict[str, int | float] | None = None,
    ga_config_factory: GAConfigFactory | None = None,
    change_factory: ChangeFactory | None = None,
) -> RepeatedRecoveryResult:
    """Run restart vs warm-start across multiple seeds and aggregate.

    ``ga_config_factory(seed)`` must return a :class:`GAConfig` bound to that
    seed. ``change_factory(problem)`` must return a single
    :class:`ProblemChange` applied to that seed's problem.
    """
    if not seeds:
        raise ValueError("seeds must not be empty")
    ga_factory: GAConfigFactory = ga_config_factory or _default_ga_config_factory
    change_fac: ChangeFactory = change_factory or _default_change_factory
    spec_kw: dict[str, int | float] = (
        spec_kwargs if spec_kwargs is not None else {"n_resources": 2, "n_jobs": 6}
    )

    restart_metrics: list[RecoveryMetrics] = []
    warm_metrics: list[RecoveryMetrics] = []
    paired_saved: list[int] = []
    change_descriptions: list[str] = []

    for seed in seeds:
        result = _run_single_seed(seed, spec_kw, change_fac, ga_factory)
        restart_metrics.append(result.restart)
        warm_metrics.append(result.warm_start)
        change_descriptions.append(
            _describe_change(change_fac(result.problem_before))
        )
        if result.iterations_saved is not None:
            paired_saved.append(int(result.iterations_saved))

    restart_agg = _aggregate("restart", restart_metrics)
    warm_agg = _aggregate("warm_start", warm_metrics)
    paired_mean = mean(paired_saved) if paired_saved else None
    paired_median = float(median(paired_saved)) if paired_saved else None

    return RepeatedRecoveryResult(
        seeds=tuple(seeds),
        change_descriptions=tuple(change_descriptions),
        restart=restart_agg,
        warm_start=warm_agg,
        paired_iterations_saved=tuple(paired_saved),
        paired_mean_iterations_saved=paired_mean,
        paired_median_iterations_saved=paired_median,
    )
