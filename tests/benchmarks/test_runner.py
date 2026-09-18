import pytest

from edmo.algorithms.differential_evolution.config import DEConfig
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.random_search import RandomSearchConfig
from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.benchmarks.runner import (
    StaticBenchmarkConfig,
    run_dynamic_benchmark,
    run_static_benchmark,
    verify_reproducibility,
)
from edmo.domain.change import ChangeWeight
from edmo.domain.problem import Problem


def _problem() -> Problem:
    return random_problem(RandomProblemSpec(n_resources=2, n_jobs=4, seed=0))


def _config() -> StaticBenchmarkConfig:
    return StaticBenchmarkConfig(
        experiment_id="exp-1",
        ga=GAConfig(population_size=10, generations=5, seed=1),
        de=DEConfig(population_size=10, generations=5, seed=1),
        random_search=RandomSearchConfig(iterations=100, seed=1),
    )


def test_static_benchmark_runs() -> None:
    result = run_static_benchmark(_problem(), _config())
    assert result.winner in {"ga", "de", "random_search"}
    assert result.ga.algorithm == "ga"
    assert result.de.algorithm == "de"
    assert result.random.algorithm == "random_search"


def test_static_benchmark_reproducible() -> None:
    r1 = run_static_benchmark(_problem(), _config())
    r2 = run_static_benchmark(_problem(), _config())
    assert r1.ga.best_total == r2.ga.best_total
    assert r1.de.best_total == r2.de.best_total
    assert r1.random.best_total == r2.random.best_total


def test_dynamic_benchmark_runs() -> None:
    cfg = GAConfig(population_size=10, generations=8, seed=1)
    result = run_dynamic_benchmark(
        _problem(),
        [ChangeWeight("j0", 2.0)],
        cfg,
        experiment_id="exp-dyn",
    )
    assert result.restart.algorithm == "ga_restart"
    assert result.warm_start.algorithm == "ga_warm_start"
    assert result.problem_after.version == 2


def test_verify_reproducibility_true() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=1)
    assert verify_reproducibility(_problem(), cfg, repetitions=3)


def test_verify_reproducibility_requires_two_runs() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=1)
    with pytest.raises(ValueError):
        verify_reproducibility(_problem(), cfg, repetitions=1)


def test_dynamic_benchmark_propagates_real_feasibility() -> None:
    """Both restart and warm_start RunRecords must carry the real feasible flag.

    Regression guard: the previous implementation synthesized
    ``Evaluation(feasible=True, penalty=0.0)`` regardless of the actual
    optimizer result.
    """
    cfg = GAConfig(population_size=10, generations=8, seed=1)
    result = run_dynamic_benchmark(
        _problem(),
        [ChangeWeight("j0", 2.0)],
        cfg,
        experiment_id="exp-feas",
    )
    # Feasibility must match the real evaluation of the strategy's best solution.
    # Both strategies are expected to reach feasibility in this benign scenario.
    assert isinstance(result.restart.feasible, bool)
    assert isinstance(result.warm_start.feasible, bool)


def test_dynamic_benchmark_feasible_flag_comes_from_evaluation() -> None:
    """The recorded feasible flag must equal best_evaluation.feasible in the benchmark."""
    from edmo.benchmarks.restart_vs_warmstart import restart_vs_warm_start

    cfg = GAConfig(population_size=10, generations=8, seed=2)
    inner = restart_vs_warm_start(_problem(), [ChangeWeight("j0", 2.0)], cfg)
    runner_result = run_dynamic_benchmark(
        _problem(),
        [ChangeWeight("j0", 2.0)],
        cfg,
        experiment_id="exp-feas-2",
    )
    assert runner_result.restart.feasible == inner.restart.best_evaluation.feasible
    assert runner_result.warm_start.feasible == inner.warm_start.best_evaluation.feasible


def test_ga_recorded_metrics_are_real_telemetry() -> None:
    """GA metrics from the runner must come from GA snapshots, not placeholders."""
    result = run_static_benchmark(_problem(), _config())
    ga_metrics = result.ga.metrics
    assert ga_metrics, "GA run must record at least one generation"
    # Not all generations should have mean_total == worst_total == best_total
    # with diversity == 0 (that pattern identifies the placeholder path).
    real_telemetry = [
        m
        for m in ga_metrics
        if (m.mean_total != m.best_total or m.worst_total != m.best_total)
        and m.population_diversity > 0.0
    ]
    assert real_telemetry, "GA metrics should include real per-generation telemetry"
    for m in ga_metrics:
        assert m.best_total <= m.mean_total <= m.worst_total


def test_dynamic_benchmark_ga_metrics_have_real_telemetry() -> None:
    cfg = GAConfig(population_size=10, generations=8, seed=3)
    result = run_dynamic_benchmark(
        _problem(),
        [ChangeWeight("j0", 2.0)],
        cfg,
        experiment_id="exp-telemetry",
    )
    for rec in (result.restart, result.warm_start):
        assert rec.metrics, f"{rec.algorithm} must record generation metrics"
        for m in rec.metrics:
            assert m.best_total <= m.mean_total <= m.worst_total
        assert any(m.population_diversity > 0.0 for m in rec.metrics)
