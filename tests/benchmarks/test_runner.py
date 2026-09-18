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
