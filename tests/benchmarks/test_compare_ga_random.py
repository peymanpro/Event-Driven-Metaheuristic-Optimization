import pytest

from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.random_search import RandomSearchConfig
from edmo.benchmarks.compare_ga_random import ComparisonResult, compare_ga_vs_random
from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.domain.problem import Problem


def _problem() -> Problem:
    return random_problem(RandomProblemSpec(n_resources=2, n_jobs=6, seed=42))


def test_compare_returns_result() -> None:
    p = _problem()
    result = compare_ga_vs_random(
        p,
        RandomSearchConfig(iterations=100, seed=1),
        GAConfig(population_size=20, generations=20, seed=1),
    )
    assert isinstance(result, ComparisonResult)
    assert result.problem_id == p.id
    assert result.random_search.iterations == 100
    assert result.ga.generations == 20


def test_compare_is_reproducible() -> None:
    p = _problem()
    r1 = compare_ga_vs_random(
        p,
        RandomSearchConfig(iterations=200, seed=7),
        GAConfig(population_size=20, generations=20, seed=7),
    )
    r2 = compare_ga_vs_random(
        p,
        RandomSearchConfig(iterations=200, seed=7),
        GAConfig(population_size=20, generations=20, seed=7),
    )
    assert r1.ga.best_evaluation.total == r2.ga.best_evaluation.total
    assert r1.random_search.best_evaluation.total == r2.random_search.best_evaluation.total


def test_ga_beats_random_on_reasonable_budget() -> None:
    """With the same seed and a comparable budget, GA should not lose to random."""
    p = _problem()
    result = compare_ga_vs_random(
        p,
        RandomSearchConfig(iterations=300, seed=3),
        GAConfig(population_size=30, generations=40, seed=3),
    )
    # GA must be at least as good as random search here.
    assert result.ga.best_evaluation.total <= result.random_search.best_evaluation.total


def test_improvement_relative_matches_absolute() -> None:
    p = _problem()
    result = compare_ga_vs_random(
        p,
        RandomSearchConfig(iterations=50, seed=1),
        GAConfig(population_size=10, generations=5, seed=1),
    )
    rs_total = result.random_search.best_evaluation.total
    ga_total = result.ga.best_evaluation.total
    assert result.improvement_absolute == pytest.approx(rs_total - ga_total)
    expected_rel = (rs_total - ga_total) / (abs(rs_total) + 1e-12)
    assert result.improvement_relative == pytest.approx(expected_rel)
