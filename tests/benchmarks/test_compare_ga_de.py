import pytest

from edmo.algorithms.differential_evolution.config import DEConfig
from edmo.algorithms.genetic.config import GAConfig
from edmo.benchmarks.compare_ga_de import GADEResult, compare_ga_vs_de
from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.domain.problem import Problem


def _problem() -> Problem:
    return random_problem(RandomProblemSpec(n_resources=2, n_jobs=5, seed=42))


def test_compare_returns_result() -> None:
    result = compare_ga_vs_de(
        _problem(),
        GAConfig(population_size=15, generations=10, seed=1),
        DEConfig(population_size=15, generations=10, seed=1),
    )
    assert isinstance(result, GADEResult)
    assert result.winner in {"ga", "de", "tie"}


def test_compare_is_reproducible() -> None:
    p = _problem()
    r1 = compare_ga_vs_de(
        p,
        GAConfig(population_size=20, generations=15, seed=7),
        DEConfig(population_size=20, generations=15, seed=7),
    )
    r2 = compare_ga_vs_de(
        p,
        GAConfig(population_size=20, generations=15, seed=7),
        DEConfig(population_size=20, generations=15, seed=7),
    )
    assert r1.ga_total == r2.ga_total
    assert r1.de_total == r2.de_total
    assert r1.winner == r2.winner


def test_gap_signs_match_winner() -> None:
    result = compare_ga_vs_de(
        _problem(),
        GAConfig(population_size=20, generations=10, seed=1),
        DEConfig(population_size=20, generations=10, seed=1),
    )
    if result.winner == "ga":
        assert result.gap_absolute > 0.0
    elif result.winner == "de":
        assert result.gap_absolute < 0.0
    else:
        assert result.gap_absolute == 0.0


def test_gap_relative_matches_absolute() -> None:
    result = compare_ga_vs_de(
        _problem(),
        GAConfig(population_size=10, generations=5, seed=3),
        DEConfig(population_size=10, generations=5, seed=3),
    )
    expected = (result.de_total - result.ga_total) / (abs(result.ga_total) + 1e-12)
    assert result.gap_relative == pytest.approx(expected)
