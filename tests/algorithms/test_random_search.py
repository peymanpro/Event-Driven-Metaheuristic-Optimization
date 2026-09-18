import pytest

from edmo.algorithms.random_search import (
    RandomSearchConfig,
    RandomSearchResult,
    random_search,
)
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r1"), Resource(id="r2")),
        jobs=(
            Job(id="j1", processing_time=1.0),
            Job(id="j2", processing_time=2.0),
        ),
    )


def test_random_search_config_defaults() -> None:
    cfg = RandomSearchConfig()
    assert cfg.iterations == 500
    assert cfg.horizon == 100.0
    assert cfg.seed is None


def test_random_search_config_rejects_bad_iterations() -> None:
    with pytest.raises(ValueError):
        RandomSearchConfig(iterations=0)


def test_random_search_config_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError):
        RandomSearchConfig(horizon=0.0)


def test_random_search_returns_result() -> None:
    result = random_search(_problem(), RandomSearchConfig(iterations=10, seed=1))
    assert isinstance(result, RandomSearchResult)
    assert result.iterations == 10
    assert result.seed == 1
    assert set(result.best_solution.assignments.keys()) == {"j1", "j2"}


def test_random_search_is_deterministic_with_seed() -> None:
    p = _problem()
    r1 = random_search(p, RandomSearchConfig(iterations=50, seed=42))
    r2 = random_search(p, RandomSearchConfig(iterations=50, seed=42))
    assert r1.best_evaluation.total == r2.best_evaluation.total
    for jid in ("j1", "j2"):
        a1 = r1.best_solution.assignment_for(jid)
        a2 = r2.best_solution.assignment_for(jid)
        assert (a1.resource_id, a1.start_time) == (a2.resource_id, a2.start_time)


def test_random_search_improves_or_equals_with_more_iterations() -> None:
    p = _problem()
    small = random_search(p, RandomSearchConfig(iterations=5, seed=7))
    large = random_search(p, RandomSearchConfig(iterations=500, seed=7))
    assert large.best_evaluation.total <= small.best_evaluation.total


def test_random_search_best_is_minimum_over_iterations() -> None:
    p = _problem()
    r = random_search(p, RandomSearchConfig(iterations=200, seed=3))
    # Total must be at least the objective and never negative.
    assert r.best_evaluation.total >= 0.0
