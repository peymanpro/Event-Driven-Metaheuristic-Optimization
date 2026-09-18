
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.optimizer import GAResult, run_ga
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"), Resource(id="r1")),
        jobs=(
            Job(id="j1", processing_time=1.0, deadline=10.0),
            Job(id="j2", processing_time=2.0, deadline=10.0),
            Job(id="j3", processing_time=1.5, deadline=10.0),
        ),
    )


def test_run_ga_returns_result() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=1)
    result = run_ga(_problem(), cfg)
    assert isinstance(result, GAResult)
    assert result.generations == 5
    assert result.population_size == 10
    assert result.seed == 1
    assert len(result.history) == 6


def test_run_ga_is_deterministic_with_seed() -> None:
    cfg = GAConfig(population_size=20, generations=10, seed=42)
    r1 = run_ga(_problem(), cfg)
    r2 = run_ga(_problem(), cfg)
    assert r1.history == r2.history
    assert r1.best_chromosome == r2.best_chromosome
    assert r1.best_evaluation.total == r2.best_evaluation.total


def test_run_ga_history_is_non_increasing() -> None:
    cfg = GAConfig(population_size=20, generations=20, seed=7)
    result = run_ga(_problem(), cfg)
    # history stores the best-so-far in each generation (elitism guarantees
    # that the running minimum never increases).
    for prev, curr in zip(result.history, result.history[1:], strict=False):
        assert curr <= prev


def test_run_ga_improves_over_random_population() -> None:
    cfg = GAConfig(population_size=30, generations=30, seed=3)
    result = run_ga(_problem(), cfg)
    assert result.history[-1] <= result.history[0]


def test_run_ga_zero_generations() -> None:
    cfg = GAConfig(population_size=5, generations=0, seed=0)
    result = run_ga(_problem(), cfg)
    assert len(result.history) == 1


def test_run_ga_produces_valid_solution() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=0)
    result = run_ga(_problem(), cfg)
    assert set(result.best_solution.assignments.keys()) == {"j1", "j2", "j3"}
