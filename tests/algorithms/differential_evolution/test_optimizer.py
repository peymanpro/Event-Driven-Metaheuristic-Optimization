
from edmo.algorithms.common.interface import OptimizerResult
from edmo.algorithms.differential_evolution.config import DEConfig
from edmo.algorithms.differential_evolution.optimizer import DEResult, run_de
from edmo.algorithms.genetic.termination import TerminationPolicy
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"), Resource(id="r1")),
        jobs=(
            Job(id="j1", processing_time=1.0),
            Job(id="j2", processing_time=2.0),
            Job(id="j3", processing_time=1.5),
        ),
    )


def _problem_with_precedence() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"), Resource(id="r1")),
        jobs=(
            Job(id="j1", processing_time=2.0),
            Job(id="j2", processing_time=1.0, predecessors=frozenset({"j1"})),
        ),
    )


def test_run_de_returns_result() -> None:
    cfg = DEConfig(population_size=10, generations=5, seed=1)
    result = run_de(_problem(), cfg)
    assert isinstance(result, DEResult)
    assert isinstance(result.optimizer_result, OptimizerResult)
    assert result.generations == 5
    assert result.seed == 1
    assert len(result.history) == 6


def test_run_de_deterministic_with_seed() -> None:
    cfg = DEConfig(population_size=20, generations=10, seed=7)
    r1 = run_de(_problem(), cfg)
    r2 = run_de(_problem(), cfg)
    assert r1.history == r2.history
    assert r1.best_vector == r2.best_vector


def test_run_de_history_is_non_increasing() -> None:
    cfg = DEConfig(population_size=20, generations=20, seed=3)
    result = run_de(_problem(), cfg)
    for prev, curr in zip(result.history, result.history[1:], strict=False):
        assert curr <= prev


def test_run_de_improves_over_initial() -> None:
    cfg = DEConfig(population_size=30, generations=30, seed=9)
    result = run_de(_problem(), cfg)
    assert result.history[-1] <= result.history[0]


def test_run_de_produces_valid_solution() -> None:
    cfg = DEConfig(population_size=10, generations=5, seed=0)
    result = run_de(_problem(), cfg)
    assert set(result.best_solution.assignments.keys()) == {"j1", "j2", "j3"}


def test_run_de_repair_enables_precedence_feasibility() -> None:
    cfg = DEConfig(population_size=30, generations=30, seed=11, apply_repair=True)
    result = run_de(_problem_with_precedence(), cfg)
    assert result.best_evaluation.feasible


def test_run_de_respects_termination_policy() -> None:
    cfg = DEConfig(
        population_size=10,
        generations=200,
        seed=5,
        termination=TerminationPolicy(max_generations=4),
    )
    result = run_de(_problem(), cfg)
    assert result.stopped_reason == "max_generations"
    assert len(result.history) <= 5
