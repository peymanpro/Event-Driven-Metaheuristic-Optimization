import pytest

from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.optimizer import run_ga
from edmo.algorithms.genetic.state import GAState
from edmo.algorithms.genetic.stateful import (
    adapt_chromosome,
    adapt_state,
    run_ga_stateful,
)
from edmo.domain.change import AddJob, AddResource, ChangeWeight, apply_change
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
        ),
    )


def test_fresh_stateful_run_matches_run_ga() -> None:
    cfg = GAConfig(population_size=15, generations=10, seed=3)
    _, r = run_ga_stateful(_problem(), state=None, config=cfg)
    r_direct = run_ga(_problem(), cfg)
    assert r.history == r_direct.history


def test_stateful_run_returns_state() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=0)
    state, result = run_ga_stateful(_problem(), None, cfg)
    assert isinstance(state, GAState)
    assert state.problem_version == 1
    assert state.generation == 5
    assert result.generations == 5


def test_stateful_continue_same_problem_extends_history() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=1)
    state, r1 = run_ga_stateful(_problem(), None, cfg)
    _, r2 = run_ga_stateful(_problem(), state, cfg)
    assert r2.history[: len(r1.history)] == r1.history
    assert len(r2.history) > len(r1.history)


def test_stateful_mismatched_version_rejected() -> None:
    cfg = GAConfig(population_size=10, generations=2, seed=0)
    state, _ = run_ga_stateful(_problem(), None, cfg)
    p2 = apply_change(_problem(), ChangeWeight("j1", 2.0))
    with pytest.raises(ValueError):
        run_ga_stateful(p2, state, cfg)


def test_adapt_chromosome_added_job() -> None:
    from random import Random

    old = _problem()
    new = apply_change(old, AddJob(Job(id="j3", processing_time=1.0)))
    state, _ = run_ga_stateful(
        old, None, GAConfig(population_size=4, generations=0, seed=0)
    )
    c = state.population[0]
    adapted = adapt_chromosome(old, new, c, Random(0), 100.0)
    assert adapted.length == 3


def test_adapt_chromosome_removed_job() -> None:
    from random import Random

    old = _problem()
    new = apply_change(old, ChangeWeight("j1", 5.0))  # version bump only
    state, _ = run_ga_stateful(
        old, None, GAConfig(population_size=4, generations=0, seed=0)
    )
    c = state.population[0]
    adapted = adapt_chromosome(old, new, c, Random(0), 100.0)
    assert adapted.length == c.length


def test_adapt_chromosome_resource_addition_keeps_existing() -> None:
    from random import Random

    old = _problem()
    new = apply_change(old, AddResource(Resource(id="r2")))
    state, _ = run_ga_stateful(
        old, None, GAConfig(population_size=4, generations=0, seed=0)
    )
    c = state.population[0]
    adapted = adapt_chromosome(old, new, c, Random(0), 100.0)
    # Resource indices for old jobs stay within [0, 1] since old resources kept positions.
    for idx in adapted.resource_idx[:2]:
        assert idx in {0, 1}


def test_adapt_state_preserves_generation_and_history() -> None:
    cfg = GAConfig(population_size=8, generations=4, seed=2)
    state, _ = run_ga_stateful(_problem(), None, cfg)
    new_problem = apply_change(_problem(), ChangeWeight("j1", 3.0))
    adapted = adapt_state(_problem(), new_problem, state, cfg)
    assert adapted.generation == state.generation
    assert adapted.history == state.history
    assert adapted.problem_version == new_problem.version
    assert adapted.best_evaluation is not None


def test_warm_start_after_change_runs() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=4)
    state, _ = run_ga_stateful(_problem(), None, cfg)
    old = _problem()
    new = apply_change(old, AddJob(Job(id="j3", processing_time=1.5)))
    adapted = adapt_state(old, new, state, cfg)
    new_state, result = run_ga_stateful(new, adapted, cfg)
    assert new_state.problem_version == new.version
    assert len(result.best_solution.assignments) == 3


def test_target_total_stops_early() -> None:
    cfg = GAConfig(population_size=20, generations=100, seed=0)
    _, r = run_ga_stateful(
        _problem(),
        None,
        cfg,
        target_total=1.0e9,
    )
    assert r.stopped_reason == "target_reached"


def test_fresh_run_generation_equals_steps_performed() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=1)
    state, result = run_ga_stateful(_problem(), None, cfg)
    assert state.generation == 5
    assert result.generations == 5
    # history includes generation 0 plus 5 new generations
    assert len(state.history) == 6


def test_fresh_run_zero_generations() -> None:
    cfg = GAConfig(population_size=10, generations=0, seed=1)
    state, result = run_ga_stateful(_problem(), None, cfg)
    assert state.generation == 0
    assert result.generations == 0
    assert len(state.history) == 1


def test_warm_start_continues_generation_correctly() -> None:
    cfg = GAConfig(population_size=10, generations=5, seed=1)
    state, _ = run_ga_stateful(_problem(), None, cfg)
    assert state.generation == 5

    state2, result2 = run_ga_stateful(_problem(), state, cfg)
    # 5 additional steps -> generation 10.
    assert state2.generation == 10
    assert result2.generations == 5
    # Warm start does not re-record the inherited initial population.
    assert len(state2.history) == len(state.history) + 5


def test_two_consecutive_warm_starts_no_double_counting() -> None:
    cfg = GAConfig(population_size=10, generations=10, seed=2)
    s0, _ = run_ga_stateful(_problem(), None, cfg)
    assert s0.generation == 10

    s1, r1 = run_ga_stateful(_problem(), s0, cfg)
    assert s1.generation == 20
    assert r1.generations == 10

    s2, r2 = run_ga_stateful(_problem(), s1, cfg)
    assert s2.generation == 30
    assert r2.generations == 10

    # History length: 1 (initial) + 10 + 10 + 10 = 31.
    assert len(s2.history) == 31


def test_warm_start_history_extends_by_exact_number_of_steps() -> None:
    cfg_a = GAConfig(population_size=8, generations=4, seed=3)
    state_a, _ = run_ga_stateful(_problem(), None, cfg_a)

    cfg_b = GAConfig(population_size=8, generations=6, seed=3)
    state_b, _ = run_ga_stateful(_problem(), state_a, cfg_b)

    assert state_b.generation == 4 + 6
    # initial record + 4 + 6
    assert len(state_b.history) == 1 + 4 + 6


def test_warm_start_zero_generations_preserves_generation() -> None:
    cfg = GAConfig(population_size=8, generations=3, seed=4)
    state, _ = run_ga_stateful(_problem(), None, cfg)
    assert state.generation == 3

    cfg0 = GAConfig(population_size=8, generations=0, seed=4)
    state2, result2 = run_ga_stateful(_problem(), state, cfg0)
    assert state2.generation == 3
    assert result2.generations == 0
    assert len(state2.history) == len(state.history)
