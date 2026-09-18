from random import Random

import pytest

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.operators import (
    init_population,
    mutate,
    tournament_selection,
    uniform_crossover,
)
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"), Resource(id="r1"), Resource(id="r2")),
        jobs=(
            Job(id="j1", processing_time=1.0),
            Job(id="j2", processing_time=2.0),
            Job(id="j3", processing_time=1.5),
        ),
    )


def test_init_population_size_and_shape() -> None:
    cfg = GAConfig(population_size=10, start_time_horizon=50.0, seed=0)
    pop = init_population(_problem(), cfg, Random(0))
    assert len(pop) == 10
    for c in pop:
        assert c.length == 3
        for idx in c.resource_idx:
            assert 0 <= idx < 3
        for t in c.start_time:
            assert 0.0 <= t <= 50.0


def test_init_population_is_deterministic() -> None:
    cfg = GAConfig(population_size=5, seed=0)
    p1 = init_population(_problem(), cfg, Random(123))
    p2 = init_population(_problem(), cfg, Random(123))
    assert p1 == p2


def test_tournament_selection_picks_best() -> None:
    a = Chromosome((0, 0, 0), (0.0, 0.0, 0.0))
    b = Chromosome((1, 1, 1), (1.0, 1.0, 1.0))
    c = Chromosome((2, 2, 2), (2.0, 2.0, 2.0))
    pop = [a, b, c]
    fitness = {a: 3.0, b: 1.0, c: 2.0}
    # With tournament_size >= len(pop) * many draws, best should always win.
    winner = tournament_selection(pop, fitness, tournament_size=10, rng=Random(0))
    assert winner is b


def test_tournament_selection_rejects_empty_population() -> None:
    with pytest.raises(ValueError):
        tournament_selection([], {}, 3, Random(0))


def test_tournament_selection_rejects_bad_size() -> None:
    a = Chromosome((0,), (0.0,))
    with pytest.raises(ValueError):
        tournament_selection([a], {a: 1.0}, 0, Random(0))


def test_uniform_crossover_swaps_genes() -> None:
    a = Chromosome((0, 0, 0), (0.0, 0.0, 0.0))
    b = Chromosome((1, 1, 1), (1.0, 1.0, 1.0))
    child = uniform_crossover(a, b, rate=0.5, rng=Random(0))
    assert child.length == 3
    for idx, t in zip(child.resource_idx, child.start_time, strict=True):
        assert (idx, t) in {(0, 0.0), (1, 1.0)}


def test_uniform_crossover_rate_one_takes_a() -> None:
    a = Chromosome((0, 0), (1.0, 2.0))
    b = Chromosome((1, 1), (3.0, 4.0))
    child = uniform_crossover(a, b, rate=1.0, rng=Random(0))
    assert child.resource_idx == a.resource_idx
    assert child.start_time == a.start_time


def test_uniform_crossover_rate_zero_takes_b() -> None:
    a = Chromosome((0, 0), (1.0, 2.0))
    b = Chromosome((1, 1), (3.0, 4.0))
    child = uniform_crossover(a, b, rate=0.0, rng=Random(0))
    assert child.resource_idx == b.resource_idx
    assert child.start_time == b.start_time


def test_uniform_crossover_rejects_mismatched_lengths() -> None:
    a = Chromosome((0,), (0.0,))
    b = Chromosome((0, 0), (0.0, 0.0))
    with pytest.raises(ValueError):
        uniform_crossover(a, b, 0.5, Random(0))


def test_uniform_crossover_rejects_bad_rate() -> None:
    a = Chromosome((0,), (0.0,))
    with pytest.raises(ValueError):
        uniform_crossover(a, a, -0.1, Random(0))


def test_mutate_rate_zero_returns_equal() -> None:
    cfg = GAConfig(mutation_rate=0.0)
    a = Chromosome((0, 1, 0), (1.0, 2.0, 3.0))
    m = mutate(a, n_resources=3, config=cfg, rng=Random(0))
    assert m.resource_idx == a.resource_idx
    assert m.start_time == a.start_time


def test_mutate_rate_one_changes_resource_indices() -> None:
    cfg = GAConfig(mutation_rate=1.0, mutation_time_sigma=0.001)
    a = Chromosome((0, 0, 0, 0, 0, 0, 0, 0), (1.0,) * 8)
    m = mutate(a, n_resources=4, config=cfg, rng=Random(0))
    # Resource indices must be within range.
    for idx in m.resource_idx:
        assert 0 <= idx < 4
    # Start times must remain non-negative.
    for t in m.start_time:
        assert t >= 0.0


def test_mutate_clamps_negative_start_times() -> None:
    cfg = GAConfig(mutation_rate=1.0, mutation_time_sigma=100.0)
    a = Chromosome((0,), (0.5,))
    for seed in range(20):
        m = mutate(a, n_resources=1, config=cfg, rng=Random(seed))
        assert m.start_time[0] >= 0.0
