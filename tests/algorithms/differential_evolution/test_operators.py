from random import Random

import pytest

from edmo.algorithms.differential_evolution.operators import (
    crossover_binomial,
    init_population,
    mutate_rand_1,
)
from edmo.algorithms.differential_evolution.vector import VectorBounds


def _bounds() -> VectorBounds:
    return VectorBounds(lower=(0.0, 0.0), upper=(10.0, 10.0))


def test_init_population_shape_and_bounds() -> None:
    pop = init_population(_bounds(), population_size=10, rng=Random(0))
    assert len(pop) == 10
    for v in pop:
        assert len(v) == 2
        for x in v:
            assert 0.0 <= x <= 10.0


def test_init_population_deterministic() -> None:
    p1 = init_population(_bounds(), 5, Random(123))
    p2 = init_population(_bounds(), 5, Random(123))
    assert p1 == p2


def test_init_population_single_value_bounds() -> None:
    b = VectorBounds(lower=(0.0, 0.0), upper=(0.0, 0.0))
    pop = init_population(b, 4, Random(0))
    for v in pop:
        assert v == (0.0, 0.0)


def test_mutate_rand_1_shape_and_bounds() -> None:
    pop = init_population(_bounds(), 6, Random(0))
    mutant = mutate_rand_1(pop, target_idx=0, mutation_factor=0.5, bounds=_bounds(), rng=Random(0))
    assert len(mutant) == 2
    for x in mutant:
        assert 0.0 <= x <= 10.0


def test_mutate_rand_1_rejects_small_population() -> None:
    pop = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    with pytest.raises(ValueError):
        mutate_rand_1(pop, 0, 0.5, _bounds(), Random(0))


def test_mutate_rand_1_rejects_bad_f() -> None:
    pop = [(0.0, 0.0)] * 4
    with pytest.raises(ValueError):
        mutate_rand_1(pop, 0, 0.0, _bounds(), Random(0))


def test_crossover_rate_one_takes_mutant() -> None:
    target = (0.0, 0.0)
    mutant = (1.0, 1.0)
    trial = crossover_binomial(target, mutant, 1.0, Random(0))
    assert trial == mutant


def test_crossover_rate_zero_takes_target() -> None:
    target = (0.0, 0.0)
    mutant = (1.0, 1.0)
    trial = crossover_binomial(target, mutant, 0.0, Random(0))
    # j_rand guarantees at least one mutant gene.
    assert sum(1 for a, b in zip(trial, mutant, strict=True) if a == b) >= 1
    assert len(trial) == 2


def test_crossover_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        crossover_binomial((0.0,), (1.0, 1.0), 0.5, Random(0))


def test_crossover_rejects_bad_rate() -> None:
    with pytest.raises(ValueError):
        crossover_binomial((0.0,), (1.0,), -0.1, Random(0))
