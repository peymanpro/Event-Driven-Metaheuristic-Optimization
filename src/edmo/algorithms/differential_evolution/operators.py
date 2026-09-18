from __future__ import annotations

from collections.abc import Sequence
from random import Random

from edmo.algorithms.differential_evolution.vector import VectorBounds, clip_to_bounds


def init_population(
    bounds: VectorBounds,
    population_size: int,
    rng: Random,
) -> list[tuple[float, ...]]:
    population: list[tuple[float, ...]] = []
    for _ in range(population_size):
        population.append(
            tuple(
                rng.uniform(lo, hi) if hi > lo else lo
                for lo, hi in zip(bounds.lower, bounds.upper, strict=True)
            )
        )
    return population


def mutate_rand_1(
    population: Sequence[tuple[float, ...]],
    target_idx: int,
    mutation_factor: float,
    bounds: VectorBounds,
    rng: Random,
) -> tuple[float, ...]:
    """DE/rand/1: ``v = x_a + F * (x_b - x_c)`` with indices distinct from target."""
    n = len(population)
    if n < 4:
        raise ValueError("population must have at least 4 individuals")
    if mutation_factor <= 0.0:
        raise ValueError("mutation_factor must be positive")

    candidates = [i for i in range(n) if i != target_idx]
    a, b, c = rng.sample(candidates, 3)
    xa = population[a]
    xb = population[b]
    xc = population[c]
    trial = tuple(
        xa_i + mutation_factor * (xb_i - xc_i)
        for xa_i, xb_i, xc_i in zip(xa, xb, xc, strict=True)
    )
    return clip_to_bounds(trial, bounds)


def crossover_binomial(
    target: tuple[float, ...],
    mutant: tuple[float, ...],
    crossover_rate: float,
    rng: Random,
) -> tuple[float, ...]:
    """Binomial crossover: at least one gene from the mutant."""
    if len(target) != len(mutant):
        raise ValueError("target and mutant must have equal length")
    if not 0.0 <= crossover_rate <= 1.0:
        raise ValueError("crossover_rate must be in [0, 1]")
    n = len(target)
    j_rand = rng.randrange(n)
    trial: list[float] = []
    for j in range(n):
        if rng.random() < crossover_rate or j == j_rand:
            trial.append(mutant[j])
        else:
            trial.append(target[j])
    return tuple(trial)
