from __future__ import annotations

from collections.abc import Mapping
from random import Random

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.config import GAConfig
from edmo.domain.problem import Problem


def init_population(
    problem: Problem,
    config: GAConfig,
    rng: Random,
) -> list[Chromosome]:
    """Random initial population over (resource_idx, start_time) genes."""
    n_resources = len(problem.resources)
    n_jobs = len(problem.jobs)
    population: list[Chromosome] = []
    for _ in range(config.population_size):
        ridx = tuple(rng.randrange(n_resources) for _ in range(n_jobs))
        starts = tuple(
            rng.uniform(0.0, config.start_time_horizon) for _ in range(n_jobs)
        )
        population.append(Chromosome(resource_idx=ridx, start_time=starts))
    return population


def tournament_selection(
    population: list[Chromosome],
    fitness: Mapping[Chromosome, float],
    tournament_size: int,
    rng: Random,
) -> Chromosome:
    """Return the best individual from ``tournament_size`` random draws."""
    if not population:
        raise ValueError("population must not be empty")
    if tournament_size < 1:
        raise ValueError("tournament_size must be >= 1")
    best = population[rng.randrange(len(population))]
    best_score = fitness[best]
    for _ in range(tournament_size - 1):
        candidate = population[rng.randrange(len(population))]
        score = fitness[candidate]
        if score < best_score:
            best = candidate
            best_score = score
    return best


def uniform_crossover(
    parent_a: Chromosome,
    parent_b: Chromosome,
    rate: float,
    rng: Random,
) -> Chromosome:
    """Per-gene uniform crossover: pick gene from A with probability ``rate``."""
    if parent_a.length != parent_b.length:
        raise ValueError("parents must have equal length")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rate must be in [0, 1]")
    n = parent_a.length
    ridx: list[int] = []
    starts: list[float] = []
    for i in range(n):
        if rng.random() < rate:
            ridx.append(parent_a.resource_idx[i])
            starts.append(parent_a.start_time[i])
        else:
            ridx.append(parent_b.resource_idx[i])
            starts.append(parent_b.start_time[i])
    return Chromosome(resource_idx=tuple(ridx), start_time=tuple(starts))


def mutate(
    chromosome: Chromosome,
    n_resources: int,
    config: GAConfig,
    rng: Random,
) -> Chromosome:
    """Per-gene mutation: resample resource and/or perturb start time."""
    if n_resources < 1:
        raise ValueError("n_resources must be >= 1")
    ridx = list(chromosome.resource_idx)
    starts = list(chromosome.start_time)
    for i in range(len(ridx)):
        if rng.random() < config.mutation_rate:
            ridx[i] = rng.randrange(n_resources)
        if rng.random() < config.mutation_rate:
            perturbed = starts[i] + rng.gauss(0.0, config.mutation_time_sigma)
            starts[i] = perturbed if perturbed > 0.0 else 0.0
    return Chromosome(resource_idx=tuple(ridx), start_time=tuple(starts))
