from __future__ import annotations

from dataclasses import dataclass
from random import Random

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.operators import (
    init_population,
    mutate,
    tournament_selection,
    uniform_crossover,
)
from edmo.domain.evaluation import Evaluation, PenaltyWeights, evaluate
from edmo.domain.problem import Problem
from edmo.domain.solution import Solution


@dataclass(frozen=True, slots=True)
class GAResult:
    best_chromosome: Chromosome
    best_solution: Solution
    best_evaluation: Evaluation
    generations: int
    population_size: int
    seed: int | None
    history: tuple[float, ...]


def run_ga(
    problem: Problem,
    config: GAConfig | None = None,
    weights: PenaltyWeights | None = None,
) -> GAResult:
    """Run the genetic algorithm and return the best individual found.

    ``history[i]`` is the best total score in generation ``i`` (0-based),
    including the initial random population as generation 0.
    """
    cfg = config or GAConfig()
    rng = Random(cfg.seed)
    cache: dict[Chromosome, Evaluation] = {}

    def evaluate_chromosome(chromosome: Chromosome) -> Evaluation:
        cached = cache.get(chromosome)
        if cached is not None:
            return cached
        solution = chromosome.to_solution(problem)
        result = (
            evaluate(problem, solution)
            if weights is None
            else evaluate(problem, solution, weights)
        )
        cache[chromosome] = result
        return result

    best: tuple[Chromosome, Evaluation] | None = None
    history: list[float] = []

    def record(population: list[Chromosome]) -> list[tuple[Chromosome, Evaluation]]:
        nonlocal best
        scored: list[tuple[Chromosome, Evaluation]] = [
            (c, evaluate_chromosome(c)) for c in population
        ]
        scored.sort(key=lambda ce: ce[1].total)
        top_c, top_e = scored[0]
        history.append(top_e.total)
        if best is None or top_e.total < best[1].total:
            best = (top_c, top_e)
        return scored

    population = init_population(problem, cfg, rng)
    scored = record(population)

    for _ in range(cfg.generations):
        fitness = {c: e.total for c, e in scored}
        new_population: list[Chromosome] = [
            c for c, _ in scored[: cfg.elite_count]
        ]
        while len(new_population) < cfg.population_size:
            parent_a = tournament_selection(population, fitness, cfg.tournament_size, rng)
            parent_b = tournament_selection(population, fitness, cfg.tournament_size, rng)
            child = uniform_crossover(parent_a, parent_b, cfg.crossover_rate, rng)
            child = mutate(child, len(problem.resources), cfg, rng)
            new_population.append(child)
        population = new_population[: cfg.population_size]
        scored = record(population)

    assert best is not None
    best_c, best_e = best
    return GAResult(
        best_chromosome=best_c,
        best_solution=best_c.to_solution(problem),
        best_evaluation=best_e,
        generations=cfg.generations,
        population_size=cfg.population_size,
        seed=cfg.seed,
        history=tuple(history),
    )
