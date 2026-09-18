from __future__ import annotations

from dataclasses import dataclass
from random import Random

from edmo.algorithms.common.interface import OptimizerResult
from edmo.algorithms.differential_evolution.config import DEConfig
from edmo.algorithms.differential_evolution.operators import (
    crossover_binomial,
    init_population,
    mutate_rand_1,
)
from edmo.algorithms.differential_evolution.vector import (
    VectorBounds,
    clip_to_bounds,
    decode_vector,
    vector_bounds,
)
from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.repair import repair
from edmo.domain.evaluation import Evaluation, PenaltyWeights, evaluate
from edmo.domain.problem import Problem


@dataclass(frozen=True, slots=True)
class DEResult:
    best_vector: tuple[float, ...]
    best_chromosome: Chromosome
    optimizer_result: OptimizerResult

    @property
    def best_solution(self):  # type: ignore[no-untyped-def]
        return self.optimizer_result.best_solution

    @property
    def best_evaluation(self) -> Evaluation:
        return self.optimizer_result.best_evaluation

    @property
    def history(self) -> tuple[float, ...]:
        return self.optimizer_result.history

    @property
    def generations(self) -> int:
        return self.optimizer_result.iterations

    @property
    def seed(self) -> int | None:
        return self.optimizer_result.seed

    @property
    def stopped_reason(self) -> str:
        return self.optimizer_result.stopped_reason


def run_de(
    problem: Problem,
    config: DEConfig | None = None,
    weights: PenaltyWeights | None = None,
) -> DEResult:
    """Run differential evolution (DE/rand/1/bin) on a scheduling problem."""
    cfg = config or DEConfig()
    rng = Random(cfg.seed)
    bounds: VectorBounds = vector_bounds(problem, cfg.start_time_horizon)
    cache: dict[Chromosome, Evaluation] = {}

    def prepare(chromosome: Chromosome) -> Chromosome:
        if not cfg.apply_repair:
            return chromosome
        return repair(problem, chromosome)

    def evaluate_vector(vector: tuple[float, ...]) -> tuple[Chromosome, Evaluation]:
        chromosome = prepare(decode_vector(problem, vector))
        cached = cache.get(chromosome)
        if cached is not None:
            return chromosome, cached
        solution = chromosome.to_solution(problem)
        result = (
            evaluate(problem, solution)
            if weights is None
            else evaluate(problem, solution, weights)
        )
        cache[chromosome] = result
        return chromosome, result

    population = init_population(bounds, cfg.population_size, rng)
    fitness: list[float] = []
    chromosomes: list[Chromosome] = []
    for v in population:
        c, e = evaluate_vector(v)
        chromosomes.append(c)
        fitness.append(e.total)

    best_idx = min(range(len(fitness)), key=lambda i: fitness[i])
    best_vector = population[best_idx]
    best_chromosome = chromosomes[best_idx]
    best_evaluation = evaluate_vector(best_vector)[1]
    history: list[float] = [best_evaluation.total]
    stopped_reason = "max_generations"

    policy = cfg.termination
    for _ in range(cfg.generations):
        if policy is not None:
            decision = policy.decide(history)
            if decision.should_stop:
                stopped_reason = decision.reason
                break
        for i in range(cfg.population_size):
            mutant = mutate_rand_1(population, i, cfg.mutation_factor, bounds, rng)
            trial = crossover_binomial(population[i], mutant, cfg.crossover_rate, rng)
            trial = clip_to_bounds(trial, bounds)
            _, trial_eval = evaluate_vector(trial)
            if trial_eval.total < fitness[i]:
                population[i] = trial
                fitness[i] = trial_eval.total
                chromosomes[i] = prepare(decode_vector(problem, trial))
                if trial_eval.total < best_evaluation.total:
                    best_vector = trial
                    best_chromosome = chromosomes[i]
                    best_evaluation = trial_eval
        history.append(best_evaluation.total)

    optimizer_result = OptimizerResult(
        best_solution=best_chromosome.to_solution(problem),
        best_evaluation=best_evaluation,
        history=tuple(history),
        iterations=len(history) - 1,
        seed=cfg.seed,
        stopped_reason=stopped_reason,
    )
    return DEResult(
        best_vector=best_vector,
        best_chromosome=best_chromosome,
        optimizer_result=optimizer_result,
    )
