from __future__ import annotations

from random import Random

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.operators import (
    init_population,
    mutate,
    tournament_selection,
    uniform_crossover,
)
from edmo.algorithms.genetic.optimizer import GAResult
from edmo.algorithms.genetic.repair import repair
from edmo.algorithms.genetic.state import GAState
from edmo.algorithms.genetic.termination import TerminationPolicy
from edmo.domain.evaluation import Evaluation, PenaltyWeights, evaluate
from edmo.domain.problem import Problem


def _random_gene(n_resources: int, horizon: float, rng: Random) -> tuple[int, float]:
    return rng.randrange(n_resources), rng.uniform(0.0, horizon)


def adapt_chromosome(
    old_problem: Problem,
    new_problem: Problem,
    chromosome: Chromosome,
    rng: Random,
    horizon: float,
) -> Chromosome:
    """Re-shape a chromosome to the structure of ``new_problem``.

    Jobs present in both problems keep their gene. New jobs receive random
    genes. Removed jobs are dropped. Resource indices are remapped by id;
    removed resources cause the affected genes to be resampled.
    """
    old_job_index = {job.id: i for i, job in enumerate(old_problem.jobs)}
    new_resource_index = {r.id: i for i, r in enumerate(new_problem.resources)}
    n_new_resources = len(new_problem.resources)

    ridx: list[int] = []
    starts: list[float] = []
    for job in new_problem.jobs:
        old_i = old_job_index.get(job.id)
        if old_i is None or old_i >= chromosome.length:
            idx, t = _random_gene(n_new_resources, horizon, rng)
            ridx.append(idx)
            starts.append(t)
            continue
        old_resource_pos = chromosome.resource_idx[old_i]
        if old_resource_pos >= len(old_problem.resources):
            idx, t = _random_gene(n_new_resources, horizon, rng)
            ridx.append(idx)
            starts.append(t)
            continue
        old_resource_id = old_problem.resources[old_resource_pos].id
        new_pos = new_resource_index.get(old_resource_id)
        if new_pos is None:
            idx, t = _random_gene(n_new_resources, horizon, rng)
            ridx.append(idx)
            starts.append(t)
        else:
            ridx.append(new_pos)
            starts.append(chromosome.start_time[old_i])
    return Chromosome(resource_idx=tuple(ridx), start_time=tuple(starts))


def adapt_state(
    old_problem: Problem,
    new_problem: Problem,
    state: GAState,
    config: GAConfig,
) -> GAState:
    """Re-shape every chromosome in ``state`` to the new problem structure."""
    population = tuple(
        adapt_chromosome(old_problem, new_problem, c, state.rng, config.start_time_horizon)
        for c in state.population
    )
    best = adapt_chromosome(
        old_problem, new_problem, state.best_chromosome, state.rng, config.start_time_horizon
    )
    if config.apply_repair:
        population = tuple(repair(new_problem, c) for c in population)
        best = repair(new_problem, best)
    best_eval = evaluate(new_problem, best.to_solution(new_problem))
    return GAState(
        population=population,
        best_chromosome=best,
        best_evaluation=best_eval,
        generation=state.generation,
        history=state.history,
        problem_version=new_problem.version,
        rng=state.rng,
    )


def run_ga_stateful(
    problem: Problem,
    state: GAState | None = None,
    config: GAConfig | None = None,
    weights: PenaltyWeights | None = None,
    target_total: float | None = None,
) -> tuple[GAState, GAResult]:
    """Run GA, optionally warm-starting from ``state``.

    If ``state`` is provided and its ``problem_version`` matches the problem,
    evolution continues from the existing population. If versions differ, the
    state is adapted to the new problem structure first. A fresh run is
    triggered when ``state`` is ``None``.

    ``target_total`` stops evolution early once the best total reaches it,
    which is used by recovery benchmarks.
    """
    cfg = config or GAConfig()
    rng = Random(cfg.seed) if state is None else state.rng

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

    def prepare(chromosome: Chromosome) -> Chromosome:
        if not cfg.apply_repair:
            return chromosome
        return repair(problem, chromosome)

    if state is None:
        population = [prepare(c) for c in init_population(problem, cfg, rng)]
        generation = 0
        history: list[float] = []
        best: tuple[Chromosome, Evaluation] | None = None
        problem_version = problem.version
    else:
        if state.problem_version != problem.version:
            # Reconstruct the "before" problem from the state's perspective is
            # not possible; the caller must have already produced an adapted
            # state via adapt_state. Refuse silently wrong shapes.
            raise ValueError(
                "state.problem_version does not match problem.version; "
                "call adapt_state before run_ga_stateful"
            )
        population = [prepare(c) for c in state.population]
        generation = state.generation
        history = list(state.history)
        best = (state.best_chromosome, state.best_evaluation)
        problem_version = state.problem_version

    def record(pop: list[Chromosome]) -> list[tuple[Chromosome, Evaluation]]:
        nonlocal best
        scored = [(c, evaluate_chromosome(c)) for c in pop]
        scored.sort(key=lambda ce: ce[1].total)
        top_c, top_e = scored[0]
        history.append(top_e.total)
        if best is None or top_e.total < best[1].total:
            best = (top_c, top_e)
        return scored

    scored = record(population)

    policy: TerminationPolicy | None = cfg.termination
    stopped_reason = "max_generations"
    for _ in range(cfg.generations):
        if target_total is not None and history[-1] <= target_total:
            stopped_reason = "target_reached"
            break
        if policy is not None:
            decision = policy.decide(history)
            if decision.should_stop:
                stopped_reason = decision.reason
                break
        fitness = {c: e.total for c, e in scored}
        new_population: list[Chromosome] = [c for c, _ in scored[: cfg.elite_count]]
        while len(new_population) < cfg.population_size:
            pa = tournament_selection(population, fitness, cfg.tournament_size, rng)
            pb = tournament_selection(population, fitness, cfg.tournament_size, rng)
            child = uniform_crossover(pa, pb, cfg.crossover_rate, rng)
            child = mutate(child, len(problem.resources), cfg, rng)
            child = prepare(child)
            new_population.append(child)
        population = new_population[: cfg.population_size]
        scored = record(population)

    assert best is not None
    best_c, best_e = best
    new_state = GAState(
        population=tuple(population),
        best_chromosome=best_c,
        best_evaluation=best_e,
        generation=generation + (len(history) - 1),
        history=tuple(history),
        problem_version=problem_version,
        rng=rng,
    )
    result = GAResult(
        best_chromosome=best_c,
        best_solution=best_c.to_solution(problem),
        best_evaluation=best_e,
        generations=len(history) - 1,
        population_size=cfg.population_size,
        seed=cfg.seed,
        history=tuple(history),
        stopped_reason=stopped_reason,
    )
    return new_state, result
