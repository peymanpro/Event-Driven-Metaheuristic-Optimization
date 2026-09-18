from __future__ import annotations

from dataclasses import dataclass
from random import Random

from edmo.domain.evaluation import Evaluation, PenaltyWeights, evaluate
from edmo.domain.problem import Problem
from edmo.domain.solution import Assignment, Solution


@dataclass(frozen=True, slots=True)
class RandomSearchConfig:
    """Configuration for the random-search baseline."""

    iterations: int = 500
    horizon: float = 100.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.iterations < 1:
            raise ValueError("RandomSearchConfig.iterations must be >= 1")
        if self.horizon <= 0.0:
            raise ValueError("RandomSearchConfig.horizon must be positive")


@dataclass(frozen=True, slots=True)
class RandomSearchResult:
    best_solution: Solution
    best_evaluation: Evaluation
    iterations: int
    seed: int | None


def _random_solution(problem: Problem, rng: Random, horizon: float) -> Solution:
    assignments: dict[str, Assignment] = {}
    for job in problem.jobs:
        resource = problem.resources[rng.randrange(len(problem.resources))]
        start = rng.uniform(0.0, horizon)
        assignments[job.id] = Assignment(resource_id=resource.id, start_time=start)
    return Solution(assignments=assignments)


def random_search(
    problem: Problem,
    config: RandomSearchConfig | None = None,
    weights: PenaltyWeights | None = None,
) -> RandomSearchResult:
    """Uniform random sampling over (resource, start_time) per job.

    Returns the best sampled solution under the ``total`` score
    (objective + penalty).
    """
    cfg = config or RandomSearchConfig()
    rng = Random(cfg.seed)
    effective_weights = weights

    best_solution: Solution | None = None
    best_evaluation: Evaluation | None = None

    for _ in range(cfg.iterations):
        candidate = _random_solution(problem, rng, cfg.horizon)
        if effective_weights is None:
            evaluation = evaluate(problem, candidate)
        else:
            evaluation = evaluate(problem, candidate, effective_weights)

        if best_evaluation is None or evaluation.total < best_evaluation.total:
            best_solution = candidate
            best_evaluation = evaluation

    assert best_solution is not None
    assert best_evaluation is not None
    return RandomSearchResult(
        best_solution=best_solution,
        best_evaluation=best_evaluation,
        iterations=cfg.iterations,
        seed=cfg.seed,
    )
