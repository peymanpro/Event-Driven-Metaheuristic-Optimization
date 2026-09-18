from __future__ import annotations

from dataclasses import dataclass

from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.optimizer import GAResult, run_ga
from edmo.algorithms.random_search import (
    RandomSearchConfig,
    RandomSearchResult,
    random_search,
)
from edmo.domain.evaluation import PenaltyWeights
from edmo.domain.problem import Problem


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    problem_id: str
    random_search: RandomSearchResult
    ga: GAResult
    improvement_absolute: float
    improvement_relative: float


def compare_ga_vs_random(
    problem: Problem,
    random_config: RandomSearchConfig | None = None,
    ga_config: GAConfig | None = None,
    weights: PenaltyWeights | None = None,
) -> ComparisonResult:
    """Run both algorithms with the same seed and compare their best totals."""
    rc = random_config or RandomSearchConfig(iterations=1000, seed=1)
    gc = ga_config or GAConfig(population_size=30, generations=50, seed=1)

    rs = random_search(problem, rc, weights)
    ga = run_ga(problem, gc, weights)

    rs_total = rs.best_evaluation.total
    ga_total = ga.best_evaluation.total
    improvement = rs_total - ga_total
    relative = improvement / (abs(rs_total) + 1.0e-12)

    return ComparisonResult(
        problem_id=problem.id,
        random_search=rs,
        ga=ga,
        improvement_absolute=improvement,
        improvement_relative=relative,
    )
