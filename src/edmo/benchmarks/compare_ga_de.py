from __future__ import annotations

from dataclasses import dataclass

from edmo.algorithms.differential_evolution.config import DEConfig
from edmo.algorithms.differential_evolution.optimizer import DEResult, run_de
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.optimizer import GAResult, run_ga
from edmo.domain.evaluation import PenaltyWeights
from edmo.domain.problem import Problem


@dataclass(frozen=True, slots=True)
class GADEResult:
    problem_id: str
    ga: GAResult
    de: DEResult
    ga_total: float
    de_total: float
    winner: str
    gap_absolute: float
    gap_relative: float


def compare_ga_vs_de(
    problem: Problem,
    ga_config: GAConfig | None = None,
    de_config: DEConfig | None = None,
    weights: PenaltyWeights | None = None,
) -> GADEResult:
    """Run GA and DE with the same seed and compare their best totals.

    ``winner`` is ``"ga"``, ``"de"`` or ``"tie"``. ``gap_*`` is expressed
    as ``de_total - ga_total`` so positive values mean GA is better.
    """
    gc = ga_config or GAConfig(population_size=30, generations=50, seed=1)
    dc = de_config or DEConfig(population_size=30, generations=50, seed=1)

    ga = run_ga(problem, gc, weights)
    de = run_de(problem, dc, weights)

    ga_total = ga.best_evaluation.total
    de_total = de.best_evaluation.total
    gap = de_total - ga_total
    rel = gap / (abs(ga_total) + 1.0e-12)

    if gap > 0.0:
        winner = "ga"
    elif gap < 0.0:
        winner = "de"
    else:
        winner = "tie"

    return GADEResult(
        problem_id=problem.id,
        ga=ga,
        de=de,
        ga_total=ga_total,
        de_total=de_total,
        winner=winner,
        gap_absolute=gap,
        gap_relative=rel,
    )
