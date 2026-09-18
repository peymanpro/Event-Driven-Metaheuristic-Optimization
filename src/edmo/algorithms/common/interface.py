from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from edmo.domain.evaluation import Evaluation, PenaltyWeights
from edmo.domain.problem import Problem
from edmo.domain.solution import Solution


@dataclass(frozen=True, slots=True)
class OptimizerResult:
    """Uniform result shape shared by all optimizers.

    ``history[i]`` is the best-so-far total (objective + penalty) observed
    after generation/iteration ``i``. ``stopped_reason`` documents why the
    optimizer returned.
    """

    best_solution: Solution
    best_evaluation: Evaluation
    history: tuple[float, ...]
    iterations: int
    seed: int | None
    stopped_reason: str


@runtime_checkable
class Optimizer(Protocol):
    """Solver protocol for the project's metaheuristics.

    Each optimizer must be a pure function of ``(problem, config, weights)``
    and must be reproducible given the same seed in ``config``.
    """

    name: str

    def __call__(
        self,
        problem: Problem,
        config: object | None = ...,
        weights: PenaltyWeights | None = ...,
    ) -> OptimizerResult: ...


OptimizerFactory = Callable[[Problem, object | None, PenaltyWeights | None], OptimizerResult]
