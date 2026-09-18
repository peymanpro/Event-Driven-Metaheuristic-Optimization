from __future__ import annotations

from dataclasses import dataclass

from edmo.domain.evaluation import PenaltyWeights
from edmo.domain.problem import Problem
from edmo.domain.solution import Solution


@dataclass(frozen=True, slots=True)
class FitnessJob:
    """A unit of work submitted to a fitness worker.

    ``job_id`` is a caller-assigned identifier so results can be matched
    back even when the executor reorders them. The payload is intentionally
    value-only so it can be serialized and shipped across processes.
    """

    job_id: str
    problem: Problem
    solution: Solution
    weights: PenaltyWeights | None = None

    def __post_init__(self) -> None:
        if not self.job_id:
            raise ValueError("FitnessJob.job_id must be non-empty")


@dataclass(frozen=True, slots=True)
class FitnessResult:
    job_id: str
    objective: float
    penalty: float
    total: float
    feasible: bool
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None
