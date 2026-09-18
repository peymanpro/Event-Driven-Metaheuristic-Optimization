from __future__ import annotations

from dataclasses import dataclass
from random import Random

from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


@dataclass(frozen=True, slots=True)
class RandomProblemSpec:
    """Specification for a reproducible random problem instance."""

    n_resources: int = 3
    n_jobs: int = 8
    max_processing_time: float = 5.0
    max_release_time: float = 2.0
    max_deadline_horizon: float = 30.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_resources < 1:
            raise ValueError("n_resources must be >= 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if self.max_processing_time <= 0.0:
            raise ValueError("max_processing_time must be positive")
        if self.max_release_time < 0.0:
            raise ValueError("max_release_time must be non-negative")
        if self.max_deadline_horizon <= 0.0:
            raise ValueError("max_deadline_horizon must be positive")


def random_problem(spec: RandomProblemSpec | None = None) -> Problem:
    """Generate a reproducible random problem instance."""
    s = spec or RandomProblemSpec()
    rng = Random(s.seed)

    resources = tuple(
        Resource(id=f"r{i}", speed=round(rng.uniform(0.8, 1.5), 3))
        for i in range(s.n_resources)
    )

    jobs: list[Job] = []
    for i in range(s.n_jobs):
        processing = round(rng.uniform(0.5, s.max_processing_time), 3)
        release = round(rng.uniform(0.0, s.max_release_time), 3)
        deadline = round(release + rng.uniform(processing, s.max_deadline_horizon), 3)
        demand = {"cpu": round(rng.uniform(0.5, 2.0), 3)}
        jobs.append(
            Job(
                id=f"j{i}",
                processing_time=processing,
                demand=demand,
                release_time=release,
                deadline=deadline,
                weight=round(rng.uniform(0.5, 3.0), 3),
            )
        )

    return Problem(
        id=f"random-{s.seed}",
        resources=resources,
        jobs=tuple(jobs),
        metadata={"seed": str(s.seed)},
    )
