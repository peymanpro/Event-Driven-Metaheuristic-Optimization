from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True, eq=False)
class Job:
    """A unit of work that must be assigned to and scheduled on a resource.

    ``processing_time`` is expressed in base work units. ``demand`` describes
    the resource quantities required while the job is running. ``weight``
    expresses priority in the objective. ``predecessors`` are the ids of jobs
    that must finish before this job can start.
    """

    id: str
    processing_time: float
    demand: Mapping[str, float] = field(default_factory=dict)
    release_time: float = 0.0
    deadline: float | None = None
    weight: float = 1.0
    predecessors: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Job.id must be a non-empty string")
        if self.processing_time <= 0.0:
            raise ValueError("Job.processing_time must be positive")
        if self.release_time < 0.0:
            raise ValueError("Job.release_time must be non-negative")
        if self.deadline is not None and self.deadline < self.release_time:
            raise ValueError("Job.deadline must be >= release_time")
        if self.weight <= 0.0:
            raise ValueError("Job.weight must be positive")
        for name, value in self.demand.items():
            if not name:
                raise ValueError("demand dimension name must be non-empty")
            if value < 0.0:
                raise ValueError(f"demand[{name!r}] must be non-negative")
        object.__setattr__(self, "demand", MappingProxyType(dict(self.demand)))
        if not isinstance(self.predecessors, frozenset):
            object.__setattr__(self, "predecessors", frozenset(self.predecessors))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            Job,
            (
                self.id,
                self.processing_time,
                dict(self.demand),
                self.release_time,
                self.deadline,
                self.weight,
                self.predecessors,
            ),
        )
