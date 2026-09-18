from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class Assignment:
    """Placement of one job on one resource at one start time."""

    resource_id: str
    start_time: float

    def __post_init__(self) -> None:
        if not self.resource_id:
            raise ValueError("Assignment.resource_id must be a non-empty string")
        if self.start_time < 0.0:
            raise ValueError("Assignment.start_time must be non-negative")


@dataclass(frozen=True, slots=True, eq=False)
class Solution:
    """A candidate solution: one assignment per job, keyed by job id.

    The solution does not know the problem it belongs to. Completeness,
    resource existence and precedence are validated by the evaluator against a
    concrete :class:`~edmo.domain.problem.Problem`.
    """

    assignments: Mapping[str, Assignment] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for job_id, a in self.assignments.items():
            if not job_id:
                raise ValueError("Solution assignment keys must be non-empty strings")
            if not isinstance(a, Assignment):
                raise TypeError(
                    f"Solution.assignments[{job_id!r}] must be an Assignment"
                )
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType(dict(self.assignments)),
        )

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (Solution, (dict(self.assignments),))

    def __contains__(self, job_id: object) -> bool:
        return job_id in self.assignments

    def __len__(self) -> int:
        return len(self.assignments)

    def assignment_for(self, job_id: str) -> Assignment:
        return self.assignments[job_id]

    def with_assignment(self, job_id: str, assignment: Assignment) -> Solution:
        updated = dict(self.assignments)
        updated[job_id] = assignment
        return Solution(assignments=updated)
