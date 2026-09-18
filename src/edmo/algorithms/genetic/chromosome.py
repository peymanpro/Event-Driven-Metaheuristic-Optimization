from __future__ import annotations

from dataclasses import dataclass

from edmo.domain.problem import Problem
from edmo.domain.solution import Assignment, Solution


@dataclass(frozen=True, slots=True, eq=False)
class Chromosome:
    """Fixed-length gene vector for the genetic algorithm.

    Gene ``i`` corresponds to ``problem.jobs[i]`` and encodes
    ``(resource_idx, start_time)`` where ``resource_idx`` indexes into
    ``problem.resources``. The chromosome carries no reference to the problem;
    callers must convert with the same problem instance they evolved against.
    """

    resource_idx: tuple[int, ...]
    start_time: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.resource_idx) != len(self.start_time):
            raise ValueError("resource_idx and start_time must have equal length")
        for idx in self.resource_idx:
            if idx < 0:
                raise ValueError("resource_idx entries must be non-negative")
        for t in self.start_time:
            if t < 0.0:
                raise ValueError("start_time entries must be non-negative")

    @property
    def length(self) -> int:
        return len(self.resource_idx)

    def to_solution(self, problem: Problem) -> Solution:
        if self.length != len(problem.jobs):
            raise ValueError(
                f"chromosome length {self.length} does not match "
                f"number of jobs {len(problem.jobs)}"
            )
        n_resources = len(problem.resources)
        assignments: dict[str, Assignment] = {}
        for i, job in enumerate(problem.jobs):
            idx = self.resource_idx[i]
            if idx >= n_resources:
                raise ValueError(
                    f"resource index {idx} out of range for {n_resources} resources"
                )
            assignments[job.id] = Assignment(
                resource_id=problem.resources[idx].id,
                start_time=self.start_time[i],
            )
        return Solution(assignments=assignments)

    @classmethod
    def from_solution(cls, problem: Problem, solution: Solution) -> Chromosome:
        if len(solution) != len(problem.jobs):
            raise ValueError(
                f"solution has {len(solution)} assignments but problem has "
                f"{len(problem.jobs)} jobs"
            )
        index_by_resource = {r.id: i for i, r in enumerate(problem.resources)}
        ridx: list[int] = []
        starts: list[float] = []
        for job in problem.jobs:
            try:
                assignment = solution.assignment_for(job.id)
            except KeyError as exc:
                raise ValueError(
                    f"solution is missing assignment for job {job.id!r}"
                ) from exc
            try:
                i = index_by_resource[assignment.resource_id]
            except KeyError as exc:
                raise ValueError(
                    f"solution references unknown resource "
                    f"{assignment.resource_id!r}"
                ) from exc
            ridx.append(i)
            starts.append(assignment.start_time)
        return cls(resource_idx=tuple(ridx), start_time=tuple(starts))
