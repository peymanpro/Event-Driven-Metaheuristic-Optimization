from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


@dataclass(frozen=True, slots=True)
class AddJob:
    job: Job


@dataclass(frozen=True, slots=True)
class RemoveJob:
    job_id: str


@dataclass(frozen=True, slots=True)
class ChangeDeadline:
    job_id: str
    deadline: float | None


@dataclass(frozen=True, slots=True)
class ChangeWeight:
    job_id: str
    weight: float


@dataclass(frozen=True, slots=True)
class AddResource:
    resource: Resource


@dataclass(frozen=True, slots=True)
class RemoveResource:
    resource_id: str


@dataclass(frozen=True, slots=True)
class ChangeResourceCapacity:
    resource_id: str
    capacity: dict[str, float]


@dataclass(frozen=True, slots=True)
class ChangeResourceSpeed:
    resource_id: str
    speed: float


type ProblemChange = (
    AddJob
    | RemoveJob
    | ChangeDeadline
    | ChangeWeight
    | AddResource
    | RemoveResource
    | ChangeResourceCapacity
    | ChangeResourceSpeed
)


def apply_change(problem: Problem, change: ProblemChange) -> Problem:
    """Return a new Problem with ``change`` applied and version incremented.

    The original problem is never mutated. Structural errors (unknown ids,
    duplicates, dangling predecessors) raise ``ValueError``.
    """
    if isinstance(change, AddJob):
        return _add_job(problem, change.job)
    if isinstance(change, RemoveJob):
        return _remove_job(problem, change.job_id)
    if isinstance(change, ChangeDeadline):
        return _change_deadline(problem, change.job_id, change.deadline)
    if isinstance(change, ChangeWeight):
        return _change_weight(problem, change.job_id, change.weight)
    if isinstance(change, AddResource):
        return _add_resource(problem, change.resource)
    if isinstance(change, RemoveResource):
        return _remove_resource(problem, change.resource_id)
    if isinstance(change, ChangeResourceCapacity):
        return _change_resource_capacity(problem, change.resource_id, change.capacity)
    if isinstance(change, ChangeResourceSpeed):
        return _change_resource_speed(problem, change.resource_id, change.speed)
    raise TypeError(f"unsupported change type: {type(change).__name__}")


def apply_changes(
    problem: Problem,
    changes: list[ProblemChange] | tuple[ProblemChange, ...],
) -> Problem:
    """Apply a sequence of changes in order, returning the final problem."""
    current = problem
    for change in changes:
        current = apply_change(current, change)
    return current


def _bump(problem: Problem, **overrides: object) -> Problem:
    fields: dict[str, object] = {
        "id": problem.id,
        "resources": problem.resources,
        "jobs": problem.jobs,
        "version": problem.version + 1,
        "metadata": dict(problem.metadata),
    }
    fields.update(overrides)
    return Problem(**fields)  # type: ignore[arg-type]


def _add_job(problem: Problem, job: Job) -> Problem:
    if any(j.id == job.id for j in problem.jobs):
        raise ValueError(f"job id {job.id!r} already exists")
    known = {j.id for j in problem.jobs}
    unknown = job.predecessors - known
    if unknown:
        raise ValueError(
            f"new job {job.id!r} references unknown predecessors: {sorted(unknown)}"
        )
    return _bump(problem, jobs=(*problem.jobs, job))


def _remove_job(problem: Problem, job_id: str) -> Problem:
    if not any(j.id == job_id for j in problem.jobs):
        raise ValueError(f"job id {job_id!r} does not exist")
    remaining = tuple(j for j in problem.jobs if j.id != job_id)
    for j in remaining:
        if job_id in j.predecessors:
            raise ValueError(
                f"cannot remove job {job_id!r}: it is a predecessor of {j.id!r}"
            )
    return _bump(problem, jobs=remaining)


def _change_deadline(problem: Problem, job_id: str, deadline: float | None) -> Problem:
    found = False
    new_jobs: list[Job] = []
    for j in problem.jobs:
        if j.id != job_id:
            new_jobs.append(j)
            continue
        found = True
        new_jobs.append(
            Job(
                id=j.id,
                processing_time=j.processing_time,
                demand=dict(j.demand),
                release_time=j.release_time,
                deadline=deadline,
                weight=j.weight,
                predecessors=j.predecessors,
            )
        )
    if not found:
        raise ValueError(f"job id {job_id!r} does not exist")
    return _bump(problem, jobs=tuple(new_jobs))


def _change_weight(problem: Problem, job_id: str, weight: float) -> Problem:
    if weight <= 0.0:
        raise ValueError("weight must be positive")
    found = False
    new_jobs: list[Job] = []
    for j in problem.jobs:
        if j.id != job_id:
            new_jobs.append(j)
            continue
        found = True
        new_jobs.append(
            Job(
                id=j.id,
                processing_time=j.processing_time,
                demand=dict(j.demand),
                release_time=j.release_time,
                deadline=j.deadline,
                weight=weight,
                predecessors=j.predecessors,
            )
        )
    if not found:
        raise ValueError(f"job id {job_id!r} does not exist")
    return _bump(problem, jobs=tuple(new_jobs))


def _add_resource(problem: Problem, resource: Resource) -> Problem:
    if any(r.id == resource.id for r in problem.resources):
        raise ValueError(f"resource id {resource.id!r} already exists")
    return _bump(problem, resources=(*problem.resources, resource))


def _remove_resource(problem: Problem, resource_id: str) -> Problem:
    if not any(r.id == resource_id for r in problem.resources):
        raise ValueError(f"resource id {resource_id!r} does not exist")
    remaining = tuple(r for r in problem.resources if r.id != resource_id)
    return _bump(problem, resources=remaining)


def _change_resource_capacity(
    problem: Problem,
    resource_id: str,
    capacity: dict[str, float],
) -> Problem:
    for name, value in capacity.items():
        if not name:
            raise ValueError("capacity dimension name must be non-empty")
        if value < 0.0:
            raise ValueError(f"capacity[{name!r}] must be non-negative")
    found = False
    new_resources: list[Resource] = []
    for r in problem.resources:
        if r.id != resource_id:
            new_resources.append(r)
            continue
        found = True
        new_resources.append(
            Resource(id=r.id, capacity=MappingProxyType(dict(capacity)), speed=r.speed)
        )
    if not found:
        raise ValueError(f"resource id {resource_id!r} does not exist")
    return _bump(problem, resources=tuple(new_resources))


def _change_resource_speed(problem: Problem, resource_id: str, speed: float) -> Problem:
    if speed <= 0.0:
        raise ValueError("speed must be positive")
    found = False
    new_resources: list[Resource] = []
    for r in problem.resources:
        if r.id != resource_id:
            new_resources.append(r)
            continue
        found = True
        new_resources.append(
            Resource(id=r.id, capacity=dict(r.capacity), speed=speed)
        )
    if not found:
        raise ValueError(f"resource id {resource_id!r} does not exist")
    return _bump(problem, resources=tuple(new_resources))
