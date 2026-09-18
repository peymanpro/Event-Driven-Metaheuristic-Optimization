from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from edmo.domain.problem import Problem
from edmo.domain.solution import Solution


@dataclass(frozen=True, slots=True)
class Schedule:
    """Derived timing for assignments with a resolvable resource.

    ``resource_id``, ``start`` and ``completion`` are keyed by job id and are
    only populated for jobs whose assignment references an existing resource.
    """

    resource_id: Mapping[str, str] = field(default_factory=dict)
    start: Mapping[str, float] = field(default_factory=dict)
    completion: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "resource_id", MappingProxyType(dict(self.resource_id))
        )
        object.__setattr__(self, "start", MappingProxyType(dict(self.start)))
        object.__setattr__(
            self, "completion", MappingProxyType(dict(self.completion))
        )


@dataclass(frozen=True, slots=True)
class Violation:
    """A single atomic constraint violation with an explicit magnitude."""

    kind: str
    magnitude: float
    message: str
    job_id: str | None = None
    resource_id: str | None = None


@dataclass(frozen=True, slots=True)
class PenaltyWeights:
    """Positive weights applied to normalized magnitudes of each violation kind."""

    missing_assignment: float = 1.0e6
    unknown_resource: float = 1.0e6
    release_time: float = 1.0e3
    precedence: float = 1.0e3
    deadline: float = 1.0e3
    capacity: float = 1.0e3


DEFAULT_WEIGHTS = PenaltyWeights()


@dataclass(frozen=True, slots=True)
class Evaluation:
    objective: float
    penalty: float
    total: float
    feasible: bool
    violations: tuple[Violation, ...]
    schedule: Schedule


def compute_schedule(problem: Problem, solution: Solution) -> Schedule:
    """Build a schedule from a solution, skipping unknown resources."""
    resource_id: dict[str, str] = {}
    start: dict[str, float] = {}
    completion: dict[str, float] = {}
    for job in problem.jobs:
        assignment = solution.assignments.get(job.id)
        if assignment is None:
            continue
        try:
            resource = problem.resource_by_id(assignment.resource_id)
        except KeyError:
            continue
        duration = job.processing_time / resource.speed
        resource_id[job.id] = resource.id
        start[job.id] = assignment.start_time
        completion[job.id] = assignment.start_time + duration
    return Schedule(resource_id=resource_id, start=start, completion=completion)


def compute_objective(problem: Problem, schedule: Schedule) -> float:
    """Weighted sum of completion times plus the makespan."""
    weighted_sum = 0.0
    makespan = 0.0
    for job in problem.jobs:
        completion = schedule.completion.get(job.id)
        if completion is None:
            continue
        weighted_sum += job.weight * completion
        if completion > makespan:
            makespan = completion
    return weighted_sum + makespan


def _max_overlap_demand(intervals: Sequence[tuple[float, float, float]]) -> float:
    """Peak concurrent demand over half-open intervals ``[start, end)``."""
    events: list[tuple[float, int, float]] = []
    for start, end, demand in intervals:
        events.append((start, 1, demand))
        events.append((end, -1, demand))
    events.sort(key=lambda e: (e[0], e[1]))
    current = 0.0
    peak = 0.0
    for _, sign, demand in events:
        current += sign * demand
        if current > peak:
            peak = current
    return peak


def collect_violations(
    problem: Problem,
    solution: Solution,
    schedule: Schedule,
) -> tuple[Violation, ...]:
    out: list[Violation] = []

    for job in problem.jobs:
        assignment = solution.assignments.get(job.id)
        if assignment is None:
            out.append(
                Violation(
                    kind="missing_assignment",
                    magnitude=1.0,
                    message=f"job {job.id!r} has no assignment",
                    job_id=job.id,
                )
            )
            continue

        if job.id not in schedule.start:
            out.append(
                Violation(
                    kind="unknown_resource",
                    magnitude=1.0,
                    message=(
                        f"job {job.id!r} references unknown resource "
                        f"{assignment.resource_id!r}"
                    ),
                    job_id=job.id,
                    resource_id=assignment.resource_id,
                )
            )
            continue

        start = schedule.start[job.id]
        if start < job.release_time:
            out.append(
                Violation(
                    kind="release_time",
                    magnitude=job.release_time - start,
                    message=(
                        f"job {job.id!r} starts at {start} before release_time "
                        f"{job.release_time}"
                    ),
                    job_id=job.id,
                )
            )

        if job.deadline is not None:
            completion = schedule.completion[job.id]
            if completion > job.deadline:
                out.append(
                    Violation(
                        kind="deadline",
                        magnitude=completion - job.deadline,
                        message=(
                            f"job {job.id!r} completes at {completion} after "
                            f"deadline {job.deadline}"
                        ),
                        job_id=job.id,
                    )
                )

    for job in problem.jobs:
        if job.id not in schedule.start:
            continue
        job_start = schedule.start[job.id]
        for pred_id in sorted(job.predecessors):
            pred_completion = schedule.completion.get(pred_id)
            if pred_completion is None:
                continue
            if job_start < pred_completion:
                out.append(
                    Violation(
                        kind="precedence",
                        magnitude=pred_completion - job_start,
                        message=(
                            f"job {job.id!r} starts at {job_start} before "
                            f"predecessor {pred_id!r} completes at "
                            f"{pred_completion}"
                        ),
                        job_id=job.id,
                    )
                )

    for resource in problem.resources:
        assigned = [j for j in problem.jobs if schedule.resource_id.get(j.id) == resource.id]
        dims: set[str] = set(resource.capacity.keys())
        for job in assigned:
            dims.update(job.demand.keys())
        for dim in sorted(dims):
            cap = resource.capacity.get(dim, 0.0)
            intervals = [
                (schedule.start[j.id], schedule.completion[j.id], j.demand.get(dim, 0.0))
                for j in assigned
                if j.demand.get(dim, 0.0) > 0.0
            ]
            peak = _max_overlap_demand(intervals)
            if peak > cap:
                out.append(
                    Violation(
                        kind="capacity",
                        magnitude=peak - cap,
                        message=(
                            f"resource {resource.id!r} dimension {dim!r} demand "
                            f"{peak} exceeds capacity {cap}"
                        ),
                        resource_id=resource.id,
                    )
                )

    return tuple(out)


def compute_penalty(
    violations: Sequence[Violation],
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
) -> float:
    total = 0.0
    for v in violations:
        try:
            weight = getattr(weights, v.kind)
        except AttributeError as exc:
            raise ValueError(
                f"no penalty weight configured for violation kind {v.kind!r}"
            ) from exc
        total += weight * v.magnitude
    return total


def evaluate(
    problem: Problem,
    solution: Solution,
    weights: PenaltyWeights = DEFAULT_WEIGHTS,
) -> Evaluation:
    schedule = compute_schedule(problem, solution)
    objective = compute_objective(problem, schedule)
    violations = collect_violations(problem, solution, schedule)
    penalty = compute_penalty(violations, weights)
    return Evaluation(
        objective=objective,
        penalty=penalty,
        total=objective + penalty,
        feasible=len(violations) == 0,
        violations=violations,
        schedule=schedule,
    )
