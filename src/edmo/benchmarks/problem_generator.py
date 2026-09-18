from __future__ import annotations

from dataclasses import dataclass
from random import Random

from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.domain.solution import Assignment, Solution


@dataclass(frozen=True, slots=True)
class RandomProblemSpec:
    """Specification for a reproducible random problem instance.

    Capacity factors are expressed relative to ``max_demand``. A factor of
    1.05 guarantees that any single job fits on any resource, while a factor
    well below ``n_jobs`` keeps capacity constraints meaningful: scheduling
    many high-demand jobs on the same resource at the same time is infeasible.

    Deadlines are constructed so that a canonical round-robin sequential
    schedule is feasible; a slack of 0 makes the instance tight, larger slack
    relaxes the deadline constraint.
    """

    n_resources: int = 3
    n_jobs: int = 8
    max_processing_time: float = 5.0
    max_release_time: float = 2.0
    min_demand: float = 0.5
    max_demand: float = 2.0
    min_capacity_factor: float = 1.05
    max_capacity_factor: float = 2.5
    min_speed: float = 0.8
    max_speed: float = 1.5
    max_deadline_slack: float = 10.0
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
        if self.min_demand <= 0.0:
            raise ValueError("min_demand must be positive")
        if self.max_demand < self.min_demand:
            raise ValueError("max_demand must be >= min_demand")
        if self.min_capacity_factor < 1.0:
            raise ValueError(
                "min_capacity_factor must be >= 1.0 so single jobs always fit"
            )
        if self.max_capacity_factor < self.min_capacity_factor:
            raise ValueError("max_capacity_factor must be >= min_capacity_factor")
        if self.min_speed <= 0.0:
            raise ValueError("min_speed must be positive")
        if self.max_speed < self.min_speed:
            raise ValueError("max_speed must be >= min_speed")
        if self.max_deadline_slack < 0.0:
            raise ValueError("max_deadline_slack must be non-negative")


def known_feasible_solution(problem: Problem) -> Solution | None:
    """Construct a canonical feasible solution if one can be built.

    Strategy: round-robin jobs across resources, place them sequentially in
    each resource at ``max(release_time, previous_completion)``. Since no job
    has predecessors in generated instances, this respects precedence. If the
    resulting solution is feasible under :func:`~edmo.domain.evaluation.evaluate`,
    it is returned; otherwise ``None``.
    """
    if not problem.resources:
        return None

    from edmo.domain.evaluation import evaluate

    # Order jobs by release time then by id for determinism.
    jobs = sorted(problem.jobs, key=lambda j: (j.release_time, j.id))
    next_available = {r.id: 0.0 for r in problem.resources}
    resources_cycle = list(problem.resources)

    assignments: dict[str, Assignment] = {}
    for index, job in enumerate(jobs):
        resource = resources_cycle[index % len(resources_cycle)]
        start = max(job.release_time, next_available[resource.id])
        assignments[job.id] = Assignment(resource_id=resource.id, start_time=start)
        next_available[resource.id] = start + job.processing_time / resource.speed

    candidate = Solution(assignments=assignments)
    if evaluate(problem, candidate).feasible:
        return candidate
    return None


def random_problem(spec: RandomProblemSpec | None = None) -> Problem:
    """Generate a reproducible random problem instance.

    Demands are generated first; each resource's capacity is set to
    ``max_demand * factor`` with a factor drawn from the configured range.
    Deadlines are derived from a canonical round-robin sequential schedule
    plus a non-negative random slack, guaranteeing that at least one feasible
    solution exists.
    """
    s = spec or RandomProblemSpec()
    rng = Random(s.seed)

    demands = [
        round(rng.uniform(s.min_demand, s.max_demand), 3) for _ in range(s.n_jobs)
    ]
    max_demand = max(demands)

    resources = tuple(
        Resource(
            id=f"r{i}",
            capacity={
                "cpu": round(
                    max_demand
                    * rng.uniform(s.min_capacity_factor, s.max_capacity_factor),
                    3,
                )
            },
            speed=round(rng.uniform(s.min_speed, s.max_speed), 3),
        )
        for i in range(s.n_resources)
    )

    processings = [
        round(rng.uniform(0.5, s.max_processing_time), 3) for _ in range(s.n_jobs)
    ]
    releases = [
        round(rng.uniform(0.0, s.max_release_time), 3) for _ in range(s.n_jobs)
    ]

    # Canonical schedule drives deadlines so that a feasible solution exists.
    order = sorted(range(s.n_jobs), key=lambda i: (releases[i], f"j{i}"))
    completions = [0.0] * s.n_jobs
    next_available = [0.0] * s.n_resources
    for position, i in enumerate(order):
        r_idx = position % s.n_resources
        start = max(releases[i], next_available[r_idx])
        completions[i] = start + processings[i] / resources[r_idx].speed
        next_available[r_idx] = completions[i]

    jobs: list[Job] = []
    for i in range(s.n_jobs):
        slack = rng.uniform(0.0, s.max_deadline_slack)
        deadline = round(completions[i] + slack, 3)
        jobs.append(
            Job(
                id=f"j{i}",
                processing_time=processings[i],
                demand={"cpu": demands[i]},
                release_time=releases[i],
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
