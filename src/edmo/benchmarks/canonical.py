from __future__ import annotations

from edmo.benchmarks.problem_generator import known_feasible_solution
from edmo.domain.change import ChangeDeadline, ChangeWeight, ProblemChange
from edmo.domain.evaluation import evaluate
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.domain.solution import Solution


def canonical_resource_scheduling() -> Problem:
    """A small, hand-readable scheduling instance.

    Three resources, five jobs. Every job fits on every resource, but the
    total demand exceeds any single resource's capacity, so scheduling all
    jobs on one resource at the same time is infeasible. The canonical
    round-robin sequential schedule is feasible, so the instance has a
    non-empty feasible region.
    """
    resources = (
        Resource(id="r0", capacity={"cpu": 3.0}, speed=1.0),
        Resource(id="r1", capacity={"cpu": 3.0}, speed=1.2),
        Resource(id="r2", capacity={"cpu": 3.0}, speed=0.9),
    )
    jobs = (
        Job(
            id="j0",
            processing_time=2.0,
            demand={"cpu": 1.0},
            release_time=0.0,
            deadline=10.0,
            weight=1.0,
        ),
        Job(
            id="j1",
            processing_time=3.0,
            demand={"cpu": 2.0},
            release_time=1.0,
            deadline=12.0,
            weight=2.0,
        ),
        Job(
            id="j2",
            processing_time=1.5,
            demand={"cpu": 1.5},
            release_time=0.5,
            deadline=9.0,
            weight=1.5,
        ),
        Job(
            id="j3",
            processing_time=2.5,
            demand={"cpu": 2.5},
            release_time=2.0,
            deadline=15.0,
            weight=1.0,
        ),
        Job(
            id="j4",
            processing_time=1.0,
            demand={"cpu": 1.0},
            release_time=0.0,
            deadline=6.0,
            weight=3.0,
        ),
    )
    return Problem(
        id="canonical_resource_scheduling",
        resources=resources,
        jobs=jobs,
        metadata={"kind": "canonical"},
    )


def canonical_dynamic_changes() -> tuple[ProblemChange, ...]:
    """The dynamic change set used by the canonical benchmark."""
    return (
        ChangeWeight("j1", 4.0),
        ChangeDeadline("j0", 5.0),
    )


def is_canonical_instance_feasible() -> bool:
    """Return True when the canonical instance has at least one feasible solution."""
    problem = canonical_resource_scheduling()
    solution = known_feasible_solution(problem)
    if solution is None:
        return False
    return evaluate(problem, solution).feasible


def canonical_feasible_solution() -> Solution:
    """Return a canonical feasible solution or raise."""
    problem = canonical_resource_scheduling()
    solution = known_feasible_solution(problem)
    if solution is None:
        raise RuntimeError("canonical instance has no feasible solution")
    return solution
