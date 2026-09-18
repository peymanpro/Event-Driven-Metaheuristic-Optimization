import pytest

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.repair import (
    repair,
    repair_precedence,
    repair_release_times,
)
from edmo.domain.evaluation import evaluate
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def test_repair_release_times() -> None:
    problem = Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(
            Job(id="j1", processing_time=1.0, release_time=5.0),
            Job(id="j2", processing_time=1.0, release_time=2.0),
        ),
    )
    c = Chromosome(resource_idx=(0, 0), start_time=(0.0, 0.0))
    fixed = repair_release_times(problem, c)
    assert fixed.start_time == (5.0, 2.0)


def test_repair_release_times_noop() -> None:
    problem = Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(Job(id="j1", processing_time=1.0, release_time=0.0),),
    )
    c = Chromosome(resource_idx=(0,), start_time=(3.0,))
    assert repair_release_times(problem, c) is c


def test_repair_precedence() -> None:
    problem = Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(
            Job(id="j1", processing_time=2.0),
            Job(id="j2", processing_time=1.0, predecessors=frozenset({"j1"})),
        ),
    )
    c = Chromosome(resource_idx=(0, 0), start_time=(0.0, 0.0))
    fixed = repair_precedence(problem, c)
    # j2 must start at >= finish(j1) = 2.0.
    assert fixed.start_time[1] >= 2.0


def test_repair_precedence_chain() -> None:
    problem = Problem(
        id="p1",
        resources=(Resource(id="r0"), Resource(id="r1", speed=2.0)),
        jobs=(
            Job(id="j1", processing_time=2.0),
            Job(id="j2", processing_time=2.0, predecessors=frozenset({"j1"})),
            Job(id="j3", processing_time=2.0, predecessors=frozenset({"j2"})),
        ),
    )
    c = Chromosome(resource_idx=(0, 1, 0), start_time=(0.0, 0.0, 0.0))
    fixed = repair_precedence(problem, c)
    assert fixed.start_time[0] == 0.0
    assert fixed.start_time[1] >= 2.0  # j1 finishes at 2.0
    assert fixed.start_time[2] >= fixed.start_time[1] + 2.0 / 2.0


def test_repair_combined_produces_feasible() -> None:
    problem = Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(
            Job(id="j1", processing_time=1.0, release_time=1.0, deadline=10.0),
            Job(
                id="j2",
                processing_time=1.0,
                release_time=2.0,
                deadline=10.0,
                predecessors=frozenset({"j1"}),
            ),
        ),
    )
    c = Chromosome(resource_idx=(0, 0), start_time=(0.0, 0.0))
    fixed = repair(problem, c)
    ev = evaluate(problem, fixed.to_solution(problem))
    assert ev.feasible


def test_repair_detects_cycle() -> None:
    # Constructing a problem with a cycle is prevented at the Problem level
    # only for unknown predecessors; the cycle is detected during repair.
    problem = Problem.__new__(Problem)
    object.__setattr__(problem, "id", "p1")
    object.__setattr__(problem, "version", 1)
    object.__setattr__(
        problem,
        "resources",
        (Resource(id="r0"),),
    )
    object.__setattr__(
        problem,
        "jobs",
        (
            Job(id="j1", processing_time=1.0, predecessors=frozenset({"j2"})),
            Job(id="j2", processing_time=1.0, predecessors=frozenset({"j1"})),
        ),
    )
    c = Chromosome(resource_idx=(0, 0), start_time=(0.0, 0.0))
    with pytest.raises(ValueError):
        repair_precedence(problem, c)
