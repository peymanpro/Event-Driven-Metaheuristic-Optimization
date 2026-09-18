import pytest

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.domain.solution import Assignment, Solution


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"), Resource(id="r1")),
        jobs=(
            Job(id="j1", processing_time=1.0),
            Job(id="j2", processing_time=2.0),
            Job(id="j3", processing_time=1.5),
        ),
    )


def test_chromosome_valid() -> None:
    c = Chromosome(resource_idx=(0, 1, 0), start_time=(0.0, 1.0, 2.0))
    assert c.length == 3


def test_chromosome_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        Chromosome(resource_idx=(0, 1), start_time=(0.0,))


def test_chromosome_rejects_negative_index() -> None:
    with pytest.raises(ValueError):
        Chromosome(resource_idx=(-1,), start_time=(0.0,))


def test_chromosome_rejects_negative_start() -> None:
    with pytest.raises(ValueError):
        Chromosome(resource_idx=(0,), start_time=(-0.5,))


def test_chromosome_to_solution() -> None:
    problem = _problem()
    c = Chromosome(resource_idx=(0, 1, 0), start_time=(0.0, 1.0, 2.0))
    s = c.to_solution(problem)
    assert set(s.assignments.keys()) == {"j1", "j2", "j3"}
    assert s.assignment_for("j1") == Assignment(resource_id="r0", start_time=0.0)
    assert s.assignment_for("j2") == Assignment(resource_id="r1", start_time=1.0)


def test_chromosome_to_solution_length_mismatch() -> None:
    problem = _problem()
    c = Chromosome(resource_idx=(0,), start_time=(0.0,))
    with pytest.raises(ValueError):
        c.to_solution(problem)


def test_chromosome_to_solution_index_out_of_range() -> None:
    problem = _problem()
    c = Chromosome(resource_idx=(0, 5, 0), start_time=(0.0, 1.0, 2.0))
    with pytest.raises(ValueError):
        c.to_solution(problem)


def test_chromosome_roundtrip_from_solution() -> None:
    problem = _problem()
    c0 = Chromosome(resource_idx=(0, 1, 1), start_time=(0.0, 2.5, 4.0))
    s = c0.to_solution(problem)
    c1 = Chromosome.from_solution(problem, s)
    assert c1.resource_idx == c0.resource_idx
    assert c1.start_time == c0.start_time


def test_chromosome_from_solution_missing_assignment() -> None:
    problem = _problem()
    partial = Solution(
        assignments={
            "j1": Assignment(resource_id="r0", start_time=0.0),
        }
    )
    with pytest.raises(ValueError):
        Chromosome.from_solution(problem, partial)


def test_chromosome_from_solution_unknown_resource() -> None:
    problem = _problem()
    bad = Solution(
        assignments={
            "j1": Assignment(resource_id="ghost", start_time=0.0),
            "j2": Assignment(resource_id="r1", start_time=0.0),
            "j3": Assignment(resource_id="r0", start_time=0.0),
        }
    )
    with pytest.raises(ValueError):
        Chromosome.from_solution(problem, bad)
