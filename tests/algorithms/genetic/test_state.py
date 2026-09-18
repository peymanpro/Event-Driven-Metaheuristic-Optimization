from random import Random

import pytest

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.state import GAState
from edmo.domain.evaluation import evaluate
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.domain.solution import Assignment, Solution


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(Resource(id="r0"),),
        jobs=(Job(id="j1", processing_time=1.0),),
    )


def _state() -> GAState:
    problem = _problem()
    c = Chromosome((0,), (0.0,))
    sol = Solution(assignments={"j1": Assignment(resource_id="r0", start_time=0.0)})
    ev = evaluate(problem, sol)
    return GAState(
        population=(c,),
        best_chromosome=c,
        best_evaluation=ev,
        generation=0,
        history=(ev.total,),
        problem_version=1,
        rng=Random(0),
    )


def test_state_valid() -> None:
    s = _state()
    assert s.generation == 0
    assert s.problem_version == 1


def test_state_rejects_negative_generation() -> None:
    problem = _problem()
    c = Chromosome((0,), (0.0,))
    sol = Solution(assignments={"j1": Assignment(resource_id="r0", start_time=0.0)})
    ev = evaluate(problem, sol)
    with pytest.raises(ValueError):
        GAState(
            population=(c,),
            best_chromosome=c,
            best_evaluation=ev,
            generation=-1,
            history=(ev.total,),
            problem_version=1,
            rng=Random(0),
        )


def test_state_rejects_empty_population() -> None:
    problem = _problem()
    c = Chromosome((0,), (0.0,))
    sol = Solution(assignments={"j1": Assignment(resource_id="r0", start_time=0.0)})
    ev = evaluate(problem, sol)
    with pytest.raises(ValueError):
        GAState(
            population=(),
            best_chromosome=c,
            best_evaluation=ev,
            generation=0,
            history=(ev.total,),
            problem_version=1,
            rng=Random(0),
        )
