import pytest

from edmo.algorithms.common.interface import Optimizer, OptimizerResult
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


def _result() -> OptimizerResult:
    problem = _problem()
    solution = Solution(assignments={"j1": Assignment(resource_id="r0", start_time=0.0)})
    ev = evaluate(problem, solution)
    return OptimizerResult(
        best_solution=solution,
        best_evaluation=ev,
        history=(ev.total,),
        iterations=1,
        seed=0,
        stopped_reason="test",
    )


def test_result_is_immutable() -> None:
    r = _result()
    with pytest.raises(AttributeError):
        r.iterations = 5  # type: ignore[misc]


def test_optimizer_protocol_requires_name_and_callable() -> None:
    class Dummy:
        name = "dummy"

        def __call__(self, problem, config=None, weights=None):  # type: ignore[no-untyped-def]
            return _result()

    assert isinstance(Dummy(), Optimizer)
