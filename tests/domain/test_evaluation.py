import pytest

from edmo.domain.evaluation import (
    Evaluation,
    PenaltyWeights,
    Schedule,
    collect_violations,
    compute_objective,
    compute_penalty,
    compute_schedule,
    evaluate,
)
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.domain.solution import Assignment, Solution


def _problem(
    resources: tuple[Resource, ...] | None = None,
    jobs: tuple[Job, ...] | None = None,
) -> Problem:
    return Problem(
        id="p1",
        resources=resources or (Resource(id="r1"), Resource(id="r2")),
        jobs=jobs
        or (
            Job(id="j1", processing_time=1.0),
            Job(id="j2", processing_time=2.0),
        ),
    )


def test_compute_schedule_basic() -> None:
    problem = _problem()
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=1.0),
        }
    )
    sched = compute_schedule(problem, solution)
    assert sched.resource_id["j1"] == "r1"
    assert sched.start["j1"] == 0.0
    assert sched.completion["j1"] == 1.0
    assert sched.completion["j2"] == 3.0


def test_compute_schedule_respects_resource_speed() -> None:
    problem = _problem(resources=(Resource(id="r1", speed=2.0),))
    solution = Solution(assignments={"j1": Assignment(resource_id="r1", start_time=0.0)})
    sched = compute_schedule(problem, solution)
    assert sched.completion["j1"] == 0.5


def test_compute_schedule_skips_unknown_resource() -> None:
    problem = _problem()
    solution = Solution(assignments={"j1": Assignment(resource_id="nope", start_time=0.0)})
    sched = compute_schedule(problem, solution)
    assert "j1" not in sched.start


def test_compute_objective_weighted_sum_and_makespan() -> None:
    problem = _problem()
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=0.0),
        }
    )
    sched = compute_schedule(problem, solution)
    # weighted_sum = 1*1 + 1*2 = 3; makespan = 2; total = 5
    assert compute_objective(problem, sched) == 5.0


def test_evaluate_feasible() -> None:
    problem = _problem()
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=0.0),
        }
    )
    ev = evaluate(problem, solution)
    assert isinstance(ev, Evaluation)
    assert ev.feasible
    assert ev.violations == ()
    assert ev.penalty == 0.0
    assert ev.total == ev.objective


def test_evaluate_missing_assignments_infeasible() -> None:
    problem = _problem()
    solution = Solution()
    ev = evaluate(problem, solution)
    assert not ev.feasible
    kinds = [v.kind for v in ev.violations]
    assert kinds.count("missing_assignment") == 2


def test_evaluate_unknown_resource_violation() -> None:
    problem = _problem()
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="ghost", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=0.0),
        }
    )
    ev = evaluate(problem, solution)
    kinds = [v.kind for v in ev.violations]
    assert "unknown_resource" in kinds


def test_evaluate_release_time_violation() -> None:
    jobs = (
        Job(id="j1", processing_time=1.0, release_time=2.0),
        Job(id="j2", processing_time=1.0),
    )
    problem = _problem(jobs=jobs)
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=0.0),
        }
    )
    ev = evaluate(problem, solution)
    kinds = [v.kind for v in ev.violations]
    assert "release_time" in kinds


def test_evaluate_deadline_violation() -> None:
    jobs = (
        Job(id="j1", processing_time=5.0, deadline=2.0),
        Job(id="j2", processing_time=1.0),
    )
    problem = _problem(jobs=jobs)
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=0.0),
        }
    )
    ev = evaluate(problem, solution)
    kinds = [v.kind for v in ev.violations]
    assert "deadline" in kinds


def test_evaluate_precedence_violation() -> None:
    jobs = (
        Job(id="j1", processing_time=2.0),
        Job(id="j2", processing_time=1.0, predecessors=frozenset({"j1"})),
    )
    problem = _problem(jobs=jobs)
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=0.0),
        }
    )
    ev = evaluate(problem, solution)
    kinds = [v.kind for v in ev.violations]
    assert "precedence" in kinds


def test_evaluate_precedence_satisfied() -> None:
    jobs = (
        Job(id="j1", processing_time=2.0),
        Job(id="j2", processing_time=1.0, predecessors=frozenset({"j1"})),
    )
    problem = _problem(jobs=jobs)
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r2", start_time=2.0),
        }
    )
    ev = evaluate(problem, solution)
    assert ev.feasible


def test_evaluate_capacity_violation() -> None:
    resources = (Resource(id="r1", capacity={"cpu": 2.0}),)
    jobs = (
        Job(id="j1", processing_time=2.0, demand={"cpu": 1.5}),
        Job(id="j2", processing_time=2.0, demand={"cpu": 1.5}),
    )
    problem = _problem(resources=resources, jobs=jobs)
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r1", start_time=0.0),
        }
    )
    ev = evaluate(problem, solution)
    kinds = [v.kind for v in ev.violations]
    assert "capacity" in kinds


def test_evaluate_capacity_ok_when_non_overlapping() -> None:
    resources = (Resource(id="r1", capacity={"cpu": 2.0}),)
    jobs = (
        Job(id="j1", processing_time=1.0, demand={"cpu": 1.5}),
        Job(id="j2", processing_time=1.0, demand={"cpu": 1.5}),
    )
    problem = _problem(resources=resources, jobs=jobs)
    solution = Solution(
        assignments={
            "j1": Assignment(resource_id="r1", start_time=0.0),
            "j2": Assignment(resource_id="r1", start_time=1.0),
        }
    )
    ev = evaluate(problem, solution)
    assert ev.feasible


def test_evaluate_capacity_undeclared_dimension_violation() -> None:
    resources = (Resource(id="r1", capacity={"cpu": 4.0}),)
    jobs = (Job(id="j1", processing_time=1.0, demand={"gpu": 1.0}),)
    problem = _problem(resources=resources, jobs=jobs)
    solution = Solution(assignments={"j1": Assignment(resource_id="r1", start_time=0.0)})
    ev = evaluate(problem, solution)
    kinds = [v.kind for v in ev.violations]
    assert "capacity" in kinds


def test_compute_penalty_applies_weights() -> None:
    problem = _problem()
    solution = Solution()
    ev = evaluate(problem, solution)
    # Two missing assignments at default weight 1e6 each
    assert ev.penalty == 2.0e6
    custom = PenaltyWeights(missing_assignment=1.0)
    assert compute_penalty(ev.violations, custom) == 2.0


def test_evaluation_total_is_objective_plus_penalty() -> None:
    problem = _problem()
    solution = Solution(
        assignments={"j1": Assignment(resource_id="r1", start_time=0.0)}
    )
    ev = evaluate(problem, solution)
    assert ev.total == ev.objective + ev.penalty


def test_collect_violations_and_schedule_types() -> None:
    problem = _problem()
    solution = Solution()
    sched = compute_schedule(problem, solution)
    assert isinstance(sched, Schedule)
    violations = collect_violations(problem, solution, sched)
    assert isinstance(violations, tuple)


def test_schedule_mappings_are_immutable() -> None:
    sched = Schedule(resource_id={"j": "r"}, start={"j": 0.0}, completion={"j": 1.0})
    with pytest.raises(TypeError):
        sched.start["j"] = 5.0  # type: ignore[index]
