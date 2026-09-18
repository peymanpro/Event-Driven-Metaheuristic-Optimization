import pytest

from edmo.benchmarks.problem_generator import (
    RandomProblemSpec,
    known_feasible_solution,
    random_problem,
)
from edmo.domain.evaluation import evaluate


def test_spec_defaults() -> None:
    s = RandomProblemSpec()
    assert s.n_resources == 3
    assert s.n_jobs == 8
    assert s.seed == 0
    assert s.min_capacity_factor >= 1.0


def test_spec_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        RandomProblemSpec(n_resources=0)
    with pytest.raises(ValueError):
        RandomProblemSpec(n_jobs=0)
    with pytest.raises(ValueError):
        RandomProblemSpec(max_processing_time=0.0)
    with pytest.raises(ValueError):
        RandomProblemSpec(max_release_time=-1.0)
    with pytest.raises(ValueError):
        RandomProblemSpec(min_demand=0.0)
    with pytest.raises(ValueError):
        RandomProblemSpec(min_demand=2.0, max_demand=1.0)
    with pytest.raises(ValueError):
        RandomProblemSpec(min_capacity_factor=0.5)
    with pytest.raises(ValueError):
        RandomProblemSpec(min_speed=0.0)
    with pytest.raises(ValueError):
        RandomProblemSpec(max_deadline_slack=-1.0)


def test_random_problem_shape() -> None:
    p = random_problem(RandomProblemSpec(n_resources=2, n_jobs=5, seed=0))
    assert len(p.resources) == 2
    assert len(p.jobs) == 5
    assert p.metadata["seed"] == "0"


def test_random_problem_is_deterministic() -> None:
    p1 = random_problem(RandomProblemSpec(seed=7))
    p2 = random_problem(RandomProblemSpec(seed=7))
    for r1, r2 in zip(p1.resources, p2.resources, strict=True):
        assert r1.id == r2.id
        assert r1.speed == r2.speed
        assert dict(r1.capacity) == dict(r2.capacity)
    for j1, j2 in zip(p1.jobs, p2.jobs, strict=True):
        assert j1.id == j2.id
        assert j1.processing_time == j2.processing_time
        assert j1.release_time == j2.release_time
        assert j1.deadline == j2.deadline
        assert j1.weight == j2.weight
        assert dict(j1.demand) == dict(j2.demand)


def test_random_problem_different_seeds_differ() -> None:
    p1 = random_problem(RandomProblemSpec(seed=1))
    p2 = random_problem(RandomProblemSpec(seed=2))
    assert [j.processing_time for j in p1.jobs] != [j.processing_time for j in p2.jobs]


def test_resources_declare_cpu_capacity() -> None:
    p = random_problem(RandomProblemSpec(seed=3))
    for r in p.resources:
        assert "cpu" in r.capacity
        assert r.capacity["cpu"] > 0.0


def test_every_single_job_fits_on_every_resource() -> None:
    p = random_problem(RandomProblemSpec(n_resources=3, n_jobs=10, seed=4))
    for r in p.resources:
        for j in p.jobs:
            assert r.capacity["cpu"] >= j.demand["cpu"]


def test_known_feasible_solution_exists_for_default_spec() -> None:
    p = random_problem(RandomProblemSpec(seed=11))
    sol = known_feasible_solution(p)
    assert sol is not None
    ev = evaluate(p, sol)
    assert ev.feasible


def test_known_feasible_solution_exists_across_many_seeds() -> None:
    for seed in range(20):
        p = random_problem(RandomProblemSpec(seed=seed))
        sol = known_feasible_solution(p)
        assert sol is not None, f"seed={seed} produced a problem without feasible solution"
        assert evaluate(p, sol).feasible


def test_capacity_constraints_are_meaningful() -> None:
    """Overloading a resource with concurrent high-demand jobs must be infeasible."""
    from edmo.domain.solution import Assignment, Solution

    p = random_problem(RandomProblemSpec(n_resources=1, n_jobs=6, seed=2))
    # Force all jobs onto the single resource at time 0 with overlapping demand.
    assignments = {
        j.id: Assignment(resource_id=p.resources[0].id, start_time=0.0)
        for j in p.jobs
    }
    dense = Solution(assignments=assignments)
    ev = evaluate(p, dense)
    # With min_capacity_factor <= max_capacity_factor and multiple jobs,
    # the total concurrent demand should exceed the resource capacity.
    total_demand = sum(j.demand["cpu"] for j in p.jobs)
    assert total_demand > p.resources[0].capacity["cpu"]
    assert not ev.feasible
    kinds = {v.kind for v in ev.violations}
    assert "capacity" in kinds
