from edmo.benchmarks.canonical import (
    canonical_dynamic_changes,
    canonical_feasible_solution,
    canonical_resource_scheduling,
    is_canonical_instance_feasible,
)
from edmo.domain.change import apply_changes
from edmo.domain.evaluation import evaluate


def test_canonical_instance_shape() -> None:
    p = canonical_resource_scheduling()
    assert p.id == "canonical_resource_scheduling"
    assert len(p.resources) == 3
    assert len(p.jobs) == 5
    assert p.metadata["kind"] == "canonical"


def test_canonical_instance_has_feasible_solution() -> None:
    assert is_canonical_instance_feasible()


def test_canonical_feasible_solution_is_actually_feasible() -> None:
    problem = canonical_resource_scheduling()
    sol = canonical_feasible_solution()
    assert evaluate(problem, sol).feasible


def test_canonical_capacity_constraints_bite() -> None:
    from edmo.domain.solution import Assignment, Solution

    problem = canonical_resource_scheduling()
    dense = Solution(
        assignments={
            j.id: Assignment(resource_id="r0", start_time=0.0) for j in problem.jobs
        }
    )
    ev = evaluate(problem, dense)
    assert not ev.feasible
    assert any(v.kind == "capacity" for v in ev.violations)


def test_canonical_dynamic_changes_apply() -> None:
    problem = canonical_resource_scheduling()
    changes = canonical_dynamic_changes()
    assert len(changes) == 2
    updated = apply_changes(problem, list(changes))
    assert updated.version == problem.version + 2
    assert updated.job_by_id("j1").weight == 4.0
    assert updated.job_by_id("j0").deadline == 5.0


def test_canonical_restart_vs_warm_start_runs() -> None:
    from edmo.algorithms.genetic.config import GAConfig
    from edmo.benchmarks.restart_vs_warmstart import restart_vs_warm_start

    cfg = GAConfig(population_size=12, generations=10, seed=0)
    result = restart_vs_warm_start(
        canonical_resource_scheduling(),
        list(canonical_dynamic_changes()),
        cfg,
    )
    # Both strategies must produce real evaluations with a consistent total.
    for metrics in (result.restart, result.warm_start):
        ev = metrics.best_evaluation
        assert ev.total == ev.objective + ev.penalty
        assert metrics.best_total == ev.total


def test_canonical_repeated_recovery_runs() -> None:
    from edmo.algorithms.genetic.config import GAConfig
    from edmo.benchmarks.repeated_recovery import repeated_recovery_benchmark
    from edmo.domain.change import ProblemChange
    from edmo.domain.problem import Problem

    def change_factory(problem: Problem) -> ProblemChange:
        from edmo.domain.change import ChangeWeight

        return ChangeWeight(problem.jobs[0].id, 2.5)

    def config_factory(seed: int) -> GAConfig:
        return GAConfig(population_size=10, generations=8, seed=seed)

    result = repeated_recovery_benchmark(
        [0, 1, 2],
        spec_kwargs={"n_resources": 2, "n_jobs": 4},
        ga_config_factory=config_factory,
        change_factory=change_factory,
    )
    assert result.restart.runs == 3
    assert result.warm_start.runs == 3
