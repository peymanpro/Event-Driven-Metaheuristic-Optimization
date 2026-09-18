import pytest

from edmo.algorithms.genetic.config import GAConfig
from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.benchmarks.restart_vs_warmstart import (
    RecoveryMetrics,
    RestartWarmStartResult,
    restart_vs_warm_start,
)
from edmo.domain.change import AddJob, ChangeWeight
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem


def _problem() -> Problem:
    return random_problem(RandomProblemSpec(n_resources=2, n_jobs=4, seed=42))


def _cfg() -> GAConfig:
    return GAConfig(population_size=15, generations=15, seed=1)


def test_result_structure() -> None:
    result = restart_vs_warm_start(
        _problem(),
        [ChangeWeight("j0", 2.0)],
        _cfg(),
    )
    assert isinstance(result, RestartWarmStartResult)
    assert isinstance(result.restart, RecoveryMetrics)
    assert isinstance(result.warm_start, RecoveryMetrics)
    assert result.restart.label == "restart"
    assert result.warm_start.label == "warm_start"
    assert result.target_total == result.pre_change_best_total


def test_result_reproducible() -> None:
    p = _problem()
    r1 = restart_vs_warm_start(p, [ChangeWeight("j0", 2.0)], _cfg())
    r2 = restart_vs_warm_start(p, [ChangeWeight("j0", 2.0)], _cfg())
    assert r1.restart.history == r2.restart.history
    assert r1.warm_start.history == r2.warm_start.history


def test_structural_change_runs() -> None:
    result = restart_vs_warm_start(
        _problem(),
        [AddJob(Job(id="extra", processing_time=1.5))],
        _cfg(),
    )
    assert result.impact.is_structural
    assert result.impact.jobs_added == ("extra",)


def test_target_total_override() -> None:
    result = restart_vs_warm_start(
        _problem(),
        [ChangeWeight("j0", 1.5)],
        _cfg(),
        target_total=1.0e12,
    )
    assert result.target_total == 1.0e12
    assert result.restart.reached_target
    assert result.restart.iterations_to_target == 0


def test_iterations_saved_consistent() -> None:
    result = restart_vs_warm_start(
        _problem(),
        [ChangeWeight("j0", 2.5)],
        _cfg(),
    )
    saved = result.iterations_saved
    if result.restart.reached_target and result.warm_start.reached_target:
        r = result.restart.iterations_to_target
        w = result.warm_start.iterations_to_target
        assert r is not None and w is not None
        assert saved == (r - w)


def test_warm_start_initial_advantage_sign() -> None:
    result = restart_vs_warm_start(
        _problem(),
        [ChangeWeight("j0", 2.0)],
        _cfg(),
    )
    expected = result.restart.initial_total - result.warm_start.initial_total
    assert result.warm_start_initial_advantage == pytest.approx(expected)


def test_metrics_have_consistent_fields() -> None:
    result = restart_vs_warm_start(
        _problem(),
        [ChangeWeight("j0", 2.0)],
        _cfg(),
    )
    for m in (result.restart, result.warm_start):
        assert m.best_total == min(m.history)
        assert m.initial_total == m.history[0]
        assert 0.0 <= m.final_diversity <= 1.0
