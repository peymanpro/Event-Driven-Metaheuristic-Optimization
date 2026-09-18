import math

import pytest

from edmo.algorithms.genetic.config import GAConfig
from edmo.benchmarks.repeated_recovery import (
    RepeatedRecoveryResult,
    StrategyAggregate,
    repeated_recovery_benchmark,
)
from edmo.domain.change import ChangeWeight, ProblemChange
from edmo.domain.problem import Problem


def _small_config_factory(seed: int) -> GAConfig:
    return GAConfig(population_size=10, generations=10, seed=seed)


def _change_factory(problem: Problem) -> ProblemChange:
    return ChangeWeight(problem.jobs[0].id, 2.0)


def _run(seeds: list[int]) -> RepeatedRecoveryResult:
    return repeated_recovery_benchmark(
        seeds,
        spec_kwargs={"n_resources": 2, "n_jobs": 5},
        ga_config_factory=_small_config_factory,
        change_factory=_change_factory,
    )


def test_repeated_recovery_empty_seeds_rejected() -> None:
    with pytest.raises(ValueError):
        repeated_recovery_benchmark([])


def test_repeated_recovery_returns_result_shape() -> None:
    result = _run([0, 1, 2])
    assert isinstance(result, RepeatedRecoveryResult)
    assert isinstance(result.restart, StrategyAggregate)
    assert isinstance(result.warm_start, StrategyAggregate)
    assert result.seeds == (0, 1, 2)
    assert result.restart.runs == 3
    assert result.warm_start.runs == 3


def test_repeated_recovery_is_reproducible() -> None:
    r1 = _run([0, 1, 2, 3])
    r2 = _run([0, 1, 2, 3])
    assert r1.restart.success_count == r2.restart.success_count
    assert r1.warm_start.success_count == r2.warm_start.success_count
    assert r1.paired_iterations_saved == r2.paired_iterations_saved
    assert r1.restart.mean_best_total == r2.restart.mean_best_total
    assert r1.warm_start.mean_best_total == r2.warm_start.mean_best_total


def test_repeated_recovery_aggregates_consistently() -> None:
    result = _run([0, 1, 2, 3, 4])
    for agg in (result.restart, result.warm_start):
        assert agg.runs == 5
        assert agg.success_count <= agg.runs
        assert 0.0 <= agg.success_rate <= 1.0
        assert 0.0 <= agg.feasible_fraction <= 1.0
        if agg.success_count > 0:
            assert agg.median_iterations_to_target is not None
            assert agg.mean_iterations_to_target is not None
            assert agg.best_iterations_to_target is not None
            assert agg.worst_iterations_to_target is not None
            assert agg.best_iterations_to_target <= agg.worst_iterations_to_target
        # If success_count < 2, stdev must be None.
        if agg.success_count < 2:
            assert agg.stdev_iterations_to_target is None


def test_repeated_recovery_no_nan_or_inf_in_summaries() -> None:
    result = _run([0, 1, 2, 3])
    for agg in (result.restart, result.warm_start):
        for value in (
            agg.mean_initial_total,
            agg.mean_best_total,
            agg.mean_final_diversity,
            agg.success_rate,
            agg.feasible_fraction,
        ):
            assert math.isfinite(value)
        if agg.mean_iterations_to_target is not None:
            assert math.isfinite(agg.mean_iterations_to_target)
        if agg.median_iterations_to_target is not None:
            assert math.isfinite(agg.median_iterations_to_target)
        if agg.stdev_iterations_to_target is not None:
            assert math.isfinite(agg.stdev_iterations_to_target)


def test_repeated_recovery_paired_saved_signs() -> None:
    result = _run([0, 1, 2, 3])
    if result.paired_mean_iterations_saved is not None:
        assert math.isfinite(result.paired_mean_iterations_saved)
    # paired_iterations_saved should equal restart - warm_start for each pair.
    # We cannot recompute directly, but the tuple length must be <= seeds.
    assert len(result.paired_iterations_saved) <= len(result.seeds)


def test_repeated_recovery_single_seed_has_no_stdev() -> None:
    result = _run([7])
    for agg in (result.restart, result.warm_start):
        if agg.success_count < 2:
            assert agg.stdev_iterations_to_target is None
