from random import Random

import pytest

from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.operators import init_population
from edmo.algorithms.genetic.repair import repair
from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
from edmo.algorithms.genetic.termination import TerminationPolicy
from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.benchmarks.restart_vs_warmstart import (
    RecoveryMetrics,
    RestartWarmStartResult,
    restart_vs_warm_start,
)
from edmo.domain.change import AddJob, ChangeWeight, ProblemChange, apply_changes
from edmo.domain.evaluation import evaluate
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem


def _problem() -> Problem:
    return random_problem(RandomProblemSpec(n_resources=2, n_jobs=5, seed=42))


def _cfg() -> GAConfig:
    return GAConfig(population_size=15, generations=15, seed=1)


def _direct_totals(problem: Problem, chromosomes) -> list[float]:  # type: ignore[no-untyped-def]
    return [
        evaluate(problem, c.to_solution(problem)).total for c in chromosomes
    ]


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
    assert len(r1.restart.snapshots) == len(r2.restart.snapshots)
    assert len(r1.warm_start.snapshots) == len(r2.warm_start.snapshots)


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
    assert result.warm_start.reached_target
    assert result.warm_start.iterations_to_target == 0


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
        # Issue 2 invariant: history and snapshots are the same invocation,
        # so lengths must match for both strategies.
        assert len(m.history) == len(m.snapshots)


# ---- Regression tests for local recovery history (Issue 1) ----


def test_warm_start_history_starts_post_change() -> None:
    """Test A: warm_result.history[0] equals the direct post-change initial.

    The adapted population is evaluated on ``problem_after`` and its minimum
    is the local run's generation-0 entry. This must match the direct
    evaluation, and it must be strictly greater than the pre-change best
    when the change is a large weight increase (guaranteed by construction).
    """
    problem_before = _problem()
    cfg = _cfg()
    pre_state, pre_result = run_ga_stateful(problem_before, None, cfg)
    pre_best = pre_result.best_evaluation.total

    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 50.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)

    # Direct expectation for the local generation-0 entry.
    if cfg.apply_repair:
        prepared = [repair(problem_after, c) for c in adapted.population]
    else:
        prepared = list(adapted.population)
    expected_initial = min(_direct_totals(problem_after, prepared))

    _, warm_result = run_ga_stateful(problem_after, adapted, cfg)

    # Direct equality between the reported post-change initial and the
    # independently computed post-change initial.
    assert warm_result.history[0] == pytest.approx(expected_initial)
    # Length is local: 1 (initial) + cfg.generations steps.
    assert len(warm_result.history) == cfg.generations + 1
    assert len(warm_result.snapshots) == len(warm_result.history)
    # A weight increase on j0 makes the post-change evaluation strictly
    # larger than the pre-change evaluation of the same chromosome.
    assert warm_result.history[0] > pre_best


def test_warm_start_history_has_local_length() -> None:
    """Warm-start result.history is local (not cumulative across runs)."""
    cfg = GAConfig(population_size=10, generations=4, seed=3)
    state, r1 = run_ga_stateful(_problem(), None, cfg)
    assert len(r1.history) == 5
    assert len(r1.snapshots) == 5

    state2, r2 = run_ga_stateful(_problem(), state, cfg)
    # Warm-start result history/snapshots are local: 1 + 4.
    assert len(r2.history) == 5
    assert len(r2.snapshots) == 5
    # Cumulative state grows by new steps only.
    assert len(state2.history) == 9
    assert len(state2.snapshots) == 9
    assert state2.generation == state.generation + 4


def test_target_does_not_trigger_from_old_history() -> None:
    """Test B: the initial target check uses only the local post-change history.

    A weight increase on j0 strictly raises every chromosome's total on
    ``problem_after`` compared to ``problem_before``, so the adapted initial
    is strictly greater than the pre-change best. With ``target = pre_best``
    the run must therefore take at least one evolution step, proving that
    the initial target check did not use the pre-change history.
    """
    problem_before = _problem()
    cfg = GAConfig(population_size=10, generations=5, seed=5)
    pre_state, pre_result = run_ga_stateful(problem_before, None, cfg)
    pre_best = pre_result.best_evaluation.total

    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 50.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)

    _, warm_result = run_ga_stateful(
        problem_after, adapted, cfg, target_total=pre_best
    )

    # Post-change initial must be worse than the pre-change target; otherwise
    # the scenario is not meaningful for this regression.
    assert warm_result.history[0] > pre_best
    # Therefore the warm start had to perform at least one evolution step.
    assert warm_result.generations >= 1
    assert len(warm_result.history) >= 2
    # If the run eventually reached the target, that must be after a genuine
    # post-change step, not at index 0.
    if warm_result.stopped_reason == "target_reached":
        assert len(warm_result.history) - 1 >= 1


def test_zero_step_recovery_is_valid() -> None:
    """Test C: an adapted population that already beats the target reports 0 steps."""
    problem_before = _problem()
    cfg = GAConfig(population_size=10, generations=5, seed=7)
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)

    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 1.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)
    _, warm_result = run_ga_stateful(
        problem_after, adapted, cfg, target_total=1.0e12
    )
    assert warm_result.generations == 0
    assert warm_result.stopped_reason == "target_reached"
    assert len(warm_result.history) == 1
    assert len(warm_result.snapshots) == 1


def test_restart_and_warm_start_baselines_are_post_change() -> None:
    """Test D: initial totals are independent post-change evaluations.

    We recompute the restart initial directly from a fresh initial population
    on ``problem_after`` and compare to the reported value. We also verify
    that evaluating the same chromosomes on ``problem_before`` gives a
    different total, confirming the change is real and the benchmark uses
    post-change evaluations.
    """
    problem_before = _problem()
    cfg = _cfg()
    changes: list[ProblemChange] = [ChangeWeight("j0", 50.0)]
    result = restart_vs_warm_start(problem_before, changes, cfg)
    problem_after = apply_changes(problem_before, list(changes))

    # Independently reproduce the restart initial population.
    rng = Random(cfg.seed)
    population_after = [
        repair(problem_after, c) for c in init_population(problem_after, cfg, rng)
    ]
    expected_restart_initial = min(_direct_totals(problem_after, population_after))
    assert result.restart.initial_total == pytest.approx(expected_restart_initial)

    # The same initial population evaluated on problem_before has different
    # totals because the weight of j0 changed.
    rng_b = Random(cfg.seed)
    population_before = [
        repair(problem_before, c) for c in init_population(problem_before, cfg, rng_b)
    ]
    totals_before = _direct_totals(problem_before, population_before)
    totals_after = _direct_totals(problem_after, population_before)
    assert totals_before != totals_after

    # Both strategies must carry real Evaluations from problem_after.
    assert result.restart.best_evaluation.total == result.restart.best_total
    assert result.warm_start.best_evaluation.total == result.warm_start.best_total
    assert result.restart.initial_total == result.restart.history[0]
    assert result.warm_start.initial_total == result.warm_start.history[0]


def test_iterations_saved_excludes_pre_change_history() -> None:
    """Test E: iterations_saved is computed from local recovery steps only."""
    problem_before = _problem()
    cfg = GAConfig(population_size=12, generations=8, seed=11)
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)

    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 4.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)

    _, warm_result = run_ga_stateful(problem_after, adapted, cfg)

    # Local history length must never exceed cfg.generations + 1.
    assert len(warm_result.history) <= cfg.generations + 1
    assert len(warm_result.snapshots) == len(warm_result.history)

    # Full comparison via restart_vs_warm_start.
    changes_for_benchmark: list[ProblemChange] = [ChangeWeight("j0", 4.0)]
    result = restart_vs_warm_start(problem_before, changes_for_benchmark, cfg)
    assert len(result.warm_start.history) <= cfg.generations + 1
    if (
        result.restart.reached_target
        and result.warm_start.reached_target
        and result.iterations_saved is not None
    ):
        assert result.iterations_saved == (
            (result.restart.iterations_to_target or 0)
            - (result.warm_start.iterations_to_target or 0)
        )
        # The saved count cannot exceed cfg.generations (local bound).
        assert result.iterations_saved <= cfg.generations


def test_termination_not_triggered_by_pre_change_stagnation() -> None:
    """Test F: pre-change stagnation must not terminate a fresh warm-start run.

    The stagnation policy fires only when the local history has at least
    ``stagnation_window + 1`` entries. At generation 0, the warm-start local
    history has exactly one entry, so stagnation cannot trigger. If the
    cumulative pre-change history were incorrectly used instead, the very
    first check would see a stagnated window and terminate at 0 steps.
    """
    cfg = GAConfig(
        population_size=10,
        generations=100,
        seed=13,
        termination=TerminationPolicy(
            max_generations=100, stagnation_window=3, stagnation_tolerance=1e-6
        ),
    )
    problem_before = _problem()
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)
    # Pre-change history is long enough to satisfy the stagnation window if
    # it were reused verbatim.
    termination = cfg.termination
    assert termination is not None
    window = termination.stagnation_window
    assert window is not None
    assert len(pre_state.history) >= window + 1

    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 5.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)
    _, warm_result = run_ga_stateful(problem_after, adapted, cfg)

    # If the run stopped at generation 0, stagnation was not the cause.
    if warm_result.generations == 0:
        assert warm_result.stopped_reason != "stagnation"
    # If stagnation fired, it must have observed enough local entries.
    if warm_result.stopped_reason == "stagnation":
        assert len(warm_result.history) >= window + 1


def test_warm_start_metrics_use_local_snapshots() -> None:
    """Test 7: RecoveryMetrics.snapshots comes from the local invocation.

    The cumulative ``GAState.snapshots`` grows across problem versions,
    whereas ``GAResult.snapshots`` (and therefore the metrics) reflects only
    the current post-change invocation.
    """
    problem_before = _problem()
    cfg = GAConfig(population_size=10, generations=4, seed=17)
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)
    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 2.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)
    warm_state, warm_result = run_ga_stateful(problem_after, adapted, cfg)

    # Cumulative state has more snapshots than the local result.
    assert len(warm_state.snapshots) > len(warm_result.snapshots)
    # Local result invariant.
    assert len(warm_result.snapshots) == len(warm_result.history)
    assert len(warm_result.snapshots) == cfg.generations + 1

    # Same check through the benchmark API.
    changes_for_benchmark2: list[ProblemChange] = [ChangeWeight("j0", 2.0)]
    result = restart_vs_warm_start(problem_before, changes_for_benchmark2, cfg)
    assert len(result.restart.snapshots) == len(result.restart.history)
    assert len(result.warm_start.snapshots) == len(result.warm_start.history)
    assert len(result.warm_start.snapshots) == cfg.generations + 1
    # Warm-start metrics snapshots must not include pre-change snapshots.
    # Snapshot generations use the cumulative global counter, so we assert
    # that the local sequence starts at the generation reached before the
    # warm start and increases one-by-one, without gaps or regressions.
    start_generation = pre_state.generation
    generations_seq = [snap.generation for snap in warm_result.snapshots]
    assert generations_seq[0] == start_generation
    assert generations_seq[-1] == start_generation + cfg.generations
    assert generations_seq == list(
        range(start_generation, start_generation + cfg.generations + 1)
    )
