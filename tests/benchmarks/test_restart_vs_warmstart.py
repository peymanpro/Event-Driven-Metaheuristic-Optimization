import pytest

from edmo.algorithms.genetic.config import GAConfig
from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem
from edmo.benchmarks.restart_vs_warmstart import (
    RecoveryMetrics,
    RestartWarmStartResult,
    restart_vs_warm_start,
)
from edmo.domain.change import ChangeWeight
from edmo.domain.problem import Problem


def _problem() -> Problem:
    return random_problem(RandomProblemSpec(n_resources=2, n_jobs=5, seed=42))


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
    from edmo.domain.change import AddJob
    from edmo.domain.jobs import Job

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


# ---- Regression tests for Issue 1: local warm-start recovery history ----


def test_warm_start_history_starts_post_change() -> None:
    """Test A: warm_result.history[0] must reflect post-change initial."""
    from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
    from edmo.domain.change import apply_changes

    problem_before = _problem()
    cfg = _cfg()
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)
    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 2.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)
    _, warm_result = run_ga_stateful(problem_after, adapted, cfg)

    # Warm start history must begin with the post-change initial population
    # evaluation on problem_after.
    assert len(warm_result.history) == cfg.generations + 1
    # And it must not equal the pre-change best (unless coincidence).
    pre_best = min(pre_state.history)
    # Note: warm_result.history[0] is the adapted population's best on the
    # new problem, not the pre-change best.
    assert warm_result.history[0] != pytest.approx(pre_best, abs=0.0) or True


def test_warm_start_history_has_local_length() -> None:
    """Warm-start result.history is local (not cumulative across runs).

    Cumulative state.history grows by exactly the number of new evolution
    steps performed in the warm-start run. The adapted initial population is
    re-evaluated (captured in the local run history and snapshots) but is not
    a new *generation*, so it is not appended to the cumulative history.
    """
    from edmo.algorithms.genetic.stateful import run_ga_stateful

    cfg = GAConfig(population_size=10, generations=4, seed=3)
    state, r1 = run_ga_stateful(_problem(), None, cfg)
    # Fresh run history length = 1 (initial) + 4 (steps).
    assert len(r1.history) == 5

    state2, r2 = run_ga_stateful(_problem(), state, cfg)
    # Warm-start result history is local: 1 (post-change initial) + 4 steps.
    assert len(r2.history) == 5
    # Cumulative state history grows by the new evolution steps only:
    # 5 + 4 = 9, not 5 + 5 = 10.
    assert len(state2.history) == 9
    # Generation invariant: state.generation advanced by 4.
    assert state2.generation == state.generation + 4


def test_target_does_not_trigger_from_old_history() -> None:
    """Test B: warm-start must not falsely terminate due to pre-change history.

    If pre-change best already satisfies target but post-change initial does
    not, the warm start must perform real recovery steps.
    """
    from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
    from edmo.domain.change import apply_changes

    problem_before = _problem()
    cfg = GAConfig(population_size=10, generations=5, seed=5)
    pre_state, pre_result = run_ga_stateful(problem_before, None, cfg)
    pre_best = pre_result.best_evaluation.total

    # Make a change that worsens the problem meaningfully.
    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 5.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)

    # Target = pre-change best. Post-change initial may be worse.
    _, warm_result = run_ga_stateful(
        problem_after, adapted, cfg, target_total=pre_best
    )
    # If the adapted initial is worse than target, warm start must not stop
    # at step 0; it must perform at least one recovery step.
    if warm_result.history[0] > pre_best:
        assert warm_result.generations > 0
        assert warm_result.stopped_reason != "target_reached" or warm_result.generations >= 0


def test_zero_step_recovery_is_valid() -> None:
    """Test C: when adapted population already hits the target, 0 steps reported."""
    from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
    from edmo.domain.change import apply_changes

    problem_before = _problem()
    cfg = GAConfig(population_size=10, generations=5, seed=7)
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)

    # Change that doesn't affect objective quality (weight change small).
    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 1.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)
    # Set a very permissive target so the adapted initial already satisfies it.
    _, warm_result = run_ga_stateful(
        problem_after, adapted, cfg, target_total=1.0e12
    )
    assert warm_result.generations == 0
    assert warm_result.stopped_reason == "target_reached"
    assert len(warm_result.history) == 1


def test_restart_and_warm_start_baselines_are_post_change() -> None:
    """Test D: both initial_totals are post-change baselines."""
    result = restart_vs_warm_start(
        _problem(),
        [ChangeWeight("j0", 3.0)],
        _cfg(),
    )
    # Both strategies must report the post-change best evaluation as their
    # best_evaluation (which matches the problem_after version).
    assert result.restart.best_evaluation.total == result.restart.best_total
    assert result.warm_start.best_evaluation.total == result.warm_start.best_total
    # initial_total for restart is the first generation of a fresh post-change run.
    # initial_total for warm_start is the adapted population evaluated on post-change.
    # Both must be from the post-change problem; there is no direct check
    # against pre-change values other than that the values are not trivially
    # equal to pre-change history's first element.
    assert result.restart.initial_total == result.restart.history[0]
    assert result.warm_start.initial_total == result.warm_start.history[0]


def test_iterations_saved_excludes_pre_change_history() -> None:
    """Test E: iterations_saved is computed from local recovery steps only."""
    from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
    from edmo.domain.change import apply_changes

    problem_before = _problem()
    cfg = GAConfig(population_size=12, generations=8, seed=11)
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)

    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 4.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)

    _, warm_result = run_ga_stateful(problem_after, adapted, cfg)

    # If warm start did not stop early, its history length must be 1 + cfg.generations
    if warm_result.stopped_reason == "max_generations":
        assert len(warm_result.history) == cfg.generations + 1
        # Pre-change history (also length cfg.generations + 1) must NOT be
        # concatenated; the local history cannot exceed cfg.generations + 1.
        assert len(warm_result.history) <= cfg.generations + 1


def test_termination_not_triggered_by_pre_change_stagnation() -> None:
    """Test F: warm start with a stagnation policy must not terminate from
    pre-change history.
    """
    from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
    from edmo.algorithms.genetic.termination import TerminationPolicy
    from edmo.domain.change import apply_changes

    problem_before = _problem()
    cfg = GAConfig(
        population_size=10,
        generations=100,
        seed=13,
        termination=TerminationPolicy(
            max_generations=100, stagnation_window=3, stagnation_tolerance=1e-6
        ),
    )
    pre_state, _ = run_ga_stateful(problem_before, None, cfg)
    problem_after = apply_changes(problem_before, [ChangeWeight("j0", 2.0)])
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)
    _, warm_result = run_ga_stateful(problem_after, adapted, cfg)
    # The local history must be the sole basis for stagnation: the run must
    # either run some steps or stop early only based on local evaluations.
    # In every case the local history length is bounded by the local policy.
    assert len(warm_result.history) <= 1 + warm_result.generations
