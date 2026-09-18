from __future__ import annotations

from dataclasses import dataclass

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.metrics import GenerationSnapshot, population_diversity
from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
from edmo.domain.change import ProblemChange, apply_changes
from edmo.domain.evaluation import Evaluation
from edmo.domain.impact import ChangeImpact, analyze_change
from edmo.domain.problem import Problem


@dataclass(frozen=True, slots=True)
class RecoveryMetrics:
    """Per-strategy recovery summary.

    ``initial_total`` is the best total at the first evaluated generation.
    ``iterations_to_target`` is the first generation index where
    ``total <= target_total``, or ``None`` if never reached within budget.
    ``final_diversity`` is the population diversity at the end of the run.
    ``best_evaluation`` is the real :class:`Evaluation` for the best solution
    found by this strategy. ``snapshots`` carries the per-generation real
    telemetry emitted by the optimizer.
    """

    label: str
    initial_total: float
    best_total: float
    target_total: float
    reached_target: bool
    iterations_to_target: int | None
    final_diversity: float
    history: tuple[float, ...]
    best_evaluation: Evaluation
    snapshots: tuple[GenerationSnapshot, ...] = ()


@dataclass(frozen=True, slots=True)
class RestartWarmStartResult:
    problem_before: Problem
    problem_after: Problem
    impact: ChangeImpact
    pre_change_best_total: float
    target_total: float
    restart: RecoveryMetrics
    warm_start: RecoveryMetrics

    @property
    def iterations_saved(self) -> int | None:
        """Iterations the warm start saved relative to a restart, if both reached."""
        r = self.restart.iterations_to_target
        w = self.warm_start.iterations_to_target
        if r is None or w is None:
            return None
        return r - w

    @property
    def warm_start_initial_advantage(self) -> float:
        """How much better the warm-start initial best is versus restart."""
        return self.restart.initial_total - self.warm_start.initial_total


def _metrics(
    label: str,
    history: tuple[float, ...],
    target_total: float,
    final_population: tuple[Chromosome, ...],
    best_evaluation: Evaluation,
    snapshots: tuple[GenerationSnapshot, ...],
) -> RecoveryMetrics:
    initial_total = history[0] if history else float("inf")
    best_total = min(history) if history else float("inf")
    iterations_to_target: int | None = None
    for i, value in enumerate(history):
        if value <= target_total:
            iterations_to_target = i
            break
    diversity = population_diversity(list(final_population))
    return RecoveryMetrics(
        label=label,
        initial_total=initial_total,
        best_total=best_total,
        target_total=target_total,
        reached_target=iterations_to_target is not None,
        iterations_to_target=iterations_to_target,
        final_diversity=diversity,
        history=history,
        best_evaluation=best_evaluation,
        snapshots=snapshots,
    )


def restart_vs_warm_start(
    problem_before: Problem,
    changes: tuple[ProblemChange, ...] | list[ProblemChange],
    ga_config: GAConfig | None = None,
    target_total: float | None = None,
    pre_change_generations: int | None = None,
) -> RestartWarmStartResult:
    """Compare restarting from scratch versus warm-starting on a new problem.

    Procedure:

    1. Run GA on ``problem_before`` to obtain a pre-change state and best total.
    2. Apply ``changes`` to obtain ``problem_after``.
    3. Restart: run GA from scratch on ``problem_after`` with the same seed.
    4. Warm start: adapt the pre-change state and continue on ``problem_after``.

    ``target_total`` defaults to the pre-change best total. Both strategies
    then report the first iteration where they reach or beat that target.
    """
    cfg = ga_config or GAConfig()
    pre_cfg = (
        cfg
        if pre_change_generations is None
        else GAConfig(
            population_size=cfg.population_size,
            generations=pre_change_generations,
            crossover_rate=cfg.crossover_rate,
            mutation_rate=cfg.mutation_rate,
            tournament_size=cfg.tournament_size,
            elite_count=cfg.elite_count,
            start_time_horizon=cfg.start_time_horizon,
            mutation_time_sigma=cfg.mutation_time_sigma,
            seed=cfg.seed,
            apply_repair=cfg.apply_repair,
            termination=cfg.termination,
        )
    )

    pre_state, pre_result = run_ga_stateful(problem_before, None, pre_cfg)
    pre_best = pre_result.best_evaluation.total

    problem_after = apply_changes(problem_before, list(changes))
    impact = analyze_change(problem_before, problem_after)

    target = pre_best if target_total is None else target_total

    # Restart from scratch.
    restart_state, restart_result = run_ga_stateful(problem_after, None, cfg)
    restart_metrics = _metrics(
        "restart",
        restart_result.history,
        target,
        restart_state.population,
        restart_result.best_evaluation,
        restart_state.snapshots,
    )

    # Warm start from adapted state.
    adapted = adapt_state(problem_before, problem_after, pre_state, cfg)
    warm_state, warm_result = run_ga_stateful(problem_after, adapted, cfg)
    warm_metrics = _metrics(
        "warm_start",
        warm_result.history,
        target,
        warm_state.population,
        warm_result.best_evaluation,
        warm_state.snapshots,
    )

    return RestartWarmStartResult(
        problem_before=problem_before,
        problem_after=problem_after,
        impact=impact,
        pre_change_best_total=pre_best,
        target_total=target,
        restart=restart_metrics,
        warm_start=warm_metrics,
    )
