from __future__ import annotations

from dataclasses import dataclass

from edmo.experiments.record import RunRecord


@dataclass(frozen=True, slots=True)
class RunRow:
    """One flattened analytical row per run."""

    run_id: str
    experiment_id: str
    algorithm: str
    problem_id: str
    problem_version: int
    seed: int | None
    best_total: float
    feasible: bool
    iterations: int
    stopped_reason: str
    wall_time_seconds: float
    final_diversity: float
    initial_best_total: float
    improvement_ratio: float


@dataclass(frozen=True, slots=True)
class GenerationRow:
    """One flattened analytical row per generation of a run."""

    run_id: str
    experiment_id: str
    algorithm: str
    generation: int
    best_total: float
    mean_total: float
    worst_total: float
    population_diversity: float
    elapsed_seconds: float


def to_run_row(run: RunRecord) -> RunRow:
    if run.metrics:
        initial_best = run.metrics[0].best_total
        final_div = run.metrics[-1].population_diversity
    else:
        initial_best = run.best_total
        final_div = 0.0
    denom = abs(initial_best) + 1.0e-12
    improvement = (initial_best - run.best_total) / denom
    return RunRow(
        run_id=run.run_id,
        experiment_id=run.experiment_id,
        algorithm=run.algorithm,
        problem_id=run.problem_id,
        problem_version=run.problem_version,
        seed=run.seed,
        best_total=run.best_total,
        feasible=run.feasible,
        iterations=run.iterations,
        stopped_reason=run.stopped_reason,
        wall_time_seconds=run.wall_time_seconds,
        final_diversity=final_div,
        initial_best_total=initial_best,
        improvement_ratio=improvement,
    )


def to_generation_rows(run: RunRecord) -> tuple[GenerationRow, ...]:
    return tuple(
        GenerationRow(
            run_id=run.run_id,
            experiment_id=run.experiment_id,
            algorithm=run.algorithm,
            generation=m.generation,
            best_total=m.best_total,
            mean_total=m.mean_total,
            worst_total=m.worst_total,
            population_diversity=m.population_diversity,
            elapsed_seconds=m.elapsed_seconds,
        )
        for m in run.metrics
    )


def build_run_dataset(runs: list[RunRecord] | tuple[RunRecord, ...]) -> tuple[RunRow, ...]:
    return tuple(to_run_row(r) for r in runs)


def build_generation_dataset(
    runs: list[RunRecord] | tuple[RunRecord, ...],
) -> tuple[GenerationRow, ...]:
    rows: list[GenerationRow] = []
    for r in runs:
        rows.extend(to_generation_rows(r))
    return tuple(rows)
