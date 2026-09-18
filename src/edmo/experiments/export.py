from __future__ import annotations

from pathlib import Path
from typing import Any

from edmo.experiments.dataset import GenerationRow, RunRow


def _load_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa  # type: ignore[import-not-found]
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is not installed. Install with "
            "'pip install -e \".[analysis]\"' to use parquet export."
        ) from exc
    return pa, pq


def export_runs_to_parquet(rows: list[RunRow] | tuple[RunRow, ...], path: Path) -> Path:
    """Write run-level rows to a Parquet file. Returns the resolved path."""
    pa, pq = _load_pyarrow()
    table = pa.Table.from_pylist(
        [
            {
                "run_id": r.run_id,
                "experiment_id": r.experiment_id,
                "algorithm": r.algorithm,
                "problem_id": r.problem_id,
                "problem_version": r.problem_version,
                "seed": r.seed,
                "best_total": r.best_total,
                "feasible": r.feasible,
                "iterations": r.iterations,
                "stopped_reason": r.stopped_reason,
                "wall_time_seconds": r.wall_time_seconds,
                "final_diversity": r.final_diversity,
                "initial_best_total": r.initial_best_total,
                "improvement_ratio": r.improvement_ratio,
            }
            for r in rows
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)
    return target


def export_generations_to_parquet(
    rows: list[GenerationRow] | tuple[GenerationRow, ...],
    path: Path,
) -> Path:
    """Write generation-level rows to a Parquet file. Returns the resolved path."""
    pa, pq = _load_pyarrow()
    table = pa.Table.from_pylist(
        [
            {
                "run_id": r.run_id,
                "experiment_id": r.experiment_id,
                "algorithm": r.algorithm,
                "generation": r.generation,
                "best_total": r.best_total,
                "mean_total": r.mean_total,
                "worst_total": r.worst_total,
                "population_diversity": r.population_diversity,
                "elapsed_seconds": r.elapsed_seconds,
            }
            for r in rows
        ]
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target)
    return target
