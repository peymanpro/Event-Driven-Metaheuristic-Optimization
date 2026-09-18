from pathlib import Path

import pytest

from edmo.experiments.dataset import GenerationRow, RunRow
from edmo.experiments.export import (
    export_generations_to_parquet,
    export_runs_to_parquet,
)


def _run_row() -> RunRow:
    return RunRow(
        run_id="r1",
        experiment_id="exp",
        algorithm="ga",
        problem_id="p1",
        problem_version=1,
        seed=1,
        best_total=1.0,
        feasible=True,
        iterations=3,
        stopped_reason="max_generations",
        wall_time_seconds=0.5,
        final_diversity=0.5,
        initial_best_total=2.0,
        improvement_ratio=0.5,
    )


def _gen_row(g: int) -> GenerationRow:
    return GenerationRow(
        run_id="r1",
        experiment_id="exp",
        algorithm="ga",
        generation=g,
        best_total=2.0 - g,
        mean_total=3.0 - g,
        worst_total=4.0 - g,
        population_diversity=0.5,
        elapsed_seconds=0.01,
    )


def test_export_runs_requires_pyarrow(tmp_path: Path) -> None:
    # pyarrow is optional and not installed in the dev env.
    with pytest.raises(RuntimeError):
        export_runs_to_parquet([_run_row()], tmp_path / "runs.parquet")


def test_export_generations_requires_pyarrow(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError):
        export_generations_to_parquet([_gen_row(0)], tmp_path / "gens.parquet")
