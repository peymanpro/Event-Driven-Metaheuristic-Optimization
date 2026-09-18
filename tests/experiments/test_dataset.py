from datetime import UTC, datetime

from edmo.experiments.dataset import (
    GenerationRow,
    RunRow,
    build_generation_dataset,
    build_run_dataset,
    to_generation_rows,
    to_run_row,
)
from edmo.experiments.record import GenerationMetric, RunRecord


def _run() -> RunRecord:
    now = datetime.now(tz=UTC)
    metrics = tuple(
        GenerationMetric(
            generation=g,
            best_total=10.0 - g,
            mean_total=11.0 - g,
            worst_total=12.0 - g,
            population_diversity=0.5,
            elapsed_seconds=0.01,
        )
        for g in range(4)
    )
    return RunRecord(
        run_id="r1",
        experiment_id="exp",
        algorithm="ga",
        problem_id="p1",
        problem_version=1,
        seed=1,
        started_at=now,
        finished_at=now,
        best_total=7.0,
        feasible=True,
        iterations=3,
        stopped_reason="max_generations",
        metrics=metrics,
    )


def test_run_row_fields() -> None:
    row = to_run_row(_run())
    assert isinstance(row, RunRow)
    assert row.run_id == "r1"
    assert row.initial_best_total == 10.0
    assert row.final_diversity == 0.5
    expected = (10.0 - 7.0) / 10.0
    assert abs(row.improvement_ratio - expected) < 1e-9


def test_generation_rows() -> None:
    rows = to_generation_rows(_run())
    assert len(rows) == 4
    assert all(isinstance(r, GenerationRow) for r in rows)
    assert rows[0].generation == 0
    assert rows[-1].generation == 3


def test_datasets() -> None:
    run = _run()
    run_rows = build_run_dataset([run])
    gen_rows = build_generation_dataset([run])
    assert len(run_rows) == 1
    assert len(gen_rows) == 4


def test_run_row_without_metrics() -> None:
    now = datetime.now(tz=UTC)
    run = RunRecord(
        run_id="r2",
        experiment_id="exp",
        algorithm="ga",
        problem_id="p1",
        problem_version=1,
        seed=1,
        started_at=now,
        finished_at=now,
        best_total=5.0,
        feasible=True,
        iterations=0,
        stopped_reason="x",
    )
    row = to_run_row(run)
    assert row.initial_best_total == 5.0
    assert row.final_diversity == 0.0
    assert row.improvement_ratio == 0.0
