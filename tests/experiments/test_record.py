from datetime import UTC, datetime, timedelta

import pytest

from edmo.experiments.record import (
    GenerationMetric,
    RunRecord,
    new_run_record,
)


def _metric(g: int) -> GenerationMetric:
    return GenerationMetric(
        generation=g,
        best_total=float(g),
        mean_total=float(g) + 1.0,
        worst_total=float(g) + 2.0,
        population_diversity=0.5,
        elapsed_seconds=0.01,
    )


def _run() -> RunRecord:
    now = datetime.now(tz=UTC)
    return new_run_record(
        experiment_id="exp",
        algorithm="ga",
        problem_id="p1",
        problem_version=1,
        seed=1,
        best_total=1.0,
        feasible=True,
        iterations=3,
        stopped_reason="max_generations",
        started_at=now,
        finished_at=now + timedelta(seconds=2),
        metrics=(_metric(0), _metric(1), _metric(2)),
        metadata={"scenario": "small"},
    )


def test_metric_valid() -> None:
    m = _metric(0)
    assert m.generation == 0


def test_metric_rejects_negative_generation() -> None:
    with pytest.raises(ValueError):
        GenerationMetric(
            generation=-1,
            best_total=0.0,
            mean_total=0.0,
            worst_total=0.0,
            population_diversity=0.0,
            elapsed_seconds=0.0,
        )


def test_metric_rejects_bad_diversity() -> None:
    with pytest.raises(ValueError):
        GenerationMetric(0, 0.0, 0.0, 0.0, 1.5, 0.0)


def test_run_valid() -> None:
    r = _run()
    assert r.algorithm == "ga"
    assert len(r.metrics) == 3
    assert r.wall_time_seconds == 2.0


def test_run_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError):
        RunRecord(
            run_id="r",
            experiment_id="exp",
            algorithm="ga",
            problem_id="p1",
            problem_version=1,
            seed=None,
            started_at=datetime(2026, 1, 1),
            finished_at=datetime(2026, 1, 1),
            best_total=0.0,
            feasible=True,
            iterations=0,
            stopped_reason="x",
        )


def test_run_rejects_finished_before_started() -> None:
    now = datetime.now(tz=UTC)
    with pytest.raises(ValueError):
        RunRecord(
            run_id="r",
            experiment_id="exp",
            algorithm="ga",
            problem_id="p1",
            problem_version=1,
            seed=None,
            started_at=now,
            finished_at=now - timedelta(seconds=1),
            best_total=0.0,
            feasible=True,
            iterations=0,
            stopped_reason="x",
        )


def test_run_roundtrip() -> None:
    r = _run()
    d = r.to_dict()
    r2 = RunRecord.from_dict(d)
    assert r2.run_id == r.run_id
    assert r2.algorithm == r.algorithm
    assert r2.best_total == r.best_total
    assert r2.iterations == r.iterations
    assert len(r2.metrics) == len(r.metrics)
    assert dict(r2.metadata) == dict(r.metadata)


def test_run_from_dict_missing_fields() -> None:
    with pytest.raises(ValueError):
        RunRecord.from_dict({"run_id": "x"})


def test_metadata_is_immutable() -> None:
    r = _run()
    with pytest.raises(TypeError):
        r.metadata["new"] = "v"  # type: ignore[index]
