from datetime import UTC, datetime

import pytest

from edmo.experiments.history import RunHistory
from edmo.experiments.record import RunRecord


def _run(run_id: str, exp: str = "exp", alg: str = "ga", problem: str = "p1") -> RunRecord:
    now = datetime.now(tz=UTC)
    return RunRecord(
        run_id=run_id,
        experiment_id=exp,
        algorithm=alg,
        problem_id=problem,
        problem_version=1,
        seed=1,
        started_at=now,
        finished_at=now,
        best_total=1.0,
        feasible=True,
        iterations=3,
        stopped_reason="max_generations",
    )


def test_add_and_get() -> None:
    h = RunHistory()
    h.add(_run("r1"))
    assert len(h) == 1
    assert h.get("r1").run_id == "r1"


def test_add_duplicate_rejected() -> None:
    h = RunHistory()
    h.add(_run("r1"))
    with pytest.raises(ValueError):
        h.add(_run("r1"))


def test_filters() -> None:
    h = RunHistory()
    h.add(_run("r1", exp="a", alg="ga", problem="p1"))
    h.add(_run("r2", exp="a", alg="de", problem="p2"))
    h.add(_run("r3", exp="b", alg="ga", problem="p2"))
    assert {r.run_id for r in h.by_experiment("a")} == {"r1", "r2"}
    assert {r.run_id for r in h.by_algorithm("ga")} == {"r1", "r3"}
    assert {r.run_id for r in h.by_problem("p2")} == {"r2", "r3"}


def test_get_unknown_raises() -> None:
    h = RunHistory()
    with pytest.raises(KeyError):
        h.get("ghost")


def test_clear() -> None:
    h = RunHistory()
    h.add(_run("r1"))
    h.clear()
    assert len(h) == 0
