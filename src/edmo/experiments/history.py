from __future__ import annotations

from collections.abc import Iterator

from edmo.experiments.record import RunRecord


class RunHistory:
    """In-memory store of completed runs.

    Keyed by ``run_id``. Lookups by experiment, algorithm, or problem are
    provided as iterators so callers can stream without materializing a list.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, RunRecord] = {}

    def add(self, run: RunRecord) -> None:
        if run.run_id in self._by_id:
            raise ValueError(f"run_id {run.run_id!r} already present")
        self._by_id[run.run_id] = run

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self) -> Iterator[RunRecord]:
        return iter(self._by_id.values())

    def get(self, run_id: str) -> RunRecord:
        try:
            return self._by_id[run_id]
        except KeyError as exc:
            raise KeyError(run_id) from exc

    def by_experiment(self, experiment_id: str) -> Iterator[RunRecord]:
        for r in self._by_id.values():
            if r.experiment_id == experiment_id:
                yield r

    def by_algorithm(self, algorithm: str) -> Iterator[RunRecord]:
        for r in self._by_id.values():
            if r.algorithm == algorithm:
                yield r

    def by_problem(self, problem_id: str) -> Iterator[RunRecord]:
        for r in self._by_id.values():
            if r.problem_id == problem_id:
                yield r

    def clear(self) -> None:
        self._by_id.clear()
