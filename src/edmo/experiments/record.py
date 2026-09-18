from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class GenerationMetric:
    """One row of per-generation telemetry.

    ``generation`` is 0-based. ``best_total`` is the best-so-far total score
    observed at the end of that generation. ``population_diversity`` is in
    ``[0, 1]``. ``elapsed_seconds`` is wall time spent in that generation.
    """

    generation: int
    best_total: float
    mean_total: float
    worst_total: float
    population_diversity: float
    elapsed_seconds: float

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("generation must be >= 0")
        if not 0.0 <= self.population_diversity <= 1.0:
            raise ValueError("population_diversity must be in [0, 1]")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "best_total": self.best_total,
            "mean_total": self.mean_total,
            "worst_total": self.worst_total,
            "population_diversity": self.population_diversity,
            "elapsed_seconds": self.elapsed_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationMetric:
        return cls(
            generation=int(data["generation"]),
            best_total=float(data["best_total"]),
            mean_total=float(data["mean_total"]),
            worst_total=float(data["worst_total"]),
            population_diversity=float(data["population_diversity"]),
            elapsed_seconds=float(data["elapsed_seconds"]),
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    """A single optimizer run on a single problem version.

    ``metadata`` is opaque to the pipeline and used for tags (scenario,
    git sha, hostname, ...). ``metrics`` is the per-generation history.
    """

    run_id: str
    experiment_id: str
    algorithm: str
    problem_id: str
    problem_version: int
    seed: int | None
    started_at: datetime
    finished_at: datetime
    best_total: float
    feasible: bool
    iterations: int
    stopped_reason: str
    metrics: tuple[GenerationMetric, ...] = ()
    metadata: MappingProxyType[str, str] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("RunRecord.run_id must be non-empty")
        if not self.experiment_id:
            raise ValueError("RunRecord.experiment_id must be non-empty")
        if not self.algorithm:
            raise ValueError("RunRecord.algorithm must be non-empty")
        if not self.problem_id:
            raise ValueError("RunRecord.problem_id must be non-empty")
        if self.problem_version < 1:
            raise ValueError("RunRecord.problem_version must be >= 1")
        if self.iterations < 0:
            raise ValueError("RunRecord.iterations must be >= 0")
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("RunRecord timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("RunRecord.finished_at must be >= started_at")
        object.__setattr__(self, "metrics", tuple(self.metrics))
        if not isinstance(self.metadata, MappingProxyType):
            object.__setattr__(
                self,
                "metadata",
                MappingProxyType(dict(self.metadata)),
            )

    @property
    def wall_time_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment_id": self.experiment_id,
            "algorithm": self.algorithm,
            "problem_id": self.problem_id,
            "problem_version": self.problem_version,
            "seed": self.seed,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "best_total": self.best_total,
            "feasible": self.feasible,
            "iterations": self.iterations,
            "stopped_reason": self.stopped_reason,
            "metrics": [m.to_dict() for m in self.metrics],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        required = {
            "run_id",
            "experiment_id",
            "algorithm",
            "problem_id",
            "problem_version",
            "started_at",
            "finished_at",
            "best_total",
            "feasible",
            "iterations",
            "stopped_reason",
        }
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"missing run fields: {sorted(missing)}")
        metrics = tuple(
            GenerationMetric.from_dict(m) for m in data.get("metrics", [])
        )
        return cls(
            run_id=str(data["run_id"]),
            experiment_id=str(data["experiment_id"]),
            algorithm=str(data["algorithm"]),
            problem_id=str(data["problem_id"]),
            problem_version=int(data["problem_version"]),
            seed=None if data.get("seed") is None else int(data["seed"]),
            started_at=datetime.fromisoformat(str(data["started_at"])),
            finished_at=datetime.fromisoformat(str(data["finished_at"])),
            best_total=float(data["best_total"]),
            feasible=bool(data["feasible"]),
            iterations=int(data["iterations"]),
            stopped_reason=str(data["stopped_reason"]),
            metrics=metrics,
            metadata=MappingProxyType(dict(data.get("metadata", {}))),
        )


def new_run_record(
    experiment_id: str,
    algorithm: str,
    problem_id: str,
    problem_version: int,
    seed: int | None,
    best_total: float,
    feasible: bool,
    iterations: int,
    stopped_reason: str,
    started_at: datetime,
    finished_at: datetime | None = None,
    metrics: tuple[GenerationMetric, ...] = (),
    metadata: dict[str, str] | None = None,
) -> RunRecord:
    """Construct a :class:`RunRecord` with a freshly generated ``run_id``."""
    from uuid import uuid4

    return RunRecord(
        run_id=str(uuid4()),
        experiment_id=experiment_id,
        algorithm=algorithm,
        problem_id=problem_id,
        problem_version=problem_version,
        seed=seed,
        started_at=started_at,
        finished_at=finished_at or datetime.now(tz=UTC),
        best_total=best_total,
        feasible=feasible,
        iterations=iterations,
        stopped_reason=stopped_reason,
        metrics=metrics,
        metadata=MappingProxyType(dict(metadata or {})),
    )
