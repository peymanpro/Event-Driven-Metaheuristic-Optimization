from __future__ import annotations

from dataclasses import dataclass

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.domain.problem import Problem


@dataclass(frozen=True, slots=True)
class VectorBounds:
    """Per-dimension bounds for the real-valued DE vector."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper):
            raise ValueError("lower and upper must have equal length")
        for lo, hi in zip(self.lower, self.upper, strict=True):
            if lo > hi:
                raise ValueError("lower bound must be <= upper bound")


def vector_bounds(problem: Problem, start_time_horizon: float) -> VectorBounds:
    """Construct bounds for the DE vector.

    Length is ``2 * n_jobs``: first ``n_jobs`` dims are start times in
    ``[0, start_time_horizon]``; last ``n_jobs`` dims are resource selectors
    in ``[0, n_resources - 1]``. When there is only one resource, the upper
    bound is a tiny epsilon so that ``int(floor(x))`` maps to 0.
    """
    if start_time_horizon <= 0.0:
        raise ValueError("start_time_horizon must be positive")
    n_jobs = len(problem.jobs)
    n_resources = len(problem.resources)
    if n_resources < 1:
        raise ValueError("problem must have at least one resource")

    start_lower = (0.0,) * n_jobs
    start_upper = (start_time_horizon,) * n_jobs

    if n_resources == 1:
        res_lower = (0.0,) * n_jobs
        res_upper = (0.0,) * n_jobs
    else:
        res_lower = (0.0,) * n_jobs
        res_upper = (float(n_resources - 1),) * n_jobs

    return VectorBounds(lower=start_lower + res_lower, upper=start_upper + res_upper)


def decode_vector(problem: Problem, vector: tuple[float, ...]) -> Chromosome:
    """Decode a real-valued DE vector into a :class:`Chromosome`."""
    n_jobs = len(problem.jobs)
    if len(vector) != 2 * n_jobs:
        raise ValueError(
            f"vector length {len(vector)} does not match 2 * n_jobs = {2 * n_jobs}"
        )
    n_resources = len(problem.resources)
    starts = vector[:n_jobs]
    selectors = vector[n_jobs:]

    ridx: list[int] = []
    for s in selectors:
        idx = int(s // 1.0)
        if idx < 0:
            idx = 0
        elif idx >= n_resources:
            idx = n_resources - 1
        ridx.append(idx)

    start_times: list[float] = []
    for t in starts:
        start_times.append(t if t > 0.0 else 0.0)

    return Chromosome(resource_idx=tuple(ridx), start_time=tuple(start_times))


def clip_to_bounds(vector: tuple[float, ...], bounds: VectorBounds) -> tuple[float, ...]:
    if len(vector) != len(bounds.lower):
        raise ValueError("vector length does not match bounds")
    return tuple(
        min(max(v, lo), hi)
        for v, lo, hi in zip(vector, bounds.lower, bounds.upper, strict=True)
    )
