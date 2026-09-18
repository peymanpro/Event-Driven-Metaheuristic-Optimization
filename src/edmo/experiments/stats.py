from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SummaryStats:
    count: int
    mean: float
    median: float
    variance: float
    stdev: float
    minimum: float
    maximum: float


def summarize(values: Sequence[float] | Iterable[float]) -> SummaryStats:
    """Compute count/mean/median/variance/stdev/min/max.

    Uses sample variance (denominator n-1) when ``count >= 2`` and returns 0.0
    for variance/stdev when ``count < 2``. Empty input raises ``ValueError``.
    """
    xs = list(values)
    if not xs:
        raise ValueError("cannot summarize empty input")
    n = len(xs)
    mean = sum(xs) / n
    sorted_xs = sorted(xs)
    median = (
        sorted_xs[n // 2]
        if n % 2 == 1
        else (sorted_xs[n // 2 - 1] + sorted_xs[n // 2]) / 2.0
    )
    variance = (
        sum((x - mean) ** 2 for x in xs) / (n - 1) if n >= 2 else 0.0
    )
    stdev = variance ** 0.5
    return SummaryStats(
        count=n,
        mean=mean,
        median=median,
        variance=variance,
        stdev=stdev,
        minimum=sorted_xs[0],
        maximum=sorted_xs[-1],
    )


def convergence_distribution(histories: Sequence[Sequence[float]]) -> SummaryStats:
    """Summarize final best totals across runs."""
    finals = [h[-1] for h in histories if h]
    if not finals:
        raise ValueError("no non-empty histories provided")
    return summarize(finals)


def runtime_distribution(runtimes: Sequence[float]) -> SummaryStats:
    return summarize(runtimes)


@dataclass(frozen=True, slots=True)
class StabilityReport:
    success_rate: float
    mean_best_total: float
    stdev_best_total: float
    worst_case_best_total: float
    best_case_best_total: float


def stability_report(
    best_totals: Sequence[float],
    feasible_flags: Sequence[bool],
) -> StabilityReport:
    """Summarize how stable a strategy is across repeated runs.

    ``success_rate`` is the fraction of runs that were feasible. An empty
    input raises ``ValueError``. Lengths of both inputs must match.
    """
    if len(best_totals) != len(feasible_flags):
        raise ValueError("best_totals and feasible_flags must have equal length")
    if not best_totals:
        raise ValueError("cannot compute stability on empty input")
    stats = summarize(best_totals)
    successes = sum(1 for f in feasible_flags if f)
    return StabilityReport(
        success_rate=successes / len(feasible_flags),
        mean_best_total=stats.mean,
        stdev_best_total=stats.stdev,
        worst_case_best_total=stats.maximum,
        best_case_best_total=stats.minimum,
    )
