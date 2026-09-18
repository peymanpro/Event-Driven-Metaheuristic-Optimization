from __future__ import annotations

from collections.abc import Sequence

from edmo.algorithms.genetic.chromosome import Chromosome


def population_diversity(population: Sequence[Chromosome]) -> float:
    """Mean per-gene normalized diversity across a population.

    Resource genes contribute ``1 - 1 / n_distinct`` style fractional entropy
    normalized to ``[0, 1]``; start-time genes contribute normalized standard
    deviation over a per-population range. The result is the average of all
    per-gene diversity values, in ``[0, 1]``.
    """
    if len(population) < 2:
        return 0.0
    n_genes = population[0].length
    if n_genes == 0:
        return 0.0

    resource_scores: list[float] = []
    time_scores: list[float] = []

    for i in range(n_genes):
        idx_values = [c.resource_idx[i] for c in population]
        distinct = len(set(idx_values))
        if distinct <= 1:
            resource_scores.append(0.0)
        else:
            resource_scores.append((distinct - 1) / (len(population) - 1))

        times = [c.start_time[i] for c in population]
        t_min = min(times)
        t_max = max(times)
        if t_max <= t_min:
            time_scores.append(0.0)
            continue
        mean = sum(times) / len(times)
        variance = sum((t - mean) ** 2 for t in times) / len(times)
        std = variance ** 0.5
        time_scores.append(min(1.0, std / (t_max - t_min)))

    all_scores = resource_scores + time_scores
    return sum(all_scores) / len(all_scores)


def convergence_span(history: Sequence[float]) -> float:
    """Relative improvement from the first to the best value in ``history``.

    Returns ``(first - best) / (|first| + eps)`` in ``[0, 1]``; 0 means no
    improvement, values closer to 1 mean a large relative improvement.
    """
    if not history:
        raise ValueError("history must not be empty")
    first = history[0]
    best = min(history)
    denom = abs(first) + 1.0e-12
    span = (first - best) / denom
    return max(0.0, span)


def stagnation_length(history: Sequence[float], tolerance: float = 0.0) -> int:
    """Number of trailing entries with no improvement beyond ``tolerance``.

    Returns the count of generations (including the current) since the last
    strict improvement larger than ``tolerance`` relative to the running best.
    """
    if not history:
        return 0
    best = history[0]
    last_improvement = 0
    for i, value in enumerate(history):
        if value < best - tolerance:
            best = value
            last_improvement = i
    return len(history) - 1 - last_improvement
