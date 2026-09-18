from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TerminationDecision:
    should_stop: bool
    reason: str


@dataclass(frozen=True, slots=True)
class TerminationPolicy:
    """Combined termination criteria for the genetic algorithm.

    The optimizer stops when any of the following holds:

    - ``max_generations`` reached
    - the best-so-far has not improved (beyond ``stagnation_tolerance``)
      for ``stagnation_window`` consecutive generations
    - the relative improvement over the whole history reaches
      ``convergence_threshold``
    """

    max_generations: int
    stagnation_window: int | None = None
    stagnation_tolerance: float = 1.0e-6
    convergence_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.max_generations < 0:
            raise ValueError("max_generations must be >= 0")
        if self.stagnation_window is not None and self.stagnation_window < 1:
            raise ValueError("stagnation_window must be >= 1")
        if self.stagnation_tolerance < 0.0:
            raise ValueError("stagnation_tolerance must be >= 0")
        if self.convergence_threshold is not None and not (
            0.0 < self.convergence_threshold <= 1.0
        ):
            raise ValueError("convergence_threshold must be in (0, 1]")

    def decide(self, history: Sequence[float]) -> TerminationDecision:
        n = len(history)
        if n == 0:
            return TerminationDecision(should_stop=False, reason="empty")
        if n - 1 >= self.max_generations:
            return TerminationDecision(should_stop=True, reason="max_generations")

        if self.stagnation_window is not None and n - 1 >= self.stagnation_window:
            window = history[-(self.stagnation_window + 1):]
            best_before = window[0]
            improved = any(v < best_before - self.stagnation_tolerance for v in window[1:])
            if not improved:
                return TerminationDecision(should_stop=True, reason="stagnation")

        if self.convergence_threshold is not None:
            first = history[0]
            best = min(history)
            span = (
                (1.0 if best < 0.0 else 0.0)
                if first == 0.0
                else (first - best) / abs(first)
            )
            if span >= self.convergence_threshold:
                return TerminationDecision(should_stop=True, reason="convergence")

        return TerminationDecision(should_stop=False, reason="continue")
