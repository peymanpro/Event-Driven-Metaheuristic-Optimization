from __future__ import annotations

from dataclasses import dataclass

from edmo.algorithms.genetic.termination import TerminationPolicy


@dataclass(frozen=True, slots=True)
class DEConfig:
    """Configuration of Differential Evolution (DE/rand/1/bin).

    Each individual is a real-valued vector of length ``2 * n_jobs``:
    the first block encodes start times in ``[0, start_time_horizon]`` and
    the second block encodes resource selection in
    ``[0, n_resources - 1]`` (decoded by ``int(floor(x))``).

    ``mutation_factor`` is the DE ``F`` and ``crossover_rate`` is the DE
    ``CR``.
    """

    population_size: int = 50
    generations: int = 100
    mutation_factor: float = 0.5
    crossover_rate: float = 0.9
    start_time_horizon: float = 100.0
    seed: int | None = None
    apply_repair: bool = True
    termination: TerminationPolicy | None = None

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4 for DE/rand/1")
        if self.generations < 0:
            raise ValueError("generations must be >= 0")
        if not 0.0 < self.mutation_factor <= 2.0:
            raise ValueError("mutation_factor must be in (0, 2]")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if self.start_time_horizon <= 0.0:
            raise ValueError("start_time_horizon must be positive")
