from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GAConfig:
    """Configuration of the genetic algorithm.

    All rates are per-gene, applied independently. ``elite_count`` is the
    number of best individuals copied unchanged into the next generation.
    ``mutation_time_sigma`` is the standard deviation of the Gaussian used to
    perturb start times.
    """

    population_size: int = 50
    generations: int = 100
    crossover_rate: float = 0.9
    mutation_rate: float = 0.1
    tournament_size: int = 3
    elite_count: int = 2
    start_time_horizon: float = 100.0
    mutation_time_sigma: float = 5.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.population_size < 1:
            raise ValueError("population_size must be >= 1")
        if self.generations < 0:
            raise ValueError("generations must be >= 0")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be in [0, 1]")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mutation_rate must be in [0, 1]")
        if self.tournament_size < 2:
            raise ValueError("tournament_size must be >= 2")
        if not 0 <= self.elite_count <= self.population_size:
            raise ValueError("elite_count must be in [0, population_size]")
        if self.start_time_horizon <= 0.0:
            raise ValueError("start_time_horizon must be positive")
        if self.mutation_time_sigma <= 0.0:
            raise ValueError("mutation_time_sigma must be positive")
