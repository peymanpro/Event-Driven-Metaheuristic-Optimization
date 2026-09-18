from __future__ import annotations

from dataclasses import dataclass
from random import Random

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.metrics import GenerationSnapshot
from edmo.domain.evaluation import Evaluation


@dataclass(frozen=True, slots=True)
class GAState:
    """Persistent optimizer state.

    The state is self-contained: given the same problem version, replaying
    from this state must produce the same trajectory as a fresh run that
    reached it. ``rng`` is the live random generator so that warm-start
    executions continue the same stochastic stream.

    ``generation`` is the number of the last completed generation. Generation
    0 corresponds to the initial evaluated population before any evolution
    step. ``snapshots`` holds one :class:`GenerationSnapshot` per completed
    generation in order, including generation 0 for a fresh run.
    """

    population: tuple[Chromosome, ...]
    best_chromosome: Chromosome
    best_evaluation: Evaluation
    generation: int
    history: tuple[float, ...]
    problem_version: int
    rng: Random
    snapshots: tuple[GenerationSnapshot, ...] = ()

    def __post_init__(self) -> None:
        if self.generation < 0:
            raise ValueError("GAState.generation must be >= 0")
        if self.problem_version < 1:
            raise ValueError("GAState.problem_version must be >= 1")
        if not self.population:
            raise ValueError("GAState.population must not be empty")
        if self.snapshots and len(self.snapshots) != len(self.history):
            raise ValueError(
                "GAState.snapshots and GAState.history must have equal length"
            )
