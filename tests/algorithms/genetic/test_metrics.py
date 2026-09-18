import pytest

from edmo.algorithms.genetic.chromosome import Chromosome
from edmo.algorithms.genetic.metrics import (
    convergence_span,
    population_diversity,
    stagnation_length,
)


def _pop_low_diversity() -> list[Chromosome]:
    return [
        Chromosome((0, 0), (1.0, 1.0)),
        Chromosome((0, 0), (1.0, 1.0)),
        Chromosome((0, 0), (1.0, 1.0)),
    ]


def _pop_high_diversity() -> list[Chromosome]:
    return [
        Chromosome((0, 0), (0.0, 0.0)),
        Chromosome((1, 1), (5.0, 5.0)),
        Chromosome((2, 2), (10.0, 10.0)),
    ]


def test_population_diversity_singleton() -> None:
    assert population_diversity([Chromosome((0,), (0.0,))]) == 0.0


def test_population_diversity_identical_is_zero() -> None:
    assert population_diversity(_pop_low_diversity()) == 0.0


def test_population_diversity_grows_with_variation() -> None:
    low = population_diversity(_pop_low_diversity())
    high = population_diversity(_pop_high_diversity())
    assert high > low
    assert 0.0 <= high <= 1.0


def test_population_diversity_empty() -> None:
    assert population_diversity([]) == 0.0


def test_convergence_span_no_improvement() -> None:
    assert convergence_span([5.0, 5.0, 5.0]) == 0.0


def test_convergence_span_improvement() -> None:
    span = convergence_span([10.0, 8.0, 5.0])
    assert 0.4 < span < 0.6


def test_convergence_span_rejects_empty() -> None:
    with pytest.raises(ValueError):
        convergence_span([])


def test_stagnation_length_counts_trailing_flat() -> None:
    assert stagnation_length([10.0, 5.0, 5.0, 5.0]) == 2
    assert stagnation_length([10.0, 8.0, 6.0]) == 0


def test_stagnation_length_with_tolerance() -> None:
    # Improvements smaller than tolerance do not reset the counter.
    assert stagnation_length([10.0, 5.0, 4.9999995, 4.999999], tolerance=1e-3) >= 2


def test_stagnation_length_empty() -> None:
    assert stagnation_length([]) == 0
