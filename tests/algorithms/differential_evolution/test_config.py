import pytest

from edmo.algorithms.differential_evolution.config import DEConfig


def test_defaults() -> None:
    cfg = DEConfig()
    assert cfg.population_size == 50
    assert cfg.generations == 100
    assert cfg.mutation_factor == 0.5
    assert cfg.crossover_rate == 0.9
    assert cfg.seed is None


def test_rejects_small_population() -> None:
    with pytest.raises(ValueError):
        DEConfig(population_size=3)


def test_rejects_bad_generations() -> None:
    with pytest.raises(ValueError):
        DEConfig(generations=-1)


def test_rejects_bad_mutation_factor() -> None:
    with pytest.raises(ValueError):
        DEConfig(mutation_factor=0.0)
    with pytest.raises(ValueError):
        DEConfig(mutation_factor=2.5)


def test_rejects_bad_crossover_rate() -> None:
    with pytest.raises(ValueError):
        DEConfig(crossover_rate=-0.1)
    with pytest.raises(ValueError):
        DEConfig(crossover_rate=1.1)


def test_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError):
        DEConfig(start_time_horizon=0.0)
