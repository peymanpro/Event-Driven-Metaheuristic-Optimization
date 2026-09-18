import pytest

from edmo.algorithms.genetic.config import GAConfig


def test_defaults() -> None:
    cfg = GAConfig()
    assert cfg.population_size == 50
    assert cfg.generations == 100
    assert cfg.elite_count == 2
    assert cfg.tournament_size == 3
    assert cfg.seed is None


def test_rejects_bad_population_size() -> None:
    with pytest.raises(ValueError):
        GAConfig(population_size=0)


def test_rejects_bad_generations() -> None:
    with pytest.raises(ValueError):
        GAConfig(generations=-1)


def test_rejects_bad_crossover_rate() -> None:
    with pytest.raises(ValueError):
        GAConfig(crossover_rate=-0.1)
    with pytest.raises(ValueError):
        GAConfig(crossover_rate=1.1)


def test_rejects_bad_mutation_rate() -> None:
    with pytest.raises(ValueError):
        GAConfig(mutation_rate=-0.1)
    with pytest.raises(ValueError):
        GAConfig(mutation_rate=1.1)


def test_rejects_bad_tournament_size() -> None:
    with pytest.raises(ValueError):
        GAConfig(tournament_size=1)


def test_rejects_bad_elite_count() -> None:
    with pytest.raises(ValueError):
        GAConfig(population_size=5, elite_count=6)
    with pytest.raises(ValueError):
        GAConfig(elite_count=-1)


def test_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError):
        GAConfig(start_time_horizon=0.0)


def test_rejects_bad_sigma() -> None:
    with pytest.raises(ValueError):
        GAConfig(mutation_time_sigma=0.0)
