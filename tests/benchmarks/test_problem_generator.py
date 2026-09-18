import pytest

from edmo.benchmarks.problem_generator import RandomProblemSpec, random_problem


def test_spec_defaults() -> None:
    s = RandomProblemSpec()
    assert s.n_resources == 3
    assert s.n_jobs == 8
    assert s.seed == 0


def test_spec_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        RandomProblemSpec(n_resources=0)
    with pytest.raises(ValueError):
        RandomProblemSpec(n_jobs=0)
    with pytest.raises(ValueError):
        RandomProblemSpec(max_processing_time=0.0)
    with pytest.raises(ValueError):
        RandomProblemSpec(max_release_time=-1.0)
    with pytest.raises(ValueError):
        RandomProblemSpec(max_deadline_horizon=0.0)


def test_random_problem_shape() -> None:
    p = random_problem(RandomProblemSpec(n_resources=2, n_jobs=5, seed=0))
    assert len(p.resources) == 2
    assert len(p.jobs) == 5
    assert p.metadata["seed"] == "0"


def test_random_problem_is_deterministic() -> None:
    p1 = random_problem(RandomProblemSpec(seed=7))
    p2 = random_problem(RandomProblemSpec(seed=7))
    for r1, r2 in zip(p1.resources, p2.resources, strict=True):
        assert r1.id == r2.id
        assert r1.speed == r2.speed
    for j1, j2 in zip(p1.jobs, p2.jobs, strict=True):
        assert j1.id == j2.id
        assert j1.processing_time == j2.processing_time
        assert j1.release_time == j2.release_time
        assert j1.deadline == j2.deadline
        assert j1.weight == j2.weight


def test_random_problem_different_seeds_differ() -> None:
    p1 = random_problem(RandomProblemSpec(seed=1))
    p2 = random_problem(RandomProblemSpec(seed=2))
    assert [j.processing_time for j in p1.jobs] != [j.processing_time for j in p2.jobs]
