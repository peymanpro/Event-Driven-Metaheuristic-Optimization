import pytest

from edmo.algorithms.differential_evolution.vector import (
    VectorBounds,
    clip_to_bounds,
    decode_vector,
    vector_bounds,
)
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource


def _problem(n_resources: int = 3, n_jobs: int = 4) -> Problem:
    return Problem(
        id="p1",
        resources=tuple(Resource(id=f"r{i}") for i in range(n_resources)),
        jobs=tuple(Job(id=f"j{i}", processing_time=1.0) for i in range(n_jobs)),
    )


def test_vector_bounds_length() -> None:
    b = vector_bounds(_problem(n_resources=3, n_jobs=4), 50.0)
    assert len(b.lower) == 8
    assert len(b.upper) == 8


def test_vector_bounds_single_resource() -> None:
    b = vector_bounds(_problem(n_resources=1, n_jobs=2), 10.0)
    assert b.upper[2] == 0.0
    assert b.upper[3] == 0.0


def test_vector_bounds_rejects_bad_horizon() -> None:
    with pytest.raises(ValueError):
        vector_bounds(_problem(), 0.0)


def test_vector_bounds_mismatched() -> None:
    with pytest.raises(ValueError):
        VectorBounds(lower=(0.0,), upper=(1.0, 2.0))


def test_vector_bounds_lower_gt_upper() -> None:
    with pytest.raises(ValueError):
        VectorBounds(lower=(2.0,), upper=(1.0,))


def test_decode_vector_length_mismatch() -> None:
    with pytest.raises(ValueError):
        decode_vector(_problem(n_jobs=3), (0.0, 1.0))


def test_decode_vector_basic() -> None:
    p = _problem(n_resources=3, n_jobs=2)
    # starts = [0.5, 2.5], selectors = [0.2, 2.9]
    c = decode_vector(p, (0.5, 2.5, 0.2, 2.9))
    assert c.start_time == (0.5, 2.5)
    assert c.resource_idx == (0, 2)


def test_decode_vector_clamps() -> None:
    p = _problem(n_resources=2, n_jobs=2)
    c = decode_vector(p, (-1.0, 5.0, -3.0, 100.0))
    assert c.start_time[0] == 0.0
    assert c.start_time[1] == 5.0
    assert c.resource_idx[0] == 0
    assert c.resource_idx[1] == 1


def test_clip_to_bounds() -> None:
    b = VectorBounds(lower=(0.0, 0.0), upper=(1.0, 2.0))
    assert clip_to_bounds((-1.0, 5.0), b) == (0.0, 2.0)
    assert clip_to_bounds((0.5, 1.0), b) == (0.5, 1.0)


def test_clip_to_bounds_length_mismatch() -> None:
    b = VectorBounds(lower=(0.0,), upper=(1.0,))
    with pytest.raises(ValueError):
        clip_to_bounds((0.0, 1.0), b)
