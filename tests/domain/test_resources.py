import pytest

from edmo.domain.resources import Resource


def test_resource_defaults() -> None:
    r = Resource(id="r1")
    assert r.id == "r1"
    assert r.speed == 1.0
    assert dict(r.capacity) == {}


def test_resource_valid_with_capacity_and_speed() -> None:
    r = Resource(id="r1", capacity={"cpu": 4.0, "mem": 8.0}, speed=2.0)
    assert r.capacity["cpu"] == 4.0
    assert r.capacity["mem"] == 8.0
    assert r.speed == 2.0


def test_resource_capacity_is_immutable() -> None:
    r = Resource(id="r1", capacity={"cpu": 4.0})
    with pytest.raises(TypeError):
        r.capacity["cpu"] = 8.0  # type: ignore[index]


def test_resource_rejects_empty_id() -> None:
    with pytest.raises(ValueError):
        Resource(id="")


def test_resource_rejects_non_positive_speed() -> None:
    with pytest.raises(ValueError):
        Resource(id="r1", speed=0.0)
    with pytest.raises(ValueError):
        Resource(id="r1", speed=-1.0)


def test_resource_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError):
        Resource(id="r1", capacity={"cpu": -1.0})
