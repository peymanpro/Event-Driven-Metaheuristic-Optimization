import pytest

from edmo.infrastructure.kafka.retry import RetryPolicy


def test_success_first_attempt() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(max_attempts=3, sleep=sleeps.append)
    assert policy.run(lambda: 42) == 42
    assert sleeps == []


def test_retry_until_success() -> None:
    sleeps: list[float] = []
    calls = {"n": 0}

    def op() -> int:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("fail")
        return 7

    policy = RetryPolicy(max_attempts=5, base_delay=0.1, backoff=2.0, sleep=sleeps.append)
    assert policy.run(op) == 7
    assert calls["n"] == 3
    assert sleeps == [0.1, 0.2]


def test_retry_exhausts_and_raises() -> None:
    sleeps: list[float] = []

    def op() -> int:
        raise RuntimeError("permanent")

    policy = RetryPolicy(max_attempts=2, base_delay=0.05, sleep=sleeps.append)
    with pytest.raises(RuntimeError):
        policy.run(op)
    assert sleeps == [0.05]


def test_policy_validation() -> None:
    with pytest.raises(ValueError):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError):
        RetryPolicy(base_delay=-1.0)
    with pytest.raises(ValueError):
        RetryPolicy(backoff=0.0)
    with pytest.raises(ValueError):
        RetryPolicy(max_delay=-1.0)


def test_delay_capped_at_max_delay() -> None:
    sleeps: list[float] = []
    policy = RetryPolicy(
        max_attempts=5,
        base_delay=1.0,
        backoff=10.0,
        max_delay=2.0,
        sleep=sleeps.append,
    )
    with pytest.raises(RuntimeError):
        policy.run(lambda: (_ for _ in ()).throw(RuntimeError("x")))
    assert all(s <= 2.0 for s in sleeps)
