import pytest

from edmo.algorithms.genetic.termination import (
    TerminationDecision,
    TerminationPolicy,
)


def test_policy_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        TerminationPolicy(max_generations=-1)
    with pytest.raises(ValueError):
        TerminationPolicy(max_generations=1, stagnation_window=0)
    with pytest.raises(ValueError):
        TerminationPolicy(max_generations=1, stagnation_tolerance=-1.0)
    with pytest.raises(ValueError):
        TerminationPolicy(max_generations=1, convergence_threshold=0.0)
    with pytest.raises(ValueError):
        TerminationPolicy(max_generations=1, convergence_threshold=1.5)


def test_policy_continue_on_improvement() -> None:
    policy = TerminationPolicy(max_generations=10)
    d = policy.decide([10.0, 8.0, 6.0])
    assert isinstance(d, TerminationDecision)
    assert not d.should_stop


def test_policy_stops_at_max_generations() -> None:
    policy = TerminationPolicy(max_generations=2)
    d = policy.decide([10.0, 8.0, 6.0])
    assert d.should_stop
    assert d.reason == "max_generations"


def test_policy_stops_on_stagnation() -> None:
    policy = TerminationPolicy(max_generations=100, stagnation_window=3)
    d = policy.decide([10.0, 5.0, 5.0, 5.0, 5.0])
    assert d.should_stop
    assert d.reason == "stagnation"


def test_policy_does_not_stop_early_with_improvement() -> None:
    policy = TerminationPolicy(max_generations=100, stagnation_window=3)
    d = policy.decide([10.0, 5.0, 4.5, 4.0, 3.5])
    assert not d.should_stop


def test_policy_stops_on_convergence() -> None:
    policy = TerminationPolicy(max_generations=100, convergence_threshold=0.5)
    d = policy.decide([10.0, 5.0])
    assert d.should_stop
    assert d.reason == "convergence"


def test_policy_empty_history() -> None:
    policy = TerminationPolicy(max_generations=1)
    d = policy.decide([])
    assert not d.should_stop
