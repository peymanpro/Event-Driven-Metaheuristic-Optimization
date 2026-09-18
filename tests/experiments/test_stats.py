import pytest

from edmo.experiments.stats import (
    StabilityReport,
    convergence_distribution,
    runtime_distribution,
    stability_report,
    summarize,
)


def test_summarize_basic() -> None:
    s = summarize([1.0, 2.0, 3.0, 4.0])
    assert s.count == 4
    assert s.mean == 2.5
    assert s.median == 2.5
    assert s.minimum == 1.0
    assert s.maximum == 4.0


def test_summarize_single_value() -> None:
    s = summarize([7.0])
    assert s.variance == 0.0
    assert s.stdev == 0.0
    assert s.median == 7.0


def test_summarize_empty_rejected() -> None:
    with pytest.raises(ValueError):
        summarize([])


def test_summarize_odd_length_median() -> None:
    s = summarize([3.0, 1.0, 2.0])
    assert s.median == 2.0


def test_sample_variance() -> None:
    s = summarize([1.0, 2.0, 3.0])
    # sample variance = ((1-2)^2 + (0)^2 + (1)^2) / 2 = 1.0
    assert abs(s.variance - 1.0) < 1e-12
    assert abs(s.stdev - 1.0) < 1e-12


def test_convergence_distribution() -> None:
    s = convergence_distribution([[10.0, 8.0, 5.0], [10.0, 6.0], [10.0, 9.0, 7.0]])
    assert s.count == 3
    assert s.minimum == 5.0
    assert s.maximum == 7.0


def test_convergence_distribution_empty() -> None:
    with pytest.raises(ValueError):
        convergence_distribution([])


def test_runtime_distribution() -> None:
    s = runtime_distribution([0.1, 0.2, 0.3])
    assert s.count == 3


def test_stability_report() -> None:
    rep = stability_report([10.0, 8.0, 9.0], [True, True, False])
    assert isinstance(rep, StabilityReport)
    assert abs(rep.success_rate - (2.0 / 3.0)) < 1e-12
    assert rep.best_case_best_total == 8.0
    assert rep.worst_case_best_total == 10.0


def test_stability_report_length_mismatch() -> None:
    with pytest.raises(ValueError):
        stability_report([1.0, 2.0], [True])


def test_stability_report_empty() -> None:
    with pytest.raises(ValueError):
        stability_report([], [])
