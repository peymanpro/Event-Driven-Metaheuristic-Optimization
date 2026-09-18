import pytest

from edmo.domain.solution import Assignment, Solution


def test_assignment_valid() -> None:
    a = Assignment(resource_id="r1", start_time=0.0)
    assert a.resource_id == "r1"
    assert a.start_time == 0.0


def test_assignment_rejects_empty_resource() -> None:
    with pytest.raises(ValueError):
        Assignment(resource_id="", start_time=0.0)


def test_assignment_rejects_negative_start() -> None:
    with pytest.raises(ValueError):
        Assignment(resource_id="r1", start_time=-0.1)


def test_solution_defaults_empty() -> None:
    s = Solution()
    assert len(s) == 0
    assert dict(s.assignments) == {}


def test_solution_holds_assignments() -> None:
    a = Assignment(resource_id="r1", start_time=1.0)
    s = Solution(assignments={"j1": a})
    assert len(s) == 1
    assert "j1" in s
    assert s.assignment_for("j1") is a


def test_solution_assignments_are_immutable() -> None:
    a = Assignment(resource_id="r1", start_time=0.0)
    s = Solution(assignments={"j1": a})
    with pytest.raises(TypeError):
        s.assignments["j2"] = a  # type: ignore[index]


def test_solution_rejects_empty_job_key() -> None:
    a = Assignment(resource_id="r1", start_time=0.0)
    with pytest.raises(ValueError):
        Solution(assignments={"": a})


def test_solution_rejects_non_assignment_value() -> None:
    with pytest.raises(TypeError):
        Solution(assignments={"j1": "not-an-assignment"})  # type: ignore[dict-item]


def test_solution_with_assignment_returns_new_instance() -> None:
    a1 = Assignment(resource_id="r1", start_time=0.0)
    a2 = Assignment(resource_id="r2", start_time=5.0)
    s = Solution(assignments={"j1": a1})
    s2 = s.with_assignment("j2", a2)
    assert s is not s2
    assert "j2" not in s
    assert s2.assignment_for("j2") is a2
    assert s2.assignment_for("j1") is a1
