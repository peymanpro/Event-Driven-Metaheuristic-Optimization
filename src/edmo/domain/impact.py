from __future__ import annotations

from dataclasses import dataclass

from edmo.domain.problem import Problem


@dataclass(frozen=True, slots=True)
class ChangeImpact:
    """Structural diff between two problem versions.

    ``is_structural`` is True when jobs or resources were added or removed,
    which typically changes the dimensionality of the search space and thus
    may invalidate warm-start state.
    """

    version_before: int
    version_after: int
    jobs_added: tuple[str, ...]
    jobs_removed: tuple[str, ...]
    jobs_with_changed_deadline: tuple[str, ...]
    jobs_with_changed_weight: tuple[str, ...]
    resources_added: tuple[str, ...]
    resources_removed: tuple[str, ...]
    resources_with_changed_capacity: tuple[str, ...]
    resources_with_changed_speed: tuple[str, ...]

    @property
    def is_structural(self) -> bool:
        return bool(
            self.jobs_added
            or self.jobs_removed
            or self.resources_added
            or self.resources_removed
        )

    @property
    def job_count_delta(self) -> int:
        return len(self.jobs_added) - len(self.jobs_removed)

    @property
    def resource_count_delta(self) -> int:
        return len(self.resources_added) - len(self.resources_removed)


def analyze_change(before: Problem, after: Problem) -> ChangeImpact:
    """Diff two problem versions and report structural and parametric changes."""
    jobs_before = {j.id: j for j in before.jobs}
    jobs_after = {j.id: j for j in after.jobs}
    resources_before = {r.id: r for r in before.resources}
    resources_after = {r.id: r for r in after.resources}

    jobs_added = tuple(sorted(jobs_after.keys() - jobs_before.keys()))
    jobs_removed = tuple(sorted(jobs_before.keys() - jobs_after.keys()))
    resources_added = tuple(sorted(resources_after.keys() - resources_before.keys()))
    resources_removed = tuple(sorted(resources_before.keys() - resources_after.keys()))

    changed_deadline: list[str] = []
    changed_weight: list[str] = []
    for job_id in sorted(jobs_before.keys() & jobs_after.keys()):
        jb = jobs_before[job_id]
        ja = jobs_after[job_id]
        if jb.deadline != ja.deadline:
            changed_deadline.append(job_id)
        if jb.weight != ja.weight:
            changed_weight.append(job_id)

    changed_capacity: list[str] = []
    changed_speed: list[str] = []
    for rid in sorted(resources_before.keys() & resources_after.keys()):
        rb = resources_before[rid]
        ra = resources_after[rid]
        if dict(rb.capacity) != dict(ra.capacity):
            changed_capacity.append(rid)
        if rb.speed != ra.speed:
            changed_speed.append(rid)

    return ChangeImpact(
        version_before=before.version,
        version_after=after.version,
        jobs_added=jobs_added,
        jobs_removed=jobs_removed,
        jobs_with_changed_deadline=tuple(changed_deadline),
        jobs_with_changed_weight=tuple(changed_weight),
        resources_added=resources_added,
        resources_removed=resources_removed,
        resources_with_changed_capacity=tuple(changed_capacity),
        resources_with_changed_speed=tuple(changed_speed),
    )
