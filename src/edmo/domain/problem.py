from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from edmo.domain.jobs import Job
from edmo.domain.resources import Resource


@dataclass(frozen=True, slots=True, eq=False)
class Problem:
    """A static resource-allocation and scheduling problem instance.

    The problem is defined by a set of resources and a set of jobs. It carries
    an integer ``version`` so that later phases can distinguish instances of the
    same logical problem across time (``f_t`` versus ``f_{t+1}``).

    ``metadata`` is an opaque mapping for tags such as scenario names or seeds;
    it is not interpreted by the domain.
    """

    id: str
    resources: tuple[Resource, ...]
    jobs: tuple[Job, ...]
    version: int = 1
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Problem.id must be a non-empty string")
        if self.version < 1:
            raise ValueError("Problem.version must be >= 1")
        if not self.resources:
            raise ValueError("Problem.resources must not be empty")
        if not self.jobs:
            raise ValueError("Problem.jobs must not be empty")

        resource_ids = [r.id for r in self.resources]
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("Problem.resources must have unique ids")

        job_ids = [j.id for j in self.jobs]
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("Problem.jobs must have unique ids")

        known_jobs = set(job_ids)
        for job in self.jobs:
            unknown = job.predecessors - known_jobs
            if unknown:
                raise ValueError(
                    f"Job {job.id!r} references unknown predecessors: {sorted(unknown)}"
                )

        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "jobs", tuple(self.jobs))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return (
            Problem,
            (
                self.id,
                self.resources,
                self.jobs,
                self.version,
                dict(self.metadata),
            ),
        )

    def resource_by_id(self, resource_id: str) -> Resource:
        for r in self.resources:
            if r.id == resource_id:
                return r
        raise KeyError(resource_id)

    def job_by_id(self, job_id: str) -> Job:
        for j in self.jobs:
            if j.id == job_id:
                return j
        raise KeyError(job_id)
