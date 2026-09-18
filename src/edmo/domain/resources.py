from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True, slots=True, eq=False)
class Resource:
    """A machine or agent that can execute jobs.

    ``capacity`` maps a resource dimension name (e.g. ``"cpu"``, ``"memory"``)
    to a non-negative quantity. ``speed`` is a dimensionless multiplier applied
    to a job's ``processing_time`` when scheduled on this resource.
    """

    id: str
    capacity: Mapping[str, float] = field(default_factory=dict)
    speed: float = 1.0

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Resource.id must be a non-empty string")
        if self.speed <= 0.0:
            raise ValueError("Resource.speed must be positive")
        for name, value in self.capacity.items():
            if not name:
                raise ValueError("capacity dimension name must be non-empty")
            if value < 0.0:
                raise ValueError(f"capacity[{name!r}] must be non-negative")
        object.__setattr__(self, "capacity", MappingProxyType(dict(self.capacity)))
