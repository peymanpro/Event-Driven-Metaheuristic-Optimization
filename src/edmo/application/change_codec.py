from __future__ import annotations

from typing import Any

from edmo.domain.change import (
    AddJob,
    AddResource,
    ChangeDeadline,
    ChangeResourceCapacity,
    ChangeResourceSpeed,
    ChangeWeight,
    ProblemChange,
    RemoveJob,
    RemoveResource,
)
from edmo.domain.jobs import Job
from edmo.domain.resources import Resource


def _job_to_payload(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "processing_time": job.processing_time,
        "demand": dict(job.demand),
        "release_time": job.release_time,
        "deadline": job.deadline,
        "weight": job.weight,
        "predecessors": sorted(job.predecessors),
    }


def _job_from_payload(data: dict[str, Any]) -> Job:
    demand_raw = data.get("demand", {})
    demand = {str(k): float(v) for k, v in dict(demand_raw).items()}
    predecessors = frozenset(str(x) for x in data.get("predecessors", []))
    deadline_raw = data.get("deadline")
    return Job(
        id=str(data["id"]),
        processing_time=float(data["processing_time"]),
        demand=demand,
        release_time=float(data.get("release_time", 0.0)),
        deadline=None if deadline_raw is None else float(deadline_raw),
        weight=float(data.get("weight", 1.0)),
        predecessors=predecessors,
    )


def _resource_to_payload(resource: Resource) -> dict[str, Any]:
    return {
        "id": resource.id,
        "capacity": dict(resource.capacity),
        "speed": resource.speed,
    }


def _resource_from_payload(data: dict[str, Any]) -> Resource:
    capacity_raw = data.get("capacity", {})
    capacity = {str(k): float(v) for k, v in dict(capacity_raw).items()}
    return Resource(
        id=str(data["id"]),
        capacity=capacity,
        speed=float(data.get("speed", 1.0)),
    )


def encode_change(change: ProblemChange) -> dict[str, Any]:
    """Encode a :class:`ProblemChange` into a JSON-friendly payload dict."""
    if isinstance(change, AddJob):
        return {"kind": "AddJob", "job": _job_to_payload(change.job)}
    if isinstance(change, RemoveJob):
        return {"kind": "RemoveJob", "job_id": change.job_id}
    if isinstance(change, ChangeDeadline):
        return {
            "kind": "ChangeDeadline",
            "job_id": change.job_id,
            "deadline": change.deadline,
        }
    if isinstance(change, ChangeWeight):
        return {
            "kind": "ChangeWeight",
            "job_id": change.job_id,
            "weight": change.weight,
        }
    if isinstance(change, AddResource):
        return {"kind": "AddResource", "resource": _resource_to_payload(change.resource)}
    if isinstance(change, RemoveResource):
        return {"kind": "RemoveResource", "resource_id": change.resource_id}
    if isinstance(change, ChangeResourceCapacity):
        return {
            "kind": "ChangeResourceCapacity",
            "resource_id": change.resource_id,
            "capacity": dict(change.capacity),
        }
    if isinstance(change, ChangeResourceSpeed):
        return {
            "kind": "ChangeResourceSpeed",
            "resource_id": change.resource_id,
            "speed": change.speed,
        }
    raise TypeError(f"unsupported change type: {type(change).__name__}")


def decode_change(data: dict[str, Any]) -> ProblemChange:
    """Decode a payload dict produced by :func:`encode_change`."""
    kind = data.get("kind")
    if kind == "AddJob":
        return AddJob(_job_from_payload(dict(data["job"])))
    if kind == "RemoveJob":
        return RemoveJob(str(data["job_id"]))
    if kind == "ChangeDeadline":
        deadline_raw = data.get("deadline")
        return ChangeDeadline(
            str(data["job_id"]),
            None if deadline_raw is None else float(deadline_raw),
        )
    if kind == "ChangeWeight":
        return ChangeWeight(str(data["job_id"]), float(data["weight"]))
    if kind == "AddResource":
        return AddResource(_resource_from_payload(dict(data["resource"])))
    if kind == "RemoveResource":
        return RemoveResource(str(data["resource_id"]))
    if kind == "ChangeResourceCapacity":
        capacity_raw = data.get("capacity", {})
        capacity = {str(k): float(v) for k, v in dict(capacity_raw).items()}
        return ChangeResourceCapacity(str(data["resource_id"]), capacity)
    if kind == "ChangeResourceSpeed":
        return ChangeResourceSpeed(str(data["resource_id"]), float(data["speed"]))
    raise ValueError(f"unsupported change kind: {kind!r}")
