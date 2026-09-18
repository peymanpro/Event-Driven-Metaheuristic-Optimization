import pytest

from edmo.algorithms.genetic.config import GAConfig
from edmo.application.optimization_coordinator import (
    EVENT_TYPE_CHANGE_REQUESTED,
    EVENT_TYPE_OPTIMIZATION_COMPLETED,
    OptimizationCoordinator,
)
from edmo.domain.change import AddJob, ChangeWeight
from edmo.domain.events.event import DomainEvent
from edmo.domain.jobs import Job
from edmo.domain.problem import Problem
from edmo.domain.resources import Resource
from edmo.infrastructure.event_bus.in_memory import InMemoryTransport
from edmo.infrastructure.event_bus.topics import (
    OPTIMIZATION_EVENTS,
    OPTIMIZATION_RESULTS,
)


def _problem() -> Problem:
    return Problem(
        id="p1",
        resources=(
            Resource(id="r0", capacity={"cpu": 4.0}),
            Resource(id="r1", capacity={"cpu": 4.0}),
        ),
        jobs=(
            Job(id="j1", processing_time=1.0, demand={"cpu": 1.0}, deadline=20.0),
            Job(id="j2", processing_time=2.0, demand={"cpu": 1.0}, deadline=20.0),
        ),
    )


def _config() -> GAConfig:
    return GAConfig(population_size=10, generations=4, seed=0)


def _build() -> tuple[InMemoryTransport, OptimizationCoordinator, list[DomainEvent]]:
    transport = InMemoryTransport()
    coord = OptimizationCoordinator(
        publisher=transport,
        problem=_problem(),
        ga_config=_config(),
        experiment_id="e2e",
    )
    transport.subscribe(OPTIMIZATION_EVENTS, coord.handle_event)
    results: list[DomainEvent] = []
    transport.subscribe(OPTIMIZATION_RESULTS, results.append)
    return transport, coord, results


def test_initial_run_publishes_fresh_completed() -> None:
    transport, coord, results = _build()
    result = coord.run_initial()
    assert result.generations == 4
    assert coord.state is not None
    assert coord.state.generation == 4
    assert transport.drain(OPTIMIZATION_RESULTS) == 1
    assert len(results) == 1
    ev = results[0]
    assert ev.event_type == EVENT_TYPE_OPTIMIZATION_COMPLETED
    assert ev.payload["algorithm"] == "ga_fresh"
    assert ev.payload["problem_version"] == 1
    assert ev.payload["iterations"] == 4
    assert ev.payload["generation_after"] == 4
    assert isinstance(ev.payload["feasible"], bool)


def test_change_event_triggers_warm_start() -> None:
    transport, coord, results = _build()
    coord.run_initial()
    transport.drain(OPTIMIZATION_RESULTS)
    results.clear()

    change_event = coord.make_change_event(ChangeWeight("j1", 2.0))
    assert change_event.event_type == EVENT_TYPE_CHANGE_REQUESTED
    transport.publish(OPTIMIZATION_EVENTS, change_event)

    invocations = transport.drain(OPTIMIZATION_EVENTS)
    assert invocations == 1
    assert coord.problem.version == 2
    assert coord.warm_start_used_in_last_run is True
    assert coord.state is not None
    # Generation must continue, not restart: 4 + 4 = 8.
    assert coord.state.generation == 8

    assert transport.drain(OPTIMIZATION_RESULTS) == 1
    completed = results[-1]
    assert completed.payload["algorithm"] == "ga_warm_start"
    assert completed.payload["problem_version"] == 2
    assert completed.payload["generation_after"] == 8
    assert completed.payload["iterations"] == 4


def test_structural_change_and_warm_start() -> None:
    transport, coord, _results = _build()
    coord.run_initial()
    transport.drain(OPTIMIZATION_RESULTS)

    job = Job(id="j3", processing_time=1.5, demand={"cpu": 1.0})
    change_event = coord.make_change_event(AddJob(job))
    transport.publish(OPTIMIZATION_EVENTS, change_event)
    transport.drain(OPTIMIZATION_EVENTS)

    assert coord.problem.version == 2
    assert {j.id for j in coord.problem.jobs} == {"j1", "j2", "j3"}
    assert coord.state is not None
    assert coord.state.generation == 8
    assert coord.state.best_chromosome.length == 3


def test_duplicate_change_event_is_idempotent() -> None:
    transport, coord, _results = _build()
    coord.run_initial()
    transport.drain(OPTIMIZATION_RESULTS)

    event = coord.make_change_event(ChangeWeight("j1", 2.0))
    transport.publish(OPTIMIZATION_EVENTS, event)
    transport.publish(OPTIMIZATION_EVENTS, event)
    invocations = transport.drain(OPTIMIZATION_EVENTS)
    assert invocations == 1
    assert coord.problem.version == 2
    assert coord.state is not None
    assert coord.state.generation == 8


def test_consecutive_changes_accumulate_generation() -> None:
    transport, coord, _results = _build()
    coord.run_initial()
    transport.drain(OPTIMIZATION_RESULTS)

    e1 = coord.make_change_event(ChangeWeight("j1", 2.0))
    transport.publish(OPTIMIZATION_EVENTS, e1)
    transport.drain(OPTIMIZATION_EVENTS)

    e2 = coord.make_change_event(ChangeWeight("j1", 3.0))
    transport.publish(OPTIMIZATION_EVENTS, e2)
    transport.drain(OPTIMIZATION_EVENTS)

    assert coord.problem.version == 3
    assert coord.state is not None
    assert coord.state.generation == 12
    assert coord.run_count == 3
    assert coord.received_change_kinds == ["ChangeWeight", "ChangeWeight"]


def test_unrelated_event_type_is_ignored() -> None:
    transport, coord, results = _build()
    coord.run_initial()
    transport.drain(OPTIMIZATION_RESULTS)
    results.clear()

    unrelated = DomainEvent(event_type="SomethingElse", aggregate_id="p1", payload={})
    transport.publish(OPTIMIZATION_EVENTS, unrelated)
    invocations = transport.drain(OPTIMIZATION_EVENTS)
    assert invocations == 1  # handler invoked
    # But no change was applied and no new run happened.
    assert coord.problem.version == 1
    assert coord.run_count == 1
    assert transport.drain(OPTIMIZATION_RESULTS) == 0


def test_change_event_without_state_starts_fresh() -> None:
    transport, coord, results = _build()
    # No run_initial; directly publish a change event.
    event = coord.make_change_event(ChangeWeight("j1", 2.0))
    transport.publish(OPTIMIZATION_EVENTS, event)
    transport.drain(OPTIMIZATION_EVENTS)
    assert coord.problem.version == 2
    assert coord.state is not None
    assert coord.warm_start_used_in_last_run is False
    assert transport.drain(OPTIMIZATION_RESULTS) == 1
    assert results[-1].payload["algorithm"] == "ga_fresh"


def test_change_event_missing_payload_rejected() -> None:
    transport, _coord, _results = _build()
    bad = DomainEvent(event_type=EVENT_TYPE_CHANGE_REQUESTED, aggregate_id="p1", payload={})
    transport.publish(OPTIMIZATION_EVENTS, bad)
    with pytest.raises(ValueError):
        transport.drain(OPTIMIZATION_EVENTS)
