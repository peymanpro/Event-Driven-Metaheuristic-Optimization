from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from edmo.algorithms.genetic.config import GAConfig
from edmo.algorithms.genetic.optimizer import GAResult
from edmo.algorithms.genetic.state import GAState
from edmo.algorithms.genetic.stateful import adapt_state, run_ga_stateful
from edmo.application.change_codec import decode_change, encode_change
from edmo.domain.change import ProblemChange, apply_change
from edmo.domain.evaluation import PenaltyWeights
from edmo.domain.events.event import DomainEvent
from edmo.domain.problem import Problem
from edmo.infrastructure.event_bus.topics import (
    OPTIMIZATION_EVENTS,
    OPTIMIZATION_RESULTS,
)

EVENT_TYPE_CHANGE_REQUESTED = "ProblemChangeRequested"
EVENT_TYPE_OPTIMIZATION_COMPLETED = "OptimizationCompleted"


class _Publisher(Protocol):
    def publish(self, topic: str, event: DomainEvent) -> None: ...


@dataclass(slots=True)
class OptimizationCoordinator:
    """Application-level coordinator wiring domain events to the optimizer.

    The coordinator is transport-agnostic: it only needs a publisher with a
    ``publish(topic, event)`` method. Handlers are registered by the caller
    (e.g. ``transport.subscribe(OPTIMIZATION_EVENTS, coordinator.handle_event)``).

    Flow:

        ProblemChangeRequested
            -> apply change
            -> adapt optimizer state (if any)
            -> warm-start optimization
            -> OptimizationCompleted

    Semantics:

    - ``state`` is ``None`` until ``run_initial`` is called.
    - ``warm_start_used_in_last_run`` reports whether the last run continued
      from an existing state or started fresh.
    - ``run_count`` is the number of completed runs handled by this coordinator.
    """

    publisher: _Publisher
    problem: Problem
    ga_config: GAConfig
    weights: PenaltyWeights | None = None
    experiment_id: str = "default"
    events_topic: str = OPTIMIZATION_EVENTS
    results_topic: str = OPTIMIZATION_RESULTS
    state: GAState | None = None
    last_result: GAResult | None = None
    last_run_id: str | None = None
    warm_start_used_in_last_run: bool = False
    run_count: int = 0
    received_change_kinds: list[str] = field(default_factory=list)

    def run_initial(self) -> GAResult:
        """Run the optimizer from scratch on the current problem."""
        state, result = run_ga_stateful(
            self.problem, None, self.ga_config, self.weights
        )
        self.state = state
        self.last_result = result
        self.warm_start_used_in_last_run = False
        self.run_count += 1
        self._publish_completed(warm_start=False, result=result, state=state)
        return result

    def apply_change(self, change: ProblemChange) -> None:
        """Apply a domain change and adapt the optimizer state in place."""
        old_problem = self.problem
        new_problem = apply_change(old_problem, change)
        if self.state is not None:
            self.state = adapt_state(old_problem, new_problem, self.state, self.ga_config)
        self.problem = new_problem
        self.received_change_kinds.append(type(change).__name__)

    def handle_event(self, event: DomainEvent) -> None:
        """Handle a single domain event.

        Only ``ProblemChangeRequested`` is handled here; any other event type
        is ignored so the coordinator can be wired to a shared topic safely.
        """
        if event.event_type != EVENT_TYPE_CHANGE_REQUESTED:
            return
        payload = event.payload
        change_payload = payload.get("change")
        if not isinstance(change_payload, dict):
            raise ValueError(
                "ProblemChangeRequested payload must contain a 'change' object"
            )
        change = decode_change(dict(change_payload))
        self.apply_change(change)
        warm_start = self.state is not None
        state, result = run_ga_stateful(
            self.problem, self.state, self.ga_config, self.weights
        )
        self.state = state
        self.last_result = result
        self.warm_start_used_in_last_run = warm_start
        self.run_count += 1
        self._publish_completed(warm_start=warm_start, result=result, state=state)

    def make_change_event(self, change: ProblemChange) -> DomainEvent:
        """Build a ``ProblemChangeRequested`` event for ``change``."""
        return DomainEvent(
            event_type=EVENT_TYPE_CHANGE_REQUESTED,
            aggregate_id=self.problem.id,
            payload={"change": encode_change(change)},
        )

    def _publish_completed(
        self,
        warm_start: bool,
        result: GAResult,
        state: GAState,
    ) -> None:
        run_id = str(uuid4())
        self.last_run_id = run_id
        payload = {
            "run_id": run_id,
            "experiment_id": self.experiment_id,
            "algorithm": "ga_warm_start" if warm_start else "ga_fresh",
            "problem_id": self.problem.id,
            "problem_version": self.problem.version,
            "best_total": result.best_evaluation.total,
            "objective": result.best_evaluation.objective,
            "penalty": result.best_evaluation.penalty,
            "feasible": result.best_evaluation.feasible,
            "iterations": result.generations,
            "stopped_reason": result.stopped_reason,
            "generation_after": state.generation,
        }
        event = DomainEvent(
            event_type=EVENT_TYPE_OPTIMIZATION_COMPLETED,
            aggregate_id=self.problem.id,
            payload=payload,
        )
        self.publisher.publish(self.results_topic, event)
