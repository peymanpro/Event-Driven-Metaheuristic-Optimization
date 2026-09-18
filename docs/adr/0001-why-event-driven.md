# ADR-0001: Why Event-Driven?

## Status

Accepted.

## Context

The optimization problem changes over time (`f_t`, `g_{i,t}`). Jobs are
added or removed, deadlines and priorities shift, and resources are rescaled.
The optimizer must react to those changes without losing the current solution
state and without hard-coupling the algorithmic core to a specific transport.

## Decision

We model domain changes and optimizer progress as explicit events with a
stable envelope (event_id, event_type, timestamp, aggregate_id, version,
payload). The domain layer owns the event model; infrastructure adapters
(in-memory bus, Kafka) transport them.

## Consequences

- The algorithmic core (GA, DE, state) stays free of transport concerns.
- Events are replayable and analyzable offline via the EventLog JSONL.
- A new transport can be added without touching domain or algorithm code.
- We accept the cost of serialization boundaries and per-process idempotency
  filtering. Durable, cross-restart exactly-once delivery is explicitly out
  of scope; handlers that need it must persist their own state.