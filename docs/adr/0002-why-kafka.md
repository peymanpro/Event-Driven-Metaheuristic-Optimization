# ADR-0002: Why Kafka?

## Status

Accepted.

## Context

The system produces a stream of domain events (job lifecycle, resource
changes, run results) that may be consumed by independent readers: the
optimizer state, the experiment pipeline, and operational telemetry. We need
a durable, ordered, replayable log with consumer-group semantics.

## Decision

Kafka is used as the durable transport for the two topics declared in
`edmo.infrastructure.event_bus.topics`: `optimization-events` and
`optimization-results`. Aggregates are used as message keys so ordering is
preserved per aggregate.

## Consequences

- Replay and consumer groups are available without building them ourselves.
- We do not convert internal optimizer steps into messages; only domain
  events and run results travel over Kafka.
- Tests run without a broker by injecting fake producer/consumer clients.
- `kafka-python` is an optional dependency; the algorithmic core never
  requires Kafka.