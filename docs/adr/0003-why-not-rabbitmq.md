# ADR-0003: Why Not RabbitMQ?

## Status

Accepted.

## Context

RabbitMQ is a mature message broker with strong routing semantics. However,
the project's event needs are: durable log, replay, per-aggregate ordering,
consumer groups for parallel consumers, and a well-defined retention story.

## Decision

We chose Kafka over RabbitMQ for the following reasons:

- RabbitMQ is fundamentally a message queue with routing; durable ordered
  logs and replay are supported through streams/plugins but require more
  configuration.
- Consumer group semantics and offset management are first-class in Kafka.
- Partitioned ordering by aggregate is trivial in Kafka and non-trivial in
  RabbitMQ.

## Consequences

- We accept a heavier local setup cost when running against a real broker.
- Tests are written so they do not require a running broker.