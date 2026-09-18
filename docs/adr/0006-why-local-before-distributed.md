# ADR-0006: Why Local Execution Before Distributed Workers?

## Status

Accepted.

## Context

Fitness evaluation is CPU-bound per candidate but embarrassingly parallel
across candidates. A distributed worker model would introduce broker
coupling, serialization overhead, and operational complexity.

## Decision

The default worker model is a local process pool
(`ProcessPoolExecutor`) with a thread option for debugging. Kafka-based
workers are not implemented. Any move to distributed workers must be
justified by a measured bottleneck (see `sequential_vs_parallel` benchmark).

## Consequences

- The parallel path is testable without a broker.
- Domain value types implement `__reduce__` so they pickle cleanly.
- Scaling to a cluster remains possible later without changing the domain.