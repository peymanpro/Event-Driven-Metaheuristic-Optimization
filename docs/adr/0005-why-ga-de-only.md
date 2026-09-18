# ADR-0005: Why GA + DE Only?

## Status

Accepted.

## Context

The project must remain deep rather than broad. Each additional metaheuristic
would multiply implementation, tuning, benchmarking, and maintenance costs.

## Decision

We keep exactly two metaheuristics: Genetic Algorithm and Differential
Evolution. They cover two distinct search paradigms (discrete combinatorics
with a mixed-integer chromosome; continuous vector search with a decoded
discrete projection). Both are required for a meaningful GA vs DE benchmark.

## Consequences

- PSO, ACO, SA and similar algorithms are explicitly out of scope.
- Both algorithms share a common `OptimizerResult` interface, a common
  repair path, and a common termination policy.
- Any future addition must justify itself against this ADR.