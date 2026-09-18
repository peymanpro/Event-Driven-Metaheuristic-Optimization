# ADR-0004: Why Warm-Start?

## Status

Accepted.

## Context

When a problem changes, restarting the optimizer from a random population
throws away information. The pre-change state contains a population that
already encodes partial structure of the new problem (shared jobs, shared
resources) and a known best solution.

## Decision

We keep optimizer state explicit (`GAState`) and provide `adapt_state` plus
`run_ga_stateful`. The warm-start path reuses the existing population,
remaps resource indices by id, resamples only affected genes, and continues
the same RNG stream.

## Consequences

- The central research question becomes testable with a reproducible
  benchmark (`restart_vs_warm_start`).
- Warm-start benefits depend on change structure; structural changes require
  re-shaping genes and cannot reuse the full population.
- State is deliberately persisted as an immutable snapshot, so it can be
  serialized or shipped without hidden mutable coupling.