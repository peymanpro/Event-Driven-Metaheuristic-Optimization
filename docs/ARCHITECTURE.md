# Architecture

## Overview

The project solves a time-varying constrained optimization problem:

    min_x f_t(x)   subject to   g_{i,t}(x) <= 0

It is organized as a layered Python package under `src/edmo`. Each layer has
a single responsibility and depends only downward.

    domain         -> pure problem, solution, evaluation, events
    algorithms     -> GA, DE, random search, state, repair, metrics
    benchmarks     -> reproducible comparisons and problem generators
    experiments    -> run records, history, event log, dataset, stats, export
    infrastructure -> event bus, kafka adapter, workers

## Domain layer

- `Problem`, `Resource`, `Job`, `Solution`, `Assignment` are immutable
  value objects with validation in `__post_init__` and pickle support via
  `__reduce__`.
- `evaluation.py` computes the objective (weighted sum of completion times
  plus makespan), collects hard-constraint violations, and applies the
  penalty model. Feasibility is `len(violations) == 0`.
- `change.py` and `impact.py` model dynamic problem changes and the
  structural / parametric diff between two problem versions.
- `events/` contains the transport-agnostic `DomainEvent` and the
  in-memory `InMemoryEventBus`.

## Algorithm layer

- `algorithms/genetic/` contains the GA: chromosome, operators, repair,
  metrics, termination, optimizer, and `state.py` / `stateful.py` for
  warm-start.
- `algorithms/differential_evolution/` mirrors the GA structure with a
  real-valued vector coding and DE/rand/1/bin operators.
- `algorithms/common/interface.py` defines a single `OptimizerResult` used
  by both GA and DE.
- `algorithms/random_search.py` is the deterministic baseline.

Both GA and DE share the same repair path (`repair.py`), the same
termination policy (`termination.py`), and the same evaluation entry point.

## Benchmark layer

- `problem_generator.py` creates reproducible random problems from a seed.
- `compare_ga_random.py`, `compare_ga_de.py`, `restart_vs_warmstart.py`,
  `sequential_vs_parallel.py` are the reproducible benchmarks.
- `runner.py` ties everything together: it produces `RunRecord` values,
  computes winners, and offers `verify_reproducibility`.

## Experiment layer

- `record.py` defines `GenerationMetric` and `RunRecord` with JSON round-trip.
- `history.py` stores completed runs in memory with filters.
- `event_log.py` is an append-only JSONL log of `DomainEvent`.
- `dataset.py` flattens runs and generations into analytical rows.
- `stats.py` computes summary, stability, and distribution statistics.
- `export.py` writes Parquet files when the optional `analysis` extra is
  installed.

## Infrastructure layer

- `event_bus/` contains transport protocols, the in-memory transport, topic
  names, and JSON serialization.
- `kafka/` contains the Kafka adapter: retry policy, producer, consumer,
  config, and a factory that lazy-imports `kafka-python`.
- `workers/` contains the fitness job model and the local parallel executor
  (`ProcessPoolExecutor` by default, thread option for debugging).

## Cross-cutting properties

- Reproducibility: every optimizer accepts a `seed`, and RNG state is
  captured in `GAState`.
- Idempotency: the in-memory bus and the Kafka consumer both drop duplicate
  `event_id` values per subscription / group.
- Type safety: the codebase is fully typed under `mypy --strict`.
- Optional dependencies: `kafka` and `analysis` extras are not required for
  tests or CI.
