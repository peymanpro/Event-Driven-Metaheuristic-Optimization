# Event-Driven Metaheuristic Optimization

Dynamic resource allocation and scheduling via event-driven metaheuristics
(Genetic Algorithm + Differential Evolution) with stateful / warm-start
optimization.

## Problem

We solve a time-varying constrained optimization problem:

    min_x  f_t(x)
    s.t.   g_{i,t}(x) <= 0

where the objective f_t and the constraints g_{i,t} may change during
execution (jobs added/removed, deadlines and priorities updated, resources
rescaled). The central research question is:

> When the optimization problem changes during execution, does warm-start
> adaptation recover a high-quality feasible solution faster than restarting
> the optimizer from scratch?

The claim is supported by a reproducible benchmark in
`src/edmo/benchmarks/restart_vs_warmstart.py`.

## Algorithmic core

Only two metaheuristics are in scope:

- Genetic Algorithm (GA), mixed-integer chromosome, per-gene uniform crossover
  and Gaussian mutation.
- Differential Evolution (DE/rand/1/bin) over a real-valued vector with a
  decoded discrete projection.

Both share a single evaluation entry point, a single repair path, and a
single termination policy. A deterministic Random Search is used as the
baseline.

## Architecture

    src/edmo/
      domain/            problem, solution, evaluation, change, events
      algorithms/        genetic, differential_evolution, random_search, common
      benchmarks/        problem_generator, compare_*, restart_vs_warmstart,
                         sequential_vs_parallel, runner
      experiments/       record, history, event_log, dataset, stats, export
      infrastructure/    event_bus, kafka, workers

Detailed description in `docs/ARCHITECTURE.md`.

## Reproducible benchmarks

- `compare_ga_vs_random` - GA vs Random Search on random problems.
- `compare_ga_vs_de` - GA vs DE on random problems.
- `restart_vs_warm_start` - recovery after a dynamic problem change.
- `sequential_vs_parallel` - fitness evaluation throughput and speedup.
- `runner.run_static_benchmark` - GA, DE, Random Search on a static problem.
- `runner.run_dynamic_benchmark` - restart vs warm-start with records.
- `runner.verify_reproducibility` - identical history for identical seed.

Every benchmark accepts a seed and is deterministic given that seed.

## Event-driven architecture

Domain events use a stable envelope (`event_id`, `event_type`, `timestamp`,
`aggregate_id`, `version`, `payload`). The bus and transport protocols are
transport-agnostic. Kafka is used as durable infrastructure for two topics
(`optimization-events`, `optimization-results`). Internal optimizer steps are
never turned into messages. See ADR-0001 and ADR-0002.

## Parallel fitness evaluation

Fitness jobs are submitted to a local process pool by default, with a thread
backend for debugging. Per-job timeouts and retries are supported. Kafka-based
workers are intentionally not implemented until a measured bottleneck
justifies them (see ADR-0006 and the `sequential_vs_parallel` benchmark).

## Experiments

Runs are captured as immutable `RunRecord` values with per-generation
`GenerationMetric` rows. Run history and event logs are append-only. When the
optional `analysis` extra is installed, datasets can be exported to Parquet
and summarized with `stats.summarize`, `stats.stability_report`, and
`stats.convergence_distribution`.

## Development

    python -m venv .venv
    . .\.venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"
    ruff check .
    mypy .
    pytest

Optional extras:

    python -m pip install -e ".[kafka]"      # enable Kafka adapter
    python -m pip install -e ".[analysis]"   # enable Parquet export

## Scope boundaries

Deliberately out of scope unless a concrete need is demonstrated:

- Additional metaheuristics (PSO, ACO, SA, ...)
- Kubernetes, Airflow, Snowflake, Databricks
- Cloud deployment and web dashboards
- Premature microservices and abstractions

## Documentation

- `ROADMAP.md` - phase plan.
- `PROJECT-STATE.md` - current handoff snapshot.
- `docs/ARCHITECTURE.md` - layered architecture.
- `docs/FAILURE-MODES.md` - failure modes and handling.
- `docs/adr/` - architecture decision records.

## License

MIT (to be finalized).
