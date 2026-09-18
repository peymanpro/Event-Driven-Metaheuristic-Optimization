# Event-Driven Metaheuristic Optimization

Dynamic resource allocation and scheduling via event-driven metaheuristics
with stateful / warm-start optimization.

## Problem

We solve a time-varying constrained optimization problem:

    min_x  f_t(x)
    s.t.   g_{i,t}(x) <= 0

where the objective `f_t` and the constraints `g_{i,t}` may change during
execution (jobs added/removed, deadlines and priorities updated, resources
rescaled). The central research question of this project is:

> When the optimization problem changes during execution, does warm-start
> adaptation recover a high-quality feasible solution faster than restarting
> the optimizer from scratch?

This claim must be supported by a reproducible benchmark shipped in this
repository.

## Algorithmic core

Only two metaheuristics are in scope:

- Genetic Algorithm (GA)
- Differential Evolution (DE)

## Architectural pillars

- Dynamic optimization (`f_t`, `g_{i,t}`)
- Stateful / warm-start optimization
- Event-driven architecture (domain events decoupled from transport)
- Parallel fitness evaluation (local first, distributed only if justified)
- Reproducible experiment and benchmark pipeline

Kafka is used strictly as infrastructure for event ingestion, job
distribution, result events and telemetry. Internal optimizer steps are not
turned into messages.

## Scope boundaries

Deliberately out of scope unless a concrete need is demonstrated:

- Additional metaheuristics (PSO, ACO, SA, ...)
- Kubernetes, Airflow, Snowflake, Databricks
- Cloud deployment and web dashboards
- Premature microservices and abstractions

## Status

Phase 0 - Foundation & Scope (packaging, tests, linting, type checking, CI).
See `ROADMAP.md` for the phase plan and `PROJECT-STATE.md` for the current
handoff snapshot.

## Development

    python -m venv .venv
    . .\.venv\Scripts\Activate.ps1
    python -m pip install -e ".[dev]"
    ruff check .
    mypy .
    pytest

## License

MIT (to be finalized).