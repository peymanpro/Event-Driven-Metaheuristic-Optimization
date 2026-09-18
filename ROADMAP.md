# ROADMAP

Phases are executed strictly in order. Each phase must reach a milestone with
green tests, lint, and type checks before the next phase begins.

## Phase 0 - Foundation & Scope

- 0.1 Inspect local repository and remote
- 0.2 Initialize project structure
- 0.3 Python packaging
- 0.4 Pytest
- 0.5 Ruff
- 0.6 Mypy
- 0.7 CI
- 0.8 README
- 0.9 ROADMAP
- 0.10 PROJECT-STATE / HANDOFF

## Phase 1 - Static Optimization Core

Problem model, decision variables, resources, jobs, objective, hard and soft
constraints, feasibility, penalty model, random-search baseline, tests.
No Kafka in this phase.

## Phase 2 - Genetic Algorithm

Solution representation, population init, fitness evaluation, tournament
selection, crossover, mutation, elitism, generation loop, best tracking,
seed/reproducibility, tests.

## Phase 3 - Search Quality & Constraint Handling

Hard-constraint handling, penalty strategy, repair operators, diversity
metric, convergence and stagnation detection, termination policies,
benchmark against Random Search.

## Phase 4 - Differential Evolution

Vector representation, mutation, crossover, selection, constraint handling,
convergence, common optimizer interface, GA vs DE benchmark.

## Phase 5 - Dynamic Optimization

From `f(x)` to `f_t(x)` and `g_i(x)` to `g_{i,t}(x)`: objective changes,
constraint changes, resource changes, job add/remove, deadline and priority
changes, problem versioning, change impact analysis.

## Phase 6 - Stateful / Warm-Start Optimization

Optimizer state (population, best, generation, metrics, random state, problem
version), state transition, population repair, partial re-init, stagnation
recovery, restart strategy, warm-start strategy, and the restart-vs-warm-start
benchmark.

## Phase 7 - Event-Driven Architecture

Domain event model, validation, handlers, in-memory bus, Kafka adapter,
producer, consumer, topics, consumer groups, idempotency, error and retry
policy.

## Phase 8 - Parallel Fitness Evaluation

Fitness job model, local process-based execution, parallel evaluation, result
aggregation, failure handling, timeout, retry policy, Kafka-based workers
(only if justified), sequential vs parallel benchmark.

## Phase 9 - Experiment Data Pipeline

Experiment record model, run history, generation metrics, event persistence,
analytical dataset, export to Parquet, batch analysis, statistical analysis.

## Phase 10 - Final Benchmark & Documentation

Static benchmark, dynamic benchmark, GA vs DE, restart vs warm-start,
sequential vs parallel, reproducibility verification, architecture docs,
ADRs, failure modes, final README.