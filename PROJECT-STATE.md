# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Review pass 2 (post-review hardening). Both remaining issues fixed and
verified with regression tests. Project status: COMPLETE.

## Review pass 2 fixes

1. Warm-start recovery history is now local to the current run.
   `GAState.history` remains cumulative for persistence/audit;
   `GAResult.history` is local: index 0 is the post-change initial
   evaluation, index k is after the k-th new evolution step. Target checks
   and termination policies operate on the local history, so a warm start
   cannot be falsely terminated by pre-change history. Recovery metrics
   (`initial_total`, `best_total`, `iterations_to_target`,
   `iterations_saved`, `warm_start_initial_advantage`) are computed only
   from the local post-change run.
2. `KafkaEventConsumer` no longer marks events as seen inside `poll`.
   `poll()` returns the next event unmarked; `mark_processed(event)` must
   be called after handler success. `run_consumer` does this in the correct
   order, so a failed handler leaves the event unmarked and a redelivery is
   returned again. Successful processing dedups later redeliveries of the
   same `event_id` for the lifetime of the consumer instance.

## Earlier review fixes (still in place)

- Feasible scheduling problem generation with meaningful capacities
- GA generation accounting across warm starts
- Real `Evaluation` propagation and real generation telemetry
- End-to-end event-driven coordinator
- `PollingSubscriber` / `PushSubscriber` protocol split
- Kafka at-least-once documented without exactly-once overclaim
- Repeated-seed research benchmark
- Canonical scheduling scenario

## Local commands

    . .\.venv\Scripts\Activate.ps1
    ruff check .
    mypy .
    pytest

## Answer to the central research question

The empirical answer comes from `repeated_recovery_benchmark`, which runs
restart and warm-start across many seeds and reports per-strategy
aggregation. The benchmark is descriptive: the project does not claim
"warm start always wins".

## Open items

- None required for completion.
