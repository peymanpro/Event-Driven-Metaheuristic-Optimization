# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Review & hardening pass (post-phase-10). All issues from the independent
review are fixed and verified. Project status: COMPLETE.

## Review fixes (in order)

1. Problem generator produces feasible instances with meaningful capacity
   constraints; `known_feasible_solution` proves feasibility with a canonical
   round-robin schedule; tested across 20 seeds.
2. `GAState.generation` semantics are explicit: the number of the last
   completed generation, incremented by exactly the number of new evolution
   steps across warm starts. No double counting; warm start does not
   re-record the inherited population in history.
3. `RecoveryMetrics` and `RunRecord` carry the real `Evaluation` produced by
   the optimizer. No synthesized Evaluation objects remain on the core path.
4. GA records real per-generation telemetry: best, mean, worst totals,
   population diversity, elapsed seconds. Invariant
   `best <= mean <= worst` is enforced.
5. `OptimizationCoordinator` wires dynamic events end-to-end:
   `ProblemChangeRequested` -> apply change -> adapt state -> warm start
   optimization -> `OptimizationCompleted`. In-memory path is fully tested.
6. `EventSubscriber` is split into `PollingSubscriber` (Kafka) and
   `PushSubscriber` (in-memory bus). Runtime protocol tests verify each
   implementation matches its contract.
7. Documentation states Kafka semantics accurately: at-least-once transport,
   in-process duplicate filtering only, no exactly-once claim.
8. `repeated_recovery_benchmark` aggregates restart vs warm-start across
   multiple seeds with success rate, median/mean/stdev iterations-to-target,
   best/worst recovery, and feasible fraction.
9. `canonical_resource_scheduling` provides a hand-readable benchmark
   scenario with a proven feasible solution and a dynamic change set.

Tests: 378 passing.

## Local commands

    . .\.venv\Scripts\Activate.ps1
    ruff check .
    mypy .
    pytest

## Answer to the central research question

The empirical answer comes from `repeated_recovery_benchmark`, which runs
restart and warm-start across many seeds and reports per-strategy statistics.
The benchmark is descriptive, not prescriptive: the README does not claim
"warm start always wins". Interpretation is left to whoever reads the
aggregated metrics.

## Open items

- None required for completion. Optional future work remains documented in
  ROADMAP.md and the ADRs.
