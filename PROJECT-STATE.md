# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 8 - Parallel Fitness Evaluation (completed).
Ready to start Phase 9 (Experiment Data Pipeline).

## Completed

- Phase 0 foundation
- Phase 1 static optimization core
- Phase 2 genetic algorithm
- Phase 3 search quality & constraint handling
- Phase 4 differential evolution + common OptimizerResult
- Phase 5 dynamic optimization
- Phase 6 stateful / warm-start + restart vs warm-start benchmark
- Phase 7 event-driven architecture (DomainEvent, bus, transport, Kafka adapter)
- Phase 8 parallel fitness evaluation:
  - FitnessJob and FitnessResult value types
  - WorkerError / JobTimeoutError / JobExecutionError
  - WorkerConfig (process or thread, max_workers, timeout, retries)
  - evaluate_job (pure, picklable, error-capturing)
  - submit_all with retries, order preservation, and per-job timeouts
  - Pickle support for Resource / Job / Problem / Solution / Schedule
  - sequential_vs_parallel benchmark with TimingResult and speedup

Tests: 292 passing.

## Local commands

    . .\.venv\Scripts\Activate.ps1
    ruff check .
    mypy .
    pytest

## Conventions

- PowerShell-safe commands, no bash here-strings
- Files written via System.Text.UTF8Encoding($false) to avoid BOM
- Atomic commits, one milestone per commit
- After each milestone: git status, git log -1 --oneline, git push

## Open items

- Phase 9 (experiment data pipeline) not started
- Phase 10 (final benchmark + docs + ADRs) not started
- Kafka-based workers not added (no justified need yet)