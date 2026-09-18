# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 7 - Event-Driven Architecture (completed).
Ready to start Phase 8 (Parallel Fitness Evaluation, local first).

## Completed

- Phase 0 foundation
- Phase 1 static optimization core
- Phase 2 genetic algorithm
- Phase 3 search quality & constraint handling
- Phase 4 differential evolution + common OptimizerResult
- Phase 5 dynamic optimization
- Phase 6 stateful / warm-start + restart vs warm-start benchmark
- Phase 7 event-driven architecture:
  - DomainEvent envelope (UUID, timestamp, version, payload)
  - InMemoryEventBus with idempotency per event_id
  - EventPublisher / EventSubscriber protocols
  - Topics module (optimization-events, optimization-results)
  - JSON serialization (encode_event / decode_event)
  - InMemoryTransport with fan-out and per-subscriber dedup
  - Kafka adapter: RetryPolicy, KafkaEventPublisher, KafkaEventConsumer
  - KafkaConfig + factory with lazy import of kafka-python
  - Optional `kafka` extra in pyproject.toml

Tests: 278 passing.

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

- Phase 8 (parallel fitness evaluation) not started
- Phase 9 (experiment data pipeline) not started
- Phase 10 (final benchmark + docs) not started