# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 9 - Experiment Data Pipeline (completed).
Ready to start Phase 10 (Final Benchmark & Documentation).

## Completed

- Phase 0 foundation
- Phase 1 static optimization core
- Phase 2 genetic algorithm
- Phase 3 search quality & constraint handling
- Phase 4 differential evolution
- Phase 5 dynamic optimization
- Phase 6 stateful / warm-start + restart vs warm-start benchmark
- Phase 7 event-driven architecture (bus + Kafka adapter)
- Phase 8 parallel fitness evaluation + sequential vs parallel benchmark
- Phase 9 experiment data pipeline:
  - GenerationMetric and RunRecord (immutable, JSON round-trip)
  - RunHistory store with experiment/algorithm/problem filters
  - EventLog append-only JSONL
  - Analytical dataset (RunRow, GenerationRow, builders)
  - Parquet export (optional pyarrow extra)
  - Statistical analysis: summarize, convergence_distribution,
    runtime_distribution, stability_report

Tests: 327 passing.

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

- Phase 10 (benchmark runner, reproducibility, docs, ADRs) not started