# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 5 - Dynamic Optimization (completed).
Ready to start Phase 6 (Stateful / Warm-Start Optimization).

## Completed

- Phase 0 foundation (packaging, pytest, ruff, mypy, CI, docs)
- Phase 1 static optimization core
- Phase 2 genetic algorithm
- Phase 3 search quality & constraint handling
- Phase 4 differential evolution + common OptimizerResult interface
- Phase 5 dynamic optimization:
  - Domain change events (add/remove job, deadline, weight, add/remove
    resource, capacity, speed)
  - apply_change / apply_changes with version bumping and no mutation
  - Structural diff via analyze_change producing ChangeImpact
  - Structural vs parametric change classification

Tests: 203 passing.

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

- Phase 6 (state, warm-start vs restart benchmark) not started
- No events, no Kafka, no parallel execution yet