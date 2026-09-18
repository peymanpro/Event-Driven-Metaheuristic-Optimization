# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 4 - Differential Evolution (completed).
Ready to start Phase 5 (Dynamic Optimization: f_t, g_{i,t}).

## Completed

- Phase 0 foundation (packaging, pytest, ruff, mypy, CI, docs)
- Phase 1 static optimization core (problem, solution, evaluation, penalty,
  random search baseline)
- Phase 2 genetic algorithm (chromosome, operators, elitism, seed-based
  reproducibility)
- Phase 3 search quality & constraint handling (repair operators, diversity,
  convergence, stagnation, termination policies, GA vs Random Search benchmark)
- Phase 4 differential evolution:
  - Vector coding with VectorBounds and decode/clip
  - DE/rand/1/bin operators
  - Optimizer with repair and termination reuse
  - Common OptimizerResult interface shared by GA/DE
  - Reproducible GA vs DE benchmark

Tests: 173 passing.

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

- Phase 5 (Dynamic optimization f_t) not started
- No events, no Kafka, no parallel execution yet
- No warm-start or restart strategies yet