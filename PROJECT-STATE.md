# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 3 - Search Quality & Constraint Handling (completed).
Ready to start Phase 4 (Differential Evolution).

## Completed

- Phase 0 foundation (packaging, pytest, ruff, mypy, CI, docs)
- Phase 1 static optimization core:
  - Problem, Resource, Job, Solution, Assignment
  - Objective (weighted completion + makespan)
  - Hard constraints (missing, unknown resource, release, precedence,
    deadline, capacity)
  - Penalty model with configurable weights
  - Deterministic Random Search baseline
- Phase 2 genetic algorithm:
  - Chromosome (value object), operators (init, tournament, uniform crossover,
    gaussian mutation), elitism, generation loop, seed-based reproducibility
- Phase 3 search quality & constraint handling:
  - Repair operators (release time, precedence via topological order)
  - Diversity metric, convergence span, stagnation length
  - Termination policy (max_generations, stagnation, convergence)
  - Repair and termination wired into GA optimizer
  - Reproducible random problem generator
  - GA vs Random Search comparison benchmark

Tests: 134 passing across domain, algorithms, benchmarks.

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

- Phase 4 (Differential Evolution) not started
- No events, no Kafka, no parallel execution yet
- Dynamic optimization (f_t) not implemented