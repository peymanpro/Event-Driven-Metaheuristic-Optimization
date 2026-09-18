# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 1 - Static Optimization Core (completed). Ready to start Phase 2.

## Completed

- 0.1..0.10 Phase 0 foundation (packaging, pytest, ruff, mypy, CI, docs)
- 1.1 Problem model with validation and lookups
- 1.2 Decision variables (Solution, Assignment)
- 1.3 Resource model
- 1.4 Job model (release, deadline, weight, predecessors, demand)
- 1.5 Objective (weighted sum of completion times + makespan)
- 1.6 Hard constraints (missing assignment, unknown resource, release time,
      precedence, deadline, capacity per dimension)
- 1.7 Soft constraints (penalty weights per violation kind, default weights)
- 1.8 Feasibility evaluation (Evaluation.feasible = no violations)
- 1.9 Penalty model (weighted sum of violation magnitudes)
- 1.10 Random search baseline (deterministic with seed)
- 1.11 Tests: 60 passing across domain and algorithms

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

- Phase 2 (Genetic Algorithm) not started
- No events, no Kafka, no parallel execution yet