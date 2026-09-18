# PROJECT-STATE

Handoff snapshot. Update at the end of every milestone.

## Identity

- Repository: https://github.com/peymanpro/Event-Driven-Metaheuristic-Optimization
- Local path: E:\git-public-projects\Event-Driven-Metaheuristic-Optimization
- Branch: main
- Python: 3.12
- Virtual environment: .venv (activated per session)

## Current phase

Phase 0 - Foundation & Scope.

## Completed

- 0.1 Repository cloned and inspected (started from empty remote)
- 0.2 Project structure created (src/edmo, tests)
- 0.3 Python packaging via pyproject.toml (hatchling, src-layout)
- 0.4 Pytest configured, smoke test passing
- 0.5 Ruff configured and clean
- 0.6 Mypy configured (strict) and clean
- 0.7 CI workflow at .github/workflows/ci.yml
- 0.8 README written
- 0.9 ROADMAP written

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

- Phase 1 has not started
- No Kafka, no events, no algorithms implemented yet