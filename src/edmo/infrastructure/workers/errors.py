from __future__ import annotations


class WorkerError(RuntimeError):
    """Base error for worker execution failures."""


class JobTimeoutError(WorkerError):
    """A fitness job did not complete within its timeout budget."""


class JobExecutionError(WorkerError):
    """A fitness job raised an exception while running."""
