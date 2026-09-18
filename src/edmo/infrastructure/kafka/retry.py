from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Retry policy for producer/consumer operations.

    ``max_attempts`` is the total number of attempts (1 = no retry).
    ``base_delay`` is the initial delay in seconds; each subsequent attempt
    multiplies the delay by ``backoff`` up to ``max_delay``.
    ``sleep`` is injectable for deterministic tests.
    """

    max_attempts: int = 3
    base_delay: float = 0.05
    backoff: float = 2.0
    max_delay: float = 1.0
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_delay < 0.0:
            raise ValueError("base_delay must be >= 0")
        if self.backoff <= 0.0:
            raise ValueError("backoff must be > 0")
        if self.max_delay < 0.0:
            raise ValueError("max_delay must be >= 0")

    def run(self, op: Callable[[], T]) -> T:
        """Execute ``op`` with retries; re-raise the last exception on failure."""
        delay = self.base_delay
        last_exc: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return op()
            except Exception as exc:
                last_exc = exc
                if attempt == self.max_attempts:
                    break
                self.sleep(min(delay, self.max_delay))
                delay = min(delay * self.backoff, self.max_delay)
        assert last_exc is not None
        raise last_exc
