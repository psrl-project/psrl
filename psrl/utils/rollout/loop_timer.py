"""Lightweight per-trajectory wall-clock timing accumulator for agent loops."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


class LoopTimer:
    """Accumulates wall-clock time spent in generation and environment steps.

    One instance is created per agent-loop instance (which is itself created per
    request), so the accumulated values describe a single trajectory. The context
    managers are cheap and allocation-free, safe to call on the rollout hot path.
    """

    def __init__(self) -> None:
        self.generation_s: float = 0.0
        self.env_s: float = 0.0
        self._t0: float = time.time()

    @contextmanager
    def generation(self) -> Iterator[None]:
        """Accumulate time spent generating model tokens / chat completions."""
        start = time.time()
        try:
            yield
        finally:
            self.generation_s += time.time() - start

    @contextmanager
    def env(self) -> Iterator[None]:
        """Accumulate time spent executing environment / tool steps."""
        start = time.time()
        try:
            yield
        finally:
            self.env_s += time.time() - start

    def elapsed(self) -> float:
        """Total wall-clock time since this timer was created."""
        return time.time() - self._t0

    def as_dict(self) -> dict[str, float]:
        """Return accumulated timing as a plain dict for serialization."""
        return {
            "generation_s": self.generation_s,
            "env_s": self.env_s,
            "elapsed_s": self.elapsed(),
        }
