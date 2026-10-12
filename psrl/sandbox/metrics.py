"""Low-overhead lifecycle and operation metrics for sandbox backends."""

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OperationMetrics:
    """Aggregated latency and failure counters for one operation."""

    count: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0

    @property
    def mean_seconds(self) -> float:
        """Return mean latency, or zero when no samples exist."""
        return self.total_seconds / self.count if self.count else 0.0


@dataclass(frozen=True)
class SandboxMetricsSnapshot:
    """Immutable backend metrics suitable for logging or assertions."""

    operations: dict[str, OperationMetrics] = field(default_factory=dict)
    active_sessions: int = 0
    peak_active_sessions: int = 0
    current_memory_bytes: int = 0
    peak_memory_bytes: int = 0

    def as_dict(self) -> dict[str, object]:
        """Serialize metrics including computed operation means."""
        return {
            "operations": {
                name: {
                    "count": metric.count,
                    "failures": metric.failures,
                    "total_seconds": metric.total_seconds,
                    "mean_seconds": metric.mean_seconds,
                    "max_seconds": metric.max_seconds,
                }
                for name, metric in self.operations.items()
            },
            "active_sessions": self.active_sessions,
            "peak_active_sessions": self.peak_active_sessions,
            "current_memory_bytes": self.current_memory_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
        }


@dataclass
class _MutableOperationMetrics:
    count: int = 0
    failures: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0


class SandboxMetrics:
    """Thread-safe in-process metrics with constant work per operation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._operations: dict[str, _MutableOperationMetrics] = {}
        self._active_sessions = 0
        self._peak_active_sessions = 0
        self._current_memory_bytes = 0
        self._peak_memory_bytes = 0

    @contextmanager
    def measure(self, operation: str) -> Iterator[None]:
        """Record latency and failure state around one operation."""
        started_at = time.perf_counter()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            self.record(operation, time.perf_counter() - started_at, failed=failed)

    def record(self, operation: str, duration_s: float, *, failed: bool = False) -> None:
        """Record one completed operation."""
        with self._lock:
            metric = self._operations.setdefault(operation, _MutableOperationMetrics())
            metric.count += 1
            metric.failures += int(failed)
            metric.total_seconds += duration_s
            metric.max_seconds = max(metric.max_seconds, duration_s)

    def session_started(self) -> None:
        """Increment active and peak session counters."""
        with self._lock:
            self._active_sessions += 1
            self._peak_active_sessions = max(self._peak_active_sessions, self._active_sessions)

    def session_stopped(self) -> None:
        """Decrement the active session counter without allowing underflow."""
        with self._lock:
            self._active_sessions = max(0, self._active_sessions - 1)

    def observe_memory(self, current_bytes: int, peak_bytes: int | None = None) -> None:
        """Record current and peak sandbox memory usage."""
        with self._lock:
            self._current_memory_bytes = max(0, current_bytes)
            observed_peak = current_bytes if peak_bytes is None else peak_bytes
            self._peak_memory_bytes = max(self._peak_memory_bytes, observed_peak)

    def snapshot(self) -> SandboxMetricsSnapshot:
        """Return a point-in-time immutable copy."""
        with self._lock:
            operations = {
                name: OperationMetrics(
                    count=metric.count,
                    failures=metric.failures,
                    total_seconds=metric.total_seconds,
                    max_seconds=metric.max_seconds,
                )
                for name, metric in self._operations.items()
            }
            return SandboxMetricsSnapshot(
                operations=operations,
                active_sessions=self._active_sessions,
                peak_active_sessions=self._peak_active_sessions,
                current_memory_bytes=self._current_memory_bytes,
                peak_memory_bytes=self._peak_memory_bytes,
            )
