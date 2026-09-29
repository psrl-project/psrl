"""Node-wide weighted admission for local sandbox runtimes."""

from __future__ import annotations

import asyncio
import functools
import heapq
import logging
import math
import os
import re
import time
from collections import deque
from dataclasses import asdict, dataclass

from psrl.sandbox.core import ResourceSpec

psrl_logger = logging.getLogger(__file__)

_BYTES_PER_MIB = 1024 * 1024
_UNLIMITED_BYTES = 1 << 60
_MEMORY_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kmgt]?)b?")
# Admit a short burst around a fragmented head request, then reserve releases
# so a steady stream of small sandboxes cannot starve it indefinitely.
_MAX_HEAD_BYPASSES = 8


def parse_memory_mb(value: str | int | None) -> int | None:
    """Parse a Docker memory value into whole MiB."""
    if value is None or value == "":
        return None
    if isinstance(value, int):
        return max(1, math.ceil(value / _BYTES_PER_MIB))
    match = _MEMORY_RE.fullmatch(str(value).strip().lower())
    if match is None:
        raise ValueError(f"Invalid sandbox memory limit: {value!r}.")
    amount = float(match.group(1))
    multiplier = {"": 1 / _BYTES_PER_MIB, "k": 1 / 1024, "m": 1, "g": 1024, "t": 1024 * 1024}
    return max(1, math.ceil(amount * multiplier[match.group(2)]))


def _read_int(path: str) -> int | None:
    try:
        with open(path) as handle:
            raw = handle.read().strip()
    except OSError:
        return None
    if not raw or raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_paths(controller: str, filename: str) -> list[str]:
    paths: list[str] = []
    try:
        with open("/proc/self/cgroup") as handle:
            lines = list(handle)
    except OSError:
        lines = []
    for line in lines:
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        group = parts[2].strip("/")
        if parts[0] == "0" and parts[1] == "":
            paths.append(f"/sys/fs/cgroup/{group}/{filename}")
        elif controller in parts[1].split(","):
            paths.append(f"/sys/fs/cgroup/{controller}/{group}/{filename}")
    return paths


@functools.lru_cache(maxsize=1)
def detect_node_memory_mb() -> int | None:
    """Return the local cgroup or machine memory limit in MiB."""
    paths = _cgroup_paths("memory", "memory.max") + _cgroup_paths("memory", "memory.limit_in_bytes")
    paths += ["/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"]
    for path in paths:
        value = _read_int(path)
        if value is not None and 0 < value < _UNLIMITED_BYTES:
            return value // _BYTES_PER_MIB
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


@functools.lru_cache(maxsize=1)
def detect_node_cpu_cores() -> float | None:
    """Return the local cgroup or CPU-affinity capacity in cores."""
    candidates: list[float] = []
    try:
        candidates.append(float(len(os.sched_getaffinity(0))))
    except AttributeError:
        if os.cpu_count():
            candidates.append(float(os.cpu_count()))

    for path in _cgroup_paths("cpu", "cpu.max") + ["/sys/fs/cgroup/cpu.max"]:
        try:
            with open(path) as handle:
                quota, period = handle.read().split()[:2]
            if quota != "max":
                candidates.append(int(quota) / int(period))
        except (OSError, ValueError, IndexError):
            continue

    quota_paths = _cgroup_paths("cpu", "cpu.cfs_quota_us") + ["/sys/fs/cgroup/cpu/cpu.cfs_quota_us"]
    period_paths = _cgroup_paths("cpu", "cpu.cfs_period_us") + ["/sys/fs/cgroup/cpu/cpu.cfs_period_us"]
    for quota_path, period_path in zip(quota_paths, period_paths, strict=False):
        quota = _read_int(quota_path)
        period = _read_int(period_path)
        if quota is not None and period is not None and quota > 0 and period > 0:
            candidates.append(quota / period)
    return min(candidates) if candidates else None


@dataclass(frozen=True)
class SandboxCapacityConfig:
    """One node resource envelope shared by all local sandboxes.

    Unset memory or CPU selects node-local auto-detection. ``utilization`` is the
    only safety knob and applies to both resources.
    """

    memory_mb: int | None = None
    cpu_cores: float | None = None
    utilization: float = 0.5
    lease_ttl_s: float = 180
    heartbeat_interval_s: float = 30

    def __post_init__(self) -> None:
        if self.memory_mb is not None and self.memory_mb <= 0:
            raise ValueError("Sandbox capacity memory_mb must be greater than zero when configured.")
        if self.cpu_cores is not None and self.cpu_cores <= 0:
            raise ValueError("Sandbox capacity cpu_cores must be greater than zero when configured.")
        if not 0 < self.utilization <= 1:
            raise ValueError("Sandbox capacity utilization must be in (0, 1].")
        if self.lease_ttl_s <= 0 or not 0 < self.heartbeat_interval_s < self.lease_ttl_s:
            raise ValueError("Sandbox capacity heartbeat must be positive and shorter than lease_ttl_s.")


@dataclass(frozen=True)
class ResourceQuantity:
    """Integer resource quantity used for exact accounting."""

    memory_mb: int
    cpu_millis: int

    @classmethod
    def from_spec(cls, resources: ResourceSpec) -> ResourceQuantity:
        """Create a quantity from a complete sandbox resource request."""
        if resources.memory_mb is None or resources.cpu_count is None:
            raise ValueError("Node-capacity admission requires sandbox memory_mb and cpu_count.")
        return cls(resources.memory_mb, math.ceil(resources.cpu_count * 1000))

    def fits(self, available_capacity: ResourceQuantity) -> bool:
        return self.memory_mb <= available_capacity.memory_mb and self.cpu_millis <= available_capacity.cpu_millis

    def __add__(self, other: ResourceQuantity) -> ResourceQuantity:
        return ResourceQuantity(self.memory_mb + other.memory_mb, self.cpu_millis + other.cpu_millis)

    def __sub__(self, other: ResourceQuantity) -> ResourceQuantity:
        return ResourceQuantity(self.memory_mb - other.memory_mb, self.cpu_millis - other.cpu_millis)


@dataclass(frozen=True)
class ResolvedSandboxCapacity:
    """Resolved node envelope and its provenance."""

    resources: ResourceQuantity
    memory_source: str
    cpu_source: str


def resolve_sandbox_capacity(
    config: SandboxCapacityConfig,
    *,
    detected_memory_mb: int | None = None,
    detected_cpu_cores: float | None = None,
) -> ResolvedSandboxCapacity:
    """Resolve and scale the node envelope once on the target node."""
    memory_mb = config.memory_mb or detected_memory_mb or detect_node_memory_mb()
    cpu_cores = config.cpu_cores or detected_cpu_cores or detect_node_cpu_cores()
    if not memory_mb or not cpu_cores:
        raise RuntimeError(
            "Sandbox node capacity could not be detected; configure memory_mb and cpu_cores explicitly."
        )
    return ResolvedSandboxCapacity(
        resources=ResourceQuantity(
            memory_mb=max(1, math.floor(memory_mb * config.utilization)),
            cpu_millis=max(1, math.floor(cpu_cores * 1000 * config.utilization)),
        ),
        memory_source="configured" if config.memory_mb is not None else "detected",
        cpu_source="configured" if config.cpu_cores is not None else "detected",
    )


@dataclass
class _Allocation:
    owner_id: str
    resources: ResourceQuantity


@dataclass
class _Waiter:
    lease_id: str
    owner_id: str
    resources: ResourceQuantity
    future: asyncio.Future[None]
    queued_at: float
    dominant_share: float
    bypasses: int = 0


class SandboxCapacityCoordinator:
    """Single-node dominant-share admission coordinator with bounded starvation."""

    def __init__(self, config: SandboxCapacityConfig | dict) -> None:
        if isinstance(config, dict):
            config = SandboxCapacityConfig(**config)
        self._config = config
        self._capacity = resolve_sandbox_capacity(config)
        self.available_capacity = self._capacity.resources
        self._allocations: dict[str, _Allocation] = {}
        self._owners: dict[str, float] = {}
        self._waiters: dict[str, _Waiter] = {}
        self._waiter_heap: list[tuple[float, float, str]] = []
        self._waiter_memory_heap: list[tuple[int, str]] = []
        self._waiter_cpu_heap: list[tuple[int, str]] = []
        self._waiter_order: deque[str] = deque()
        self._lock = asyncio.Lock()
        self._sweeper: asyncio.Task[None] | None = None
        self._granted = 0
        self._released = 0
        self._expired = 0
        self._total_wait_s = 0.0
        psrl_logger.info(
            f"Sandbox node envelope: memory_mb={self.available_capacity.memory_mb}, "
            f"cpu_cores={self.available_capacity.cpu_millis / 1000:g}, "
            f"sources={self._capacity.memory_source!r}/{self._capacity.cpu_source!r}."
        )

    def _ensure_sweeper(self) -> None:
        if self._sweeper is None:
            self._sweeper = asyncio.create_task(self._sweep_expired())

    async def acquire(self, lease_id: str, owner_id: str, memory_mb: int, cpu_count: float) -> None:
        """Wait until one complete resource vector can be admitted."""
        self._ensure_sweeper()
        resources = ResourceQuantity(memory_mb, math.ceil(cpu_count * 1000))
        if not resources.fits(self._capacity.resources):
            raise ValueError(
                f"Sandbox request memory_mb={memory_mb}, cpu_count={cpu_count:g} exceeds node envelope "
                f"memory_mb={self._capacity.resources.memory_mb}, "
                f"cpu_cores={self._capacity.resources.cpu_millis / 1000:g}."
            )
        loop = asyncio.get_running_loop()
        waiter = _Waiter(
            lease_id,
            owner_id,
            resources,
            loop.create_future(),
            time.monotonic(),
            max(
                resources.memory_mb / self._capacity.resources.memory_mb,
                resources.cpu_millis / self._capacity.resources.cpu_millis,
            ),
        )
        async with self._lock:
            if lease_id in self._allocations or lease_id in self._waiters:
                raise ValueError(f"Sandbox capacity lease {lease_id!r} already exists.")
            self._owners[owner_id] = time.monotonic() + self._config.lease_ttl_s
            self._waiters[lease_id] = waiter
            self._waiter_order.append(lease_id)
            heapq.heappush(self._waiter_heap, (waiter.dominant_share, waiter.queued_at, lease_id))
            heapq.heappush(self._waiter_memory_heap, (waiter.resources.memory_mb, lease_id))
            heapq.heappush(self._waiter_cpu_heap, (waiter.resources.cpu_millis, lease_id))
            if waiter.resources.fits(self.available_capacity):
                self._drain_waiters()
        try:
            await waiter.future
        except asyncio.CancelledError:
            async with self._lock:
                if lease_id in self._waiters:
                    self._waiters.pop(lease_id)
                elif lease_id in self._allocations:
                    self._release_locked(lease_id)
                self._drain_waiters()
            raise

    async def release(self, lease_id: str) -> None:
        """Release one admitted resource vector."""
        async with self._lock:
            if self._release_locked(lease_id):
                self._released += 1
                self._drain_waiters()

    async def cancel(self, lease_id: str) -> None:
        """Remove a canceled request whether it is queued or already admitted."""
        async with self._lock:
            waiter = self._waiters.pop(lease_id, None)
            if waiter is not None:
                waiter.future.cancel()
            if self._release_locked(lease_id):
                self._released += 1
            self._drain_waiters()

    async def renew_owner(self, owner_id: str) -> None:
        """Renew every active lease belonging to one worker."""
        async with self._lock:
            if owner_id in self._owners:
                self._owners[owner_id] = time.monotonic() + self._config.lease_ttl_s

    async def release_owner(self, owner_id: str) -> None:
        """Release all capacity held by a worker that is shutting down."""
        async with self._lock:
            self._owners.pop(owner_id, None)
            lease_ids = [lease_id for lease_id, item in self._allocations.items() if item.owner_id == owner_id]
            for lease_id in lease_ids:
                self._release_locked(lease_id)
            owner_waiters = [waiter for waiter in self._waiters.values() if waiter.owner_id == owner_id]
            for waiter in owner_waiters:
                self._waiters.pop(waiter.lease_id)
                waiter.future.cancel()
            self._released += len(lease_ids)
            self._drain_waiters()

    async def snapshot(self) -> dict:
        """Return a compact accounting and queue snapshot."""
        async with self._lock:
            return {
                "capacity": asdict(self._capacity.resources),
                "available_capacity": asdict(self.available_capacity),
                "allocations": len(self._allocations),
                "owners": len(self._owners),
                "waiters": len(self._waiters),
                "granted": self._granted,
                "released": self._released,
                "expired": self._expired,
                "mean_wait_s": self._total_wait_s / self._granted if self._granted else 0.0,
                "memory_source": self._capacity.memory_source,
                "cpu_source": self._capacity.cpu_source,
            }

    async def shutdown(self) -> None:
        """Stop background expiry work."""
        if self._sweeper is not None:
            self._sweeper.cancel()
            await asyncio.gather(self._sweeper, return_exceptions=True)

    def _drain_waiters(self) -> None:
        self._compact_waiter_indexes()
        while self._waiters:
            if not self._could_fit_any_waiter():
                break
            head = self._head_waiter()
            if head is None:
                break
            if head.bypasses >= _MAX_HEAD_BYPASSES:
                waiter = head if head.resources.fits(self.available_capacity) else None
            else:
                waiter = self._pop_weighted_fit()
            if waiter is None:
                break
            if waiter is not head:
                head.bypasses += 1
            self._waiters.pop(waiter.lease_id)
            self.available_capacity -= waiter.resources
            self._allocations[waiter.lease_id] = _Allocation(
                owner_id=waiter.owner_id,
                resources=waiter.resources,
            )
            self._granted += 1
            self._total_wait_s += time.monotonic() - waiter.queued_at
            waiter.future.set_result(None)

    def _head_waiter(self) -> _Waiter | None:
        while self._waiter_order and self._waiter_order[0] not in self._waiters:
            self._waiter_order.popleft()
        return self._waiters.get(self._waiter_order[0]) if self._waiter_order else None

    def _pop_weighted_fit(self) -> _Waiter | None:
        blocked: list[tuple[float, float, str]] = []
        selected = None
        while self._waiter_heap:
            item = heapq.heappop(self._waiter_heap)
            waiter = self._waiters.get(item[2])
            if waiter is None:
                continue
            if waiter.resources.fits(self.available_capacity):
                selected = waiter
                break
            blocked.append(item)
        for item in blocked:
            heapq.heappush(self._waiter_heap, item)
        return selected

    def _compact_waiter_indexes(self) -> None:
        limit = 2 * len(self._waiters) + 1024
        index_sizes = (
            len(self._waiter_heap),
            len(self._waiter_memory_heap),
            len(self._waiter_cpu_heap),
            len(self._waiter_order),
        )
        if all(size <= limit for size in index_sizes):
            return
        self._waiter_order = deque(lease_id for lease_id in self._waiter_order if lease_id in self._waiters)
        self._waiter_heap = [
            (waiter.dominant_share, waiter.queued_at, waiter.lease_id) for waiter in self._waiters.values()
        ]
        self._waiter_memory_heap = [(waiter.resources.memory_mb, waiter.lease_id) for waiter in self._waiters.values()]
        self._waiter_cpu_heap = [(waiter.resources.cpu_millis, waiter.lease_id) for waiter in self._waiters.values()]
        heapq.heapify(self._waiter_heap)
        heapq.heapify(self._waiter_memory_heap)
        heapq.heapify(self._waiter_cpu_heap)

    def _could_fit_any_waiter(self) -> bool:
        for heap in (self._waiter_memory_heap, self._waiter_cpu_heap):
            while heap and heap[0][1] not in self._waiters:
                heapq.heappop(heap)
        return bool(
            self._waiter_memory_heap
            and self._waiter_cpu_heap
            and self._waiter_memory_heap[0][0] <= self.available_capacity.memory_mb
            and self._waiter_cpu_heap[0][0] <= self.available_capacity.cpu_millis
        )

    def _release_locked(self, lease_id: str) -> bool:
        allocation = self._allocations.pop(lease_id, None)
        if allocation is None:
            return False
        self.available_capacity += allocation.resources
        return True

    async def _sweep_expired(self) -> None:
        interval = min(self._config.heartbeat_interval_s, self._config.lease_ttl_s / 2)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            async with self._lock:
                expired_owners = [owner_id for owner_id, expires_at in self._owners.items() if expires_at <= now]
                for owner_id in expired_owners:
                    self._owners.pop(owner_id)
                expired_leases = [
                    lease_id for lease_id, item in self._allocations.items() if item.owner_id in expired_owners
                ]
                for lease_id in expired_leases:
                    self._release_locked(lease_id)
                expired_waiters = [waiter for waiter in self._waiters.values() if waiter.owner_id in expired_owners]
                for waiter in expired_waiters:
                    self._waiters.pop(waiter.lease_id)
                    waiter.future.cancel()
                self._expired += len(expired_leases)
                if expired_leases or expired_waiters:
                    self._drain_waiters()
