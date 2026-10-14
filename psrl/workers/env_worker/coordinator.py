"""Route sandbox requests without owning container or placement mechanics."""

from __future__ import annotations

import asyncio
import itertools
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

from psrl.workers.env_worker.sandbox import ExecResult, SandboxSpec

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# NOTE(claude): `_ROUND_ROBIN_COUNTER` is shared by coordinators in one process, so
# heterogeneous pools index each eligible subset independently.
_ROUND_ROBIN_COUNTER = itertools.count()
_SELECT_POLL_INTERVAL_S = 0.5


@dataclass
class WorkerSlot:
    """Capacity bookkeeping for one registered `EnvWorker`."""

    worker_id: int
    handle: Any
    cpu_slots: int
    gpu_slots: int
    node_ip: str
    used_cpu: int = 0
    used_gpu: int = 0

    @property
    def has_cpu_capacity(self) -> bool:
        return self.used_cpu < self.cpu_slots

    def has_gpu_capacity(self, requested: int) -> bool:
        return self.used_gpu + requested <= self.gpu_slots


def select_by_method(
    method: str,
    candidates: list[WorkerSlot],
    spec: SandboxSpec,
) -> WorkerSlot | None:
    """
    Choose a worker for one sandbox request.

    Args:
        method (str): Routing method, one of least_loaded, round_robin, random.
        candidates (list[WorkerSlot]): Registered workers with current load.
        spec (SandboxSpec): Sandbox being placed. Its `gpus` field filters candidates.

    Returns:
        WorkerSlot | None: The chosen worker, or None when nothing can host the
            request right now, which tells the caller to queue.

    Raises:
        ValueError: If `method` is not a known routing method.
    """
    if method not in ("least_loaded", "round_robin", "random"):
        raise ValueError(f"Unknown routing method {method!r}.")

    eligible = [slot for slot in candidates if slot.has_cpu_capacity and slot.has_gpu_capacity(spec.gpus)]
    if not eligible:
        return None

    if method == "least_loaded":
        return min(eligible, key=lambda slot: (slot.used_cpu, slot.worker_id))
    if method == "round_robin":
        return eligible[next(_ROUND_ROBIN_COUNTER) % len(eligible)]

    return random.choice(eligible)


@dataclass
class SandboxHandle:
    """
    Lightweight client-side reference to one sandbox.

    Held by the agent loop, which never learns which node hosts the sandbox.
    """

    sandbox_id: str
    worker: Any
    coordinator: Any = None
    worker_id: int = -1
    gpus: int = 0
    _destroyed: bool = field(default=False, repr=False)

    async def exec(
        self,
        command: str,
        timeout_s: float,
        no_output_timeout_s: float | None = None,
    ) -> ExecResult:
        """Run one command in this sandbox's persistent shell."""
        return await self.worker.exec.remote(self.sandbox_id, command, timeout_s, no_output_timeout_s)

    async def read_file(self, path: str) -> bytes:
        """Read one file out of this sandbox."""
        return await self.worker.read_file.remote(self.sandbox_id, path)

    async def write_file(self, path: str, data: bytes) -> None:
        """Write bytes into one file in this sandbox."""
        await self.worker.write_file.remote(self.sandbox_id, path, data)

    async def destroy(self) -> None:
        """Destroy this sandbox and release its capacity. Idempotent."""
        if self._destroyed:
            return
        self._destroyed = True
        try:
            await self.worker.destroy_sandbox.remote(self.sandbox_id)
        finally:
            if self.coordinator is not None:
                await self.coordinator.release.remote(self.worker_id, self.gpus)


class EnvWorkerCoordinator:
    """Route sandbox requests to workers and track their capacity."""

    def __init__(self, routing_method: str = "least_loaded"):
        self.routing_method = routing_method
        self._slots: dict[int, WorkerSlot] = {}
        self._lock = asyncio.Lock()

    async def register_worker(
        self,
        worker_handle: Any,
        worker_id: int,
        cpu_slots: int,
        gpu_slots: int,
        node_ip: str,
    ) -> None:
        """Add one worker to the routable pool."""
        async with self._lock:
            self._slots[worker_id] = WorkerSlot(
                worker_id=worker_id,
                handle=worker_handle,
                cpu_slots=cpu_slots,
                gpu_slots=gpu_slots,
                node_ip=node_ip,
            )
        psrl_logger.info(
            f"Registered env worker={worker_id!r} on node={node_ip!r}. "
            f"CPU slots={cpu_slots!r}, GPU slots={gpu_slots!r}."
        )

    async def select_worker(self, spec: SandboxSpec, timeout_s: float) -> tuple[Any, int]:
        """
        Wait for capacity and reserve one slot.

        Args:
            spec (SandboxSpec): Sandbox being placed.
            timeout_s (float): Maximum seconds to wait for capacity.

        Returns:
            tuple[Any, int]: The chosen worker handle and its worker id.

        Raises:
            RuntimeError: If no worker is registered, or capacity never frees up.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while True:
            async with self._lock:
                if not self._slots:
                    raise RuntimeError("No env worker is registered with the coordinator.")
                chosen = select_by_method(self.routing_method, list(self._slots.values()), spec)
                if chosen is not None:
                    chosen.used_cpu += 1
                    chosen.used_gpu += spec.gpus
                    return chosen.handle, chosen.worker_id
            if loop.time() >= deadline:
                raise RuntimeError(
                    f"No env worker capacity became available within {timeout_s}s for "
                    f"a sandbox requesting {spec.gpus} gpu(s)."
                )
            await asyncio.sleep(_SELECT_POLL_INTERVAL_S)

    async def create_sandbox(self, spec: SandboxSpec, timeout_s: float = 3600.0) -> SandboxHandle:
        """
        Place and start one sandbox, retrying once on a different worker.

        Args:
            spec (SandboxSpec): Sandbox description.
            timeout_s (float): Maximum seconds to wait for placement capacity.

        Returns:
            SandboxHandle: Handle bound to the worker that hosts the sandbox.

        Raises:
            RuntimeError: If the sandbox cannot be started on two distinct attempts.
        """
        import ray

        last_error: Exception | None = None
        for attempt in range(2):
            worker_handle, worker_id = await self.select_worker(spec, timeout_s)
            try:
                sandbox_id = await worker_handle.create_sandbox.remote(spec)
                return SandboxHandle(
                    sandbox_id=sandbox_id,
                    worker=worker_handle,
                    coordinator=ray.get_runtime_context().current_actor,
                    worker_id=worker_id,
                    gpus=spec.gpus,
                )
            except Exception as error:  # noqa: BLE001
                last_error = error
                await self.release(worker_id, spec.gpus)
                psrl_logger.warning(f"Sandbox creation attempt={attempt!r} failed on worker={worker_id!r}: {error!r}.")
        raise RuntimeError(f"Failed to create a sandbox after two attempts: {last_error}.")

    async def release(self, worker_id: int, gpus: int) -> None:
        """Return one sandbox's capacity to a worker."""
        async with self._lock:
            slot = self._slots.get(worker_id)
            if slot is None:
                return
            slot.used_cpu = max(0, slot.used_cpu - 1)
            slot.used_gpu = max(0, slot.used_gpu - gpus)

    async def stats(self) -> dict[str, Any]:
        """Return per-worker capacity for logging and diagnostics."""
        async with self._lock:
            return {
                "routing_method": self.routing_method,
                "workers": [
                    {
                        "worker_id": slot.worker_id,
                        "node_ip": slot.node_ip,
                        "cpu_slots": slot.cpu_slots,
                        "used_cpu": slot.used_cpu,
                        "gpu_slots": slot.gpu_slots,
                        "used_gpu": slot.used_gpu,
                    }
                    for slot in self._slots.values()
                ],
            }
