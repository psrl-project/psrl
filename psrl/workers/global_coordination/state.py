"""
Store model and replica state independently of coordinator implementations.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from psrl.utils.server.command import CommandExtension
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.gen.utils import RolloutInstanceId

CoordinatorScope: TypeAlias = tuple[str, str]
ROLLOUT_SCOPE: CoordinatorScope = ("rollout", "")


@dataclass
class ReplicaState:
    replica: Any
    server: Any
    tag: str
    dp_size: int
    sleeping: bool = False
    operation_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass
class ModelState:
    scope: CoordinatorScope
    gateway_url: str
    dynamic_weights: bool
    status_endpoint: str
    status_queue: Any
    replicas: dict[str, ReplicaState] = field(default_factory=dict)
    rank_to_replica: dict[int, str] = field(default_factory=dict)
    versions: dict[RolloutInstanceId, int] = field(default_factory=dict)
    target_versions: dict[RolloutInstanceId, int] = field(default_factory=dict)
    engine_status: dict[RolloutInstanceId, EngineStats] = field(default_factory=dict)
    ps_version: int = 0
    ready_buffers: set[int] = field(default_factory=set)
    commands: CommandExtension = field(default_factory=CommandExtension)
    routing_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    nixl_ready: asyncio.Event = field(default_factory=asyncio.Event)
    closing: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    sync_tasks: dict[str, asyncio.Task] = field(default_factory=dict)
    failures: list[BaseException] = field(default_factory=list)
    recorder: Any = None

    def add_worker(
        self,
        rollout_replica,
        server_handle,
        replica_id: str,
        dp_size: int,
        is_validate: bool = False,
        model_version: int = 0,
    ) -> None:
        if replica_id in self.replicas or rollout_replica.replica_rank in self.rank_to_replica:
            raise ValueError(f"Replica ID or rank is already registered in model {self.scope!r}.")
        if dp_size <= 0 or (is_validate and not self.dynamic_weights):
            raise ValueError(f"Invalid replica registration for model {self.scope!r}.")
        self.replicas[replica_id] = ReplicaState(
            rollout_replica, server_handle, "validate" if is_validate else "rollout", dp_size
        )
        self.rank_to_replica[rollout_replica.replica_rank] = replica_id
        for rank in range(dp_size):
            instance_id = (replica_id, rank)
            self.versions[instance_id] = model_version
            self.target_versions[instance_id] = model_version
            self.engine_status[instance_id] = EngineStats(
                replica_idx=rollout_replica.replica_rank,
                data_parallel_rank=rank,
                model_version=model_version,
                snapshot=EngineStats.get_default_snapshot(),
            )

    def replica_ids(self, tag: str = "all") -> list[str]:
        if tag not in {"all", "rollout", "validate"}:
            raise ValueError(f"Unknown replica tag: {tag!r}.")
        return sorted(rid for rid, replica in self.replicas.items() if tag == "all" or replica.tag == tag)

    def instances(self, tag: str = "all") -> list[RolloutInstanceId]:
        replicas = set(self.replica_ids(tag))
        return sorted(iid for iid in self.versions if iid[0] in replicas)

    def expand_instances(self, instance_ids: list[RolloutInstanceId]) -> list[RolloutInstanceId]:
        unknown = set(instance_ids) - self.versions.keys()
        if unknown:
            raise ValueError(f"Unknown instances in model {self.scope!r}: {unknown!r}.")
        replicas = {iid[0] for iid in instance_ids}
        return sorted(iid for iid in self.versions if iid[0] in replicas)

    def set_version(self, instance_id: RolloutInstanceId, version: int) -> None:
        if instance_id not in self.versions:
            raise ValueError(f"Unknown instance in model {self.scope!r}: {instance_id!r}.")
        self.versions[instance_id] = version
        self.engine_status[instance_id].model_version = version

    async def reserve(self, replica_ids: list[str]) -> AsyncExitStack:
        operations = AsyncExitStack()
        try:
            for rid in sorted(set(replica_ids)):
                await operations.enter_async_context(self.replicas[rid].operation_lock)
        except BaseException:
            await operations.aclose()
            raise
        return operations

    async def wait_interval(self, seconds: float) -> bool:
        try:
            await asyncio.wait_for(self.closing.wait(), timeout=seconds)
            return False
        except TimeoutError:
            return True

    def track(self, name: str, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine, name=f"global:{self.scope}:{name}")
        self.tasks[name] = task

        def finished(completed):
            if not completed.cancelled() and (error := completed.exception()) is not None:
                self.failures.append(error)

        task.add_done_callback(finished)
        return task
