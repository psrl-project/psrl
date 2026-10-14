"""CoordinatorBase: shared low-level helpers used by all RolloutCoordinator mixins."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack

from psrl.workers.gen.smg_adapter import (
    WORKERS_PAUSE_PATH,
    WORKERS_RESUME_PATH,
    ReplicaRoutingStatus,
    build_pause_resume_payload,
)
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class CoordinatorBase:
    """Shared HTTP gateway helpers and routing-loop control.

    All `self.*` attributes are initialized by `RolloutCoordinator.__init__`.
    this class has no ``__init__`` of its own.
    """

    async def _get_awake_instance_ids(
        self,
        instance_ids: list[RolloutInstanceId],
    ) -> list[RolloutInstanceId]:
        """
        Select awake role replicas before considering their model versions.
        """
        replica_ids = sorted({instance_id[0] for instance_id in instance_ids} - self.sleeping_replica_ids)
        sleeping = await asyncio.gather(*(
            self.server_handles[replica_id].is_sleeping.remote() for replica_id in replica_ids
        ))
        awake_replicas = {
            replica_id for replica_id, asleep in zip(replica_ids, sleeping, strict=True) if not asleep
        }
        return sorted({instance_id for instance_id in instance_ids if instance_id[0] in awake_replicas})

    async def _acquire_replica_operations(self, replica_ids: list[str]) -> AsyncExitStack:
        """Reserve replica lifecycles until memory and weight operations settle."""
        operations = AsyncExitStack()
        try:
            for replica_id in sorted(set(replica_ids)):
                lock = self.replica_operation_locks.setdefault(replica_id, asyncio.Lock())
                await operations.enter_async_context(lock)
        except BaseException:
            await operations.aclose()
            raise
        return operations

    async def _pause_replica_dispatch(self, replica_ids: list[str]) -> None:
        """Drain router handoffs before removing replicas from selection."""
        async with self.routing_operation_lock:
            try:
                await self._set_routing_loop_running(False)
                await self._set_replica_routing_status(replica_ids, ReplicaRoutingStatus.PAUSED)
            finally:
                await self._set_routing_loop_running(True)

    async def _gateway_post_json(self, path: str, payload, params: dict | None = None):
        url = f"{self.rollout_gateway_url}{path}"
        async with self.gateway_client.post(url, json=payload, params=params) as resp:
            resp.raise_for_status()
            text = await resp.text()
            if not text.strip():
                return {}
            return json.loads(text)

    async def _gateway_get_json(self, path: str, params: dict | None = None):
        url = f"{self.rollout_gateway_url}{path}"
        psrl_logger.debug(f"Gateway GET request: url={url!r}, params={params!r}.")
        async with self.gateway_client.get(url, params=params) as resp:
            resp.raise_for_status()
            text = await resp.text()
            if not text.strip():
                return {}
            return json.loads(text)

    async def _set_routing_loop_running(self, running: bool):
        path = "/routing_loop/resume" if running else "/routing_loop/pause"
        # A synchronous pause prevents routing with the pre-sync model version.
        params = {"wait": "true"} if not running else {}
        psrl_logger.info(f"Setting routing loop state: running={running}, path={path!r}.")
        data = await self._gateway_post_json(path, payload={}, params=params)
        expected_paused = not running
        actual_paused = data.get("paused")
        if actual_paused is not None and bool(actual_paused) != expected_paused:
            raise RuntimeError(
                f"Unexpected routing loop pause state after {path}: expected={expected_paused}, got={actual_paused!r}."
            )
        if not running and bool(data.get("selecting", False)):
            raise RuntimeError("Routing loop is still selecting after pause wait.")
        if not running and int(data.get("active_dispatch_handoffs", 0)) != 0:
            raise RuntimeError("Routing loop still has active dispatch handoffs after pause wait.")

    async def _set_replica_routing_status(
        self,
        replica_ids: list[str],
        status: ReplicaRoutingStatus,
    ) -> None:
        """Update routing eligibility for all DP ranks in the selected replicas."""
        replica_ids = sorted(set(replica_ids))
        if not replica_ids:
            return
        if status is ReplicaRoutingStatus.PAUSED:
            path = WORKERS_PAUSE_PATH
        elif status is ReplicaRoutingStatus.ACTIVE:
            path = WORKERS_RESUME_PATH
        else:
            raise ValueError(f"Unknown replica routing status {status!r}.")
        result = await self._gateway_post_json(
            path,
            payload=build_pause_resume_payload(replica_ids),
        )
        rejected = int(result.get("rejected", 0))
        expected = sum(instance_id[0] in replica_ids for instance_id in self.instance_ids)
        if rejected or int(result.get("updated", 0)) != expected:
            raise RuntimeError(
                f"Incomplete replica routing update: replicas={replica_ids!r}, expected={expected}, result={result!r}."
            )

    async def _is_selecting(self) -> bool:
        """Check if the router is currently selecting a worker for queued requests.

        Returns:
            bool: True if any worker-selection stage is active, False otherwise.
        """
        data = await self._gateway_get_json("/routing_loop/status")
        return bool(data.get("selecting", False))
