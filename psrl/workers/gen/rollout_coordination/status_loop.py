"""StatusMixin: engine-status queue processing and router sync loops."""

from __future__ import annotations

import asyncio
import logging
import os

from psrl.workers.gen.smg_adapter import (
    ROUTING_LOOP_STATUS_PATH,
    WORKERS_STATS_PATH,
    WORKERS_UPDATE_STATS_PATH,
    build_worker_stats_update,
)
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

STATUS_QUEUE_POLL_TIMEOUT_SECS = 0.2


class StatusMixin:
    """Provides status-collection and router-sync loops for RolloutCoordinator.

    Expects ``self`` to carry: ``status_queue``, ``replica_idx_to_replica_id``,
    ``instance_to_engine_status``, ``stop_process_status_queue``,
    ``stop_sync_status_to_router``, ``stop_stats_recorder``,
    ``config``,
    ``_stats_recorder``, and the ``_gateway_post_json`` helper.
    """

    async def _process_status_queue(self):
        psrl_logger.info("Starting to process ZMQ status stream")
        while not self.stop_process_status_queue:
            # Naive implementation version
            """
            recv_stats = await self.status_queue.get_async(block=True, timeout=None)
            replica_id = self.replica_idx_to_replica_id.get(recv_stats.replica_idx)
            if replica_id is None:
                raise RuntimeError(f"Received engine stats with unknown replica_idx {recv_stats.replica_idx}")
            instance_id = (replica_id, recv_stats.data_parallel_rank)
            self.instance_to_engine_status[instance_id] = recv_stats
            psrl_logger.debug(
                f"Updated engine status for instance "
                f"{instance_id}: {self.instance_to_engine_status[instance_id]} "
            )
            """

            # Coalesce implementation version
            try:
                recv_stats = await self.status_queue.get_async(
                    block=True,
                    timeout=STATUS_QUEUE_POLL_TIMEOUT_SECS,
                )
            except TimeoutError:
                # Periodically wake up so shutdown flag can be observed even when no messages arrive.
                continue
            latest_by_instance: dict[RolloutInstanceId, EngineStats] = {}

            # Put first blocking message
            replica_id = self.replica_idx_to_replica_id.get(recv_stats.replica_idx)
            if replica_id is None:
                raise RuntimeError(f"Received engine stats with unknown replica_idx {recv_stats.replica_idx}")
            instance_id = (replica_id, recv_stats.data_parallel_rank)
            latest_by_instance[instance_id] = recv_stats

            # Drain all currently queued messages and coalesce by instance
            while True:
                next_stats = await self.status_queue.get_async(block=False)
                if next_stats is None:
                    break
                replica_id = self.replica_idx_to_replica_id.get(next_stats.replica_idx)
                if replica_id is None:
                    raise RuntimeError(f"Received engine stats with unknown replica_idx {next_stats.replica_idx}")
                instance_id = (replica_id, next_stats.data_parallel_rank)
                latest_by_instance[instance_id] = next_stats

            # Apply only the newest snapshot per instance
            self.instance_to_engine_status.update(latest_by_instance)
        psrl_logger.info("Stopped processing ZMQ status stream.")

    def get_instance_engine_status_snapshot(self) -> dict[RolloutInstanceId, dict]:
        """Return a lightweight snapshot map for elastic scaling decisions."""
        snapshot = {}
        for instance_id, engine_status in self.instance_to_engine_status.items():
            snapshot[instance_id] = engine_status.snapshot
        return snapshot

    async def get_instance_runtime_snapshot(self) -> dict[RolloutInstanceId, dict]:
        """
        Collect live scheduler and memory state from every replica in parallel.
        """
        replica_ids = sorted(self.server_handles)
        observations = await asyncio.gather(
            *(self.server_handles[replica_id].get_engine_runtime_snapshots.remote() for replica_id in replica_ids)
        )
        result = {}
        for replica_id, engines in zip(replica_ids, observations, strict=True):
            for dp_rank, observation in engines.items():
                instance_id = (replica_id, dp_rank)
                engine_status = self.instance_to_engine_status[instance_id]
                snapshot = {
                    **engine_status.snapshot,
                    "timestamp": observation["timestamp"],
                    "scheduler_stats": observation["scheduler_stats"],
                }
                is_sleeping = observation["is_sleeping"] or replica_id in self.sleeping_replica_ids
                if is_sleeping:
                    snapshot = EngineStats.get_default_snapshot()
                stats = snapshot["scheduler_stats"]
                if stats["num_running_reqs"] + stats["num_waiting_reqs"] == 0:
                    snapshot["generation_throughput"] = 0.0
                self.instance_to_engine_status[instance_id] = EngineStats(
                    replica_idx=engine_status.replica_idx,
                    data_parallel_rank=dp_rank,
                    model_version=engine_status.model_version,
                    snapshot=snapshot,
                )
                result[instance_id] = {
                    "is_sleeping": is_sleeping,
                    "engine_stats": snapshot,
                    "model_version": (
                        self.ps_model_version if is_sleeping else self.instance_to_model_version.get(instance_id, 0)
                    ),
                    "wake_model_version": self.ps_model_version,
                }
        return result

    async def _get_training_request_snapshot(self) -> list[dict]:
        """
        Exclude dedicated validation demand from elastic rollout planning.
        """
        requests = await self._gateway_get_json("/routing_loop/snapshot")
        if not isinstance(requests, list):
            raise TypeError("Router request snapshot must be a list.")
        return [request for request in requests if not request["is_validate"]]

    async def get_router_request_snapshot(self, candidate_versions: list[int]) -> list[dict]:
        """
        Preserve queued training demand and its admission inventory in native dispatch order.
        """
        requests = await self._get_training_request_snapshot()
        if self.ps_manager is not None:
            versions = sorted(set(candidate_versions))
            new_requests = [request for request in requests if request["version_tag"] < 0]
            reservations = (
                await self.ps_manager.can_reserve_request.remote(
                    [int(request["request_id"]) for request in new_requests],
                    versions,
                    False,
                    [request["is_validate"] for request in new_requests],
                )
                if new_requests
                else []
            )
            for request, reservation in zip(new_requests, reservations, strict=True):
                request["reservable_versions"] = [
                    version for version, ok in zip(versions, reservation, strict=True) if ok
                ]
            for request in requests:
                request["reserve_indicators"] = dict(
                    zip(
                        versions,
                        await self.ps_manager.get_reserve_indicator.remote(
                            int(request["request_id"]),
                            versions,
                            request["is_validate"],
                        ),
                        strict=True,
                    )
                )
        return requests

    async def _sync_status_to_router(self):
        """Broadcast the engine status to the router."""
        while not self.stop_sync_status_to_router:
            # Broadcast the engine status to the router every coordinator sync interval
            await asyncio.sleep(self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000)
            if not self.instance_to_engine_status:
                continue
            updates = []
            for instance_id, engine_status in self.instance_to_engine_status.items():
                if not engine_status.snapshot.get("timestamp"):
                    continue
                replica_id, dp_rank = instance_id
                updates.append(build_worker_stats_update(replica_id, dp_rank, engine_status.snapshot))
            if not updates:
                continue
            try:
                result = await self._gateway_post_json(WORKERS_UPDATE_STATS_PATH, payload=updates)
                accepted = int(result.get("updated", 0)) + int(result.get("stale_ignored", 0))
                if int(result.get("rejected", 0)) or accepted != len(updates):
                    raise RuntimeError(f"Router rejected engine status updates: result={result!r}.")
            except Exception:
                psrl_logger.exception("Failed to sync engine status to the router. Retrying on the next interval.")
        psrl_logger.info("Stopped syncing engine status to router.")

    async def _stats_recorder_loop(self):
        """Periodically snapshot per-replica stats to JSONL files."""
        psrl_logger.info("Starting stats recorder loop")
        interval = self.config.psrl.status_collection.stats_recorder.interval_in_s
        while not self.stop_stats_recorder:
            await asyncio.sleep(interval)
            if not self.stop_stats_recorder:
                self._stats_recorder.record(self.instance_to_engine_status)
                routing_loop_status = await self._gateway_get_json(ROUTING_LOOP_STATUS_PATH)
                workers_stats = await self._gateway_get_json(WORKERS_STATS_PATH)
                self._stats_recorder.record_smg_routing_status(routing_loop_status, workers_stats)
        psrl_logger.info("Stopped stats recorder loop.")
