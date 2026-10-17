"""
Collect model scoped engine telemetry and gateway demand.
"""

from __future__ import annotations

import asyncio
import logging

from psrl.workers.gen.smg_adapter import WORKERS_UPDATE_STATS_PATH, build_worker_stats_update
from psrl.workers.gen.stats_collector import EngineStats

from .lifecycle import GatewayClient
from .state import ModelState

psrl_logger = logging.getLogger(__name__)


class StatusService:
    def __init__(self, config, gateway: GatewayClient, ps_manager) -> None:
        self.config = config
        self.gateway = gateway
        self.ps_manager = ps_manager

    async def collect(self, model: ModelState) -> None:
        while not model.closing.is_set():
            try:
                stats = await model.status_queue.get_async(block=True, timeout=0.2)
            except TimeoutError:
                continue
            while stats is not None:
                rid = model.rank_to_replica[stats.replica_idx]
                instance_id = (rid, stats.data_parallel_rank)
                if instance_id not in model.engine_status:
                    raise ValueError(
                        f"Unknown engine status instance: model={model.scope!r}, instance={instance_id!r}."
                    )
                model.engine_status[instance_id] = stats
                stats = await model.status_queue.get_async(block=False)

    async def broadcast(self, model: ModelState) -> None:
        interval = self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000
        while await model.wait_interval(interval):
            updates = [
                build_worker_stats_update(iid[0], iid[1], stats.snapshot)
                for iid, stats in model.engine_status.items()
                if stats.snapshot.get("timestamp")
            ]
            if not updates:
                continue
            try:
                result = await self.gateway.post(model, WORKERS_UPDATE_STATS_PATH, updates)
                accepted = int(result.get("updated", 0)) + int(result.get("stale_ignored", 0))
                if result.get("rejected", 0) or accepted != len(updates):
                    raise RuntimeError(f"Gateway rejected engine status: {result!r}.")
            except Exception:
                psrl_logger.exception(f"Failed to broadcast status for model {model.scope!r}. Retrying.")

    async def record(self, model: ModelState) -> None:
        interval = self.config.psrl.status_collection.stats_recorder.interval_in_s
        while await model.wait_interval(interval):
            model.recorder.record(model.engine_status)
            status, workers = await asyncio.gather(
                self.gateway.get(model, "/routing_loop/status"),
                self.gateway.get(model, "/workers/stats"),
            )
            model.recorder.record_smg_routing_status(status, workers)

    async def runtime_snapshot(self, model: ModelState) -> dict:
        replicas = sorted(model.replicas)
        observations = await asyncio.gather(
            *(model.replicas[rid].server.get_engine_runtime_snapshots.remote() for rid in replicas)
        )
        result = {}
        for rid, engines in zip(replicas, observations, strict=True):
            for rank, observation in engines.items():
                iid = (rid, rank)
                stats = model.engine_status[iid]
                sleeping = observation["is_sleeping"] or model.replicas[rid].sleeping
                snapshot = (
                    EngineStats.get_default_snapshot()
                    if sleeping
                    else {
                        **stats.snapshot,
                        "timestamp": observation["timestamp"],
                        "scheduler_stats": observation["scheduler_stats"],
                    }
                )
                scheduler = snapshot["scheduler_stats"]
                if scheduler["num_running_reqs"] + scheduler["num_waiting_reqs"] == 0:
                    snapshot["generation_throughput"] = 0.0
                model.engine_status[iid] = EngineStats(stats.replica_idx, rank, stats.model_version, snapshot)
                result[iid] = {
                    "is_sleeping": sleeping,
                    "engine_stats": snapshot,
                    "model_version": model.ps_version if sleeping else model.versions[iid],
                    "wake_model_version": model.ps_version,
                }
        return result

    async def requests(self, model: ModelState, candidate_versions: list[int] | None = None) -> list[dict]:
        requests = await self.gateway.get(model, "/routing_loop/snapshot")
        if not isinstance(requests, list):
            raise TypeError("Router request snapshot must be a list.")
        if not model.dynamic_weights:
            return requests
        requests = [request for request in requests if not request["is_validate"]]
        if candidate_versions is None:
            return requests
        versions = sorted(set(candidate_versions))
        new = [request for request in requests if request["version_tag"] < 0]
        if new:
            reservations = await self.ps_manager.can_reserve_request.remote(
                [int(request["request_id"]) for request in new],
                versions,
                False,
                [request["is_validate"] for request in new],
            )
            for request, reservation in zip(new, reservations, strict=True):
                request["reservable_versions"] = [v for v, ok in zip(versions, reservation, strict=True) if ok]
        for request in requests:
            indicators = await self.ps_manager.get_reserve_indicator.remote(
                int(request["request_id"]), versions, request["is_validate"]
            )
            request["reserve_indicators"] = dict(zip(versions, indicators, strict=True))
        return requests

    async def backlog(self, model: ModelState) -> int:
        if model.dynamic_weights:
            return len(await self.requests(model))
        return int((await self.gateway.get(model, "/routing_loop/status"))["pending_request_num"])

    async def backlog_summary(self, model: ModelState, top_t: int | None = None) -> dict:
        pending = await self.backlog(model)
        return {
            "pending": pending,
            "count": pending if top_t is None else min(pending, max(0, top_t)),
            "total_tokens": 0,
        }
