"""
Synchronize policy replicas and plan request redistribution from model state.
"""

from __future__ import annotations

import asyncio

from psrl.workers.gen.smg_adapter import build_weight_version_updates

from .commands import CommandService
from .lifecycle import GatewayClient, ReplicaLifecycle
from .state import ModelState


class WeightSynchronization:
    def __init__(
        self, config, ps_manager, gateway: GatewayClient, lifecycle: ReplicaLifecycle, commands: CommandService
    ) -> None:
        self.config = config
        self.ps_manager = ps_manager
        self.gateway = gateway
        self.lifecycle = lifecycle
        self.commands = commands

    def set_ps_version(self, model: ModelState, version: int) -> None:
        self.lifecycle.require_dynamic(model)
        if version <= 0 or (model.ready_buffers and version - 1 not in model.ready_buffers):
            raise ValueError(f"Invalid parameter server version {version} for ready buffers {model.ready_buffers!r}.")
        model.ps_version = version
        model.ready_buffers.discard(version - 1)

    async def quarantine(self, model: ModelState, replica_ids: list[str]) -> None:
        await asyncio.gather(
            *(model.replicas[rid].server.fail_sync.remote() for rid in replica_ids), return_exceptions=True
        )
        await self.gateway.eligibility(model, replica_ids, False)

    async def sync(
        self,
        model: ModelState,
        instance_ids: list,
        wait_model_sync: bool = False,
        wait_interrupted_partial_requests_loop_back: bool = True,
    ) -> None:
        self.lifecycle.require_dynamic(model)
        instances = await self.lifecycle.awake(model, instance_ids)
        if not instances:
            return
        replica_ids = sorted({iid[0] for iid in instances})
        operations = await model.reserve(replica_ids)
        transferred = False
        quarantined = False
        pulls = []
        try:
            instances = await self.lifecycle.awake(model, instances)
            if not instances:
                return
            replica_ids = sorted({iid[0] for iid in instances})
            version = model.ps_version
            async with model.routing_lock:
                try:
                    await self.gateway.routing(model, False)
                    await asyncio.gather(*(model.replicas[rid].server.pause_for_sync.remote() for rid in replica_ids))
                    if (
                        wait_interrupted_partial_requests_loop_back
                        and self.config.psrl.rollout_coordination.partial_rollout.enable
                    ):
                        await self.drain(model, replica_ids)
                    pulls = [model.replicas[rid].server.pull_model_for_sync.remote(version) for rid in replica_ids]
                    await self.gateway.publish_versions(model, build_weight_version_updates(instances, version))
                    for iid in instances:
                        model.target_versions[iid] = version
                except BaseException:
                    await asyncio.gather(*pulls, return_exceptions=True)
                    await self.quarantine(model, replica_ids)
                    quarantined = True
                    raise
                finally:
                    await self.gateway.routing(model, True)
            task = model.track(
                f"sync:{','.join(replica_ids)}",
                self._finish(model, replica_ids, instances, version, pulls, operations),
            )
            for rid in replica_ids:
                model.sync_tasks[rid] = task

            def finished(completed):
                for rid in replica_ids:
                    if model.sync_tasks.get(rid) is completed:
                        del model.sync_tasks[rid]

            task.add_done_callback(finished)
            transferred = True
            if wait_model_sync:
                await asyncio.shield(task)
        except BaseException:
            if not transferred:
                await asyncio.gather(*pulls, return_exceptions=True)
                if not quarantined:
                    await self.quarantine(model, replica_ids)
            raise
        finally:
            if not transferred:
                await operations.aclose()

    async def _finish(self, model, replicas, instances, advertised_version, pulls, operations) -> None:
        try:
            versions = await asyncio.gather(*pulls, return_exceptions=True)
            self.lifecycle._raise_failures("Weight pull", versions)
            corrections = []
            for rid, version in zip(replicas, versions, strict=True):
                replica_instances = [iid for iid in instances if iid[0] == rid]
                for iid in replica_instances:
                    model.set_version(iid, version)
                    model.target_versions[iid] = version
                if version != advertised_version:
                    corrections.extend(build_weight_version_updates(replica_instances, version))
            await self.gateway.publish_versions(model, corrections)
            await asyncio.gather(*(model.replicas[rid].server.resume_after_sync.remote() for rid in replicas))
        except BaseException:
            await self.quarantine(model, replicas)
            raise
        finally:
            await operations.aclose()

    async def drain(self, model: ModelState, replica_ids: list[str]) -> None:
        await asyncio.gather(*(model.replicas[rid].server.wait_for_requests_to_drain.remote() for rid in replica_ids))

    async def reservable_requests(self, model: ModelState, version: int, existing_only: bool) -> list:
        if await self.ps_manager.check_aborted_model_versions.remote(version):
            return []
        requests = await self.gateway.get(model, "/routing_loop/filter", {"version_tag": version})
        if not requests:
            return []
        ids = [int(request["request_id"]) for request in requests]
        aborted = await self.ps_manager.check_aborted_requests.remote(ids, remove=False)
        requests = [request for request, remove in zip(requests, aborted, strict=True) if not remove]
        if not requests:
            return []
        reservations = await self.ps_manager.can_reserve_request.remote(
            [int(request["request_id"]) for request in requests],
            [version],
            without_new_reserve_entry=existing_only,
            is_validate=[bool(request.get("is_validate", False)) for request in requests],
        )
        return [request for request, ok in zip(requests, reservations, strict=True) if ok == [True]]

    @staticmethod
    def indicator(stats, name):
        readers = {
            "request_num": stats.get_waiting_and_running_queue_size,
            "throughput": stats.get_generation_throughput,
            "kv_cache": stats.get_kv_cache_utilization,
        }
        if name not in readers:
            raise ValueError(f"Unknown scheduling indicator: {name!r}.")
        return readers[name]()

    async def should_sync(self, model: ModelState, instance_id: tuple) -> bool:
        stats = model.engine_status[instance_id]
        cfg = self.config.psrl.rollout_coordination.sync_and_mig_strategy.sync
        if stats.get_waiting_queue_size() > 0:
            return False
        if cfg.check_req_before_sync and await self.reservable_requests(
            model, model.target_versions[instance_id], True
        ):
            return False
        return self.indicator(stats, cfg.indicator) <= cfg.threshold

    async def check_idle(self, model: ModelState, instance_id: tuple) -> bool:
        return await model.replicas[instance_id[0]].server.get_active_task_num.remote(instance_id[1]) == 0

    async def run(self, model: ModelState) -> None:
        cfg = self.config.psrl.rollout_coordination.sync_and_mig_strategy
        partial = self.config.psrl.rollout_coordination.partial_rollout.enable
        while await model.wait_interval(cfg.check_interval_in_ms / 1000):
            awake = await self.lifecycle.awake(model, model.instances("rollout"))
            candidates = []
            selecting = (
                (await self.gateway.get(model, "/routing_loop/status")).get("selecting", False)
                if cfg.method == "status_based"
                else False
            )
            for iid in awake:
                if iid[0] in model.sync_tasks or model.versions[iid] >= model.ps_version:
                    continue
                if cfg.method == "status_based":
                    stats = model.engine_status[iid]
                    if selecting or stats.model_version != model.versions[iid]:
                        continue
                    if partial:
                        if not await self.should_sync(model, iid):
                            continue
                    elif stats.get_waiting_and_running_queue_size() > 0:
                        continue
                elif not partial and not await self.check_idle(model, iid):
                    continue
                if cfg.sync.seamless_train_version >= model.ps_version and model.ps_version not in model.ready_buffers:
                    continue
                candidates.append(iid)
            if candidates:
                await self.sync(model, candidates)
            elif not model.sync_tasks and cfg.mig.enable:
                await self.migrate(model)

    async def migrate(self, model: ModelState, wait_interrupted_partial_requests_loop_back: bool = True) -> None:
        cfg = self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig
        if not cfg.enable:
            raise ValueError("Request migration is disabled.")
        awake = await self.lifecycle.awake(model, model.instances("rollout"))
        starved = []
        for iid in awake:
            if model.engine_status[iid].get_waiting_queue_size() == 0:
                if not await self.reservable_requests(model, model.target_versions[iid], False):
                    starved.append(iid)
        candidates = []
        for idle in starved:
            low = self.indicator(model.engine_status[idle], cfg.indicator)
            for iid in awake:
                if iid == idle or model.target_versions[iid] > model.target_versions[idle]:
                    continue
                high = self.indicator(model.engine_status[iid], cfg.indicator)
                ratio = high / low if low else (float("inf") if high else 1)
                if ratio > cfg.threshold:
                    candidates.append((ratio, iid))
        if not candidates:
            return
        _, selected = max(candidates)
        if self.indicator(model.engine_status[selected], cfg.stop_indicator) < cfg.stop_threshold:
            return
        instances = model.expand_instances([selected])
        async with await model.reserve([selected[0]]):
            async with model.routing_lock:
                try:
                    await self.gateway.routing(model, False)
                    await self.commands.abort(model, instances)
                    if wait_interrupted_partial_requests_loop_back:
                        await self.drain(model, [selected[0]])
                finally:
                    await self.gateway.routing(model, True)
