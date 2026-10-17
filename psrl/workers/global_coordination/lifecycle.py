"""
Control gateway eligibility and GPU memory for every inference role.
"""

from __future__ import annotations

import asyncio
import json

import aiohttp

from psrl.workers.gen.smg_adapter import (
    WORKERS_PAUSE_PATH,
    WORKERS_RESUME_PATH,
    WORKERS_UPDATE_WEIGHT_VERSION_PATH,
    build_pause_resume_payload,
    build_weight_version_updates,
)
from psrl.workers.gen.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT

from .state import ModelState


class GatewayClient:
    def __init__(self) -> None:
        connector = aiohttp.TCPConnector(limit=DEFAULT_MAX_CONNECTIONS, ttl_dns_cache=300)
        self.session = aiohttp.ClientSession(connector=connector, timeout=aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT))

    async def request(self, method: str, url: str, **kwargs):
        async with self.session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            content = await response.text()
            return json.loads(content) if content.strip() else {}

    async def get(self, model: ModelState, path: str, params=None):
        return await self.request("GET", f"{model.gateway_url}{path}", params=params)

    async def post(self, model: ModelState, path: str, payload, params=None):
        return await self.request("POST", f"{model.gateway_url}{path}", json=payload, params=params)

    async def routing(self, model: ModelState, running: bool) -> None:
        path = "/routing_loop/resume" if running else "/routing_loop/pause"
        data = await self.post(model, path, {}, params={"wait": "true"} if not running else {})
        if "paused" in data and bool(data["paused"]) == running:
            raise RuntimeError(f"Unexpected gateway routing state: {data!r}.")
        if not running and (data.get("selecting") or data.get("active_dispatch_handoffs", 0)):
            raise RuntimeError(f"Gateway dispatch did not drain: {data!r}.")

    async def eligibility(self, model: ModelState, replica_ids: list[str], active: bool) -> None:
        if not replica_ids:
            return
        result = await self.post(
            model, WORKERS_RESUME_PATH if active else WORKERS_PAUSE_PATH, build_pause_resume_payload(replica_ids)
        )
        expected = sum(model.replicas[rid].dp_size for rid in set(replica_ids))
        if result.get("rejected", 0) or int(result.get("updated", 0)) != expected:
            raise RuntimeError(f"Incomplete gateway eligibility update: expected={expected}, result={result!r}.")

    async def pause_replicas(self, model: ModelState, replica_ids: list[str]) -> None:
        if not replica_ids:
            return
        async with model.routing_lock:
            try:
                await self.routing(model, False)
                await self.eligibility(model, replica_ids, False)
            finally:
                await self.routing(model, True)

    async def publish_versions(self, model: ModelState, updates: list[dict]) -> None:
        if not updates:
            return
        result = await self.post(model, WORKERS_UPDATE_WEIGHT_VERSION_PATH, updates)
        if result.get("rejected", 0) or int(result.get("updated", 0)) != len(updates):
            raise RuntimeError(f"Gateway rejected weight versions: {result!r}.")


class ReplicaLifecycle:
    def __init__(self, gateway: GatewayClient) -> None:
        self.gateway = gateway

    async def awake(self, model: ModelState, instance_ids: list) -> list:
        instance_ids = model.expand_instances(instance_ids)
        replica_ids = sorted({iid[0] for iid in instance_ids if not model.replicas[iid[0]].sleeping})
        sleeping = await asyncio.gather(*(model.replicas[rid].server.is_sleeping.remote() for rid in replica_ids))
        awake = {rid for rid, asleep in zip(replica_ids, sleeping, strict=True) if not asleep}
        return [iid for iid in instance_ids if iid[0] in awake]

    async def sleep(self, model: ModelState, instance_ids: list) -> bool:
        replica_ids = sorted({iid[0] for iid in model.expand_instances(instance_ids)})
        async with await model.reserve(replica_ids):
            sleeping = await asyncio.gather(*(model.replicas[rid].server.is_sleeping.remote() for rid in replica_ids))
            wake_ids = []
            for rid, asleep in zip(replica_ids, sleeping, strict=True):
                replica = model.replicas[rid]
                if asleep:
                    replica.sleeping = True
                if not replica.sleeping:
                    wake_ids.append(rid)
            await self.gateway.pause_replicas(model, replica_ids)

            async def release(rid):
                replica = model.replicas[rid]
                method = replica.server.nixl_sleep if model.dynamic_weights else replica.server.sleep
                await method.remote(level=2)
                replica.sleeping = True

            results = await asyncio.gather(*(release(rid) for rid in wake_ids), return_exceptions=True)
            self._raise_failures("Sleep", results)
        return True

    async def wake(self, model: ModelState, instance_ids: list) -> bool:
        replica_ids = sorted({iid[0] for iid in model.expand_instances(instance_ids)})
        async with await model.reserve(replica_ids):
            sleeping = await asyncio.gather(*(model.replicas[rid].server.is_sleeping.remote() for rid in replica_ids))
            wake_ids = [
                rid
                for rid, asleep in zip(replica_ids, sleeping, strict=True)
                if asleep or model.replicas[rid].sleeping
            ]
            await self.gateway.pause_replicas(model, wake_ids)
            for rid in wake_ids:
                model.replicas[rid].sleeping = True

            async def restore(rid):
                replica = model.replicas[rid]
                if model.dynamic_weights:
                    await replica.server.nixl_wake_up.remote()
                    version = await replica.server.pull_model_for_sync.remote(model.ps_version)
                    instances = [iid for iid in model.versions if iid[0] == rid]
                    await self.gateway.publish_versions(model, build_weight_version_updates(instances, version))
                    for iid in instances:
                        model.set_version(iid, version)
                        model.target_versions[iid] = version
                else:
                    await replica.server.wake_up.remote()

            results = await asyncio.gather(*(restore(rid) for rid in wake_ids), return_exceptions=True)
            failed = {rid for rid, result in zip(wake_ids, results, strict=True) if isinstance(result, BaseException)}
            ready = [rid for rid in replica_ids if rid not in failed]
            await self.gateway.eligibility(model, ready, True)
            for rid in ready:
                model.replicas[rid].sleeping = False
            self._raise_failures("Wake", results)
        return True

    @staticmethod
    def _raise_failures(operation: str, results: list) -> None:
        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise RuntimeError(f"{operation} failed for inference replicas: {failures!r}.") from failures[0]

    async def initialize_nixl(self, model: ModelState) -> None:
        self.require_dynamic(model)
        await asyncio.gather(*(replica.server.init_nixl_client.remote() for replica in model.replicas.values()))
        model.nixl_ready.set()

    async def nixl_protocol(self, model: ModelState, full_tag: str = "all") -> None:
        self.require_dynamic(model)
        if full_tag not in {"all", "rollout", "validate", "meta"}:
            raise ValueError(f"Unknown NIXL protocol tag: {full_tag!r}.")
        await model.nixl_ready.wait()
        await asyncio.gather(
            *(
                replica.server.nixl_protocol.remote("full" if full_tag in {"all", replica.tag} else "meta")
                for replica in model.replicas.values()
            )
        )

    async def convert_nixl(self, model: ModelState) -> None:
        self.require_dynamic(model)
        await model.nixl_ready.wait()
        await asyncio.gather(*(replica.server.nixl_convert_params.remote() for replica in model.replicas.values()))

    async def initial_pull(self, model: ModelState, tag: str = "rollout") -> None:
        self.require_dynamic(model)
        await asyncio.gather(*(model.replicas[rid].server.pull_model.remote() for rid in model.replica_ids(tag)))
        instances = model.instances(tag)
        await self.gateway.publish_versions(model, build_weight_version_updates(instances, model.ps_version))
        for iid in instances:
            model.set_version(iid, model.ps_version)
            model.target_versions[iid] = model.ps_version

    @staticmethod
    def require_dynamic(model: ModelState) -> None:
        if not model.dynamic_weights:
            raise ValueError(f"Static model has no parameter server weight lifecycle: {model.scope!r}.")
