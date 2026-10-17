"""
Manage session admission and cache transfer resources for policy replicas.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from pathlib import Path

import aiohttp

from psrl.utils.common.http_utils import find_available_port, get_host_info
from psrl.workers.gen.session_scheduler import InstanceCapacity, SessionInfo, ThunderAgentScheduler

from .lifecycle import GatewayClient
from .state import ModelState


class PolicyAuxiliary:
    def __init__(self, config, gateway: GatewayClient, session_router_url: str | None) -> None:
        self.config = config
        self.gateway = gateway
        self.session_url = session_router_url.rstrip("/") if session_router_url else None
        self.controller = None
        self.controller_url = None
        self.kv_capacity = {}

    async def sessions(self, model: ModelState) -> None:
        cfg = self.config.psrl.rollout_coordination.session_strategy.thunder_agent
        scheduler = ThunderAgentScheduler(
            env_token_weight=float(cfg.get("env_token_weight", 1.0)),
            buffer_per_session=int(cfg.get("buffer_per_session", 100)),
            global_scope=cfg.get("continue_scope", "bucketed") == "global",
        )
        while await model.wait_interval(cfg.get("check_interval_in_ms", 1000) / 1000):
            for rid in model.replica_ids("rollout"):
                if rid not in self.kv_capacity:
                    self.kv_capacity[rid] = int(await model.replicas[rid].server.get_total_kv_cache_tokens.remote())
            capacities = [
                InstanceCapacity(
                    iid, self.kv_capacity[iid[0]], round(stats.get_kv_cache_utilization() * self.kv_capacity[iid[0]])
                )
                for iid, stats in model.engine_status.items()
                if iid[0] in self.kv_capacity and not model.replicas[iid[0]].sleeping
            ]
            payload = await self.gateway.request("GET", f"{self.session_url}/control/sessions")
            sessions = []
            for item in payload.get("sessions", []):
                if item.get("closing"):
                    continue
                worker, rank = item.get("base_worker_id"), item.get("target_dp_rank")
                sessions.append(
                    SessionInfo(
                        str(item["session_id"]),
                        (str(worker), int(rank)) if worker is not None and rank is not None else None,
                        str(item.get("status", "env")),
                        str(item.get("hang_state", "running")),
                        int(item.get("total_tokens", 0)),
                    )
                )
            hang, resume = scheduler.decide(capacities, sessions)
            if hang:
                await self.gateway.request(
                    "POST", f"{self.session_url}/control/hang", json=[{"session_id": sid} for sid in hang]
                )
            if resume:
                updates = [
                    {
                        "session_id": sid,
                        **(
                            {"base_worker_id": str(iid[0]), "target_dp_rank": str(iid[1])}
                            if cfg.get("continue_force_pin", False)
                            else {}
                        ),
                    }
                    for sid, iid in resume
                ]
                await self.gateway.request("POST", f"{self.session_url}/control/continue", json=updates)

    async def start_cache_controller(self, model: ModelState) -> str:
        if self.controller is not None:
            if self.controller.poll() is None:
                return self.controller_url
            raise RuntimeError("LMCache controller has exited.")
        _, host = get_host_info()
        port = find_available_port(self.config.psrl.lmcache.controller_base_port)
        directory = Path(os.path.expanduser(self.config.psrl.logging_path))
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / "lmcache_controller.log"
        with log_path.open("w") as output:
            self.controller = subprocess.Popen(
                ["lmcache_controller", "--host", host, "--port", str(port)],
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        self.controller_url = f"http://{host}:{port}"
        deadline = time.monotonic() + self.config.psrl.lmcache.controller_health_timeout_s
        while self.controller.poll() is None and time.monotonic() < deadline:
            try:
                await self.gateway.request(
                    "GET", f"{self.controller_url}/openapi.json", timeout=aiohttp.ClientTimeout(total=2)
                )
                return self.controller_url
            except Exception:
                await asyncio.sleep(1)
        await self.close()
        raise RuntimeError(f"LMCache controller failed to become healthy. See {str(log_path)!r}.")

    async def initialize_cache_peers(self, model: ModelState) -> None:
        if not self.controller_url:
            raise RuntimeError("Start the LMCache controller before initializing peers.")
        await asyncio.gather(
            *(
                replica.server.set_lmcache_controller_url.remote(self.controller_url)
                for replica in model.replicas.values()
            )
        )
        registry, zmq_urls = {}, {}
        for rid, replica in model.replicas.items():
            name = f"psrl_instance_{replica.replica.replica_rank}"
            response = await self.gateway.request(
                "POST", f"{self.controller_url}/query_worker_info", json={"instance_id": name}
            )
            infos = sorted(response["worker_infos"], key=lambda info: info.get("worker_id", 0))
            if not infos:
                raise RuntimeError(f"LMCache controller has no registered ranks for {name!r}.")
            registry[name] = [info.get("peer_init_url", "") for info in infos]
            zmq_urls[rid] = [f"{info.get('ip', '')}:{info.get('port', 0)}" for info in infos]
        await asyncio.gather(
            *(
                replica.server.kv_set_peer_registry.remote(registry, zmq_urls[rid])
                for rid, replica in model.replicas.items()
            )
        )

    async def close(self) -> None:
        if self.controller is not None:
            if self.controller.poll() is None:
                self.controller.terminate()
                try:
                    await asyncio.to_thread(self.controller.wait, timeout=10)
                except subprocess.TimeoutExpired:
                    self.controller.kill()
                    await asyncio.to_thread(self.controller.wait)
            self.controller = None
