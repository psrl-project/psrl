"""
Coordinate rollout, validation, and reward replicas through one model registry.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import Any
from urllib.parse import quote

import ray
from omegaconf import DictConfig
from ray.actor import ActorHandle

from psrl.utils.common.http_utils import find_available_port
from psrl.utils.logger import DualOutputHandler
from psrl.workers.gen.stats_recorder import StatsRecorder
from psrl.workers.gen.zmq_queue import ZMQPullQueue
from psrl.workers.global_coordination.auxiliary import PolicyAuxiliary
from psrl.workers.global_coordination.commands import CommandService
from psrl.workers.global_coordination.lifecycle import GatewayClient, ReplicaLifecycle
from psrl.workers.global_coordination.state import ROLLOUT_SCOPE, CoordinatorScope, ModelState
from psrl.workers.global_coordination.status import StatusService
from psrl.workers.global_coordination.synchronization import WeightSynchronization

psrl_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _CoordinatorMethod:
    actor: ActorHandle
    scope: CoordinatorScope
    method: str

    def remote(self, *args: Any, **kwargs: Any):
        return self.actor.invoke.remote(self.scope, self.method, args, kwargs)


@dataclass(frozen=True)
class CoordinatorClient:
    """
    Address one model scope inside a shared `GlobalCoordinator` actor.

    Clients carry only the actor handle and scope, so they can be passed to the
    parameter server and elastic executor without serializing model state.
    """

    actor: ActorHandle
    scope: CoordinatorScope = ROLLOUT_SCOPE

    def __getattr__(self, method: str) -> _CoordinatorMethod:
        if method.startswith("_"):
            raise AttributeError(method)
        return _CoordinatorMethod(self.actor, self.scope, method)


class GlobalCoordinator:
    """
    Own model state and directly schedule inference control services.

    Validation shares policy weights. Static reward models use the same registry
    and lifecycle services with parameter server synchronization disabled.
    """

    def __init__(
        self,
        config: DictConfig,
        ps_manager: ActorHandle,
        rollout_gateway_url: str,
        session_router_url: str | None = None,
    ) -> None:
        self.config = config
        self.gateway = GatewayClient()
        self.lifecycle = ReplicaLifecycle(self.gateway)
        self.commands = CommandService(config, self.gateway, self.lifecycle)
        self.status = StatusService(config, self.gateway, ps_manager)
        self.weights = WeightSynchronization(config, ps_manager, self.gateway, self.lifecycle, self.commands)
        self.auxiliary = PolicyAuxiliary(config, self.gateway, session_router_url)
        self.models: dict[CoordinatorScope, ModelState] = {}
        self._shutdown_task: asyncio.Task | None = None
        self._register_model(ROLLOUT_SCOPE, rollout_gateway_url, dynamic_weights=True)
        if not any(isinstance(handler, DualOutputHandler) for handler in psrl_logger.handlers):
            psrl_logger.addHandler(DualOutputHandler(config.psrl.logging_path, "GlobalCoordinator"))

    def _register_model(self, scope: CoordinatorScope, gateway_url: str, dynamic_weights: bool) -> None:
        if not gateway_url:
            raise ValueError("Model gateway URL must not be empty.")
        if scope in self.models:
            raise ValueError(f"Model is already registered: {scope!r}.")
        host = ray.util.get_node_ip_address().strip("[]")
        endpoint = f"tcp://{host}:{find_available_port(base_port=28000)}"
        self.models[scope] = ModelState(
            scope, gateway_url.rstrip("/"), dynamic_weights, endpoint, ZMQPullQueue(endpoint=endpoint)
        )

    async def register_reward_model(self, rm_config: DictConfig, gateway_url: str) -> None:
        self._check_running()
        self._register_model(("reward", rm_config.reward_model_name), gateway_url, dynamic_weights=False)

    async def invoke(self, scope: CoordinatorScope, method: str, args: tuple, kwargs: dict) -> Any:
        """
        Resolve a model scope and dispatch directly to the owning service.
        """
        self._check_running()
        if method.startswith("_"):
            raise ValueError(f"Private coordinator methods cannot be invoked: {method!r}.")
        if scope not in self.models:
            raise ValueError(f"Unknown coordinator scope: {scope!r}.")
        model = self.models[scope]
        handlers = {
            "add_worker": model.add_worker,
            "get_status_sink_endpoint": lambda: model.status_endpoint,
            "get_all_instance_ids": model.instances,
            "world_size": lambda: sum(replica.replica.world_size for replica in model.replicas.values()),
            "exec_command": model.commands.exec_command,
            "get_command_status": model.commands.get_command_status,
            "synchronize_command": model.commands.synchronize_command,
            "sleep": partial(self._sleep_tag, model),
            "start_busy_loop": partial(self._start_model, model),
            "get_instance_engine_status_snapshot": lambda: {
                iid: stats.snapshot for iid, stats in model.engine_status.items()
            },
            "get_instance_runtime_snapshot": partial(self.status.runtime_snapshot, model),
            "get_router_backlog_size": partial(self.status.backlog, model),
            "get_router_backlog_summary": partial(self.status.backlog_summary, model),
            "get_router_request_snapshot": partial(self.status.requests, model),
            "execute_request_migrations": partial(self.commands.migrate_requests, model),
        }
        if model.dynamic_weights:
            handlers.update(
                {
                    "init_nixl_client": partial(self.lifecycle.initialize_nixl, model),
                    "nixl_protocol": partial(self.lifecycle.nixl_protocol, model),
                    "nixl_convert_params": partial(self.lifecycle.convert_nixl, model),
                    "initial_pull_from_ps": partial(self.lifecycle.initial_pull, model),
                    "sync_with_ps": partial(self.weights.sync, model),
                    "set_ps_model_version": partial(self.weights.set_ps_version, model),
                    "init_ps_model_version": partial(self._init_version, model),
                    "set_rollout_instance_model_version": lambda rollout_instance_id, version_tag: model.set_version(
                        rollout_instance_id, version_tag
                    ),
                    "update_ready_buffer": model.ready_buffers.add,
                    "check_should_sync": partial(self.weights.should_sync, model),
                    "check_no_activate_tasks": partial(self.weights.check_idle, model),
                    "check_and_migrate": partial(self.weights.migrate, model),
                    "start_lmcache_controller": partial(self.auxiliary.start_cache_controller, model),
                    "init_lmcache_p2p": partial(self.auxiliary.initialize_cache_peers, model),
                }
            )
        if method not in handlers:
            raise ValueError(f"Unknown coordinator method {method!r} for scope {scope!r}.")
        result = handlers[method](*args, **kwargs)
        return await result if inspect.isawaitable(result) else result

    async def _sleep_tag(self, model: ModelState, tag: str = "all") -> bool:
        return await self.lifecycle.sleep(model, model.instances(tag))

    @staticmethod
    def _init_version(model: ModelState, version: int) -> None:
        if version < 0:
            raise ValueError(f"Parameter server version must be nonnegative, got {version}.")
        model.ps_version = version

    async def _start_model(self, model: ModelState) -> None:
        if model.tasks:
            if model.failures:
                raise RuntimeError(f"Model background services failed: {model.failures!r}.")
            return
        if model.dynamic_weights:
            cfg = self.config.psrl.rollout_coordination.sync_and_mig_strategy
            if cfg.method not in {"greedy", "status_based"}:
                raise ValueError(f"Unsupported weight synchronization strategy: {cfg.method!r}.")
            if (cfg.method == "status_based" or cfg.mig.enable) and not self.config.psrl.status_collection.enable:
                raise ValueError("Status based synchronization and migration require status collection.")
            if cfg.mig.enable and not self.config.psrl.rollout_coordination.partial_rollout.enable:
                raise ValueError("Request migration requires partial rollout.")
        model.track("commands", self.commands.run(model))
        if self.config.psrl.status_collection.enable:
            model.track("status", self.status.collect(model))
        if model.dynamic_weights or self.config.psrl.status_collection.enable:
            model.track("broadcast", self.status.broadcast(model))
        if model.dynamic_weights:
            model.track("weights", self.weights.run(model))
            if (
                self.auxiliary.session_url
                and self.config.psrl.rollout_coordination.session_strategy.thunder_agent.enable
            ):
                model.track("sessions", self.auxiliary.sessions(model))
        if self.config.psrl.status_collection.stats_recorder.enable:
            directory = os.path.expanduser(self.config.psrl.logging_path)
            if not model.dynamic_weights:
                directory = os.path.join(directory, "reward", f"model_{quote(model.scope[1], safe='')}")
            model.recorder = StatsRecorder(self.config.psrl, directory)
            model.recorder.write_config(
                self.config.psrl.rollout_coordination.routing_strategy.method,
                self.config.psrl.rollout_coordination.partial_rollout.enable,
            )
            model.track("record", self.status.record(model))

    def _check_running(self) -> None:
        if self._shutdown_task is not None:
            raise RuntimeError("Global coordinator is shutting down.")

    async def stop_busy_loop(self) -> None:
        """
        Drain model services and release shared resources exactly once.
        """
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._shutdown_task)

    async def _shutdown(self) -> None:
        for model in self.models.values():
            model.closing.set()
        tasks = {task for model in self.models.values() for task in model.tasks.values()}
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pending_syncs = {task for model in self.models.values() for task in model.sync_tasks.values()}
        results.extend(await asyncio.gather(*pending_syncs, return_exceptions=True))
        failures = [result for result in results if isinstance(result, BaseException)]
        for model in self.models.values():
            try:
                model.status_queue.close()
                if model.recorder is not None:
                    model.recorder.close()
            except Exception as error:
                failures.append(error)
            failures.extend(model.failures)
        try:
            await self.auxiliary.close()
        finally:
            await self.gateway.session.close()
        if failures:
            raise RuntimeError(f"Global coordinator shutdown failed: {failures!r}.")
