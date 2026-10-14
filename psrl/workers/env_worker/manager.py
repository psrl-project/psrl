"""Construct env worker pools according to placement policy."""

from __future__ import annotations

import logging
import os
from typing import Any

import ray
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from psrl.workers.env_worker.coordinator import EnvWorkerCoordinator
from psrl.workers.env_worker.worker import EnvWorker

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def resolve_placement_node_ips(config: DictConfig, alive_node_ips: list[str]) -> list[str]:
    """
    Decide which nodes receive env workers.

    Args:
        config (DictConfig): The `psrl.env_worker` config group.
        alive_node_ips (list[str]): IPs of alive Ray nodes.

    Returns:
        list[str]: Node IPs that should host workers.

    Raises:
        ValueError: If the policy is unknown, or a dedicated policy names no node or
            names a node that is not alive.
    """
    placement = config.placement
    if placement == "colocated":
        return list(alive_node_ips)
    if placement == "dedicated":
        dedicated = list(config.dedicated_node_ips or [])
        if not dedicated:
            raise ValueError("placement=dedicated requires a non-empty env_worker.dedicated_node_ips list.")
        missing = [ip for ip in dedicated if ip not in alive_node_ips]
        if missing:
            raise ValueError(f"Dedicated env nodes {missing!r} are not alive in the Ray cluster.")
        return dedicated
    raise ValueError(f"Unknown placement policy {placement!r}, expected colocated or dedicated.")


class EnvWorkerManager:
    """Own the coordinator and the worker actors for one training job."""

    def __init__(self, config: DictConfig):
        """
        Build the coordinator and every worker actor.

        Args:
            config (DictConfig): The full PSRL config. Reads `psrl.env_worker`.
        """
        self.env_config = config.psrl.env_worker
        node_by_ip = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes() if node["Alive"]}
        target_ips = resolve_placement_node_ips(self.env_config, list(node_by_ip))

        self.coordinator = (
            ray.remote(EnvWorkerCoordinator)
            .options(
                name="env_worker_coordinator",
                max_concurrency=int(self.env_config.get("coordinator_max_concurrency", 64)),
            )
            .remote(routing_method=self.env_config.routing.method)
        )

        cpu_slots = int(self.env_config.cpu_slots_per_worker)
        gpu_slots = int(self.env_config.gpu_slots_per_worker)
        self.workers: list[Any] = []
        worker_id = 0

        for node_ip in target_ips:
            for _ in range(int(self.env_config.workers_per_node)):
                worker = (
                    ray.remote(EnvWorker)
                    .options(
                        name=f"env_worker_{worker_id}",
                        num_cpus=float(self.env_config.get("worker_num_cpus", 1)),
                        num_gpus=float(gpu_slots),
                        max_concurrency=cpu_slots + 4,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_by_ip[node_ip], soft=False),
                    )
                    .remote(
                        worker_id=worker_id,
                        cpu_slots=cpu_slots,
                        gpu_slots=gpu_slots,
                        exec_default_timeout_s=float(self.env_config.exec_default_timeout_s),
                        max_observation_chars=int(self.env_config.max_observation_chars),
                        idle_sandbox_timeout_s=float(self.env_config.idle_sandbox_timeout_s),
                    )
                )
                ray.get(self.coordinator.register_worker.remote(worker, worker_id, cpu_slots, gpu_slots, node_ip))
                self.workers.append(worker)
                worker_id += 1

        psrl_logger.info(
            f"Env worker pool is ready. Workers={len(self.workers)}, "
            f"nodes={len(target_ips)}, placement={self.env_config.placement!r}."
        )

    def coordinator_handle(self) -> Any:
        """Return the coordinator actor handle for agent loops."""
        return self.coordinator

    def shutdown(self) -> None:
        """Kill every worker and the coordinator."""
        for worker in self.workers:
            ray.kill(worker, no_restart=True)
        ray.kill(self.coordinator, no_restart=True)
        psrl_logger.info("Env worker pool shut down.")
