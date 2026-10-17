from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

import ray
from omegaconf import DictConfig
from verl.single_controller.ray import RayWorkerGroup
from verl.utils import hf_tokenizer, omega_conf_to_dataclass
from verl.utils.fs import copy_to_local

from psrl.workers.config import RolloutConfig
from psrl.workers.gen.vllm_async_server import GenInterface
from psrl.workers.reward.reward_model.coordinator import RewardModelCoordinator
from psrl.workers.reward.reward_model.replica import RewardModelReplica

if TYPE_CHECKING:
    from psrl.workers.global_coordinator import CoordinatorClient

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class RewardModelManager:
    """Manage all replicas for one named reward model."""

    def __init__(
        self,
        reward_model_name: str,
        config: DictConfig,
        reward_model_config: DictConfig,
        reward_model_wg_list: list[RayWorkerGroup],
        gateway_url: str,
        coordinator: ray.actor.ActorHandle | CoordinatorClient | None = None,
    ) -> None:
        self.reward_model_name = reward_model_name
        self.config = config
        self.reward_model_config = reward_model_config
        self.gateway_url = gateway_url

        model_cfg = reward_model_config.model
        tokenizer_path = model_cfg.get("tokenizer_path") or model_cfg.path
        local_path = copy_to_local(tokenizer_path, use_shm=model_cfg.get("use_shm", False))
        self.reward_model_tokenizer = hf_tokenizer(
            local_path,
            trust_remote_code=model_cfg.get("trust_remote_code", False),
        )
        self.hf_model_config = model_cfg
        self.rollout_config: RolloutConfig = omega_conf_to_dataclass(reward_model_config.rollout)

        self.reward_model_coordinator = coordinator
        if self.reward_model_coordinator is None:
            self.reward_model_coordinator = ray.remote(RewardModelCoordinator).remote(
                config,
                reward_model_config,
                rollout_gateway_url=self.gateway_url,
            )

        self.reward_model_wg_list = reward_model_wg_list
        self.replicas: list[RewardModelReplica] = []

        self._initialize_reward_replicas()
        self._register_reward_servers(self.replicas)

        ray.get(self.reward_model_coordinator.start_busy_loop.remote())

        psrl_logger.info(
            "RewardModelManager for '%s' initialized with %d replica(s).",
            reward_model_name,
            len(self.replicas),
        )

    def _initialize_reward_replicas(self):
        status_endpoint: str = ray.get(self.reward_model_coordinator.get_status_sink_endpoint.remote())
        replica_world_size = (
            self.rollout_config.tensor_model_parallel_size
            * self.rollout_config.data_parallel_size
            * self.rollout_config.pipeline_model_parallel_size
        )

        init_tasks = []
        replica_rank = 0
        for wg in self.reward_model_wg_list:
            wg_world_size = wg.world_size
            if replica_world_size <= 0 or wg_world_size % replica_world_size != 0:
                raise ValueError(
                    f"Reward worker group world_size={wg_world_size} is not divisible by "
                    f"replica world_size={replica_world_size}."
                )
            num_local_replicas = wg_world_size // replica_world_size
            if num_local_replicas < 1:
                raise ValueError(
                    f"Reward worker group has world_size={wg_world_size} but each replica needs "
                    f"{replica_world_size} workers."
                )
            for local_replica_rank in range(num_local_replicas):
                gen_interface = GenInterface(
                    role=f"reward_model_{self.reward_model_name}",
                    rollout_replica_idx=replica_rank,
                    status_endpoint=status_endpoint,
                    ps_manager_handle=None,
                )
                replica = RewardModelReplica(
                    replica_rank=replica_rank,
                    local_replica_rank=local_replica_rank,
                    psrl_config=self.config.psrl,
                    config=self.rollout_config,
                    model_config=self.hf_model_config,
                    gen_interface=gen_interface,
                    reward_model_name=self.reward_model_name,
                    gpus_per_node=self.reward_model_config.rollout_ngpus_per_instance_per_node,
                )
                init_tasks.append(replica.init_model(wg))
                self.replicas.append(replica)
                replica_rank += 1
        self._run_all(init_tasks)
        if self.config.psrl.deployment.elastic.enable:
            ray.get([replica.servers[0].sleep.remote(level=2) for replica in self.replicas])

    def _register_reward_servers(self, replicas: list[RewardModelReplica]):
        reg_futures = [
            replica.servers[0].register_server_to_gateway.remote(
                self.gateway_url,
                model_id=self.reward_model_name,
            )
            for replica in replicas
        ]
        worker_ids: list[str] = ray.get(reg_futures)
        for replica, worker_id in zip(replicas, worker_ids, strict=True):
            replica.gateway_replica_id = worker_id

        coord_futures = [
            self.reward_model_coordinator.add_worker.remote(
                replica,
                replica.servers[0],
                worker_id,
                replica.data_parallel_size,
                is_validate=False,
                model_version=0,
            )
            for replica, worker_id in zip(replicas, worker_ids)
        ]
        ray.get(coord_futures)

    def get_gateway_url(self) -> str:
        """Return the smg gateway HTTP URL for this reward model."""
        return self.gateway_url

    def get_reward_model_tokenizer(self):
        """Return the reward model tokenizer (for prompt construction in GenRewardLoop)."""
        return self.reward_model_tokenizer

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())
