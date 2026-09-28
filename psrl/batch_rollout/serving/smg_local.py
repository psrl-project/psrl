"""Serve batch rollout from a local SMG gateway over PSRL vLLM replicas.

The full RL rollout stack minus the parameter server: SMG plus SessionRouter in
front of `PSRL_vLLMReplica` engines that load the checkpoint themselves. TITO
therefore captures the token stream the engines actually processed, which is what
makes `batch_rollout.dump_tokens` meaningful here and nowhere else.

Follows `psrl/workers/reward/reward_model/`, which already runs this stack without
a PS: a `GenInterface` with `ps_manager_handle=None` turns off request-status
tracking and weight sync, and SMG's `psrl` worker selector tolerates an empty
`ps_manager_addr` because every PS call site is guarded.
"""

import asyncio
import logging
import os

import ray
from omegaconf import DictConfig
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig

from psrl.batch_rollout.serving.base import BackendHandle, ServingBackend
from psrl.workers.config import RolloutConfig
from psrl.workers.gen.rollout_gateway import RolloutGateway
from psrl.workers.gen.vllm_async_server import GenInterface, PSRL_vLLMReplica
from psrl.workers.gen.vllm_rollout import PSRL_ServerAdapter

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class BatchRolloutReplica(PSRL_vLLMReplica):
    """One vLLM replica that owns its weights and reports to no parameter server.

    Mirrors `RewardModelReplica`. The base class decides `load_format` from
    `gen_interface.ps_manager_handle`, so passing `None` is what makes the engine
    read the checkpoint instead of starting on dummy weights.
    """

    def __init__(
        self,
        replica_rank: int,
        local_replica_rank: int,
        psrl_config,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gen_interface: GenInterface,
        gpus_per_node: int = 8,
    ):
        """
        Args:
            replica_rank (int): Global replica rank, used for naming.
            local_replica_rank (int): Rank within this replica's worker group.
            psrl_config: The `psrl` config node.
            config (RolloutConfig): Rollout configuration.
            model_config (HFModelConfig): HuggingFace model configuration.
            gen_interface (GenInterface): Status reporting. `ps_manager_handle` must
                be `None`, because collection has no parameter server.
            gpus_per_node (int): GPUs per node for this replica.
        """
        if gen_interface.ps_manager_handle is not None:
            raise ValueError(
                "BatchRolloutReplica requires gen_interface.ps_manager_handle=None. A PS handle "
                "would make the engine start on dummy weights and wait for a push that never comes."
            )
        super().__init__(
            replica_rank=replica_rank,
            local_replica_rank=local_replica_rank,
            psrl_config=psrl_config,
            config=config,
            model_config=model_config,
            gen_interface=gen_interface,
            gpus_per_node=gpus_per_node,
            tag="rollout",
        )


class SMGLocalServingBackend(ServingBackend):
    """Launch SMG, the SessionRouter, and the vLLM replicas behind them."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.gateway: ray.actor.ActorHandle | None = None
        self.replicas: list[BatchRolloutReplica] = []
        self.worker_groups: list[RayWorkerGroup] = []

    def start(self) -> BackendHandle:
        """Bring the whole local stack up and return how to reach it.

        Returns:
            BackendHandle: Gateway, session router, and the served model name.
        """
        serving = self.config.batch_rollout.serving
        n_instances = int(serving.get("n_instances", 1))
        ngpus_per_node = int(serving.get("ngpus_per_node_per_instance", 2))
        nnodes = int(serving.get("nnodes_per_instance", 1))

        # Checked before the dataclass conversion and before any GPU is claimed, so a
        # bad shape is a clear error rather than a placement group that never fills.
        raw_rollout = self.config.gen_actor_rollout_ref.rollout
        world_size = (
            int(raw_rollout.tensor_model_parallel_size)
            * int(raw_rollout.pipeline_model_parallel_size)
            * int(raw_rollout.data_parallel_size)
        )
        if world_size != ngpus_per_node * nnodes:
            raise ValueError(
                f"Replica world size {world_size} (tp * pp * dp) must equal "
                f"ngpus_per_node_per_instance * nnodes_per_instance = {ngpus_per_node * nnodes}."
            )

        rollout_config: RolloutConfig = omega_conf_to_dataclass(raw_rollout)
        model_config = self.config.gen_actor_rollout_ref.model

        # NOTE(claude): An empty PS address leaves SMG's `ps_manager_client` unset.
        # Every PS call site is guarded, so version filtering falls back to the
        # worker's `weight_version` label and the other routing stages still apply.
        self.gateway = RolloutGateway.remote(self.config, "", 0)
        gateway_url = ray.get(self.gateway.launch_router.remote())
        session_router_url = ray.get(self.gateway.launch_session_router.remote())
        psrl_logger.info("SMG gateway at %s, session router at %s.", gateway_url, session_router_url)

        self._launch_replicas(n_instances, ngpus_per_node, nnodes, rollout_config, model_config)
        self._register_replicas(gateway_url)

        model_name = str(serving.get("model_name", "") or "") or str(model_config.path)
        return BackendHandle(
            api_base_url=f"{gateway_url}/v1",
            model_name=model_name,
            session_router_url=session_router_url,
            rollout_gateway_url=gateway_url,
            # TITO records what the engines processed, so a token dump is measured
            # rather than reconstructed.
            supports_token_capture=True,
        )

    def _launch_replicas(self, n_instances, ngpus_per_node, nnodes, rollout_config, model_config) -> None:
        """Create one worker group and one replica per instance."""
        # No coordinator receives engine stats, so there is no ZMQ sink to name.
        # `GenInterface` treats a None endpoint as "reporting disabled".
        status_endpoint = None

        init_tasks = []
        for replica_rank in range(n_instances):
            resource_pool = RayResourcePool(
                process_on_nodes=[ngpus_per_node] * nnodes,
                use_gpu=True,
                max_colocate_count=1,
                name_prefix=f"batch_rollout_pool_{replica_rank}",
            )
            worker_group = RayWorkerGroup(
                resource_pool=resource_pool,
                ray_cls_with_init=RayClassWithInitArgs(
                    cls=ray.remote(PSRL_ServerAdapter),
                    config=self.config.gen_actor_rollout_ref.rollout,
                    model_config=model_config,
                    device_mesh=None,
                ),
            )
            self.worker_groups.append(worker_group)

            replica = BatchRolloutReplica(
                replica_rank=replica_rank,
                local_replica_rank=0,
                psrl_config=self.config.psrl,
                config=rollout_config,
                model_config=omega_conf_to_dataclass(model_config),
                gen_interface=GenInterface(
                    role="rollout",
                    rollout_replica_idx=replica_rank,
                    status_endpoint=status_endpoint,
                    ps_manager_handle=None,
                ),
                gpus_per_node=ngpus_per_node,
            )
            self.replicas.append(replica)
            init_tasks.append(replica.init_model(worker_group))

        asyncio.run(_gather(init_tasks))
        psrl_logger.info("Launched %d vLLM replica(s) with real checkpoint weights.", len(self.replicas))

    def _register_replicas(self, gateway_url: str) -> None:
        """Register each replica's engine with SMG.

        Unlike RL there is no PS registration and no coordinator: nothing publishes
        a new weight version, so there is no sync to drive.
        """
        worker_ids = ray.get(
            [replica.servers[0].register_server_to_gateway.remote(gateway_url) for replica in self.replicas]
        )
        psrl_logger.info("Registered %d replica(s) with SMG: %s.", len(worker_ids), worker_ids)

    def stop(self) -> None:
        """Shut the gateway down. Ray reclaims the replica actors with the job."""
        if self.gateway is not None:
            try:
                ray.get(self.gateway.shutdown_router.remote(), timeout=60)
            except Exception as exc:
                psrl_logger.warning("SMG gateway shutdown failed: %s.", exc)
            self.gateway = None


async def _gather(tasks) -> None:
    """Await the replica init coroutines together."""
    await asyncio.gather(*tasks)
