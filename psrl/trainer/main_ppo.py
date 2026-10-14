import logging
import os
import random
import socket

import hydra
import numpy as np
import ray
import torch
import transfer_queue as tq
from omegaconf import OmegaConf
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.device import auto_set_device, is_cuda_available

from psrl.trainer.constants_ppo import get_ppo_ray_runtime_env
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.config import validate_config
from psrl.utils.deployment_mode import expand_ngpus_per_node
from psrl.workers.config.reward_model import resolve_active_managers
from psrl.workers.elastic.pool_layout import elastic_pool_specs, plan_replicas

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def seed_everything(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(config_path="config", config_name="ppo_trainer", version_base=None)
def main(config):
    auto_set_device(config)

    config.transfer_queue.enable = True

    # validate config
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    # AGENT(VERL): Skip migrate legacy reward impl
    run_ppo(config, task_runner_class=TaskRunner)


@ray.remote(num_gpus=1, num_cpus=0)
class _GPUSlotReserver:
    """Lightweight Ray actor that holds a GPU slot without touching CUDA.

    Used to prevent Ray from scheduling job workers onto nodes that the job
    does not intend to use, which would otherwise cause multiple validate
    instances to land on the same GPU when the cluster has excess capacity.
    """

    def ping(self) -> str:
        return "ok"


# NOTE(zyf): these two functions are used for checking plasma in ray whether is big enough to put model parameters.
#   whether needed need to be checked. (TODO)
# @ray.remote(num_cpus=0)
# def _inspect_plasma_backing_on_node() -> dict:
#     node_id = ray.get_runtime_context().get_node_id()
#     return inspect_local_plasma_backing(node_id)


# def _validate_cluster_plasma_backing(config) -> None:
#     """Fail before model startup when Plasma's mmap filesystem is overcommitted."""
#     alive_gpu_nodes = sorted(
#         [node for node in ray.nodes() if node["Alive"] and node["Resources"].get("GPU", 0) > 0],
#         key=lambda node: node["NodeID"],
#     )
#     total_nnodes = config.psrl.deployment.get("total_nnodes", None)
#     if total_nnodes is not None:
#         alive_gpu_nodes = alive_gpu_nodes[: int(total_nnodes)]
#     refs = [
#         _inspect_plasma_backing_on_node.options(
#             scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
#         ).remote()
#         for node in alive_gpu_nodes
#     ]
#     snapshots = ray.get(refs, timeout=60)
#     reserve_bytes = int(os.environ.get("PSRL_PLASMA_BACKING_RESERVE_BYTES", 16 * 1024**3))
#     if reserve_bytes < 0:
#         raise ValueError("PSRL_PLASMA_BACKING_RESERVE_BYTES must be non-negative.")
#     errors = [
#         error
#         for snapshot in snapshots
#         if (error := plasma_backing_error(snapshot, reserve_bytes)) is not None
#     ]
#     if errors:
#         details = "\n".join(f"  - {error}" for error in errors)
#         raise RuntimeError(
#             "Unsafe Ray Plasma backing configuration detected before model startup:\n"
#             f"{details}\n"
#             "Reduce Ray object-store size or clear stale files from its backing filesystem "
#             "until the reported reserve is available."
#         )


def _reserve_excess_nodes(config) -> list:
    """Block GPU slots on nodes that are not needed by this job.

    Reads psrl.deployment.total_nnodes from config. If null/None, does
    nothing. Otherwise, compares against the live cluster node count and
    creates one _GPUSlotReserver actor per GPU on every excess node to
    prevent Ray from scheduling job workers there.

    Args:
        config: Hydra config with psrl.deployment fields.

    Returns:
        list: The list of _GPUSlotReserver actor handles (kept alive by caller).
    """
    total_nnodes = config.psrl.deployment.get("total_nnodes", None)
    if total_nnodes is None:
        return []
    total_nnodes = int(total_nnodes)

    alive_gpu_nodes = sorted(
        [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0],
        key=lambda n: n["NodeID"],
    )
    cluster_nnodes = len(alive_gpu_nodes)

    if cluster_nnodes <= total_nnodes:
        psrl_logger.info(f"Cluster has {cluster_nnodes} GPU nodes, job needs {total_nnodes}. No reservation needed.")
        return []

    excess_nodes = alive_gpu_nodes[total_nnodes:]
    reservers = []
    for node in excess_nodes:
        node_id = node["NodeID"]
        n_gpus = int(node["Resources"]["GPU"])
        psrl_logger.info(
            f"Reserving {n_gpus} GPU slot(s) on excess node {node['NodeManagerAddress']} "
            f"(node_id={node_id}) to prevent stray worker placement."
        )
        for _ in range(n_gpus):
            actor = _GPUSlotReserver.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            ).remote()
            reservers.append(actor)

    # Verify actors are ready before proceeding so placement is confirmed.
    ray.get([r.ping.remote() for r in reservers])
    psrl_logger.info(
        f"[PSRL] Reserved {len(reservers)} GPU slot(s) across "
        f"{len(excess_nodes)} excess node(s) "
        f"(cluster={cluster_nnodes}, job needs={total_nnodes})."
    )
    return reservers


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    if not ray.is_initialized():
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        psrl_logger.info(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    # NOTE(claude): Keep the handle list alive for the entire job lifetime so Ray
    # does not garbage-collect the reservation actors before the job finishes.
    _slot_reservers = _reserve_excess_nodes(config)

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))
    ray.shutdown()

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


@ray.remote(num_cpus=1)
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def __init__(self):
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker (backend selected via config.actor.strategy)."""
        from psrl.workers.gen.vllm_rollout import PSRL_ServerAdapter
        from psrl.workers.train.engine_train_worker import PSRL_EngineTrainWorker as PSRL_TrainWorker

        self.role_worker_mapping[PSRL_Role.Actor] = ray.remote(PSRL_TrainWorker)
        self.role_worker_mapping[PSRL_Role.Rollout] = ray.remote(PSRL_ServerAdapter)
        if config.psrl.colocate_validate_and_train:
            self.role_worker_mapping[PSRL_Role.Validate] = ray.remote(PSRL_ServerAdapter)
        self.mapping[PSRL_Role.Actor] = ["train_pool"]

    def add_critic_worker(self, config):
        """Add critic worker using verl's unified engine TrainingWorker."""
        from verl.workers.engine_workers import TrainingWorker

        if need_critic(config):
            self.role_worker_mapping[PSRL_Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[PSRL_Role.Critic] = ["train_pool"]

    def add_ref_policy_worker(self, config):
        """Add a reference policy worker when KL loss or KL reward is enabled."""
        if not need_reference_policy(config):
            return

        from verl.workers.engine_workers import ActorRolloutRefWorker

        self.role_worker_mapping[PSRL_Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        self.mapping[PSRL_Role.RefPolicy] = ["train_pool"]

    def init_resource_pool_mgr(self, config):
        """Initialize resource pool manager."""
        deployment_config = config.psrl.deployment
        # AGENT(VERL): use train_pool instead of global_pool
        train_pool_id = "train_pool"
        train_bundle_resource_num = 0.9 if config.psrl.colocate_validate_and_train else 1.0
        resource_pool_spec = {
            train_pool_id: expand_ngpus_per_node(
                deployment_config.train_ngpus_per_node, deployment_config.train_nnodes
            ),
        }
        # Unequal bundle fractions prevent two colocated workers from claiming one GPU.
        resource_num_per_bundle = {
            train_pool_id: train_bundle_resource_num,
        }

        elastic_replica_placements = {}

        # Set the resource pool spec for each rollout instance.
        if deployment_config.elastic.enable:
            rollout_pool_id_list = ["shared_rollout_pool"]
        else:
            rollout_pool_id_list = [f"rollout_pool_{i}" for i in range(deployment_config.n_rollout_instances)]

        # If heterogeneous rollout is enabled, we will use the heterogeneous rollout configuration.
        if deployment_config.heterogeneous_rollout.enable:
            if deployment_config.elastic.enable:
                raise ValueError("heterogeneous_rollout is not supported when psrl.deployment.elastic.enable is True.")
            heterogeneous_deployment_config = deployment_config.heterogeneous_rollout
            assert (
                len(heterogeneous_deployment_config.rollout_nnodes_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of rollout nnodes per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of rollout ngpus per node per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.tensor_model_parallel_size_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of tensor model parallel size per instance must match the number of rollout instances."
            assert (
                len(heterogeneous_deployment_config.pipeline_model_parallel_size_per_instance)
                == heterogeneous_deployment_config.n_rollout_instances
            ), "The number of pipeline model parallel size per instance must match the number of rollout instances."

            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [
                    heterogeneous_deployment_config.rollout_ngpus_per_node_per_instance[i]
                ] * heterogeneous_deployment_config.rollout_nnodes_per_instance[i]
        elif deployment_config.elastic.enable:
            elastic_pools = elastic_pool_specs(deployment_config)
            rollout_placements = plan_replicas(
                elastic_pools,
                (
                    config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size
                    * config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size
                    * config.gen_actor_rollout_ref.rollout.data_parallel_size
                ),
            )
            deployment_config.n_rollout_instances = len(rollout_placements)
            psrl_logger.info(
                f"[Elastic] Maximum number of rollout instances = {deployment_config.n_rollout_instances}"
            )
            resource_pool_spec["shared_rollout_pool"] = elastic_pools["shared_rollout_pool"]
            rollout_pool_id_list = []
            for i, placement in enumerate(rollout_placements):
                tag = f"rollout_{i}"
                elastic_replica_placements[tag] = placement
                rollout_pool_id_list.append(tag)
        else:
            for i in range(deployment_config.n_rollout_instances):
                rollout_pool_id = rollout_pool_id_list[i]
                resource_pool_spec[rollout_pool_id] = [
                    deployment_config.rollout_ngpus_per_node_per_instance
                ] * deployment_config.rollout_nnodes_per_instance

            # Mode 4 (trainer_pool_only): append a fixed number of extra rollout
            # replicas on train_pool. train_pool already exists in resource_pool_spec
            # (created above for the actor); do NOT overwrite its spec. These replicas
            # are time-multiplexed with the actor via NIXL sleep/wake.
            if deployment_config.get("mode", None) == "trainer_pool_only":
                idle_rollout = int(deployment_config.get("trainer_pool_idle_rollout_instances", 0))
                if idle_rollout < 0:
                    raise ValueError("trainer_pool_idle_rollout_instances must be >= 0.")
                rollout_pool_id_list = rollout_pool_id_list + [train_pool_id] * idle_rollout
                deployment_config.n_rollout_instances = len(rollout_pool_id_list)
                print(
                    f"[Trainer-pool-only] rollout instances = {deployment_config.n_rollout_instances} "
                    f"(main={deployment_config.n_rollout_instances - idle_rollout}, "
                    f"train_pool_idle={idle_rollout})"
                )

        # Set the resource pool spec for each validation instance.
        if config.psrl.colocate_validate_and_train:
            val_pool_id_list = [f"validate_pool_{i}" for i in range(deployment_config.n_validate_instances)]
            for i in range(deployment_config.n_validate_instances):
                validate_pool_id = val_pool_id_list[i]
                resource_pool_spec[validate_pool_id] = [
                    deployment_config.validate_ngpus_per_node_per_instance
                ] * deployment_config.validate_nnodes_per_instance
                resource_num_per_bundle[validate_pool_id] = 1.0 - train_bundle_resource_num
            self.mapping[PSRL_Role.Validate] = val_pool_id_list

        self.mapping[PSRL_Role.Rollout] = rollout_pool_id_list

        # Reward model resource pool
        total_reward_pool_id_list = []
        for reward_model in resolve_active_managers(config.reward):
            if reward_model.reward_loop_type not in ("gen", "opd"):
                continue
            if deployment_config.elastic.enable:
                reward_model_name = reward_model.get("reward_model_name")
                if not reward_model_name:
                    reward_model_name = reward_model.model.path.split("/")[-1]
                reward_placements = plan_replicas(
                    elastic_pools,
                    (
                        reward_model.rollout.tensor_model_parallel_size
                        * reward_model.rollout.pipeline_model_parallel_size
                        * reward_model.rollout.data_parallel_size
                    ),
                )
                reward_model.num_replicas = len(reward_placements)
                for i, placement in enumerate(reward_placements):
                    tag = f"reward_model_{reward_model_name}_{i}"
                    elastic_replica_placements[tag] = placement
                    total_reward_pool_id_list.append(tag)
                psrl_logger.info(
                    f"[Elastic] Maximum number of reward model"
                    f"({reward_model_name}) instances = {reward_model.num_replicas}"
                )
            elif reward_model.enable_resource_pool:
                if reward_model.n_gpus_per_node <= 0:
                    raise ValueError("reward_model.n_gpus_per_node must be greater than 0")
                if reward_model.nnodes <= 0:
                    raise ValueError("reward_model.nnodes must be greater than 0")

                reward_model_instances = reward_model.get("num_replicas", 1)
                reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
                reward_pool_id_list = [f"reward_pool_{reward_model_name}_{i}" for i in range(reward_model_instances)]
                for i in range(reward_model_instances):
                    resource_pool_spec[reward_pool_id_list[i]] = [
                        reward_model.rollout_ngpus_per_instance_per_node
                    ] * reward_model.rollout_nnodes_per_instance
                total_reward_pool_id_list.extend(reward_pool_id_list)
            else:
                raise ValueError(
                    "reward_model.enable_resource_pool must be True when psrl.deployment.elastic.enable is False."
                )

        self.mapping[PSRL_Role.RewardModel] = total_reward_pool_id_list

        from psrl.trainer.ppo.utils import ResourcePoolManager

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec,
            mapping=self.mapping,
            resource_num_per_bundle=resource_num_per_bundle,
            elastic_replica_placements=elastic_replica_placements,
        )

        print(f"resource_pool_spec = {resource_pool_spec}, mapping = {self.mapping}")

        return resource_pool_manager

    def add_reward_model_worker(self, config):
        """Add reward model worker."""
        from psrl.workers.gen.vllm_rollout import PSRL_ServerAdapter

        self.role_worker_mapping[PSRL_Role.RewardModel] = ray.remote(PSRL_ServerAdapter)

    def add_dummy_worker(self, config):
        from psrl.trainer.ppo.utils import PSRL_DummyWorker

        self.role_worker_mapping[PSRL_Role.DummyPolicy] = ray.remote(PSRL_DummyWorker)
        self.mapping[PSRL_Role.DummyPolicy] = ["train_pool"]

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        from concurrent.futures import ThreadPoolExecutor

        # Overlap tq.init (network/storage I/O, ~38s) with add_workers (Python imports, ~79s)
        # since they have no shared dependencies. This reduces total from ~117s to ~79s.
        with ThreadPoolExecutor(max_workers=1) as executor:
            tq_future = executor.submit(tq.init, config.transfer_queue)
            self.add_actor_rollout_worker(config)
            self.add_critic_worker(config)
            self.add_ref_policy_worker(config)

            # AGENT(VERL): PSRL use reward model worker.
            self.add_reward_model_worker(config)

            # NOTE(linsh): Add a dummy worker so Ray does not classify fused workers as asynchronous.
            self.add_dummy_worker(config)

            tq_future.result()

        trainer = None
        try:
            resource_pool_manager = self.init_resource_pool_mgr(config)

            # NOTE(linsh): Lazily import `PSRL_RayPPOTrainer` here to avoid implicit `ray.init()`
            # during initialization of nixl modules.
            from psrl.trainer.ppo.ray_trainer import PSRL_RayPPOTrainer

            # Initialize the PPO trainer.
            trainer = PSRL_RayPPOTrainer(
                config=config,
                role_worker_mapping=self.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
            )
            # Initialize the workers of the trainer.
            trainer.init_workers()
            # Start the training process.
            trainer.fit()
        finally:
            if trainer:
                trainer.replay_buffer.close()
            tq.close()


if __name__ == "__main__":
    seed_everything(0)
    main()
