import asyncio
import json
import logging
import math
import os
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import requests
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf, open_dict
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from tensordict import TensorDict
from tqdm import tqdm
from transfer_queue import KVBatchMeta
from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import (
    SubRayResourcePool,
    create_colocated_worker_cls_fused,
    sort_placement_group_by_node_ip,
)
from verl.trainer.distillation.losses import is_distillation_enabled
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.padding_utils import upsample_batch_to_divisible_size
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, apply_kl_penalty
from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
from verl.trainer.ppo.utils import (
    WorkerType,
    need_critic,
    need_reference_policy,
    need_teacher_policy,
)
from verl.utils import hf_processor, hf_tokenizer
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.debug.metrics import calculate_debug_metrics
from verl.utils.fs import copy_to_local
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import DistillationConfig, EngineConfig
from verl.workers.utils.padding import response_from_nested, response_to_nested

from psrl.sandbox.capacity import SandboxCapacityCoordinator
from psrl.trainer.ppo.batch_schedule import (
    TRAJECTORY_AGG_MODE,
    BatchScheduleStep,
    build_batch_schedule,
    get_batch_schedule_strategy,
    resolve_sample_keys,
)
from psrl.trainer.ppo.session_loss import compute_session_loss_weights
from psrl.trainer.ppo.utils import (
    PSRL_Role,
    ResourcePoolManager,
    _compute_termination_metrics,
    compute_advantage_for_multi_trajectories,
)
from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import WorkerKey, ps_agent_name, train_client_name
from psrl.utils.dataset import DataProcessor
from psrl.utils.elastic_rm.cluster_topology import ClusterTopology
from psrl.utils.elastic_rm.elastic_executor import ElasticExecutor
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
)
from psrl.utils.post_processor import (
    load_buffer_post_processor,
    load_group_post_processor,
)
from psrl.utils.server.command import Command, CommandType
from psrl.utils.transferqueue_utils import PayloadState, clear_payload
from psrl.workers.agent_loop.manager import PSRL_AgentLoopManager
from psrl.workers.agent_loop.prometheus_utils import update_prometheus_config
from psrl.workers.agent_loop.worker import PSRL_AgentLoopWorker
from psrl.workers.config.reward_model import resolve_active_managers
from psrl.workers.gen.rollout_coordination import RolloutCoordinator
from psrl.workers.gen.rollout_gateway import RolloutGateway
from psrl.workers.gen.smg_adapter import build_pause_resume_payload
from psrl.workers.gen.vllm_async_server import GenInterface, PSRL_vLLMReplica
from psrl.workers.ps import (
    PSClassWithInitArgs,
    PSManager,
    PSResourcePool,
    PSResourceSpec,
    PSStoragePlan,
    PSStorageWorker,
    PSWorkerGroup,
)
from psrl.workers.reward.reward_manager import RewardLoopManager
from psrl.workers.reward.reward_model import RewardModelManager
from psrl.workers.reward.reward_model.gateway import RewardModelGateway
from psrl.workers.reward.reward_protocol import RewardModelRuntimeInfo
from psrl.workers.reward.reward_worker import RewardLoopWorker
from psrl.workers.train import TrainInterface

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class ReplayBuffer:
    """Replay buffer periodically polls metadata from transfer queue.

    Args:
        poll_interval (float, optional): Poll interval in seconds. Defaults to 0.5.
    """

    def __init__(self, poll_interval: float = 0.5):
        # partition_id => {key: tags}
        self.partitions: dict[str, dict[str, dict]] = defaultdict(dict)

        self.poll_interval = poll_interval
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        self.poll_thread = threading.Thread(target=self._poll_from_transfer_queue, daemon=True)

    def _poll_from_transfer_queue(self):
        """Periodically poll metadata from transfer queue."""
        try:
            while not self._stop_event.is_set():
                data = tq.kv_list()
                if data is not None:
                    for partition_id, items in data.items():
                        self.add(partition_id, items)
                self._stop_event.wait(self.poll_interval)
        except Exception as e:
            if not self._stop_event.is_set():
                psrl_logger.error(f"Error in _poll_from_transfer_queue: {e}")
                os._exit(1)

    def start_polling(self):
        """Start the background polling thread."""
        if not self.poll_thread.is_alive():
            self.poll_thread.start()
            psrl_logger.info("ReplayBuffer polling thread started.")

    def close(self):
        """Stop the background polling thread."""
        if not self.poll_thread.is_alive():
            return
        self._stop_event.set()
        self.poll_thread.join(timeout=self.poll_interval + 1.0)
        if self.poll_thread.is_alive():
            psrl_logger.warning("ReplayBuffer poll thread did not stop within timeout")

    def add(self, partition_id: str, items: dict[str, dict]):
        """Add items to the replay buffer.

        Args:
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
            items (dict[str, dict]): Items to add, e.g. {"key": {"tag": "value"}}.
        """
        with self.lock:
            partition = self.partitions[partition_id]
            for key, tags in items.items():
                if key not in partition:
                    partition[key] = {}
                partition[key].update(tags)

    def remove(self, keys: list[str], partition_id: str):
        """Remove items from the replay buffer.

        Args:
            keys (list[str]): Keys to remove.
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
        """
        with self.lock:
            partition = self.partitions[partition_id]
            for key in keys:
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    origin_key = parts[0]
                    if origin_key in partition:
                        del partition[origin_key]
                if key in partition:
                    del partition[key]

    def sample(self, keys: list[str], partition_id: str):
        """Sample a batch of data from the replay buffer.

        Args:
            keys (list[str]): Keys to sample.
            partition_id (str): Partition of transfer queue, e.g. "train" or "val".
        """
        log_interval = 10.0
        next_log_time = time.monotonic() + log_interval
        while True:
            time.sleep(self.poll_interval)
            with self.lock:
                partition = self.partitions[partition_id]
                success_count = 0
                running_count = 0
                unknown_count = 0
                missing_keys = []
                for key in keys:
                    tag = partition.get(key)
                    if tag is None:
                        missing_keys.append(key)
                        continue

                    status = tag.get("status", "running")
                    if status == "running":
                        running_count += 1
                    elif status == "success":
                        success_count += 1
                    else:
                        unknown_count += 1
                        psrl_logger.debug(f"Unknown status {status!r} for key {key!r}.")

            should_wait = bool(missing_keys) or running_count > 0
            if not should_wait:
                return

            current_time = time.monotonic()
            if current_time >= next_log_time:
                psrl_logger.warning(
                    f"ReplayBuffer is waiting for samples: partition_id={partition_id!r}, total={len(keys)}, "
                    f"success={success_count}, running={running_count}, missing={len(missing_keys)}, "
                    f"unknown={unknown_count}, missing_keys[:20]={missing_keys[:20]!r}."
                )
                next_log_time = current_time + log_interval


class PSRL_RayPPOTrainer(RayPPOTrainer):
    def __init__(
        self,
        config,
        role_worker_mapping: dict[PSRL_Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process and is responsible for managing the training process.

        Args:
            config: Configuration object containing training parameters.
            role_worker_mapping (dict[PSRL_Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resources.
        """

        # AGENT(VERL): PSRL use `config.train_actor_rollout_ref` instead of `config.actor_rollout_ref` in verl.

        self.config = config

        # AGENT(VERL): skip `hybrid_engine` in PSRL

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)
        self.use_teacher_policy = need_teacher_policy(self.config)

        # AGENT(VERL): skip `use_rm` in PSRL.

        self.use_critic = need_critic(self.config)
        self.device_name = self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.train_actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.train_actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.train_actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.train_actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.batch_agg_mode = self.config.psrl.agentic_rl.get(
            "batch_agg_mode",
            TRAJECTORY_AGG_MODE,
        )
        self.batch_schedule_strategy = get_batch_schedule_strategy(self.batch_agg_mode)

        # AGENT(VERL): skip legacy workers and use PSRL's `data_processor` instead of veRL's dataloader.
        # Load PSRL post-processors from configuration.
        self.group_post_process_fn = load_group_post_processor(config)
        self.buffer_post_process_fn = load_buffer_post_processor(config)

        # CPU workers for Streaming Rollout
        self.data_processor = None
        self.agent_loop_manager = None
        self.rollout_coordinator = None
        self.reward_manager = None
        self.reward_loop_workers = []
        self.env_worker_manager = None

        self.reward_gateways: dict[str, ray.actor.ActorHandle] = {}
        self.reward_gateway_urls: dict[str, str] = {}
        self.reward_model_to_manager = {}

        # Elastic rm
        self.elastic_rm_mode = config.psrl.deployment.elastic_rm.enable
        self.elastic_executor = None

        self.rollout_replicas = []
        # Serialize rollout and validation index allocation to prevent duplicate gateway registrations.
        self._rollout_replica_alloc_lock = threading.Lock()
        self.tag_to_server_handles = {}
        self.tag_to_base_worker_ids = {}
        self.server_addresses = []

        self.replay_buffer = ReplayBuffer(poll_interval=0.1)

        # Rollout gateway lifecycle and endpoints.
        self.rollout_gateway_url: str | None = None
        self.session_router_url: str | None = None

        # HTTP session for rollout gateway control
        self._gateway_http_session = requests.Session()
        configured_n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        configured_n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )
        gateway_pool_size = max(configured_n_rollout_instances + configured_n_validate_instances, 32)
        gateway_adapter = requests.adapters.HTTPAdapter(
            pool_connections=gateway_pool_size,
            pool_maxsize=gateway_pool_size,
            max_retries=0,
        )
        self._gateway_http_session.mount("http://", gateway_adapter)
        self._gateway_http_session.mount("https://", gateway_adapter)
        self._gateway_http_timeout = None

        # Parameter server handle for other workers to access
        self.ps_manager_handle = None
        self.ps_manager_grpc_port = None

        # Last validation metrics (set by FullBatchStepStrategy at the final step).
        # Initialised to None so callers never get AttributeError when no validation ran.
        self._last_val_metrics = None

        # Async rollout mode for training worker
        self.async_rollout_mode = False

        # Indicate whether current mode is rollout mode in actor
        self.is_rollout_mode_in_actor = (
            self.config.psrl.colocate_validate_and_train and self.config.trainer.val_before_train
        )
        psrl_logger.warning(
            f"[INIT] is_rollout_mode_in_actor: {self.is_rollout_mode_in_actor} "
            f"(colocate_validate_and_train={self.config.psrl.colocate_validate_and_train}, "
            f"val_before_train={self.config.trainer.val_before_train})"
        )

        # Mappings from WorkerKey to Ray node id and PS instance index for NIXL.
        self.worker_to_node_id: dict[WorkerKey, str] = {}
        self.worker_to_ps_idx: dict[WorkerKey, int] = {}

        self.n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        self.n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )

        self._initialize_queue_buffers()
        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.max_concurrency = (
                self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
                * self.config.psrl.rollout_coordination.redundant_rollout.redundant_global_batch_size
                * (self.config.psrl.staleness + 1)
            )
        else:
            self.max_concurrency = (
                self.rollout_n  # instance variable set earlier in __init__
                * self.config.psrl.staleness_buffer_entries
                * (self.config.psrl.staleness + 1)
            )

        # Build logger
        self.log_prefix = "MainRayTrainer"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info("Initialized major ray trainer (single controller).")

        # Cache IP-to-Ray-node-ID mapping for scheduling actors to specific nodes.
        self.ip_to_node_id: dict[str, str] = {node["NodeManagerAddress"]: node["NodeID"] for node in ray.nodes()}

        # Pin each port scanner to its node so availability checks use the correct network namespace.
        from psrl.utils.nixl.port_scanner import create_port_scanners

        self.port_scanner_handles = create_port_scanners(self.ip_to_node_id)

        self._init_ps_manager()

        # NOTE(linsh): Launch the gateway early to overlap its startup with remaining initialization.
        self.rollout_gateway = self.init_rollout_gateway()
        self._launch_gateway_future = self.rollout_gateway.launch_router.remote()

        # Initialize data first so worker optimizer schedules receive `total_training_steps`.
        self._init_tokenizer()
        self._init_data_processor()
        self._init_dump_executor()

    def _init_tokenizer(self):
        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            self.config.train_actor_rollout_ref.model.path,
            use_shm=self.config.train_actor_rollout_ref.model.get("use_shm", False),
        )

        trust_remote_code = self.config.data.get("trust_remote_code", False)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    def _initialize_queue_buffers(self):
        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        assert self.rollout_n >= self.alg_rollout_n, (
            f"Rollout n {self.rollout_n} must be greater than or equal to alg_rollout_n {self.alg_rollout_n}."
        )

        # Data queue is the communication handle between the data processor and the rollout server.
        # The size of the queue is determined by the batch size and the rollout n.
        self.data_queue_size = (
            self.config.data.get("gen_batch_size", self.config.data.train_batch_size) * self.rollout_n
        )

        psrl_logger.debug(
            "Initialized data_queue with sizes: %d.",
            self.data_queue_size,
        )

    def _init_ps_manager(self):
        """Initialize the PS manager for handling model version, requests condition and staleness."""
        # Set the validation rollout number in the config
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "psrl"):
                    self.config.psrl.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        except Exception as e:
            psrl_logger.warning(f"Could not set val_rollout_n in config. Structure missing? Error: {e}")

        ip_to_node_id = self.ip_to_node_id
        assert self.config.psrl.ps_manager_ip in ip_to_node_id, (
            f"PSManager IP {self.config.psrl.ps_manager_ip} not found in ray nodes"
        )
        psrl_logger.info("Getting the handle of the PSManager")
        self.ps_manager_handle = (
            ray.remote(PSManager)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.psrl.ps_manager_ip], soft=False
                )
            )
            .remote(self.config.psrl)
        )
        self.ps_manager_grpc_port = ray.get(self.ps_manager_handle.start_grpc_server.remote())

    def stop_ps_manager(self):
        """Stop the PS manager."""
        if self.ps_manager_handle is not None:
            psrl_logger.info("Stopping PS manager...")
            ray.get(self.ps_manager_handle.shutdown_nixl_server.remote())
            psrl_logger.info("NIXL server stopped successfully.")
            ray.get(self.ps_manager_handle.stop_grpc_server.remote())
            self.ps_manager_handle = None
            psrl_logger.info("PS manager stopped successfully.")
        else:
            psrl_logger.warning("PS manager is not initialized, skipping stop operation.")

    def _init_data_processor(self):
        """Initialize the data processor for handling data preprocessing and batching."""
        if self.data_processor is not None:
            return

        # Initialize the data processor
        self.data_processor = DataProcessor.remote(self.config, self.tokenizer, self.processor, self.ps_manager_handle)

        # Get total training steps from the data processor where dataloaders are built
        self.total_training_steps = ray.get(self.data_processor.get_total_training_steps.remote())

        psrl_logger.info(f"Total training steps: {self.total_training_steps}")

        # Set the total training steps in the config
        # AGENT(VERL): this logic is mapped to `_create_dataloader` in `verl/trainer/ppo/ray_trainer.py`
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "train_actor_rollout_ref.actor.optim"):
                    self.config.train_actor_rollout_ref.actor.optim.total_training_steps = self.total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = self.total_training_steps
        except Exception as e:
            psrl_logger.warning(f"Could not set total_training_steps in config. Structure missing? Error: {e}")

    def start_data_processor(self):
        """Launch the data processor for processing data in the background."""
        assert self.data_processor is not None, "Data processor must be initialized before starting it."

        ray.get(self.data_processor.start_busy_loop.remote())

    def stop_data_processor(self):
        """Stop the data processor."""
        if self.data_processor is not None:
            psrl_logger.debug("Stopping data processor...")
            try:
                ray.get(self.data_processor.stop_busy_loop.remote(), timeout=60)
                self.data_processor = None
                psrl_logger.debug("Data processor stopped successfully.")
            except ray.exceptions.GetTimeoutError:
                psrl_logger.error("Timeout stopping data processor (60s), force killing.")
                ray.kill(self.data_processor, no_restart=True)
                self.data_processor = None
            except Exception as e:
                psrl_logger.error(f"Error stopping data processor: {e}", exc_info=True)
                ray.kill(self.data_processor, no_restart=True)
                self.data_processor = None
        else:
            psrl_logger.warning("Data processor is not initialized, skipping stop operation.")

    def init_agent_loop_manager(self):
        if self.agent_loop_manager is not None:
            return

        # Initialize the agent loop manager
        self.agent_loop_manager = (
            ray.remote(PSRL_AgentLoopManager)
            .options(max_concurrency=self.max_concurrency)
            .remote(
                self.config,
                self.data_queue_size,
                self.agent_loop_workers,
                self.ps_manager_handle,
                self.data_processor,
                group_post_process_fn=self.group_post_process_fn,
                buffer_post_process_fn=self.buffer_post_process_fn,
            )
        )

    def start_agent_loop_manager(self):
        """Start the agent loop manager to handle agent loops in the background."""
        assert self.agent_loop_manager is not None, "Agent loop manager must be initialized before starting it."

        ray.get(self.agent_loop_manager.start_busy_loop.remote())

    def stop_agent_loop_manager(self):
        """Stop the agent loop manager."""
        if self.agent_loop_manager is not None:
            psrl_logger.debug("Stopping agent loop manager...")
            try:
                # Apply a 60-second timeout to prevent indefinite hangs
                ray.get(self.agent_loop_manager.stop_busy_loop.remote(), timeout=60)
                self.agent_loop_manager = None
                psrl_logger.debug("Agent loop manager stopped successfully.")
            except ray.exceptions.GetTimeoutError:
                psrl_logger.error(
                    "Timeout waiting for agent loop manager to stop (60s). "
                    "This may indicate a deadlock or resource contention issue."
                )
                # Force cleanup by killing the actor
                if self.agent_loop_manager is not None:
                    ray.kill(self.agent_loop_manager, no_restart=True)
                    self.agent_loop_manager = None
                    psrl_logger.warning("Agent loop manager actor was forcefully killed.")
            except Exception as e:
                psrl_logger.error(f"Error stopping agent loop manager: {e}", exc_info=True)
                # Attempt to kill the actor on any error
                if self.agent_loop_manager is not None:
                    try:
                        ray.kill(self.agent_loop_manager, no_restart=True)
                        self.agent_loop_manager = None
                    except Exception as kill_error:
                        psrl_logger.error(f"Failed to kill agent loop manager actor: {kill_error}")
        else:
            psrl_logger.warning("Agent loop manager is not initialized, skipping stop operation.")

    def init_rollout_gateway(self) -> ray.actor.ActorHandle:
        return RolloutGateway.remote(
            self.config,
            self.config.psrl.ps_manager_ip,
            self.ps_manager_grpc_port,
        )

    def init_rollout_coordinator(self):
        assert self.rollout_gateway_url is not None, (
            "Rollout gateway URL must be initialized before initializing rollout coordinator."
        )
        self.rollout_coordinator = (
            ray.remote(RolloutCoordinator)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=self.ip_to_node_id[self.config.psrl.ps_manager_ip], soft=False
                )
            )
            .remote(
                self.config,
                self.ps_manager_handle,
                self.rollout_gateway_url,
                self.session_router_url,
            )
        )

    def init_reward_gateways(self):
        """Launch one smg gateway per named generative reward model."""
        for rm_cfg in resolve_active_managers(self.config.reward):
            if rm_cfg.reward_loop_type != "gen":
                continue
            reward_model_name = rm_cfg.get("reward_model_name", rm_cfg.model.path.split("/")[-1])
            gateway = RewardModelGateway.options(name=f"reward_gateway_{reward_model_name}").remote(
                self.config, reward_model_name
            )
            url = ray.get(gateway.launch_router.remote())
            self.reward_gateways[reward_model_name] = gateway
            self.reward_gateway_urls[reward_model_name] = url
            psrl_logger.info("Reward gateway for '%s' launched at %s", reward_model_name, url)

    def init_reward_model_servers(self, all_wg: dict):
        """
        Initialize reward model HTTP servers and register them to their smg gateways.

        Args:
            all_wg: dict mapping worker-group name → RayWorkerGroup.
        """
        for rm_cfg in resolve_active_managers(self.config.reward):
            if rm_cfg.reward_loop_type != "gen":
                continue
            reward_model_name = rm_cfg.get("reward_model_name", rm_cfg.model.path.split("/")[-1])
            reward_model_wg_list = [
                all_wg[f"reward_model_{reward_model_name}_{i}"] for i in range(rm_cfg.num_replicas)
            ]
            gateway_url = self.reward_gateway_urls[reward_model_name]
            self.reward_model_to_manager[reward_model_name] = RewardModelManager(
                reward_model_name=reward_model_name,
                config=self.config,
                reward_model_config=rm_cfg,
                reward_model_wg_list=reward_model_wg_list,
                gateway_url=gateway_url,
            )
            psrl_logger.info("RewardModelManager for '%s' initialized.", reward_model_name)
        psrl_logger.info("reward_model_to_manager: %s", list(self.reward_model_to_manager))

    def _post_gateway_worker_routing_control(self, action: str, instance_ids):
        assert action in {"pause", "resume"}, f"Unsupported action: {action}"
        assert self.rollout_gateway_url is not None, "rollout_gateway_url is not initialized"

        payload = build_pause_resume_payload(instance_ids)
        url = f"{self.rollout_gateway_url.rstrip('/')}/workers/{action}"
        psrl_logger.warning(
            f"[GATEWAY-CONTROL] Posting {action} to {url} with payload ({len(payload)} workers): {payload}"
        )
        resp = self._gateway_http_session.post(
            url,
            json=payload,
            timeout=self._gateway_http_timeout,
        )
        resp.raise_for_status()
        result = resp.json() if resp.content else {}
        psrl_logger.warning(f"[GATEWAY-CONTROL] After {action} over {len(instance_ids)} instances, resp = {result}")

        rejected_base_ids = set()
        for item in result.get("results", []):
            if item.get("status") == "rejected":
                base_id = item.get("base_worker_id", item.get("worker_id", ""))
                reason = item.get("reason", "unknown")
                psrl_logger.warning(
                    f"[GATEWAY-CONTROL] Worker {base_id} rejected by gateway: {reason}. "
                    "Removing from future pause/resume calls."
                )
                rejected_base_ids.add(base_id)

        if rejected_base_ids:
            for tag, ids in self.tag_to_base_worker_ids.items():
                self.tag_to_base_worker_ids[tag] = [wid for wid in ids if wid not in rejected_base_ids]

        return result

    def start_rollout_coordinator(self):
        assert self.rollout_coordinator is not None, "Rollout coordinator must be initialized before starting it."

        ray.get(self.rollout_coordinator.start_busy_loop.remote())

    def stop_rollout_coordinator(self):
        """Stop the rollout coordinator."""
        if self.rollout_coordinator is not None:
            psrl_logger.debug("Stopping rollout coordinator...")
            try:
                ray.get(self.rollout_coordinator.stop_busy_loop.remote(), timeout=60)
                self.rollout_coordinator = None
                psrl_logger.debug("Rollout coordinator stopped successfully.")
            except ray.exceptions.GetTimeoutError:
                psrl_logger.error("Timeout stopping rollout coordinator (60s), force killing.")
                ray.kill(self.rollout_coordinator, no_restart=True)
                self.rollout_coordinator = None
            except Exception as e:
                psrl_logger.error(f"Error stopping rollout coordinator: {e}", exc_info=True)
                ray.kill(self.rollout_coordinator, no_restart=True)
                self.rollout_coordinator = None
        else:
            psrl_logger.warning("Rollout coordinator is not initialized, skipping stop operation.")

    def init_elastic_rm_runtime(self):
        if not self.elastic_rm_mode:
            return
        if self.rollout_coordinator is None:
            raise RuntimeError("Rollout coordinator must be initialized before elastic_rm init.")

        psrl_logger.info("Initializing elastic_executor runtime (coordinator sleep/wake, ElasticExecutor).")
        rollout_model_name = self.config.gen_actor_rollout_ref.model.path.split("/")[-1]
        rollout_instance_num = len(self.rollout_wg_list)
        psrl_logger.info(
            "Elastic_RM: rollout model_name=%s, n_instances=%d",
            rollout_model_name,
            rollout_instance_num,
        )

        # Enable coordinator command handling before elastic sleep/wake orchestration.
        self.start_rollout_coordinator()
        psrl_logger.info("Rollout coordinator busy loop started for elastic_rm command handling.")

        # ── Build roles / coordinators and create ElasticExecutor early ──
        reward_coordinators: dict[str, ray.actor.ActorHandle] = {}
        for reward_model_name, manager in self.reward_model_to_manager.items():
            reward_coordinators[reward_model_name] = manager.reward_model_coordinator

        roles = [(PSRL_Role.Rollout, rollout_model_name)]
        roles.extend((PSRL_Role.RewardModel, rm_name) for rm_name in reward_coordinators)
        coordinators = {
            PSRL_Role.Rollout: {rollout_model_name: self.rollout_coordinator},
            PSRL_Role.RewardModel: reward_coordinators,
        }
        elastic_rm_cfg = OmegaConf.to_container(self.config.psrl.deployment.elastic_rm, resolve=True)
        assert isinstance(elastic_rm_cfg, dict), "elastic_rm config should be resolved as dict"

        self.elastic_executor = ElasticExecutor.remote(
            config=self.config,
            roles=roles,
            coordinators=coordinators,
            agent_loop_manager=self.agent_loop_manager,
            elastic_rm_config=elastic_rm_cfg,
        )
        psrl_logger.info(
            "Elastic_RM: ElasticExecutor created (roles=%s).",
            [(r.name, m) for r, m in roles],
        )

        # ── Sleep all rollout instances and register with executor ──
        rollout_all_ids: list = ray.get(self.rollout_coordinator.get_all_instance_ids.remote())
        if rollout_all_ids:
            psrl_logger.info(
                "Elastic_RM: putting all rollout instances to sleep (instance_ids=%s).",
                rollout_all_ids,
            )
            ray.get(
                self.rollout_coordinator.exec_command.remote(
                    Command(type=CommandType.SLEEP, instance_ids=rollout_all_ids),
                    blocking=True,
                )
            )
            psrl_logger.info("Elastic_RM: all rollout instances slept.")

        rollout_gpu_slots = [ClusterTopology.collect_gpu_slots_from_worker_group(wg) for wg in self.rollout_wg_list]
        ray.get(
            self.elastic_executor.register_role.remote(
                role_name=PSRL_Role.Rollout,
                model_name=rollout_model_name,
                instance_ids=rollout_all_ids,
                gpu_slots_per_instance=rollout_gpu_slots,
            )
        )
        psrl_logger.info("Elastic_RM: registered %d rollout instances.", len(rollout_all_ids))

        # ── Sleep all reward model instances and register with executor ──
        for reward_model_name, manager in self.reward_model_to_manager.items():
            rm_all_ids: list = ray.get(manager.reward_model_coordinator.get_all_instance_ids.remote())
            psrl_logger.info(
                "Elastic_RM: putting reward model replicas to sleep (name=%s, instance_ids=%s).",
                reward_model_name,
                rm_all_ids,
            )
            ray.get(
                manager.reward_model_coordinator.exec_command.remote(
                    Command(type=CommandType.SLEEP, instance_ids=rm_all_ids),
                    blocking=True,
                )
            )
            psrl_logger.info("Elastic_RM: reward model %s replicas slept.", reward_model_name)

            rm_gpu_slots = [
                ClusterTopology.collect_gpu_slots_from_worker_group(wg)
                for wg in self.reward_model_to_wg_list[reward_model_name]
            ]
            ray.get(
                self.elastic_executor.register_role.remote(
                    role_name=PSRL_Role.RewardModel,
                    model_name=reward_model_name,
                    instance_ids=rm_all_ids,
                    gpu_slots_per_instance=rm_gpu_slots,
                )
            )
            psrl_logger.info(
                "Elastic_RM: registered %d reward model instances (%s).", len(rm_all_ids), reward_model_name
            )

        # ── Select non-conflicting initial instances and wake them up ──
        min_awake_per_role = max(0, int(self.config.psrl.deployment.elastic_rm.min_awake_per_role))

        # Wake reward-model replicas first so each RM keeps at least one awake instance.
        for reward_model_name, manager in self.reward_model_to_manager.items():
            rm_all_ids = ray.get(manager.reward_model_coordinator.get_all_instance_ids.remote())
            rm_awake_num = max(min_awake_per_role, 0)
            rm_awake_ids = ray.get(
                self.elastic_executor.select_initial_awake_ids.remote(
                    role_name=PSRL_Role.RewardModel,
                    model_name=reward_model_name,
                    target_awake_num=rm_awake_num,
                    min_awake_num=min_awake_per_role if len(rm_all_ids) > 0 else 0,
                )
            )
            if rm_awake_ids:
                psrl_logger.info(
                    "Elastic_RM: waking up reward model replicas (name=%s, count=%d, ids=%s).",
                    reward_model_name,
                    len(rm_awake_ids),
                    rm_awake_ids,
                )
                ray.get(
                    manager.reward_model_coordinator.exec_command.remote(
                        Command(type=CommandType.WAKE_UP, instance_ids=rm_awake_ids),
                        blocking=True,
                    )
                )
                psrl_logger.info("Elastic_RM: reward model %s wake_up completed.", reward_model_name)

        rollout_awake_num = max(min_awake_per_role, rollout_instance_num)
        rollout_awake_ids = ray.get(
            self.elastic_executor.select_initial_awake_ids.remote(
                role_name=PSRL_Role.Rollout,
                model_name=rollout_model_name,
                target_awake_num=rollout_awake_num,
                min_awake_num=min_awake_per_role if rollout_instance_num > 0 else 0,
            )
        )
        if rollout_awake_ids:
            psrl_logger.info(
                "Elastic_RM: waking up rollout instances (count=%d, ids=%s).",
                len(rollout_awake_ids),
                rollout_awake_ids,
            )
            ray.get(
                self.rollout_coordinator.exec_command.remote(
                    Command(type=CommandType.WAKE_UP, instance_ids=rollout_awake_ids),
                    blocking=True,
                )
            )
            psrl_logger.info("Elastic_RM: rollout wake_up completed.")

        # ── Start monitor loop ──
        ray.get(self.elastic_executor.start_busy_loop.remote())
        psrl_logger.info("Elastic_RM: ElasticExecutor busy loop started. Runtime ready.")

    def _build_reward_model_runtime_infos(self) -> dict[str, RewardModelRuntimeInfo]:
        runtime_infos = {}
        for reward_model_name, manager in self.reward_model_to_manager.items():
            runtime_infos[reward_model_name] = RewardModelRuntimeInfo(
                gateway_url=manager.get_gateway_url(),
                reward_model_tokenizer=manager.get_reward_model_tokenizer(),
            )
        return runtime_infos

    def init_reward_loop_workers(self):
        worker_cfg = self.config.reward.reward_loop_worker
        if not worker_cfg.get("enable", True):
            self.reward_loop_workers = []
            return

        num_workers = worker_cfg.num_workers
        placement = worker_cfg.get("placement", "reward_service_ip")
        ip_to_node_id = self.ip_to_node_id
        cpu_node_ids = [
            node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0
        ]
        assert cpu_node_ids, "No alive CPU Ray nodes available for reward loop workers."
        reward_model_runtime_infos = self._build_reward_model_runtime_infos()
        max_concurrency = worker_cfg.max_concurrency_per_worker

        self.reward_loop_workers = []
        for i in range(num_workers):
            if placement == "all_cpu_nodes":
                node_id = cpu_node_ids[i % len(cpu_node_ids)]
                soft = True
            elif placement == "ps_manager_ip":
                node_id = ip_to_node_id[self.config.psrl.ps_manager_ip]
                soft = False
            else:
                node_id = ip_to_node_id[self.config.psrl.reward_service_ip]
                soft = False

            self.reward_loop_workers.append(
                RewardLoopWorker.options(
                    name=f"reward_loop_worker_{i}",
                    max_concurrency=max_concurrency,
                    num_cpus=0.001,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node_id=node_id,
                        soft=soft,
                    ),
                ).remote(
                    self.config,
                    self.tokenizer,
                    self.processor,
                    resolve_active_managers(self.config.reward),
                    reward_model_runtime_infos,
                    worker_id=i,
                    worker_num=num_workers,
                )
            )

    def init_reward_manager(self):
        """Initialize a single reward manager for both training and validation reward computation."""
        ip_to_node_id = self.ip_to_node_id
        assert self.data_processor is not None, (
            "Data processor must be initialized before starting reward computation."
        )
        assert self.rollout_coordinator is not None, (
            "Rollout server must be initialized before starting reward computation."
        )

        self.init_reward_loop_workers()
        self.reward_manager = (
            ray.remote(RewardLoopManager)
            .options(
                max_concurrency=self.max_concurrency,
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=ip_to_node_id[self.config.psrl.reward_service_ip],
                    soft=False,
                ),
            )
            .remote(
                config=self.config,
                tokenizer=self.tokenizer,
                processor=self.processor,
                ps_manager_handle=self.ps_manager_handle,
                reward_model_configs=resolve_active_managers(self.config.reward),
                reward_model_to_manager=self.reward_model_to_manager,
                reward_loop_workers=self.reward_loop_workers,
            )
        )

    def start_reward_manager(self):
        """Start the reward manager to handle reward computation requests in the background."""
        assert self.reward_manager is not None, "Reward manager must be initialized before starting it."

        ray.get(self.reward_manager.start_busy_loop.remote())

    def stop_reward_manager(self):
        """Stop the reward manager."""
        if self.reward_manager is not None:
            psrl_logger.debug("Stopping reward manager...")
            try:
                ray.get(self.reward_manager.stop_busy_loop.remote(), timeout=60)
                self.reward_manager = None
                self.reward_loop_workers = []
                psrl_logger.debug("Reward manager stopped successfully.")
            except ray.exceptions.GetTimeoutError:
                psrl_logger.error("Timeout stopping reward manager (60s), force killing.")
                ray.kill(self.reward_manager, no_restart=True)
                self.reward_manager = None
                self.reward_loop_workers = []
            except Exception as e:
                psrl_logger.error(f"Error stopping reward manager: {e}", exc_info=True)
                ray.kill(self.reward_manager, no_restart=True)
                self.reward_manager = None
                self.reward_loop_workers = []
        else:
            psrl_logger.warning("Reward manager is not initialized, skipping stop operation.")

    def _log_rollout_data(self, batch: KVBatchMeta, timing_raw: dict, rollout_data_dir: str):
        """Fetch rollout data from TransferQueue and dump sorted by uid."""
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            # AGENT(VERL): add PSRL specific `log_dual_events` here.
            with log_dual_events(
                "Dump rollout generations",
                psrl_logger,
                event_type=EventType.OTHER,
            ):
                fields = ["uid", "prompts", "responses", "rm_scores", "reward_model", "reward_extra_info"]
                data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
                data["prompts"] = data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
                data["responses"] = data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)

                inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["prompts"]]
                outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in data["responses"]]
                scores = data["rm_scores"].sum(dim=1).tolist()

                reward_model = data.pop("reward_model", None)
                if reward_model is not None:
                    gts = [item.get("ground_truth", None) for item in reward_model.tolist()]
                else:
                    gts = [None] * len(data)

                uids = []
                for key, tag in zip(batch.keys, batch.tags):
                    if tag.get("is_padding", False):
                        continue
                    parts = key.rsplit("_", 1)
                    if len(parts) == 2:
                        uid = parts[0]
                    else:
                        uid = key
                    uids.append(int(uid))

                # extract reward infos
                reward_extra_infos_dict = {
                    "reward_extra_info": data["reward_extra_info"],
                    "uid": uids,
                }

                self._dump_generations(
                    inputs=inputs,
                    outputs=outputs,
                    gts=gts,
                    scores=scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=rollout_data_dir,
                )

    def _val_metrics_update(
        self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns
    ) -> dict[str, float]:
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.array(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _validate(self, merged: bool = False):
        """Validate the model using the validation dataset.

        Note that we use the training side to do val for overlapping with generation.
        """
        # AGENT(VERL): PSRL add switch between train/rollout mode.
        with log_dual_events("Switch to rollout mode", psrl_logger, event_type=EventType.SWITCH):
            self.switch_to_rollout_mode()

        data_sources = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        # AGENT(VERL): PSRL use `parent_id` to group samples of the same prompt together,
        # while verl use `uid` instead.
        sample_parent_ids = []

        dump_all_inputs: list[str] = []
        dump_all_outputs: list[str] = []
        dump_all_keys: list[str] = []
        session_to_sample_idx: dict[str, int] = {}

        val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n
        val_batch_num = ray.get(self.data_processor.get_val_batch_num.remote())
        assert self.reward_manager is not None, "Reward manager must be initialized before validation."

        for _ in range(val_batch_num):
            with log_dual_events("launch validation sequences generation", psrl_logger, event_type=EventType.WAIT):
                val_buffer_id = ray.get(self.agent_loop_manager.generate_validate_sequences.remote())

            with log_dual_events(f"Wait for validation batch {val_buffer_id}", psrl_logger, event_type=EventType.WAIT):
                test_result: KVBatchMeta = ray.get(
                    self.agent_loop_manager.wait_for_validation_batch.remote(val_buffer_id)
                )
                self.replay_buffer.sample(test_result.keys, test_result.partition_id)

            # Skip empty batches so validation retains metrics from successful rounds.
            if len(test_result.keys) == 0:
                psrl_logger.warning(
                    "Validation batch %s is empty because all groups failed. Skipping.",
                    val_buffer_id,
                )
                continue

            if self.config.reward.launch_reward_fn_async:
                with log_dual_events("Wait for reward of validate", psrl_logger, event_type=EventType.WAIT):
                    ray.get(self.reward_manager.wait_for_reward_of_requests.remote(test_result))

            # Use only the highest-indexed output in each session for metrics.
            session_max: dict[str, tuple[int, int]] = {}  # session_key -> (max_index, position)
            for pos, key in enumerate(test_result.keys):
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    uid, index = parts[0], int(parts[1])
                    session_key = uid
                    if session_key not in session_max or index > session_max[session_key][0]:
                        session_max[session_key] = (index, pos)
                else:
                    session_max[key] = (0, pos)
            sorted_sessions = sorted(session_max.items(), key=lambda x: x[1][1])
            final_indices = [pos for _, (_, pos) in sorted_sessions]
            final_keys = [test_result.keys[i] for i in final_indices]

            base_offset = len(sample_scores)
            session_to_sample_idx.update(
                {session_key: base_offset + j for j, (session_key, _) in enumerate(sorted_sessions)}
            )
            text_data = tq.kv_batch_get(
                keys=test_result.keys, partition_id=test_result.partition_id, select_fields=["prompts", "responses"]
            )
            text_data["prompts"] = text_data["prompts"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            text_data["responses"] = text_data["responses"].to_padded_tensor(padding=self.tokenizer.pad_token_id)
            all_inputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["prompts"]]
            all_outputs = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in text_data["responses"]]

            fields = [
                "uid",
                "parent_id",
                "rm_scores",
                "num_turns",
                "reward_model",
                "data_source",
                "reward_extra_info",
            ]
            data = tq.kv_batch_get(keys=final_keys, partition_id=test_result.partition_id, select_fields=fields)

            scores = data["rm_scores"].sum(dim=1).tolist()
            sample_scores.extend(scores)
            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_infos = tu.get(data, "reward_extra_info", [{}] * len(final_keys))
            for reward_extra_info in reward_extra_infos:
                # Flatten per-loop reward details so downstream metrics use a common key space.
                extra_info = {}
                for per_loop_info in reward_extra_info.values():
                    extra_info.update(per_loop_info)
                acc = extra_info.get("acc", 0.0)
                reward_extra_infos_dict["reward_extra_info"].append(extra_info)
                reward_extra_infos_dict["acc"].append(acc)

            # Store generated outputs
            sample_outputs.extend(all_outputs[i] for i in final_indices)

            # Store original inputs
            sample_inputs.extend(all_inputs[i] for i in final_indices)

            sample_parent_ids.extend(tu.get(data, "parent_id" if val_rollout_n > 1 else "uid"))

            ground_truths = [
                item.get("ground_truth", None)
                for item in (tu.get(data, "reward_model", None) or [{}] * len(final_keys))
            ]
            sample_gts.extend(ground_truths)
            sample_turns.extend(data.pop("num_turns").tolist())

            data_source = tu.get(data, "data_source") or ["unknown"] * len(final_keys)
            data_sources.extend(data_source)

            dump_all_inputs.extend(all_inputs)
            dump_all_outputs.extend(all_outputs)
            dump_all_keys.extend(test_result.keys)

            # 5. Release TQ storage for this val batch.
            clear_payload(
                keys=test_result.keys,
                partition_id=test_result.partition_id,
                state=PayloadState.CONSUMED,
            )
            self.replay_buffer.remove(test_result.keys, test_result.partition_id)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            # Sort according to uid (so that generations in the same rollout are together)
            sort_keys = []
            for key in dump_all_keys:
                parts = key.rsplit("_", 1)
                sort_keys.append((parts[0], int(parts[1])) if len(parts) == 2 else (key, 0))
            sorted_indices = sorted(range(len(dump_all_keys)), key=lambda i: sort_keys[i])
            dump_all_inputs = [dump_all_inputs[i] for i in sorted_indices]
            dump_all_outputs = [dump_all_outputs[i] for i in sorted_indices]
            dump_all_keys = [dump_all_keys[i] for i in sorted_indices]

            # For ground truths, scores and reward extra infos, find the values in the
            # lists for the final samples of each session
            dump_all_sessions = [
                f"{parts[0]}" if len(parts) == 2 else key for key in dump_all_keys for parts in [key.rsplit("_", 1)]
            ]
            session_final_indices = [session_to_sample_idx[session] for session in dump_all_sessions]
            self._dump_generations(
                inputs=dump_all_inputs,
                outputs=dump_all_outputs,
                gts=[sample_gts[i] for i in session_final_indices],
                scores=[sample_scores[i] for i in session_final_indices],
                reward_extra_infos_dict={
                    k: [v[i] for i in session_final_indices] for k, v in reward_extra_infos_dict.items()
                }
                | {"uid": dump_all_keys},
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_sources,
                "sample_parent_ids": sample_parent_ids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }

        with log_dual_events("Switch to trainer mode", psrl_logger, event_type=EventType.SWITCH):
            self.switch_to_trainer_mode()

        return self._val_metrics_update(data_sources, sample_parent_ids, reward_extra_infos_dict, sample_turns)

    @staticmethod
    def _write_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, global_steps):
        """Write generation samples as JSONL (runs in background thread)."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        def json_encode_default(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, "tolist"):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False, default=json_encode_default) + "\n")

        print(f"Dumped generations to {filename}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL asynchronously."""
        global_steps = self.global_steps
        future = self._dump_executor.submit(
            self._write_generations,
            inputs,
            outputs,
            gts,
            scores,
            reward_extra_infos_dict,
            dump_path,
            global_steps,
        )
        self._dump_futures.append(future)
        # Clean up completed futures and surface any exceptions early
        still_pending = []
        for f in self._dump_futures:
            if f.done():
                f.result()  # re-raises if the write failed
            else:
                still_pending.append(f)
        self._dump_futures = still_pending

    def _init_dump_executor(self):
        """Create or recreate the dump executor and futures list."""
        self._dump_executor = ThreadPoolExecutor(max_workers=1)
        self._dump_futures = []

    def _shutdown_dump_executor(self):
        """Drain pending dump futures and shut down the executor."""
        for f in self._dump_futures:
            f.result()
        self._dump_futures.clear()
        self._dump_executor.shutdown(wait=True)

    def _run_all(self, tasks: list[asyncio.Task]):
        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())

    def register_rollout_servers(self, rollout_replicas: list[PSRL_vLLMReplica]):
        assert self.rollout_gateway_url is not None, "Rollout gateway URL is not initialized."
        futures = []
        # Register servers with the SMG rollout gateway.
        for replica in rollout_replicas:
            futures.append(replica.servers[0].register_server_to_gateway.remote(self.rollout_gateway_url))
        results = ray.get(futures)

        # 2. register to ps manager
        futures = []
        for replica in rollout_replicas:
            futures.append(replica.servers[0].register_rollout_instances_to_ps.remote())
        ray.get(futures)

        # 3. register to rollout coordinator
        futures = []
        for replica_id, replica in zip(results, rollout_replicas):
            futures.append(
                self.rollout_coordinator.add_worker.remote(
                    replica,
                    replica.servers[0],
                    replica_id,
                    replica.data_parallel_size,
                    is_validate=False if replica.tag == "rollout" else True,
                    model_version=0,
                )
            )
            if self.tag_to_base_worker_ids.get(replica.tag) is None:
                self.tag_to_base_worker_ids[replica.tag] = []
            self.tag_to_base_worker_ids[replica.tag].append(replica_id)
        ray.get(futures)

    def init_rollout_servers(
        self,
        worker_group_list: list[RayWorkerGroup],
        tag: str = "rollout",
    ):
        status_sink_endpoint = ray.get(self.rollout_coordinator.get_status_sink_endpoint.remote())
        if tag == "validate":
            rollout_world_size = (
                self.config.train_actor_rollout_ref.rollout.tensor_model_parallel_size
                * self.config.train_actor_rollout_ref.rollout.data_parallel_size
                * self.config.train_actor_rollout_ref.rollout.pipeline_model_parallel_size
            )
            rollout_config = self.config.train_actor_rollout_ref.rollout
        else:
            rollout_world_size = (
                self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size
                * self.config.gen_actor_rollout_ref.rollout.data_parallel_size
                * self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size
            )
            rollout_config = self.config.gen_actor_rollout_ref.rollout
        model_config = self.config.train_actor_rollout_ref.model
        rollout_replicas = []
        init_tasks = []
        for worker_group in worker_group_list:
            world_size = worker_group.world_size
            num_replicas = world_size // rollout_world_size
            psrl_logger.info(f"[{tag}]: {world_size=}, {rollout_world_size=}, {num_replicas=}")
            # Allocate replica indices atomically across rollout and validation initialization.
            with self._rollout_replica_alloc_lock:
                curr_replica_num = len(self.rollout_replicas)
                new_rollout_replicas = []
                for replica_rank in range(num_replicas):
                    gen_interface = GenInterface(
                        role=tag,
                        rollout_replica_idx=curr_replica_num + replica_rank,
                        ps_manager_handle=self.ps_manager_handle,
                        status_endpoint=status_sink_endpoint,
                    )
                    new_rollout_replicas.append(
                        PSRL_vLLMReplica(
                            replica_rank=curr_replica_num + replica_rank,
                            local_replica_rank=replica_rank,
                            psrl_config=self.config.psrl,
                            config=rollout_config,
                            model_config=model_config,
                            gen_interface=gen_interface,
                            gpus_per_node=(
                                self.config.psrl.deployment.rollout_ngpus_per_node_per_instance
                                if tag == "rollout"
                                else self.config.psrl.deployment.validate_ngpus_per_node_per_instance
                            ),
                            tag=tag,
                        )
                    )
                self.rollout_replicas.extend(new_rollout_replicas)
            rollout_replicas.extend(new_rollout_replicas)
            init_tasks.extend([replica.init_model(worker_group) for replica in new_rollout_replicas])
        self._run_all(init_tasks)
        if tag not in self.tag_to_server_handles:
            self.tag_to_server_handles[tag] = []
        self.tag_to_server_handles[tag].extend([replica._server_handle for replica in rollout_replicas])
        psrl_logger.info(f"Current server num of {tag} is {len(self.tag_to_server_handles[tag])}")
        self.server_addresses.extend([replica._server_address for replica in rollout_replicas])

        # Update Prometheus configuration with server addresses
        if rollout_config.prometheus.enable:
            if rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(rollout_config.prometheus, self.server_addresses, rollout_config.name)

        self.register_rollout_servers(rollout_replicas)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)

        Note that we use multi-threading to speed up the initialization of worker groups.
        For rollout instances, we create multiple worker groups based on
        the number of instances specified in the configuration,
        instead of creating a unified worker group for all instances.
        """

        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # ---- Step 1: Create each role by mapping resource pool to worker class ----

        # create elastic reward model if needed
        elastic_shared_pool = None
        elastic_subpool_group_idx = 0
        if self.elastic_rm_mode:
            elastic_shared_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Rollout)
            # Materialize placement groups once and reuse them across all elastic sub resource pools.
            elastic_shared_pool.get_placement_groups(strategy="STRICT_PACK", device_name=self.device_name)
            psrl_logger.info(
                "Elastic RM shared RayResourcePool: name_prefix=%s world_size=%d store=%s max_colocate_count=%s",
                elastic_shared_pool.name_prefix,
                elastic_shared_pool.world_size,
                list(elastic_shared_pool.store),
                elastic_shared_pool.max_colocate_count,
            )

        def _build_elastic_sub_resource_pool(subgroup_world_size: int, tag: str) -> SubRayResourcePool:
            nonlocal elastic_subpool_group_idx
            assert elastic_shared_pool is not None, "elastic_shared_pool must be initialized in elastic_rm_mode"
            if subgroup_world_size <= 0:
                raise ValueError(f"subgroup_world_size must be > 0, but got {subgroup_world_size} ({tag})")
            if subgroup_world_size > elastic_shared_pool.world_size:
                raise ValueError(
                    f"subgroup_world_size={subgroup_world_size} exceeds shared pool world_size="
                    f"{elastic_shared_pool.world_size} ({tag})"
                )

            # SubRayResourcePool requires a contiguous bundle range [start, start + subgroup_world_size).
            # We rotate start index to reduce collisions while allowing controlled oversubscription.
            max_start = elastic_shared_pool.world_size - subgroup_world_size
            if max_start == 0:
                start_bundle_index = 0
            else:
                start_bundle_index = (elastic_subpool_group_idx * subgroup_world_size) % (max_start + 1)
            elastic_subpool_group_idx += 1

            sub_rp = SubRayResourcePool(
                process_on_nodes=elastic_shared_pool.store,
                use_gpu=elastic_shared_pool.use_gpu,
                name_prefix=f"{elastic_shared_pool.name_prefix}_{tag}",
                max_colocate_count=elastic_shared_pool.max_colocate_count,
                detached=elastic_shared_pool.detached,
                accelerator_type=elastic_shared_pool.accelerator_type,
                resource_num_per_bundle=elastic_shared_pool.resource_num_per_bundle,
                placement_groups=elastic_shared_pool.pgs,
                start_bundle_index=start_bundle_index,
                subgroup_world_size=subgroup_world_size,
            )
            end_bundle = start_bundle_index + subgroup_world_size
            pg_ids = [getattr(pg, "id", None) for pg in (elastic_shared_pool.pgs or [])]
            psrl_logger.info(
                "Elastic SubRayResourcePool[%s]: type=%s name_prefix=%s subgroup_world_size=%d "
                "start_bundle_index=%d bundle_range=[%d, %d) shared_world_size=%d "
                "shared_store=%s pg_count=%s pg_ids=%s",
                tag,
                type(sub_rp).__name__,
                sub_rp.name_prefix,
                subgroup_world_size,
                start_bundle_index,
                start_bundle_index,
                end_bundle,
                elastic_shared_pool.world_size,
                list(elastic_shared_pool.store),
                len(elastic_shared_pool.pgs) if elastic_shared_pool.pgs is not None else 0,
                pg_ids,
            )
            return sub_rp

        # create rollout
        for i in range(self.n_rollout_instances):
            if self.elastic_rm_mode:
                rollout_world_size = (
                    self.config.gen_actor_rollout_ref.rollout.tensor_model_parallel_size
                    * self.config.gen_actor_rollout_ref.rollout.pipeline_model_parallel_size
                    * self.config.gen_actor_rollout_ref.rollout.get("data_parallel_size", 1)
                )
                rollout_resource_pool = _build_elastic_sub_resource_pool(
                    subgroup_world_size=rollout_world_size,
                    tag=f"rollout_{i}",
                )
            else:
                rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Rollout, i)
            rollout_config = self.config.gen_actor_rollout_ref.rollout
            if self.config.psrl.deployment.heterogeneous_rollout.enable:
                rollout_config.rollout.tensor_model_parallel_size = (
                    self.config.psrl.deployment.heterogeneous_rollout.tensor_model_parallel_size_per_instance[i]
                )
                rollout_config.rollout.pipeline_model_parallel_size = (
                    self.config.psrl.deployment.heterogeneous_rollout.pipeline_model_parallel_size_per_instance[i]
                )

            rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Rollout],
                config=rollout_config,
                model_config=self.config.train_actor_rollout_ref.model,
                device_mesh=None,
            )
            self.resource_pool_to_cls.setdefault(rollout_resource_pool, {})
            self.resource_pool_to_cls[rollout_resource_pool][f"rollout_{i}"] = rollout_cls

        # create validate
        for i in range(self.n_validate_instances):
            val_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Validate, i)
            rollout_config = self.config.train_actor_rollout_ref.rollout
            val_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Validate],
                config=rollout_config,
                model_config=self.config.train_actor_rollout_ref.model,
                device_mesh=None,
            )
            self.resource_pool_to_cls[val_rollout_resource_pool][f"validate_{i}"] = val_rollout_cls

        # create actor
        # AGENT(VERL): PSRL does not use `hybrid_engine`
        train_interface = TrainInterface(ps_manager_handle=self.ps_manager_handle)
        actor_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Actor)
        actor_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[PSRL_Role.Actor],
            config=self.config.train_actor_rollout_ref,
            role="actor",
            psrl_config=self.config.psrl,
            train_interface=train_interface,
            distillation_config=self.config.get("distillation", None),
        )
        self.resource_pool_to_cls[actor_resource_pool]["actor"] = actor_cls

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            # convert critic_cfg into TrainingWorkerConfig
            from verl.workers.engine_workers import TrainingWorkerConfig

            orig_critic_cfg = critic_cfg
            engine_config: EngineConfig = orig_critic_cfg.engine
            engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
            engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu

            critic_cfg = TrainingWorkerConfig(
                model_type="value_model",
                model_config=orig_critic_cfg.model,
                engine_config=engine_config,
                optimizer_config=orig_critic_cfg.optim,
                checkpoint_config=orig_critic_cfg.checkpoint,
            )

            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[PSRL_Role.Critic],
                config=critic_cfg,
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and PSRL_Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PSRL_Role.RefPolicy],
                config=self.config.train_actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        if not self.use_critic and not self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.DummyPolicy)
            dummy_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[PSRL_Role.DummyPolicy],
                config=self.config.train_actor_rollout_ref,
                role="dummy",
            )
            self.resource_pool_to_cls[resource_pool]["dummy"] = dummy_policy_cls

        # Allocate resource pools for reward model replicas.
        for reward_model in resolve_active_managers(self.config.reward):
            if reward_model.reward_loop_type != "gen":
                continue
            reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
            reward_model_cfg = reward_model
            for i in range(reward_model.num_replicas):
                if self.elastic_rm_mode:
                    reward_model_world_size = (
                        reward_model_cfg.rollout.tensor_model_parallel_size
                        * reward_model_cfg.rollout.pipeline_model_parallel_size
                        * reward_model_cfg.rollout.get("data_parallel_size", 1)
                    )
                    reward_model_resource_pool = _build_elastic_sub_resource_pool(
                        subgroup_world_size=reward_model_world_size,
                        tag=f"reward_model_{reward_model_name}_{i}",
                    )
                else:
                    reward_model_resource_pool = self.resource_pool_manager.resource_pool_dict[
                        f"reward_pool_{reward_model_name}_{i}"
                    ]
                reward_cls = RayClassWithInitArgs(
                    cls=self.role_worker_mapping[PSRL_Role.RewardModel],
                    config=reward_model_cfg.rollout,
                    model_config=reward_model_cfg.model,
                    device_mesh=None,
                )
                self.resource_pool_to_cls[reward_model_resource_pool][f"reward_model_{reward_model_name}_{i}"] = (
                    reward_cls
                )

        # ---- Step 2: Create worker groups for each role in parallel ----
        psrl_logger.info("Initializing WorkerGroup for other roles")

        # initialize WorkerGroup
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(
                        self.config.global_profiler.global_tool_config.nsys,
                        "worker_nsight_options",
                    )
                )
        wg_kwargs["device_name"] = self.device_name

        # NOTE(verl): Use separate worker groups when roles need distinct resource pools or parallel sizes.
        def create_worker_group(resource_pool, class_dict, wg_kwargs=wg_kwargs):
            # A single class can use a direct worker group and its low-level APIs.
            if len(class_dict) == 1:
                role = next(iter(class_dict.keys()))
                return {
                    role: RayWorkerGroup(
                        resource_pool=resource_pool,
                        ray_cls_with_init=class_dict[role],
                        **wg_kwargs,
                    )
                }
            # colocate
            else:
                worker_dict_cls = create_colocated_worker_cls_fused(class_dict=class_dict)
                wg_dict = RayWorkerGroup(
                    resource_pool=resource_pool,
                    ray_cls_with_init=worker_dict_cls,
                    **wg_kwargs,
                )
                return wg_dict.spawn(prefix_set=class_dict.keys())

        def _run_worker_group_tasks(tasks, label: str):
            """Create worker groups with a thread pool. Safely handle empty task lists."""

            if not tasks:
                psrl_logger.info(f"No {label} worker group to create. Skipping.")
                return

            # One thread per task requires a nonempty task list.
            with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
                futures = {}
                for resource_pool, class_dict, task_wg_kwargs in tasks:
                    future = executor.submit(create_worker_group, resource_pool, class_dict, task_wg_kwargs)
                    futures[future] = (resource_pool, class_dict)
                for future in futures:
                    result = future.result()
                    all_wg.update(result)

        # multi-thread version
        train_tasks = []
        gen_tasks = []
        val_tasks = []
        reward_model_tasks = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            psrl_logger.info(f"Creating worker group for resource pool: {resource_pool}, classes: {class_dict}")
            if "ps" in class_dict:
                assert class_dict.keys() == {"ps"}, "PS resource pool should only have PS role."
                continue
            if any("rollout" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Rollout resource pool should only have one worker class."
                gen_tasks.append((resource_pool, class_dict, wg_kwargs))
            elif any("validate" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Validate resource pool should only have one worker class."
                val_tasks.append((resource_pool, class_dict, wg_kwargs))
            elif any("reward_model" in key for key in class_dict.keys()):
                assert len(class_dict) == 1, "Reward model resource pool should only have one worker class."
                reward_model_tasks.append((resource_pool, class_dict, wg_kwargs))
            else:
                # NOTE(linsh): Adapt `wg_kwargs` for fused train workers that need environment arguments.
                if self.config.psrl.tms.range in ["train", "all"] or self.config.psrl.tms.enable_nixl:
                    # add tms config to train workers
                    import torch_memory_saver

                    dynlib_path = os.path.join(
                        os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
                        "torch_memory_saver_hook_mode_preload.abi3.so",
                    )
                    assert os.path.exists(dynlib_path), f"LD_PRELOAD so file {dynlib_path} does not exist."

                    train_wg_kwargs = wg_kwargs.copy()
                    train_wg_kwargs["worker_env"] = {
                        "LD_PRELOAD": dynlib_path,
                        "TMS_INIT_ENABLE": "1",
                        "TMS_INIT_ENABLE_CPU_BACKUP": "0",
                        # NOTE(linsh): Torch memory saver is incompatible with expandable segments.
                        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                        "PSRL_TMS_ENABLE": "1" if self.config.psrl.tms.range in ["train", "all"] else "",
                    }
                else:
                    train_wg_kwargs = wg_kwargs
                    # NOTE(lhy): Expandable segments cause NIXL errors.
                train_tasks.append((resource_pool, class_dict, train_wg_kwargs))
        # Initialize train tasks first so rollout instances cannot consume their resources.
        _run_worker_group_tasks(train_tasks, label="train")
        _run_worker_group_tasks(reward_model_tasks, label="reward_model")
        _run_worker_group_tasks(gen_tasks, label="gen")
        _run_worker_group_tasks(val_tasks, label="validate")

        # Initialize reward model gateways and servers
        self.init_reward_gateways()
        self.init_reward_model_servers(all_wg)

        self.rollout_wg_list = [all_wg[f"rollout_{i}"] for i in range(self.n_rollout_instances)]
        self.validate_wg_list = [all_wg[f"validate_{i}"] for i in range(self.n_validate_instances)]
        self.reward_model_to_wg_list = {}
        for reward_model in resolve_active_managers(self.config.reward):
            if reward_model.reward_loop_type != "gen":
                continue
            reward_model_name = reward_model.get("reward_model_name", reward_model.model.path.split("/")[-1])
            self.reward_model_to_wg_list[reward_model_name] = [
                all_wg[f"reward_model_{reward_model_name}_{i}"] for i in range(reward_model.num_replicas)
            ]

        # Dispatch actor NIXL early only when actor initialization precedes validation on shared GPUs.
        if self.is_rollout_mode_in_actor:
            actor_nixl_futures = all_wg["actor"].execute_all_async("init_nixl_client")
            psrl_logger.info("Dispatched actor NIXL init early (overlapping with router wait)")
        else:
            actor_nixl_futures = None

        self.rollout_gateway_url = ray.get(self._launch_gateway_future)
        self.session_router_url = ray.get(self.rollout_gateway.launch_session_router.remote())
        psrl_logger.info(f"Rollout gateway launched at {self.rollout_gateway_url}.")
        psrl_logger.info(f"Session router launched at {self.session_router_url}.")

        # Build env worker pool if enabled.
        if self.config.psrl.env_worker.enable:
            from psrl.workers.env_worker import EnvWorkerManager

            self.env_worker_manager = EnvWorkerManager(self.config)
            psrl_logger.info("Env worker pool built.")

        # create agent loop workers
        self.agent_loop_workers = []
        num_agent_workers = self.config.gen_actor_rollout_ref.rollout.agent.num_workers
        max_concurrency_per_worker = max(1, self.max_concurrency // num_agent_workers)
        # Distribute agent loop workers across cluster nodes round-robin so that
        # Docker containers are spread across machines instead of piling up on one.
        #
        # `agent.node_ips` restricts placement to an allow list, because a node with a
        # degraded Docker daemon accepts actors and then hangs. Empty means every node.
        allowed_ips = list(self.config.gen_actor_rollout_ref.rollout.agent.get("node_ips") or [])
        alive_nodes = [n for n in ray.nodes() if n["Alive"]]
        if allowed_ips:
            selected = [n for n in alive_nodes if n["NodeManagerAddress"] in allowed_ips]
            if not selected:
                raise ValueError(
                    f"agent.node_ips={allowed_ips} matched no alive node. "
                    f"Alive: {sorted(n['NodeManagerAddress'] for n in alive_nodes)}."
                )
            psrl_logger.info(
                "Agent loop workers restricted to %d of %d nodes: %s.",
                len(selected),
                len(alive_nodes),
                sorted(n["NodeManagerAddress"] for n in selected),
            )
            alive_nodes = selected
        alive_node_ids = [n["NodeID"] for n in alive_nodes]
        worker_node_ids = [alive_node_ids[i % len(alive_node_ids)] for i in range(num_agent_workers)]
        workers_per_node = Counter(worker_node_ids)
        sandbox_config = self.config.gen_actor_rollout_ref.rollout.agent.sandbox
        use_node_capacity = any(
            str(backend.get("_target_", "")).endswith("DockerBackend") for backend in sandbox_config.backends.values()
        )
        capacity_config = (
            OmegaConf.to_container(sandbox_config.get("capacity", {}), resolve=True) if use_node_capacity else None
        )
        # Node-local capacity coordinators must share a node with their workers, and a soft
        # placement could fall back to an excluded node, so both need hard affinity.
        hard_affinity = bool(allowed_ips) or use_node_capacity
        capacity_coordinators = {}
        for i, node_id in enumerate(worker_node_ids):
            if use_node_capacity and node_id not in capacity_coordinators:
                coordinator_concurrency = workers_per_node[node_id] * (max_concurrency_per_worker + 1) + 1
                capacity_coordinators[node_id] = (
                    ray.remote(SandboxCapacityCoordinator)
                    .options(
                        num_cpus=0,
                        max_concurrency=coordinator_concurrency,
                        scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=False),
                    )
                    .remote(capacity_config)
                )
            capacity_coordinator = capacity_coordinators.get(node_id)
            self.agent_loop_workers.append(
                PSRL_AgentLoopWorker.options(
                    name=f"agent_loop_worker_{i}",
                    max_concurrency=max_concurrency_per_worker,
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=node_id, soft=not hard_affinity),
                ).remote(
                    self.config,
                    self.ps_manager_handle,
                    self.rollout_gateway_url,
                    self.session_router_url,
                    worker_id=i,
                    worker_num=num_agent_workers,
                    capacity_coordinator=capacity_coordinator,
                )
            )
            psrl_logger.info(
                f"Agent loop worker {i} scheduled on node {node_id!r} (node_capacity={use_node_capacity!r})."
            )
        self.sandbox_capacity_coordinators = capacity_coordinators
        if capacity_coordinators:
            snapshots = ray.get([coordinator.snapshot.remote() for coordinator in capacity_coordinators.values()])
            for node_id, snapshot in zip(capacity_coordinators, snapshots, strict=True):
                psrl_logger.info(f"Sandbox capacity on node {node_id!r}: {snapshot!r}.")

        # start rollout coordinator
        self.init_rollout_coordinator()
        # Bind the coordinator before PS transfers so `push_model` can publish new versions.
        ray.get(self.ps_manager_handle.set_rollout_coordinator.remote(self.rollout_coordinator))
        # Start the LMCache Controller BEFORE init_model() so that LMCache workers
        # inside EngineCore can register immediately when they start.
        if self.config.psrl.lmcache.get("enable", False) and self.config.psrl.lmcache.get("enable_p2p", False):
            ray.get(self.rollout_coordinator.start_lmcache_controller.remote())

        # Rollout uses separate GPUs, so overlap its model loading with PS and actor initialization.
        rollout_init_executor = ThreadPoolExecutor(max_workers=1)
        rollout_init_future = rollout_init_executor.submit(self.init_rollout_servers, self.rollout_wg_list, "rollout")
        psrl_logger.info("Launched rollout server init in background thread (before PS init)")

        ps_nixl_futures = []
        model_init_futures = []

        # ---- Step 3: Create and init PS WorkerGroup ----
        psrl_logger.info("Create PS WorkerGroup")
        train_model_dtype = (
            torch.bfloat16 if self.config.train_actor_rollout_ref.actor.strategy == "megatron" else torch.float32
        )
        storage_plan = PSStoragePlan(
            train_model_dtype=train_model_dtype,
            gen_model_dtype=PrecisionType.to_dtype(self.config.gen_actor_rollout_ref.rollout.dtype),
        )
        if self.config.psrl.ps_mode == "cpu" or self.config.psrl.ps_mode == "cpu_ref":
            # PSManager is used to store the model state dict
            # No need to create PS WorkerGroup
            pass
        elif self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            # PSManager hosts metadata while colocated PS workers store model state.
            assert self.config.psrl.nixl.server_ip == self.config.psrl.ps_manager_ip, (
                "PSManager IP and NIXL server IP must be the same"
            )
            if self.config.psrl.ps_mode == "nixl_cpu":
                # ps is deployed on both generation and (maybe) actor nodes
                ps_node_ids = set()

                # Read node IDs from placement metadata to avoid waiting for cold actor workers.
                def _get_node_ids_from_wg(wg: RayWorkerGroup) -> list[str]:
                    """Extract node IDs from placement group GCS metadata without calling actors."""
                    resource_pool = wg.resource_pool
                    pgs = resource_pool.pgs
                    assert pgs is not None, "Placement groups must be created before querying node IDs"
                    local_world_size = resource_pool.store[0]
                    node_ids = []
                    for pg in sort_placement_group_by_node_ip(pgs):
                        pg_data = ray._private.state.state.placement_group_table(pg.id)
                        bundles_to_node = pg_data["bundles_to_node_id"]
                        for local_rank in range(local_world_size):
                            node_ids.append(bundles_to_node[local_rank])
                    return node_ids

                # Get all rollout instances' distinct node ids
                for i in range(self.n_rollout_instances):
                    rollout_instance_node_ids = _get_node_ids_from_wg(all_wg[f"rollout_{i}"])
                    for node_id in rollout_instance_node_ids:
                        ps_node_ids.add(node_id)
                    self.worker_to_node_id.update(
                        {
                            WorkerKey("rollout", i, idx): node_id
                            for idx, node_id in enumerate(rollout_instance_node_ids)
                        }
                    )

                # Get all actor instances' distinct node ids
                actor_instance_node_ids = _get_node_ids_from_wg(all_wg["actor"])
                for node_id in actor_instance_node_ids:
                    ps_node_ids.add(node_id)
                self.worker_to_node_id.update(
                    {WorkerKey("actor", 0, idx): node_id for idx, node_id in enumerate(actor_instance_node_ids)}
                )

                # Get all validate instances' distinct node ids
                for i in range(self.n_validate_instances):
                    validate_instance_node_ids = _get_node_ids_from_wg(all_wg[f"validate_{i}"])
                    for node_id in validate_instance_node_ids:
                        ps_node_ids.add(node_id)
                    self.worker_to_node_id.update(
                        {
                            WorkerKey("validate", i, idx): node_id
                            for idx, node_id in enumerate(validate_instance_node_ids)
                        }
                    )

                ps_spec_list = []
                ps_node_ids = list(ps_node_ids)
                # Map each worker to a PS index
                for i, node_id in enumerate(ps_node_ids):
                    self.worker_to_ps_idx.update(
                        {worker: i for worker, nid in self.worker_to_node_id.items() if nid == node_id}
                    )

                for node_id in ps_node_ids:
                    ps_spec_list.append(PSResourceSpec(node_id=node_id, attached_gpu_id=None))
                ps_resource_pool = PSResourcePool(ps_spec_list=ps_spec_list)
                psrl_logger.info(f"PS resource pool: {ps_resource_pool}")
                self.ps_wg = PSWorkerGroup(
                    resource_pool=ps_resource_pool,
                    ps_cls_with_init=PSClassWithInitArgs(
                        cls=ray.remote(PSStorageWorker),
                        storage_plan=storage_plan,
                        model_config=self.config.train_actor_rollout_ref.model,
                        psrl_config=self.config.psrl,
                    ),
                )
                if self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
                    ps_nixl_futures = self.ps_wg.execute_all_async("init_nixl_client")
                # Initialize the model skeleton on meta device. Load weights after NIXL setup.
                model_init_futures.extend(self.ps_wg.execute_all_async("init_model"))
                # NOTE(claude): Queue checkpoint preload to overlap disk I/O with worker initialization.
                if self._resolve_resume_checkpoint_paths() is None:
                    preload_futures = self.ps_wg.execute_all_async("preload_checkpoint_to_cpu")
                psrl_logger.info("PS model initialized successfully!")
            elif self.config.psrl.ps_mode == "nixl_gpu":
                raise NotImplementedError("PS mode 'nixl_gpu' is not implemented yet")
        else:
            raise ValueError(f"Invalid PS mode: {self.config.psrl.ps_mode}")

        # ---- Step 4: Initialize models in all worker groups ----

        psrl_logger.info("Initializing models in all rollout instances")

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.reset()
            # assign critic loss
            from functools import partial

            from verl.workers.utils.losses import value_loss

            value_loss_ = partial(value_loss, config=orig_critic_cfg)
            self.critic_wg.set_loss_fn(value_loss_)

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if not (self.use_critic or (self.use_reference_policy and not self.ref_in_actor)):
            # NOTE(linsh): A dummy worker prevents Ray from treating the fused worker as asynchronous.
            self.dummy_wg = all_wg["dummy"]
            self.dummy_wg.init_model()

        # initialize teacher loop manager
        if self.use_teacher_policy:
            from verl.experimental.teacher_loop import TeacherModelManager

            teacher_resource_pool = self.resource_pool_manager.get_resource_pool(PSRL_Role.TeacherModel)
            self.teacher_model_manager = TeacherModelManager(
                config=self.config.distillation,
                resource_pool=teacher_resource_pool,
            )
            self.distillation_config: DistillationConfig = omega_conf_to_dataclass(self.config.distillation)
        else:
            self.teacher_model_manager = None
            self.distillation_config = None

        # AGENT(VERL): skip `async_rollout_manager`, `checkpoint_manager` in PSRL.

        # --- Step 5: Initialize NIXL clients and parameters ---

        if self.is_rollout_mode_in_actor:
            assert self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu", (
                "Fused trainer and validator only support NIXL PS mode."
            )
            # init actor wg -> offload -> init validate wg
            psrl_logger.info("Initializing actor model")
            self.actor_wg = all_wg["actor"]
            # Wait for early-dispatched actor NIXL futures (dispatched before router wait,
            # so actor cold-boot overlapped with gateway boot).
            ray.get(actor_nixl_futures)
            psrl_logger.info("Initialized NIXL client in actor worker group")
            self.actor_wg.init_model("empty")
            ray.get(self.actor_wg.execute_all_async("nixl_convert_params"))
            ray.get(self.actor_wg.execute_all_async("nixl_sleep", "meta"))

            psrl_logger.info("Initializing validation model")
            self.init_rollout_servers(self.validate_wg_list, tag="validate")
            # Wait for rollout init to complete (likely already done since it ran in parallel
            # on separate GPUs during actor + validate initialization above)
            rollout_init_future.result()
            rollout_init_executor.shutdown(wait=False)
            ray.get(model_init_futures)
            ray.get(self.rollout_coordinator.init_nixl_client.remote())
            ray.get(self.rollout_coordinator.nixl_convert_params.remote())
            # Start the shared LMCache Controller and broadcast its URL to all instances,
            # enabling cross-instance KV cache migration for partial rollout re-routes.
            if self.config.psrl.lmcache.get("enable", False) and self.config.psrl.lmcache.get("enable_p2p", False):
                ray.get(self.rollout_coordinator.init_lmcache_p2p.remote())
        else:
            # Validate and actor share GPUs and initialize sequentially while rollout initializes elsewhere.
            psrl_logger.info("Initializing validation model")
            self.init_rollout_servers(self.validate_wg_list, tag="validate")
            ray.get(self.rollout_coordinator.sleep.remote("validate"))
            # Pause validate instances in the router before sleeping them, so that the router
            # does not route rollout requests to validate instances that are in sleep state.
            paused_base_worker_ids = self.tag_to_base_worker_ids.get("validate", [])
            psrl_logger.warning(f"[INIT-PAUSE] Paused base worker IDs: {paused_base_worker_ids!r}.")
            if paused_base_worker_ids:
                psrl_logger.warning(
                    f"[INIT-PAUSE] Pausing {len(paused_base_worker_ids)} validate instances "
                    f"in the gateway after sleep: {paused_base_worker_ids!r}."
                )
                self._post_gateway_worker_routing_control("pause", paused_base_worker_ids)
                psrl_logger.warning("[INIT-PAUSE] Pause call completed successfully.")

            psrl_logger.info("Initializing actor model")
            self.actor_wg = all_wg["actor"]
            self.actor_wg.init_model("empty")
            actor_nixl_futures = self.actor_wg.execute_all_async("init_nixl_client")
            ray.get(actor_nixl_futures)
            ray.get(self.actor_wg.execute_all_async("nixl_convert_params"))

            # Wait for rollout to complete. Actor initialization overlapped with it.
            # Coordinator NIXL ops need rollout registered, so we must wait here.
            rollout_init_future.result()
            rollout_init_executor.shutdown(wait=False)

            ray.get(model_init_futures)
            ray.get(self.rollout_coordinator.init_nixl_client.remote())
            ray.get(self.rollout_coordinator.nixl_convert_params.remote())
            # Start the shared LMCache Controller and broadcast its URL to all instances,
            # enabling cross-instance KV cache migration for partial rollout re-routes.
            if self.config.psrl.lmcache.get("enable", False) and self.config.psrl.lmcache.get("enable_p2p", False):
                ray.get(self.rollout_coordinator.init_lmcache_p2p.remote())

        psrl_logger.info("All workers' models initialized successfully!")

        # initialize NIXL
        if self.config.psrl.ps_mode == "nixl_cpu" or self.config.psrl.ps_mode == "nixl_gpu":
            if ps_nixl_futures:
                ray.get(ps_nixl_futures)
            rollout_world_size = ray.get(self.rollout_coordinator.world_size.remote())
            psrl_logger.info(
                f"Initializing NIXL server with {self.ps_wg.world_size} PS workers, "
                f"{self.actor_wg.world_size} actor workers, {rollout_world_size} rollout workers"
            )
            expected_nixl_client_agents = self.ps_wg.world_size + self.actor_wg.world_size + rollout_world_size
            ray.get(self.ps_manager_handle.init_nixl_server.remote(expected_nixl_client_agents))
            actor_protocol_mode = "meta" if self.is_rollout_mode_in_actor else "full"
            rollout_full_tag = "all" if self.is_rollout_mode_in_actor else "rollout"

            with log_dual_events("Executing NIXL protocol", psrl_logger, event_type=EventType.INIT):
                futures = []
                futures.append(self.ps_manager_handle.nixl_protocol.remote())
                futures.extend(self.ps_wg.execute_all_async("nixl_protocol"))
                futures.extend(self.actor_wg.execute_all_async("nixl_protocol", actor_protocol_mode))
                futures.append(self.rollout_coordinator.nixl_protocol.remote(rollout_full_tag))
                ray.get(futures)

            # Bind the PS worker group to PSManager before initial pull, so that PSManager's
            # ps_nixl_agent_names are populated and gen workers can call get_ps_nixl_agent_names.
            psrl_logger.info("Binding PS worker group")
            ray.get(self.ps_manager_handle.bind_ps_worker_group.remote(self.ps_wg))
            psrl_logger.info("PS worker group bound successfully!")

            # NOTE(lhy): New runs load PS directly, while resumed runs load the actor and push to PS.
            if self._resolve_resume_checkpoint_paths() is None:
                # Now that all NIXL buffers are allocated (meta tensors replaced),
                # write the preloaded checkpoint tensors into the PS registered buffers.
                with log_dual_events("Loading PS checkpoint weights", psrl_logger, event_type=EventType.INIT):
                    # Wait only if checkpoint prefetch outlasted the NIXL protocol.
                    ray.get(preload_futures)
                    ray.get(self.ps_wg.execute_all_async("write_checkpoint_to_registered_tensors"))
                # Broadcast rank zero weights to avoid independent checkpoint reads on every PS worker.
                if self.config.psrl.broadcast_init.enabled:
                    with log_dual_events(
                        "Broadcast init: PS rank-0 → all workers", psrl_logger, event_type=EventType.INIT
                    ):
                        ray.get(self.ps_manager_handle._coordinate_broadcast_init.remote())
            else:
                # Push resumed actor weights before the initial pull so all workers receive them first.
                self._resume_load_and_push_to_ps()

            # Pull initial PS weights now. A sleeping colocated actor pulls later in `switch_to_trainer_mode`.
            initial_pull_tag = "all" if self.is_rollout_mode_in_actor else "rollout"
            with log_dual_events("Initial pull: PS → gen/actor workers", psrl_logger, event_type=EventType.INIT):
                initial_pull_futures = []
                initial_pull_futures.append(self.rollout_coordinator.initial_pull_from_ps.remote(initial_pull_tag))
                if not self.is_rollout_mode_in_actor:
                    initial_pull_futures.extend(self.actor_wg.execute_all_async("pull_model"))
                ray.get(initial_pull_futures)

        self.init_elastic_rm_runtime()

        self.replay_buffer.start_polling()
        psrl_logger.info("ReplayBuffer polling started after init_workers() completed.")

    def switch_to_rollout_mode(self):
        """Switch the PSRL colocate part to rollout mode for validation.

        This involves several steps to ensure that the system transitions smoothly
        from training to rollout mode, particularly when validation and training are
        colocated.
        1. Deregister actor clients from NIXL to free up resources.
        2. Wake up validation instances and allocate necessary resources.
        3. Sync with the PS manager to update client information.
        4. Broadcast updated client information to all relevant clients.
        5. Sync validation instances' model weights and status with the PS.
        6. Resume the generation process in the rollout coordinator.
        """
        psrl_logger.warning(
            f"[SWITCH] switch_to_rollout_mode called: "
            f"colocate_validate_and_train={self.config.psrl.colocate_validate_and_train}, "
            f"is_rollout_mode_in_actor={self.is_rollout_mode_in_actor}"
        )
        if not self.config.psrl.colocate_validate_and_train or self.is_rollout_mode_in_actor:
            return
        psrl_logger.warning("Switching to rollout mode...")

        psrl_logger.info("Step 1: Deregistering actor clients from NIXL...")
        # actor_wg nixl client deregister weight memory
        release_futures = self.actor_wg.execute_all_async("nixl_sleep", "full")
        ray.get(release_futures)

        psrl_logger.info("Step 2: Waking up validation instances...")
        # Allocate rollout space and register
        ray.get(
            [self.tag_to_server_handles["validate"][i].nixl_wake_up.remote() for i in range(self.n_validate_instances)]
        )

        psrl_logger.info("Step 3: Syncing with PS manager...")
        # sync with server
        updated_client_names = []  # to collect all updated client names for broadcasting
        futures = []
        for i in range(self.n_validate_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "validate" and k.instance_id == i)
            for rank in range(tp_size):
                updated_client_names.append(
                    WorkerKey("validate", i, rank).to_nixl_client_name(self.n_rollout_instances)
                )
            futures.append(
                self.tag_to_server_handles["validate"][i].nixl_send_local_info_to.remote(NIXL_META_SERVER_NAME)
            )
        # wait for ps manager to collect all infos
        futures.append(self.ps_manager_handle.nixl_wait_for_update_infos.remote(self.n_validate_instances * tp_size))
        ray.get(futures)

        # broadcast to other clients
        psrl_logger.info("Step 4: PS manager broadcasting updated client infos...")
        self._broadcast_updated_client_infos_from_ps_manager(updated_client_names)

        psrl_logger.info("Step 5: Syncing validation instances' model weights and status with PS...")
        # sync validation instances with ps
        # the generation will be resumed in the rollout coordinator
        resumed_instance_ids = []
        validate_dp_size = self.config.train_actor_rollout_ref.rollout.data_parallel_size
        for base_worker_id in self.tag_to_base_worker_ids.get("validate", []):
            resumed_instance_ids.extend((base_worker_id, i) for i in range(validate_dp_size))

        ray.get(self.rollout_coordinator.sync_with_ps.remote(resumed_instance_ids))

        psrl_logger.info("Step 6: Resuming validation instances...")
        # resume validation instances in router and coordinator
        resumed_base_worker_ids = self.tag_to_base_worker_ids.get("validate", [])
        self._post_gateway_worker_routing_control("resume", resumed_base_worker_ids)

        self.is_rollout_mode_in_actor = True

    def switch_to_trainer_mode(self):
        """Switch the PSRL colocate part to trainer mode for training.

        This involves several steps to ensure that the system transitions smoothly
        from rollout to training mode, particularly when validation and training are
        colocated.
        1. Notify agent loop workers about paused validation instances.
        2. Interrupt the generation process in validation instances.
        3. Put validation instances to sleep and deregister from NIXL.
        4. Wake up the training actor and allocate necessary resources.
        5. Sync with the PS manager to update client information.
        6. Broadcast updated client information to all relevant clients.
        7. Pull the latest model weights from the PS to the actor.
        """
        # notify coordinator + interrupt + sleep + upload actor
        psrl_logger.warning(
            f"[SWITCH] switch_to_trainer_mode called: "
            f"colocate_validate_and_train={self.config.psrl.colocate_validate_and_train}, "
            f"is_rollout_mode_in_actor={self.is_rollout_mode_in_actor}"
        )
        if not self.config.psrl.colocate_validate_and_train or not self.is_rollout_mode_in_actor:
            return
        psrl_logger.info("Switching to trainer mode...")

        psrl_logger.info("Step 1: Pausing validation instances...")
        # pause validation instances in router and coordinator
        paused_base_worker_ids = self.tag_to_base_worker_ids.get("validate", [])
        self._post_gateway_worker_routing_control("pause", paused_base_worker_ids)

        psrl_logger.info("Step 2: Interrupting generation of validation instances...")
        # interrupt generation and sleep
        ray.get(
            [
                self.tag_to_server_handles["validate"][i].pause_generation.remote(clear_cache=False)
                for i in range(self.n_validate_instances)
            ]
        )

        psrl_logger.info("Step 3: Putting validation instances to sleep...")
        # sleep validation instances and deregister from NIXL
        ray.get(
            [
                self.tag_to_server_handles["validate"][i].nixl_sleep.remote(level=2)
                for i in range(self.n_validate_instances)
            ]
        )

        psrl_logger.info("Step 4: Waking up training actor...")
        # Allocate trainer space and register
        ray.get(self.actor_wg.execute_all_async("nixl_wake_up"))

        psrl_logger.info("Step 5: Syncing with PS manager...")
        # sync with server
        update_client_names = []  # to collect all updated client names for broadcasting
        futures = []
        for i in range(self.actor_wg.world_size):
            update_client_names.append(train_client_name(i))
        # sender side: actor workers
        futures.extend(self.actor_wg.execute_all_async("nixl_send_local_info_to", NIXL_META_SERVER_NAME))
        # receiver side: ps manager
        futures.append(self.ps_manager_handle.nixl_wait_for_update_infos.remote(self.actor_wg.world_size))
        ray.get(futures)

        psrl_logger.info("Step 6: PS manager broadcasting updated client infos...")
        self._broadcast_updated_client_infos_from_ps_manager(update_client_names)

        psrl_logger.info("Step 7: Pulling actor model from PS...")
        # pull actor model
        ray.get(self.actor_wg.execute_all_async("pull_model"))

        # Apply a deferred checkpoint after actor wakeup and NIXL reconnection.
        deferred_path = getattr(self, "_deferred_actor_ckpt_path", None)
        if deferred_path is not None:
            psrl_logger.info(f"Resume (NIXL): loading deferred actor checkpoint from {deferred_path}...")
            self.actor_wg.load_checkpoint(
                deferred_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )
            self._deferred_actor_ckpt_path = None
            psrl_logger.info("Resume (NIXL): deferred actor checkpoint loaded.")

        # Worker property replacement clears paused state, so restore it after version synchronization.
        psrl_logger.info("Step 7.5: Re-pausing validation instances in gateway after version sync...")
        paused_base_worker_ids = self.tag_to_base_worker_ids.get("validate", [])
        if paused_base_worker_ids:
            self._post_gateway_worker_routing_control("pause", paused_base_worker_ids)

        self.is_rollout_mode_in_actor = False

    def _make_broadcast_plan(self, src_agent_names, dst_agent_names) -> dict:
        """Create a broadcast plan mapping source agents to destination agents.

        Args:
            src_agent_names (list): List of source agent names.
            dst_agent_names (list): List of destination agent names.
        Returns:
            dict: A dictionary mapping each source agent to a list of destination agents.
        """
        broadcast_plan = {src_agent: [] for src_agent in src_agent_names}
        for i, dst_agent in enumerate(dst_agent_names):
            src_agent = src_agent_names[i % len(src_agent_names)]
            broadcast_plan[src_agent].append(dst_agent)
        return broadcast_plan

    def _broadcast_updated_client_infos_from_ps_manager(self, updated_client_names: list):
        """Broadcast updated client infos from PS manager to all nixl clients.

        Args:
            updated_client_names (list): List of updated client names to broadcast.
        """
        src_agent_names = [NIXL_META_SERVER_NAME]
        dst_agent_names = []
        # 1. PS storage workers
        for i in range(self.ps_wg.world_size):
            dst_agent_names.append(ps_agent_name(i))
        # 2. rollout workers
        for i in range(self.n_rollout_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "rollout" and k.instance_id == i)
            for rank in range(tp_size):
                dst_agent_names.append(WorkerKey("rollout", i, rank).to_nixl_client_name())
        # 3. validate workers
        for i in range(self.n_validate_instances):
            tp_size = sum(1 for k in self.worker_to_ps_idx if k.role == "validate" and k.instance_id == i)
            for rank in range(tp_size):
                dst_agent_names.append(WorkerKey("validate", i, rank).to_nixl_client_name(self.n_rollout_instances))
        # 4. actor workers
        for i in range(self.actor_wg.world_size):
            dst_agent_names.append(train_client_name(i))
        psrl_logger.debug(f"Destination agent names for broadcasting: {dst_agent_names}")

        broadcast_plan = self._make_broadcast_plan(src_agent_names, dst_agent_names)
        futures = []
        for src_agent, dst_agents in broadcast_plan.items():
            if src_agent == NIXL_META_SERVER_NAME:
                futures.append(
                    self.ps_manager_handle.nixl_broadcast_update_client_infos.remote(dst_agents, updated_client_names)
                )
            else:
                raise NotImplementedError(
                    "Only meta server broadcasting is implemented in _broadcast_updated_client_infos_from_ps_manager"
                )

        # recv broadcast results by all clients
        # 1. ps storage workers
        futures.extend(self.ps_wg.execute_all_async("nixl_wait_for_update_infos", 1))
        # 2. rollout workers
        futures.extend(
            [
                self.tag_to_server_handles["rollout"][i].nixl_wait_for_update_infos.remote(1)
                for i in range(self.n_rollout_instances)
            ]
        )
        futures.extend(
            [
                self.tag_to_server_handles["validate"][i].nixl_wait_for_update_infos.remote(1)
                for i in range(self.n_validate_instances)
            ]
        )
        # 3. actor workers
        futures.extend(self.actor_wg.execute_all_async("nixl_wait_for_update_infos", 1))
        ray.get(futures)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        psrl_logger.info(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            psrl_logger.warning(
                "remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        # AGENT(VERL): PSRL use `actor_wg` while VERL use `actor_rollout_wg`.

        self.actor_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        # AGENT(VERL): dataloader state is saved in `data_processor` in PSRL.
        ray.get(self.data_processor.save_train_dataloader.remote(dataloader_local_path))

        # AGENT(VERL): PSRL use `config.train_actor_rollout_ref` while VERL use `config.actor_rollout_ref.actor`.
        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.train_actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.train_actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.train_actor_rollout_ref.actor.checkpoint
            and self.config.train_actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            psrl_logger.info("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _resolve_resume_checkpoint_paths(self) -> tuple[str, str] | None:
        """Resolve actor and critic checkpoint paths for resume, without side effects.

        Returns:
            (actor_path, critic_path) if a resume checkpoint exists, else None.
            Does NOT set global_steps or load any weights.
        """
        if self.config.trainer.resume_mode == "disable":
            return None

        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")

        checkpoint_folder = self.config.trainer.default_local_dir
        if not os.path.isabs(checkpoint_folder):
            checkpoint_folder = os.path.join(os.getcwd(), checkpoint_folder)
        global_step_folder = find_latest_ckpt_path(checkpoint_folder)

        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                return None
        elif self.config.trainer.resume_mode == "resume_path":
            assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
            assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
            global_step_folder = self.config.trainer.resume_from_path
            if not os.path.isabs(global_step_folder):
                global_step_folder = os.path.join(os.getcwd(), global_step_folder)

        return (
            os.path.join(global_step_folder, "actor"),
            os.path.join(global_step_folder, "critic"),
        )

    def _resume_load_and_push_to_ps(self) -> None:
        """Load actor resume checkpoint and push weights to PS before initial_pull_from_ps.

        This must be called inside init_workers() after the NIXL protocol and PS buffer
        initialisation (write_checkpoint_to_registered_tensors / broadcast_init) but
        before initial_pull_from_ps, so that gen/val workers receive resume weights on
        their very first pull.

        Two cases:
        - Path A (is_rollout_mode_in_actor=False): Actor already has a live NIXL
          connection and real GPU memory. Load directly, then push.
        - Path B (is_rollout_mode_in_actor=True): Actor is in meta-sleep with only
          meta-tensor NIXL descriptors. We perform a mini wake → load → push → sleep
          cycle. switch_to_trainer_mode() will do a proper wake later with a full
          info broadcast. The sleep here uses "full" mode so
          switch_to_trainer_mode re-registers with fresh addresses.

        Sets self._actor_resume_loaded = True when a checkpoint was applied, which
        causes _load_checkpoint() to skip the actor re-load.
        """
        is_nixl_mode = self.config.psrl.ps_mode in ["nixl_cpu", "nixl_gpu"]
        if not is_nixl_mode:
            return

        resume_paths = self._resolve_resume_checkpoint_paths()
        assert resume_paths is not None, "Resume checkpoint paths are not resolved"

        actor_path, _ = resume_paths
        resume_version = int(actor_path.split("global_step_")[-1].split("/")[0])

        if not self.is_rollout_mode_in_actor:
            # Path A has live actor NIXL, so load the checkpoint and push.
            with log_dual_events(
                "Resume (Path A): load actor ckpt + push to PS", psrl_logger, event_type=EventType.INIT
            ):
                psrl_logger.info(f"Resume (Path A): loading actor checkpoint from {actor_path}...")
                self.actor_wg.load_checkpoint(
                    actor_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )
                # Pre-initialize PS version so push_model advances to the correct version
                ray.get(self.ps_manager_handle.init_model_version_for_resume.remote(resume_version - 1))
                psrl_logger.info("Resume (Path A): pushing resume weights to PS...")
                ray.get(self.actor_wg.execute_all_async("push_model"))
                psrl_logger.info("Resume (Path A): PS now holds resume checkpoint weights.")
        else:
            # Path B wakes the sleeping actor for a load, push, and sleep cycle.
            # The actor writes to PS without an information broadcast.
            with log_dual_events(
                "Resume (Path B): mini wake-load-push-sleep cycle", psrl_logger, event_type=EventType.INIT
            ):
                psrl_logger.info("Resume (Path B): waking actor for checkpoint push...")
                ray.get(self.actor_wg.execute_all_async("nixl_wake_up"))
                psrl_logger.info(f"Resume (Path B): loading actor checkpoint from {actor_path}...")
                self.actor_wg.load_checkpoint(
                    actor_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )
                # Pre-initialize PS version so push_model advances to the correct version
                ray.get(self.ps_manager_handle.init_model_version_for_resume.remote(resume_version - 1))
                psrl_logger.info("Resume (Path B): pushing resume weights to PS...")
                ray.get(self.actor_wg.execute_all_async("push_model"))
                # Sleep with "full" mode: release GPU memory AND deregister NIXL.
                # switch_to_trainer_mode() will re-register with fresh addresses.
                ray.get(self.actor_wg.execute_all_async("nixl_sleep", "full"))
                psrl_logger.info("Resume (Path B): actor re-sleeping. PS holds resume weights.")

        self._actor_resume_loaded = True

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                psrl_logger.info("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        psrl_logger.info(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        psrl_logger.info(f"Setting global step to {self.global_steps}")
        psrl_logger.info(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")

        # Skip actor reload when initialization already pushed the checkpoint to PS.
        if getattr(self, "_actor_resume_loaded", False):
            psrl_logger.info("Resume: actor checkpoint was already loaded in init_workers(). Skipping actor reload.")
        else:
            # Defer loading while the NIXL actor is in meta sleep because its CUDA storage is unavailable.
            # Apply the checkpoint after `switch_to_trainer_mode` restores NIXL connections.
            is_nixl_mode = self.config.psrl.ps_mode in ["nixl_cpu", "nixl_gpu"]
            need_nixl_resume = is_nixl_mode and self.is_rollout_mode_in_actor
            if need_nixl_resume:
                psrl_logger.info("Resume (NIXL): deferring actor checkpoint load until after switch_to_trainer_mode.")
                self._deferred_actor_ckpt_path = actor_path
            else:
                # load actor (train only)
                self.actor_wg.load_checkpoint(
                    actor_path,
                    del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                )

        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # load dataloader
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            # AGENT(VERL): dataloader state is saved in `data_processor` in PSRL.
            ray.get(self.data_processor.load_train_dataloader.remote(dataloader_local_path))
        else:
            psrl_logger.info(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        # AGENT(VERL): PSRL use `actor_wg` while VERL use `actor_rollout_wg`.
        if do_profile:
            self.actor_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        # AGENT(VERL): PSRL use `actor_wg` while VERL use `actor_rollout_wg`.
        if do_profile:
            self.actor_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()

    def _get_required_batch_multiple(self, dp_size: int) -> int:
        """Return the global batch multiple required by downstream train steps(e.g. critics, actors)."""
        if self.batch_schedule_strategy.dispatch_steps_individually:
            # Independently padded steps require only DP divisibility for preprocessing and inference.
            return dp_size

        required_multiple = dp_size

        # If enabled with critic training, the batch should align with critic PPO mini-batches.
        if self.use_critic:
            critic_global_mini_batch_size = self.config.critic.ppo_mini_batch_size
            critic_global_mini_batch_size *= self.config.train_actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, critic_global_mini_batch_size)

        # If there is an actor update, the batch should align with actor PPO mini-batches too.
        if self.config.trainer.critic_warmup <= self.global_steps:
            actor_global_mini_batch_size = self.config.train_actor_rollout_ref.actor.ppo_mini_batch_size
            actor_global_mini_batch_size *= self.config.train_actor_rollout_ref.rollout.n
            required_multiple = math.lcm(required_multiple, actor_global_mini_batch_size)

        # Notice lcm(a, b, c) == lcm(lcm(a, b), c), so it is optimal.
        return required_multiple

    def _prepare_scheduled_batch(
        self,
        batch: KVBatchMeta,
        step: BatchScheduleStep,
        dp_size: int,
    ) -> tuple[KVBatchMeta, list[str]]:
        """Materialize one schedule step's metadata and add DP-only padding."""
        if step.sample_indices is None:
            return batch, []

        scheduled_batch = batch.select_keys(resolve_sample_keys(batch.keys, step.sample_indices))
        original_size = len(scheduled_batch)
        scheduled_batch = upsample_batch_to_divisible_size(
            scheduled_batch,
            dp_size,
            self.tokenizer.eos_token_id,
        )
        return scheduled_batch, scheduled_batch.keys[original_size:]

    @staticmethod
    def _reduce_scheduled_metrics(
        outputs: list[TensorDict],
        prefix: str,
        mfu_key: str,
    ) -> dict:
        """Reduce metrics emitted by one or more scheduled worker dispatches."""
        metrics_across_steps: dict[str, list] = defaultdict(list)
        for output in outputs:
            step_metrics = reduce_metrics(dict(output["metrics"]))
            step_metrics = rename_dict(step_metrics, prefix)
            step_metrics[mfu_key] = step_metrics.pop(f"{prefix}mfu")
            for key, value in step_metrics.items():
                metrics_across_steps[key].append(value)
        return reduce_metrics(metrics_across_steps)

    def _run_scheduled_update(
        self,
        batch: KVBatchMeta,
        *,
        training_config: DictConfig,
        dp_size: int,
        base_extra_info: dict,
        update_fn: Callable[[KVBatchMeta], TensorDict],
        metric_prefix: str,
        mfu_key: str,
        finalize_step: bool = True,
        push_model: bool = False,
    ) -> dict:
        """Build and execute a schedule with shared lifecycle and metric handling."""
        entries_per_update = self.batch_schedule_strategy.entries_per_update(
            training_config.ppo_mini_batch_size,
            rollout_n=self.config.gen_actor_rollout_ref.rollout.n,
            algorithm_rollout_n=self.alg_rollout_n,
        )
        schedule = build_batch_schedule(
            batch.tags,
            self.batch_agg_mode,
            entries_per_update,
            epochs=training_config.ppo_epochs,
            shuffle=training_config.shuffle,
            seed=training_config.data_loader_seed,
        )
        outputs: list[TensorDict] = []
        for step_index, step in enumerate(schedule.steps):
            scheduled_batch, padding_keys = self._prepare_scheduled_batch(batch, step, dp_size)
            is_final_step = step_index == len(schedule.steps) - 1
            scheduled_batch.extra_info.update(base_extra_info)
            scheduled_batch.extra_info.update(step.worker_extra_info())
            scheduled_batch.extra_info["advance_lr_scheduler"] = finalize_step and is_final_step
            if push_model:
                scheduled_batch.extra_info["push_model"] = finalize_step and is_final_step

            try:
                outputs.append(update_fn(scheduled_batch))
            finally:
                if padding_keys:
                    clear_payload(
                        keys=padding_keys,
                        partition_id=scheduled_batch.partition_id,
                        state=PayloadState.CONSUMED,
                    )
                    self.replay_buffer.remove(padding_keys, scheduled_batch.partition_id)

        psrl_logger.info(
            "%s batch schedule: aggregation_mode=%s, entries_per_update=%d, "
            "worker_dispatches=%d, physical_samples=%d.",
            metric_prefix.rstrip("/"),
            schedule.aggregation_mode,
            schedule.entries_per_update,
            len(schedule.steps),
            sum(not tag.get("is_padding", False) for tag in batch.tags),
        )
        return self._reduce_scheduled_metrics(outputs, metric_prefix, mfu_key)

    def _balance_batch(self, batch: KVBatchMeta, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        batch_size = len(batch)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_wg, "actor")

        batch_multiple = self._get_required_batch_multiple(dp_size)
        batch = upsample_batch_to_divisible_size(batch, batch_multiple, self.tokenizer.eos_token_id)
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=["seq_len"])
        global_seqlen_lst = torch.tensor(tu.get(data, "seq_len"), dtype=torch.int64)
        workload_lst = calculate_workload(global_seqlen_lst)

        # Keep samples with the same `parent_id` together for `PrefixGrouper`.
        if getattr(self, "use_prefix_grouper", False) and "parent_id" in batch.tags[0]:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = [tag["parent_id"] for tag in batch.tags]
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.train_actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return batch

    def _compute_values(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the values of the batch."""
        # 1. compute value
        output = self.critic_wg.infer_batch(batch)
        ray.get(output.futures)

        # 2. write value back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["values", "response_mask"]
        )
        data["values"] = response_from_nested(data.pop("values"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("values"))

        return batch

    def _compute_advantage(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the advantage of the batch."""
        fields = [
            "uid",
            "parent_id",
            "response_mask",
            "rm_scores",
            "rollout_log_probs",
            "old_log_probs",
            "ref_log_prob",
            "values",
        ]
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        response_mask = data["response_mask"]

        # Extract non-tensor uid/parent_id BEFORE calling to_padded_tensor() to avoid
        # dtype conversion issues (they are stored as NonTensorData/NonTensorStack in TQ).
        uids = tu.get(data, "uid")
        parent_ids = tu.get(data, "parent_id")
        # Remove non-tensor fields so to_padded_tensor() only processes tensor fields.
        tensor_fields = ["response_mask", "rm_scores", "rollout_log_probs", "old_log_probs", "ref_log_prob", "values"]
        data_tensor_only = data.select(*[f for f in tensor_fields if f in data.keys()])
        data = DataProto(batch=data_tensor_only.to_padded_tensor())
        data.batch["token_level_scores"] = data.batch["rm_scores"]
        data.non_tensor_batch["uid"] = np.array(uids, dtype=object)
        data.non_tensor_batch["parent_id"] = np.array(parent_ids, dtype=object)

        # 1. apply kl penalty to rewards
        if self.config.algorithm.use_kl_in_reward:
            data, kl_metrics = apply_kl_penalty(
                data,
                kl_ctrl=self.kl_ctrl_in_reward,
                kl_penalty=self.config.algorithm.kl_penalty,
            )
            metrics.update(kl_metrics)
        else:
            data.batch["token_level_rewards"] = data.batch["token_level_scores"]

        # Compute rollout correction once per batch against a stable old policy.
        # Bypass mode leaves the evolving policy comparison to the actor.
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        rollout_correction = (
            rollout_corr_config is not None and "rollout_log_probs" in data.batch and not bypass_recomputing_logprobs
        )
        if rollout_correction:
            data, is_metrics = compute_rollout_correction_and_add_to_batch(data, rollout_corr_config)
            metrics.update(is_metrics)

        # 3. compute advantages
        data = compute_advantage_for_multi_trajectories(
            data,
            batch_keys=batch.keys,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.gen_actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=self.config.algorithm.get("norm_adv_by_std_in_grpo", True),
            config=self.config.algorithm,
        )

        # 4. Write response fields and dense session weights back to TransferQueue.
        fields = ["advantages", "returns"]
        if self.config.algorithm.use_kl_in_reward:
            fields.append("token_level_rewards")
        if rollout_correction:
            fields.append("response_mask")
            if "rollout_is_weights" in data.batch:
                fields.append("rollout_is_weights")

        output = {}
        for field in fields:
            output[field] = response_to_nested(data.batch[field], response_mask)
        if self.config.train_actor_rollout_ref.actor.loss_agg_mode == "session-mean-token-mean":
            output["session_loss_weights"] = compute_session_loss_weights(data.batch["response_mask"], uids)
        output = TensorDict(output, batch_size=len(batch))
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=output)

        return batch

    def _compute_ref_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the reference log prob of the batch."""
        # 1. compute log probs
        metadata = {
            "calculate_entropy": False,
            "compute_loss": False,
            "temperature": self.config.gen_actor_rollout_ref.rollout.temperature,
        }
        if self.ref_in_actor:
            metadata["no_lora_adapter"] = True
        batch.extra_info.update(metadata)
        if self.ref_in_actor:
            output = self.actor_wg.compute_log_prob(batch)
        else:
            output = self.ref_policy_wg.compute_ref_log_prob(batch)
        assert len(output) == len(batch), (
            f"Reference log-probability output size {len(output)} does not match batch size {len(batch)}."
        )

        # 2. write ref_log_prob and entropy back to TransferQueue
        data = tq.kv_batch_get(
            keys=batch.keys, partition_id=batch.partition_id, select_fields=["log_probs", "response_mask"]
        )
        data["ref_log_prob"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select("ref_log_prob"))

        return batch

    def collect_prefix_match_rate(self, batch: KVBatchMeta, metrics: dict) -> None:
        """Compute and log prefix-tree compression metrics of the current global batch.

        Records the AReaL-DTA compression ratio ``C`` and sharing ratio
        ``S = 1 - 1/C`` (arXiv:2602.00482) for the full global batch, plus the
        within-group / cross-group compression distributions. Computed before
        mini-batch splitting and logged under ``pmr/*``. Gated by
        ``trainer.enable_pmr_analysis``. All samples in the batch are included.
        """
        from psrl.utils.metrics.prefix_match_rate import compute_pmr_metrics, sequences_from_batch

        data = tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["input_ids", "parent_id"],
        )
        seqs, group_ids = sequences_from_batch(data)
        pmr = compute_pmr_metrics(seqs, group_ids=group_ids)
        metrics.update({key: value for key, value in pmr.items() if value is not None})

    def _compute_old_log_prob(self, batch: KVBatchMeta, metrics: dict) -> KVBatchMeta:
        """Compute the old log prob of the batch."""
        # Bypass uses rollout log probabilities. Decoupled mode recomputes a stable proximal anchor.
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
        if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
            data = tq.kv_batch_get(
                keys=batch.keys, partition_id=batch.partition_id, select_fields=["rollout_log_probs"]
            )
            data["old_log_probs"] = data.pop("rollout_log_probs")
            tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data)

            policy_loss_config = self.config.train_actor_rollout_ref.actor.policy_loss
            with open_dict(policy_loss_config):
                # Pass rollout_correction config to actor for loss computation and metrics
                policy_loss_config["rollout_correction"] = rollout_corr_config
                # Always use bypass_mode loss function which handles both loss_types
                policy_loss_config["loss_mode"] = "bypass_mode"
            return batch

        # 1. compute log probs
        # add meta info
        calculate_sum_pi_squared = self.config.train_actor_rollout_ref.actor.get("calculate_sum_pi_squared", False)
        batch.extra_info.update(
            {
                "calculate_entropy": True,
                "compute_loss": False,
                "calculate_sum_pi_squared": calculate_sum_pi_squared,
                "temperature": self.config.gen_actor_rollout_ref.rollout.temperature,
            }
        )
        output: KVBatchMeta = self.actor_wg.compute_log_prob(batch)
        assert len(output) == len(batch), (
            f"Actor log-probability output size {len(output)} does not match batch size {len(batch)}."
        )

        actor_config = self.config.train_actor_rollout_ref.actor
        use_session_mean = actor_config.loss_agg_mode == "session-mean-token-mean"
        fields = [
            "entropy",
            "log_probs",
            "sum_pi_squared",
            "response_mask",
            "responses",
            "rollout_log_probs",
            "metrics",
        ]
        if use_session_mean:
            fields.append("uid")
        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)
        uids = tu.pop(data, "uid") if use_session_mean else None

        # Training pads log probabilities to `responses` but slices them by `response_mask`.
        # Enforce equal row lengths here so drift fails near its source.
        response_lens = data["responses"].offsets().diff()
        mask_lens = data["response_mask"].offsets().diff()
        psrl_logger.info(
            "[length-contract] old_log_prob stage: rows=%d responses[min=%d max=%d] "
            "response_mask[min=%d max=%d] equal=%s",
            len(response_lens),
            int(response_lens.min()),
            int(response_lens.max()),
            int(mask_lens.min()),
            int(mask_lens.max()),
            bool(torch.equal(response_lens, mask_lens)),
        )
        if not torch.equal(response_lens, mask_lens):
            bad = (response_lens != mask_lens).nonzero().flatten().tolist()
            raise AssertionError(
                f"responses and response_mask disagree on length for {len(bad)} of "
                f"{len(response_lens)} rows. First offenders (row, responses, mask): "
                f"{[(int(i), int(response_lens[i]), int(mask_lens[i])) for i in bad[:5]]}. "
                "Both are written per trajectory by the agent loop and must stay in lockstep, "
                "because old_log_probs is cut to the mask length while the training forward is "
                "padded to the responses length."
            )

        # 2. write old_log_probs and entropy back to TransferQueue
        data["old_log_probs"] = response_from_nested(data.pop("log_probs"), data["response_mask"])
        data["entropy"] = response_from_nested(data.pop("entropy"), data["response_mask"])
        if calculate_sum_pi_squared:
            data["sum_pi_squared"] = response_from_nested(data.pop("sum_pi_squared"), data["response_mask"])
        # old_log_prob_mfu = tu.get(data, "metrics")["mfu"]
        fields = ["old_log_probs", "entropy"]
        if calculate_sum_pi_squared:
            fields.append("sum_pi_squared")
        tq.kv_batch_put(keys=batch.keys, partition_id=batch.partition_id, fields=data.select(*fields))

        data = DataProto(batch=data.to_padded_tensor())

        # 3. calculate actor entroy metrics
        session_loss_weights = None
        if use_session_mean:
            # Diagnostic masks and chunk boundaries can differ from the later training window.
            session_loss_weights = compute_session_loss_weights(data.batch["response_mask"], uids)
        entropy_agg = agg_loss(
            loss_mat=data.batch["entropy"],
            loss_mask=data.batch["response_mask"],
            loss_agg_mode=actor_config.loss_agg_mode,
            loss_scale_factor=actor_config.loss_scale_factor,
            session_loss_weights=session_loss_weights,
        )
        old_log_prob_metrics = {
            "actor/entropy": entropy_agg.detach().item(),
            # "perf/mfu/actor_infer": old_log_prob_mfu, # TODO(linsh): no global_token_num for mfu
        }
        metrics.update(old_log_prob_metrics)

        # 4. calculate rollout vs actor logprobs diff
        if "rollout_log_probs" in data.batch:
            metrics.update(calculate_debug_metrics(data))

        return batch

    def _update_actor(
        self,
        batch: KVBatchMeta,
        metrics: dict,
        finalize_step: bool = True,
    ) -> KVBatchMeta:
        """Update the actor and finalize shared step state on the last window."""
        actor_config = self.config.train_actor_rollout_ref.actor
        calculate_entropy = actor_config.calculate_entropy or actor_config.entropy_coeff != 0.0
        distillation_use_topk = (
            self.distillation_config.distillation_loss.loss_settings.use_topk
            if is_distillation_enabled(self.config.get("distillation"))
            else False
        )
        base_extra_info = {
            "calculate_entropy": calculate_entropy,
            "multi_turn": self.config.gen_actor_rollout_ref.rollout.multi_turn.enable,
            "distillation_use_topk": distillation_use_topk,
            "compute_loss": True,
        }
        actor_metrics = self._run_scheduled_update(
            batch,
            training_config=actor_config,
            dp_size=self._get_dp_size(self.actor_wg, "actor"),
            base_extra_info=base_extra_info,
            update_fn=self.actor_wg.update_actor,
            metric_prefix="actor/",
            mfu_key="perf/mfu/actor",
            finalize_step=finalize_step,
            push_model=True,
        )
        metrics.update(actor_metrics)
        return batch

    def _update_critic(
        self,
        batch: KVBatchMeta,
        metrics: dict,
        finalize_step: bool = True,
    ) -> KVBatchMeta:
        """Update the critic and advance its scheduler on the last window."""
        critic_config = self.config.critic

        def update_critic(scheduled_batch: KVBatchMeta) -> TensorDict:
            return self.critic_wg.train_mini_batch(scheduled_batch).get()

        critic_metrics = self._run_scheduled_update(
            batch,
            training_config=critic_config,
            dp_size=self._get_dp_size(self.critic_wg, "critic"),
            base_extra_info={},
            update_fn=update_critic,
            metric_prefix="critic/",
            mfu_key="perf/mfu/critic",
            finalize_step=finalize_step,
        )
        metrics.update(critic_metrics)

        return batch

    def _compute_metrics(self, batch: KVBatchMeta, metrics, timing_raw, global_steps):
        # Fetch only real samples because schedule-local padding keys can collide after cleanup.
        real_keys = [key for key, tag in zip(batch.keys, batch.tags) if not tag.get("is_padding", False)]
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "values",
            "advantages",
            "returns",
            "rm_scores",
            "token_level_rewards",
            "num_turns",
            "terminate_reason",
            "parent_id",
        ]
        # GDPO per-component reward metrics
        gdpo_reward_keys = self.config.algorithm.get("gdpo_reward_keys", None)
        if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
            fields.extend(gdpo_reward_keys)
        data = tq.kv_batch_get(keys=real_keys, partition_id=batch.partition_id, select_fields=fields)
        num_turns = np.array(data.pop("num_turns").tolist())
        # Read before `to_padded_tensor`, which drops non-tensor columns.
        terminate_reasons = [str(v) for v in tu.get(data, "terminate_reason")]
        metric_parent_ids = [str(v) for v in tu.get(data, "parent_id")]
        prompt_length = data["prompts"].offsets().diff()
        response_length = data["responses"].offsets().diff()
        global_token_num = (prompt_length + response_length).tolist()
        data = data.to_padded_tensor()
        data["token_level_scores"] = data["rm_scores"]
        if "token_level_rewards" not in data:
            data["token_level_rewards"] = data["rm_scores"]
        data["prompt_length"] = prompt_length.float()
        data["response_length"] = response_length.float()
        batch = DataProto(
            batch=data,
            meta_info={
                "global_token_num": global_token_num,
                "max_prompt_length": self.config.data.max_prompt_length,
                "max_response_length": self.config.data.max_response_length,
            },
        )
        # `data` holds only real samples, so every metric below sees the same batch.
        # 2. compute metrics
        metrics.update({"training/global_step": global_steps})
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
        gradient_norm = metrics.get("actor/grad_norm", None)
        metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))

        # 3. other auxiliary metrics
        metrics.update(
            {
                "training/num_turns/mean": num_turns.mean(),
                "training/num_turns/max": num_turns.max(),
                "training/num_turns/min": num_turns.min(),
            }
        )

        metrics.update(
            _compute_termination_metrics(
                terminate_reasons=terminate_reasons,
                parent_ids=metric_parent_ids,
                scores=data["token_level_scores"].sum(dim=-1).tolist(),
                trained_tokens=data["response_mask"].sum(dim=-1).tolist(),
            )
        )

        # 4. GDPO per-component reward metrics
        if gdpo_reward_keys and self.config.algorithm.adv_estimator in ("gdpo", AdvantageEstimator.GDPO):
            for key in gdpo_reward_keys:
                vals = np.array(data.pop(key).tolist(), dtype=np.float32)
                metrics[f"gdpo/{key}/mean"] = float(np.mean(vals))
                metrics[f"gdpo/{key}/std"] = float(np.std(vals))
                metrics[f"gdpo/{key}/max"] = float(np.max(vals))
                metrics[f"gdpo/{key}/min"] = float(np.min(vals))

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        from verl.utils.tracking import Tracking

        if self._dump_executor._shutdown:
            self._init_dump_executor()

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        psrl_logger.info(
            f"Initialized tracking logger with project: {self.config.trainer.project_name}, "
            f"experiment: {self.config.trainer.experiment_name}"
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        # Record starting step so buffer_id is always relative to this run's step 0.
        # Subtracting the starting step keeps the first `buffer_id` at zero.
        self._start_global_steps = self.global_steps
        # AGENT(VERL): ignore checkpoint manager in PSRL

        # AGENT(VERL): earlier check in PSRL
        if self.global_steps >= self.total_training_steps:
            psrl_logger.warning(
                f"Global steps {self.global_steps} >= total training steps {self.total_training_steps}, "
                "skipping training."
            )
            return

        # AGENT(VERL): PSRL specific initialization process
        self.init_agent_loop_manager()

        # Initialize agent loop manager with the resume version so it can correctly
        # compute expected PS versions and avoid dispatching data with stale version tags
        if self.global_steps > 0:
            ray.get(self.agent_loop_manager.set_initial_ps_version.remote(self.global_steps))

        futures = []
        futures.append(self.data_processor.set_agent_loop_manager.remote(self.agent_loop_manager))
        for agent_loop_worker in self.agent_loop_workers:
            futures.append(agent_loop_worker.set_agent_loop_manager.remote(self.agent_loop_manager))
        ray.get(futures)

        self.init_reward_manager()
        futures = []
        futures.append(self.data_processor.set_reward_manager.remote(self.reward_manager))
        futures.append(self.ps_manager_handle.set_reward_manager.remote(self.reward_manager))
        futures.append(self.agent_loop_manager.set_reward_manager.remote(self.reward_manager))
        for agent_loop_worker in self.agent_loop_workers:
            futures.append(agent_loop_worker.set_reward_manager.remote(self.reward_manager))
        for reward_loop_worker in self.reward_loop_workers:
            futures.append(reward_loop_worker.set_reward_manager.remote(self.reward_manager))
        ray.get(futures)

        # Start data pipeline
        # Start rollout coordinator to handle rollouts and data generation
        psrl_logger.info("Starting rollout coordinator...")
        self.start_rollout_coordinator()
        psrl_logger.info("Rollout coordinator started successfully.")

        # Start agent loop manager to handle agent-environment interactions
        psrl_logger.info("Starting agent loop manager...")
        self.start_agent_loop_manager()
        psrl_logger.info("Agent loop manager started successfully.")

        # Start reward manager to handle reward computation requests
        psrl_logger.info("Starting reward manager...")
        self.start_reward_manager()
        psrl_logger.info("Reward manager started successfully.")

        psrl_logger.info("All data pipeline components started successfully.")

        # AGENT(VERL): continue verl's fit logics

        # perform validation before training
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            psrl_logger.info(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._shutdown_dump_executor()
                return

        # AGENT(VERL): skip RolloutSkip and start PSRL's data processor for preprocessing and batching.
        psrl_logger.info("Starting data processor...")
        self.start_data_processor()
        psrl_logger.info("Data processor started successfully.")

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        from psrl.trainer.ppo.strategies import build_step_strategy

        self._step_strategy = build_step_strategy(
            self.config.psrl.get("fine_grain_overlap", None),
            trainer=self,
        )
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        while True:
            if hasattr(self.actor_wg, "async_calls_finalize_fn_exec"):
                self.actor_wg.async_calls_finalize_fn_exec(blocking=False)
            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps == self.total_training_steps

            with marked_timer("step", timing_raw):
                # The strategy owns batch acquisition and may start before the full batch is ready.
                buffer_id = self.global_steps - 1

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch = self._step_strategy.run_step(buffer_id, metrics, timing_raw)

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            self._compute_metrics(batch, metrics, timing_raw, global_steps=self.global_steps)

            clear_payload(
                keys=batch.keys,
                partition_id=batch.partition_id,
                state=PayloadState.CONSUMED,
            )
            self.replay_buffer.remove(batch.keys, batch.partition_id)

            # AGENT(VERL): skip curriculum sampler processing here in PSRL.

            # TODO(verl): Make a canonical logger that supports every backend.
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.train_actor_rollout_ref.actor, "profiler")
                and self.config.train_actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}",
                    sub_dir=f"step{self.global_steps}",
                )

            if is_last_step:
                if hasattr(self.actor_wg, "async_calls_finalize_fn_exec"):
                    self.actor_wg.async_calls_finalize_fn_exec(blocking=True)
                self._shutdown_dump_executor()
                psrl_logger.info(f"Final validation metrics: {self._last_val_metrics}")
                progress_bar.close()
                break

        # Stop all components
        self.stop_data_processor()
        self.stop_agent_loop_manager()
        self.stop_rollout_coordinator()
        self.stop_reward_manager()
        if self.elastic_executor is not None:
            ray.get(self.elastic_executor.stop_busy_loop.remote())
            self.elastic_executor = None
        if self.env_worker_manager is not None:
            self.env_worker_manager.shutdown()
            self.env_worker_manager = None
        self.stop_ps_manager()
        self._shutdown_dump_executor()

        # Kill all PortScanner actors to free resources.
        for handle in self.port_scanner_handles.values():
            ray.kill(handle)

        # AGENT(VERL): skip `on_batch_end` processing in PSRL.

        psrl_logger.info("Training completed successfully!")
