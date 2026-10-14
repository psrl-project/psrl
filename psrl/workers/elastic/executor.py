import asyncio
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto

import ray

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import ps_agent_name, train_client_name
from psrl.utils.elastic.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import DualOutputHandler, FileOnlyHandler
from psrl.utils.server.command import Command, CommandType
from psrl.workers.elastic.cluster_topology import ClusterTopology, GPUSlot, InstanceIdentifier, InstanceStatus
from psrl.workers.elastic.policy_runtime import PolicyIdentityMap, uses_integer_solver
from psrl.workers.elastic.scaling_policy.base import InstanceSignal, ScalingPolicy
from psrl.workers.elastic.scaling_policy.factory import create_scaling_policy
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
monitor_logger = logging.getLogger("ElasticMonitor")


@dataclass(frozen=True)
class ReplicaRegistration:
    gateway_replica_id: str
    tensor_parallel_size: int
    pipeline_parallel_size: int
    data_parallel_size: int
    worker_gpu_locations: tuple[tuple[str, int], ...]
    pool_id: str

    @classmethod
    def from_replica(
        cls,
        replica,
        placement,
    ):
        """
        Capture placement metadata without transferring the replica runtime.
        """
        return cls(
            replica.gateway_replica_id,
            replica.tensor_parallel_size,
            replica.pipeline_parallel_size,
            replica.data_parallel_size,
            tuple(replica.worker_gpu_locations),
            placement.pool_id,
        )


class TrainerState(Enum):
    AWAKE = auto()
    ASLEEP = auto()
    RESTORING = auto()


@ray.remote
class ElasticExecutor:
    def __init__(
        self,
        config,
        roles: list[tuple[PSRL_Role, str]],
        coordinators: dict[PSRL_Role, dict[str, ray.actor.ActorHandle]],
        agent_loop_manager: ray.actor.ActorHandle | None = None,
        elastic_config: dict | None = None,
    ):
        self.coordinators = coordinators
        self.roles = roles
        self.elastic_config = elastic_config or {}
        self.config = config
        self.agent_loop_manager = agent_loop_manager

        self.instances_status_flags: dict[PSRL_Role, dict[str, dict[RolloutInstanceId, InstanceStatus]]] = {}
        self.instances_engine_stats: dict[PSRL_Role, dict[str, dict[RolloutInstanceId, dict]]] = {}
        self.role_model_versions: dict[tuple[PSRL_Role, str], int] = {}
        self.topology = ClusterTopology()

        for role_name, model_name in self.roles:
            self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
            self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})

        self.scale_up_task_queue: asyncio.Queue = asyncio.Queue()
        self.scale_down_task_queue: asyncio.Queue = asyncio.Queue()

        # Task management
        self.running_loop = None
        self.monitor_task = None
        self.scale_up_task = None
        self.scale_down_task = None
        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False

        # Scaling policy
        self.scaling_policy = create_scaling_policy(config=self.config, policy_config=self.elastic_config)
        self.policy_identity_map = PolicyIdentityMap()

        # Whether current decision is being executed.
        self._decision_execution_in_progress = False
        self._instance_state_epoch = 0
        self._next_decision_id = 1
        self._decision_pending_action_counts: dict[int, int] = {}
        self._decision_settled = asyncio.Event()
        self._decision_settled.set()
        self._trainer_pool_transition_lock = asyncio.Lock()
        self._trainer_pool_transition = False
        self._trainer_pool_available = False
        self._trainer_pool_epoch = 0
        self._trainer_waiting_buffer_id = None
        self.instance_pool_ids = {}
        self.actor_wg = None
        self.ps_manager_handle = None
        self.ps_wg = None
        self._trainer_state = TrainerState.AWAKE
        self._deferred_actor_ckpt_path = None
        self._nixl_servers = []
        self._nixl_destination_names = []
        self._validation_instance_ids = []
        self._validation_active = False

        # Consecutive monitor ticks where policy is blocked by an unfinished scale decision.
        self._execution_in_progress_stall_ticks: int = 0
        # Long transitions keep the decision reserved until the original command settles.
        self._decision_abandon_stall_ticks: int = int(
            self.elastic_config.get("decision_execution_abandon_stall_ticks", 0)
        )

        self._last_monitor_instance_log_ms: float = 0.0
        self._monitor_instance_log_interval_ms: int = int(
            self.elastic_config.get("monitor_instance_log_interval_ms", 5000)
        )
        self._enable_monitor_instance_log: bool = bool(self.elastic_config.get("enable_monitor_instance_log", True))
        self.router_backlog_by_role: dict[PSRL_Role, int | dict[str, int]] = {}
        self.trainer_waiting_hint: dict[str, object] = {
            "trainer_busy": True,
            "waiting_buffer_id": None,
            "waiting_on": "none",
            "breakdown": {},
        }
        self._current_training_step = -1

        # Fraction [0,1] of waiting uids to ABORT per instance after scale-up (FIFO / queue head).
        _r = float(self.elastic_config.get("post_scale_up_abort_waiting_ratio", 1.0))
        self._post_scale_up_abort_waiting_ratio = max(0.0, min(1.0, _r))
        # Per tick timeout prevents stalled coordinators from blocking the monitor loop.
        self._coordinator_sync_timeout_s = float(self.elastic_config.get("coordinator_sync_timeout_s", 60.0))
        # Optional timeout for SLEEP/WAKE_UP/ABORT issued during elastic scale (None = wait forever).
        _cmd_to = self.elastic_config.get("coordinator_command_timeout_s", None)
        if _cmd_to is None or float(_cmd_to) <= 0:
            self._coordinator_command_timeout_s: float | None = None
        else:
            self._coordinator_command_timeout_s = float(_cmd_to)

        # Ray transfers loggers by name, so configure them inside the actor process.
        self.log_prefix = "ElasticExecutor"
        psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))
        psrl_logger.propagate = False
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        monitor_logger.setLevel(logging.INFO)
        monitor_logger.propagate = False
        monitor_logger.addHandler(FileOnlyHandler(self.config.psrl.logging_path, "ElasticMonitor"))
        psrl_logger.debug(
            f"Elastic executor initialized: roles={[(role.name, model) for role, model in self.roles]!r}, "
            f"logging_level={logging.getLevelName(psrl_logger.level)!r}."
        )
        monitor_logger.info(
            f"Elastic monitor initialized: monitor_interval_ms={self.scaling_policy.monitor_interval_ms}, "
            f"instance_logging_enabled={self._enable_monitor_instance_log!r}, "
            f"instance_log_interval_ms={self._monitor_instance_log_interval_ms}."
        )

    def register_role(
        self,
        role_name: PSRL_Role,
        model_name: str,
        instance_ids: list[RolloutInstanceId],
        gpu_slots_per_instance: list[frozenset[GPUSlot]],
        pool_ids_per_instance: list[str],
    ):
        """
        Register all instances for one (role, model) pair with their GPU placements.

        Args:
            instance_ids: list of RolloutInstanceId tuples, e.g. [(replica_id, dp_rank), ...].
            gpu_slots_per_instance: parallel list of frozenset[GPUSlot], one per instance.
        """
        if len(instance_ids) != len(gpu_slots_per_instance):
            raise ValueError(
                f"instance_ids ({len(instance_ids)}) and gpu_slots_per_instance "
                f"({len(gpu_slots_per_instance)}) must have the same length."
            )
        if len(pool_ids_per_instance) != len(instance_ids):
            raise ValueError("Each registered instance must have an explicit resource pool.")
        self.instances_status_flags.setdefault(role_name, {}).setdefault(model_name, {})
        self.instances_engine_stats.setdefault(role_name, {}).setdefault(model_name, {})
        for instance_id, gpu_slots in zip(instance_ids, gpu_slots_per_instance):
            self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.ASLEEP
            self.instances_engine_stats[role_name][model_name][instance_id] = {
                **EngineStats.get_default_snapshot(),
                "elastic_model_version": -1,
            }
            self.topology.register(
                role=role_name,
                model_name=model_name,
                instance_id=instance_id,
                gpu_slots=gpu_slots,
            )
        for instance_id, pool_id in zip(instance_ids, pool_ids_per_instance, strict=True):
            self.instance_pool_ids[(role_name, model_name, instance_id)] = pool_id
        self.policy_identity_map.register(role_name, model_name, instance_ids)
        self.role_model_versions[(role_name, model_name)] = 0

    def set_agent_loop_manager(self, manager) -> None:
        """
        Bind the manager after the data pipeline is created.
        """
        self.agent_loop_manager = manager

    def set_rollout_coordinator(
        self,
        model_name: str,
        coordinator,
    ) -> None:
        """
        Bind the rollout coordinator when generation servers are created.
        """
        self.coordinators[PSRL_Role.Rollout] = {model_name: coordinator}

    @staticmethod
    def _elastic_instance_registration(
        replicas: list[ReplicaRegistration],
    ):
        """
        Match gateway identities to the GPUs assigned to each DP instance.
        """
        ids, gpu_slots, pool_ids = [], [], []
        for replica in replicas:
            workers_per_dp = replica.tensor_parallel_size * replica.pipeline_parallel_size
            if len(replica.worker_gpu_locations) != workers_per_dp * replica.data_parallel_size:
                raise ValueError("Replica GPU locations do not match its parallelism configuration.")
            for dp_rank in range(replica.data_parallel_size):
                start = dp_rank * workers_per_dp
                locations = replica.worker_gpu_locations[start : start + workers_per_dp]
                ids.append((replica.gateway_replica_id, dp_rank))
                gpu_slots.append(frozenset(GPUSlot(node, gpu) for node, gpu in locations))
                pool_ids.append(replica.pool_id)
        return ids, gpu_slots, pool_ids

    async def sleep_coordinator_instances(
        self,
        coordinator,
    ) -> list[RolloutInstanceId]:
        """
        Sleep an initialized role before another role allocates shared GPU memory.
        """
        instance_ids = await coordinator.get_all_instance_ids.remote()
        if instance_ids:
            await self._await_elastic_coordinator_command(
                coordinator,
                Command(type=CommandType.SLEEP, instance_ids=instance_ids),
                stage="INITIAL_SLEEP",
            )
        return instance_ids

    async def initialize_inference_runtime(
        self,
        coordinators,
        registrations,
        validation_instance_ids,
    ) -> None:
        """
        Register physical placements and wake non-conflicting inference roles.
        """
        self.coordinators = coordinators
        self.roles = list(registrations)
        self._validation_instance_ids = list(validation_instance_ids)
        for (role, model), replicas in registrations.items():
            all_ids = await self.sleep_coordinator_instances(coordinators[role][model])
            ids, gpu_slots, pool_ids = self._elastic_instance_registration(replicas)
            if role == PSRL_Role.Rollout:
                all_ids = [iid for iid in all_ids if iid not in self._validation_instance_ids]
            if set(ids) != set(all_ids):
                raise RuntimeError(f"Placement registrations do not match coordinator instances for {model!r}.")
            self.register_role(role, model, ids, gpu_slots, pool_ids)

        if self.elastic_config.get("enable_trainer_pool", False):
            await self.release_trainer_pool()
        minimum = max(0, int(self.elastic_config["min_awake_per_role"]))
        ordered_roles = sorted(self.roles, key=lambda pair: pair[0] == PSRL_Role.Rollout)
        for role, model in ordered_roles:
            count = len(self.instances_status_flags[role][model])
            target = max(minimum, count) if role == PSRL_Role.Rollout else minimum
            ids = self.select_initial_awake_ids(role, model, target, minimum if count else 0)
            if ids:
                await self._await_elastic_coordinator_command(
                    coordinators[role][model],
                    Command(type=CommandType.WAKE_UP, instance_ids=ids),
                    stage=f"INITIAL_WAKE role={role.name} model={model}",
                )
        await self.start_busy_loop()

    async def initialize_trainer_pool(
        self,
        actor_wg,
        ps_manager_handle,
    ) -> None:
        """
        Bind trainer resources and release their storage before inference initialization.
        """
        async with self._trainer_pool_transition_lock:
            self.actor_wg = actor_wg
            self.ps_manager_handle = ps_manager_handle
            await self._sleep_trainer(initializing=True)

    def configure_trainer_pool_clients(
        self,
        ps_wg,
        servers,
        worker_to_ps_idx,
    ) -> None:
        """
        Bind the clients that must receive trainer NIXL address updates.
        """
        self.ps_wg = ps_wg
        self._nixl_servers = [server for role_servers in servers.values() for server in role_servers]
        rollout_count = len(servers.get("rollout", []))
        self._nixl_destination_names = [ps_agent_name(rank) for rank in range(ps_wg.world_size)]
        self._nixl_destination_names.extend(
            key.to_nixl_client_name(rollout_count if key.role == "validate" else 0)
            for key in worker_to_ps_idx
            if key.role in ("rollout", "validate")
        )
        self._nixl_destination_names.extend(train_client_name(rank) for rank in range(self.actor_wg.world_size))

    async def initialize_validation_workers(
        self,
        instance_ids: list[RolloutInstanceId],
    ) -> None:
        """
        Keep validation replicas asleep until the executor reserves the trainer pool.
        """
        self._validation_instance_ids = list(instance_ids)
        await self._rollout_coordinator().sleep.remote("validate")

    def _rollout_coordinator(self):
        return next(iter(self.coordinators[PSRL_Role.Rollout].values()))

    async def _sleep_validation_workers(self) -> None:
        if self._validation_active:
            await self._await_elastic_coordinator_command(
                self._rollout_coordinator(),
                Command(type=CommandType.SLEEP, instance_ids=self._validation_instance_ids),
                stage="VALIDATION_SLEEP",
            )
            self._validation_active = False

    async def begin_validation(self) -> None:
        """
        Reserve trainer GPUs and restore validation replicas through their coordinator.
        """
        async with self._trainer_pool_transition_lock:
            await self._reclaim_trainer_pool()
            await self._sleep_trainer()
            self._validation_active = True
            await self._await_elastic_coordinator_command(
                self._rollout_coordinator(),
                Command(type=CommandType.WAKE_UP, instance_ids=self._validation_instance_ids),
                stage="VALIDATION_WAKE",
            )

    async def _sleep_trainer(
        self,
        *,
        initializing: bool = False,
    ) -> None:
        if self._trainer_state != TrainerState.ASLEEP:
            await self._await_trainer_operations(
                self.actor_wg.execute_all_async("nixl_sleep", "meta" if initializing else "full")
            )
            self._trainer_state = TrainerState.ASLEEP

    @staticmethod
    async def _await_trainer_operations(
        futures,
    ) -> None:
        """
        Settle every worker before allowing the next storage transition.
        """
        results = await asyncio.gather(*futures, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result

    async def _broadcast_trainer_client_infos(self) -> None:
        updated = [train_client_name(rank) for rank in range(self.actor_wg.world_size)]
        await self._await_trainer_operations(
            [
                self.ps_manager_handle.nixl_broadcast_update_client_infos.remote(
                    self._nixl_destination_names, updated
                ),
                *self.ps_wg.execute_all_async("nixl_wait_for_update_infos", 1),
                *(server.nixl_wait_for_update_infos.remote(1) for server in self._nixl_servers),
                *self.actor_wg.execute_all_async("nixl_wait_for_update_infos", 1),
            ]
        )

    def defer_actor_checkpoint(
        self,
        path: str,
    ) -> None:
        """
        Apply the actor checkpoint after trainer storage and NIXL connections are restored.
        """
        self._deferred_actor_ckpt_path = path

    async def load_and_push_actor_checkpoint(
        self,
        path: str,
        model_version: int,
    ) -> None:
        """
        Populate PS resume weights before inference borrowers are awakened.
        """
        async with self._trainer_pool_transition_lock:
            await self._reclaim_trainer_pool()
            self._trainer_state = TrainerState.RESTORING
            try:
                await self._await_trainer_operations(self.actor_wg.execute_all_async("nixl_wake_up"))
                await self._await_trainer_operations(
                    self.actor_wg.execute_all_async(
                        "load_checkpoint",
                        path,
                        del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                    )
                )
                await self.ps_manager_handle.init_model_version_for_resume.remote(model_version - 1)
                await self._await_trainer_operations(self.actor_wg.execute_all_async("push_model"))
            finally:
                await self._sleep_trainer()

    async def acquire_trainer_pool(self) -> None:
        """
        Reclaim inference borrowers and restore trainer storage, connections and weights.
        """
        async with self._trainer_pool_transition_lock:
            await self._reclaim_trainer_pool()
            await self._sleep_validation_workers()
            if self._trainer_state == TrainerState.AWAKE:
                return
            self._trainer_state = TrainerState.RESTORING
            await self._await_trainer_operations(self.actor_wg.execute_all_async("nixl_wake_up"))
            await self._await_trainer_operations(
                [
                    *self.actor_wg.execute_all_async("nixl_send_local_info_to", NIXL_META_SERVER_NAME),
                    self.ps_manager_handle.nixl_wait_for_update_infos.remote(self.actor_wg.world_size),
                ]
            )
            await self._broadcast_trainer_client_infos()
            await self._await_trainer_operations(self.actor_wg.execute_all_async("pull_model"))
            if self._deferred_actor_ckpt_path is not None:
                await self._await_trainer_operations(
                    self.actor_wg.execute_all_async(
                        "load_checkpoint",
                        self._deferred_actor_ckpt_path,
                        del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
                    )
                )
                self._deferred_actor_ckpt_path = None
            self._trainer_state = TrainerState.AWAKE

    async def _reclaim_trainer_pool(self) -> None:
        self._trainer_pool_available = False
        self._trainer_pool_epoch += 1
        self._trainer_pool_transition = True
        try:
            await self._decision_settled.wait()
            for role, model in self.roles:
                ids = [
                    iid
                    for iid in self.instances_status_flags[role][model]
                    if self.instance_pool_ids.get((role, model, iid)) == "train_pool"
                ]
                if not ids:
                    continue
                await self._await_elastic_coordinator_command(
                    self.coordinators[role][model],
                    Command(type=CommandType.SLEEP, instance_ids=ids),
                    stage=f"TRAIN_POOL_RECLAIM role={role.name} model={model}",
                )
                self._instance_state_epoch += 1
                for iid in ids:
                    self.instances_status_flags[role][model][iid] = InstanceStatus.ASLEEP
                    self.topology.set_status(role, model, iid, InstanceStatus.ASLEEP)
                    self._reset_sleeping_snapshot(role, model, iid)
            self._trainer_waiting_buffer_id = None
            psrl_logger.debug("Trainer pool reclaimed from all inference borrowers.")
        finally:
            self._trainer_pool_transition = False

    async def release_trainer_pool(self) -> None:
        """
        Release validation and trainer storage before making GPUs available to inference.
        """
        async with self._trainer_pool_transition_lock:
            await self._release_trainer_pool()

    async def _release_trainer_pool(self) -> None:
        await self._sleep_validation_workers()
        await self._sleep_trainer()
        self._trainer_pool_available = True
        self._trainer_pool_epoch += 1
        psrl_logger.debug("Trainer pool released for elastic inference.")

    async def release_trainer_pool_for_wait(
        self,
        buffer_id: int,
    ) -> None:
        """
        Release trainer GPU storage unless the next training batch is already ready.

        An `AWAKE` trainer polls `wait_for_training_readiness` for
        `trainer_ready_grace_s` and keeps storage if the buffer is already full.
        Otherwise the pool is released and `buffer_id` is published as the wait
        the elastic policy may borrow.

        Args:
            buffer_id (int): Training buffer this step is about to wait on.
        """
        async with self._trainer_pool_transition_lock:
            if self._trainer_state == TrainerState.AWAKE:
                grace_s = float(self.elastic_config.get("trainer_ready_grace_s", 0.0))
                if await self.agent_loop_manager.wait_for_training_readiness.remote(buffer_id, grace_s):
                    return
            await self._release_trainer_pool()
            self._trainer_waiting_buffer_id = buffer_id

    def select_initial_awake_ids(
        self,
        role_name: PSRL_Role,
        model_name: str,
        target_awake_num: int,
        min_awake_num: int = 0,
    ) -> list[RolloutInstanceId]:
        """
        Select non-conflicting instances to wake during initial bootstrap.

        The topology derives the occupied GPU set from all currently-AWAKEN
        instances, so successive calls for different roles automatically avoid
        GPUs already claimed by earlier calls because this method marks each
        selection AWAKEN before returning.

        Returns:
            list of RolloutInstanceId that should be woken up.
        """
        status_dict = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        all_ids = list(status_dict.keys())
        if not self._trainer_pool_available:
            all_ids = [iid for iid in all_ids if self.instance_pool_ids[(role_name, model_name, iid)] != "train_pool"]
        selected_ids = self.topology.select_non_conflicting_awake_ids(
            role=role_name,
            model_name=model_name,
            instance_ids=all_ids,
            target_awake_num=target_awake_num,
            min_awake_num=min_awake_num,
        )
        for instance_id in selected_ids:
            self.instances_status_flags[role_name][model_name][instance_id] = InstanceStatus.AWAKEN
            self.topology.set_status(role_name, model_name, instance_id, InstanceStatus.AWAKEN)
        return selected_ids

    def snapshot(self) -> dict:
        return {
            "status": self.instances_status_flags,
            "topology": self.topology.snapshot(),
        }

    def set_current_training_step(self, step: int) -> None:
        """
        Tag subsequently accepted policy actions with the trainer step.
        """
        self._current_training_step = int(step)

    async def start_busy_loop(self):
        if self.monitor_task is not None and not self.monitor_task.done():
            return

        self.running_loop = asyncio.get_running_loop()
        self.stop_monitor = False
        self.stop_scale_up = False
        self.stop_scale_down = False

        self.monitor_task = self.running_loop.create_task(self._monitor_loop())
        self.monitor_task.add_done_callback(lambda f: f.result())

        self.scale_up_task = self.running_loop.create_task(self._scale_up_handler_loop())
        self.scale_up_task.add_done_callback(lambda f: f.result())

        self.scale_down_task = self.running_loop.create_task(self._scale_down_handler_loop())
        self.scale_down_task.add_done_callback(lambda f: f.result())
        psrl_logger.debug("Elastic executor loops started.")
        monitor_logger.info("Elastic monitor loop started.")

    async def stop_busy_loop(self):
        if self.monitor_task is None and self.scale_up_task is None and self.scale_down_task is None:
            self.scaling_policy.close()
            return

        self.stop_monitor = True
        self.stop_scale_up = True
        self.stop_scale_down = True

        tasks_to_wait = []
        if self.monitor_task is not None:
            tasks_to_wait.append(self.monitor_task)
        if self.scale_up_task is not None:
            tasks_to_wait.append(self.scale_up_task)
        if self.scale_down_task is not None:
            tasks_to_wait.append(self.scale_down_task)
        if tasks_to_wait:
            await asyncio.gather(*tasks_to_wait, return_exceptions=True)
        self.scaling_policy.close()
        psrl_logger.debug("Elastic executor loops stopped.")
        monitor_logger.info("Elastic monitor loop stopped.")

    async def _monitor_loop(self):
        while not self.stop_monitor:
            try:
                if self._trainer_pool_transition:
                    await asyncio.sleep(max(self.scaling_policy.monitor_interval_ms / 1000, 0.01))
                    continue
                pool_epoch = self._trainer_pool_epoch
                instance_epoch = self._instance_state_epoch
                # Pull fresh engine status from coordinators first, then decide scaling.
                engine_ready = await self._sync_engine_status_from_coordinators()
                router_ready = await self._sync_router_backlog_from_coordinators()
                signals = self._build_instance_signals()
                self._maybe_log_instance_signals(signals, engine_ready=engine_ready, router_ready=router_ready)
                if not engine_ready or not router_ready:
                    await asyncio.sleep(max(self.scaling_policy.monitor_interval_ms / 1000, 0.01))
                    continue
                await self._sync_trainer_waiting_hint()
                if (
                    self._trainer_pool_transition
                    or self._trainer_pool_epoch != pool_epoch
                    or self._instance_state_epoch != instance_epoch
                ):
                    continue
                policy_signals = [
                    signal
                    for signal in signals
                    if self._trainer_pool_available
                    or self.instance_pool_ids.get((signal.role_name, signal.model_name, signal.instance_id))
                    != "train_pool"
                ]
                integer_solver = uses_integer_solver(self.scaling_policy)
                if integer_solver:
                    policy_signals = self.policy_identity_map.signals(policy_signals)
                request_snapshots = (
                    await self._fetch_request_level_snapshots(policy_signals)
                    if not self._decision_execution_in_progress
                    else None
                )
                if (
                    self._trainer_pool_transition
                    or self._trainer_pool_epoch != pool_epoch
                    or self._instance_state_epoch != instance_epoch
                ):
                    continue
                policy_kwargs = {}
                if request_snapshots is not None:
                    policy_kwargs["request_level_snapshots_by_role"] = request_snapshots
                decision = self.scaling_policy.decide(
                    policy_signals,
                    execution_in_progress=self._decision_execution_in_progress,
                    router_backlog_by_role=self.router_backlog_by_role,
                    trainer_waiting_hint=self.trainer_waiting_hint,
                    **policy_kwargs,
                )
                if integer_solver:
                    decision = self.policy_identity_map.decision(decision)
                if not decision.actions and decision.reason == "decision_execution_in_progress":
                    self._execution_in_progress_stall_ticks += 1
                    if self._decision_abandon_stall_ticks > 0 and (
                        self._execution_in_progress_stall_ticks >= self._decision_abandon_stall_ticks
                    ):
                        psrl_logger.error(
                            f"Elastic decision remains in flight after {self._execution_in_progress_stall_ticks} "
                            f"monitor ticks: pending={self._decision_pending_action_counts!r}."
                        )
                    elif self._execution_in_progress_stall_ticks in (1, 30, 60) or (
                        self._execution_in_progress_stall_ticks > 60
                        and self._execution_in_progress_stall_ticks % 120 == 0
                    ):
                        psrl_logger.debug(
                            "elastic_rm: policy blocked by in-flight scaling for %d monitor ticks. "
                            "pending=%s. Likely causes: (1) coordinator.exec_command stuck inside "
                            "SLEEP/WAKE_UP/ABORT (gen/RM workers or router not returning). "
                            "(2) another client's command ahead in the same coordinator queue never finishes "
                            "(head-of-line blocking). Check ElasticExecutor.log for the last "
                            "coordinator_cmd START line without matching END.",
                            self._execution_in_progress_stall_ticks,
                            dict(self._decision_pending_action_counts),
                        )
                else:
                    self._execution_in_progress_stall_ticks = 0

                if decision.actions:
                    decision_id = self._next_decision_id
                    self._next_decision_id += 1
                    self._decision_execution_in_progress = True
                    self._decision_settled.clear()
                    self._decision_pending_action_counts[decision_id] = len(decision.actions)
                    actions = [
                        {
                            "action_type": action.action_type,
                            "role_name": action.role_name.name,
                            "model_name": action.model_name,
                            "num_instances": action.num_instances,
                            "preferred_instance_ids": action.preferred_instance_ids,
                        }
                        for action in decision.actions
                    ]
                    psrl_logger.info(
                        f"Policy decision accepted: decision_id={decision_id}, reason={decision.reason!r}, "
                        f"actions={actions!r}."
                    )
                    psrl_logger.debug("elastic_rm trainer_waiting_hint=%s", self.trainer_waiting_hint)
                    for action in decision.actions:
                        task = {
                            "role_name": action.role_name,
                            "model_name": action.model_name,
                            "num_instances": action.num_instances,
                            "preferred_instance_ids": action.preferred_instance_ids or [],
                            "reason": action.reason,
                            "decision_id": decision_id,
                            "pre_sleep_other_preferred": action.pre_sleep_other_preferred,
                            "pre_wake_other_preferred": action.pre_wake_other_preferred,
                            "planned_request_migrations": action.planned_request_migrations,
                        }
                        if action.action_type == "scale_up":
                            self.scale_up_task_queue.put_nowait(task)
                        elif action.action_type == "scale_down":
                            self.scale_down_task_queue.put_nowait(task)
                        else:
                            psrl_logger.warning(
                                "Unknown action_type=%s for decision_id=%d, mark as finished immediately.",
                                action.action_type,
                                decision_id,
                            )
                            self._mark_decision_action_finished(decision_id)
            except Exception:
                # Keep transient sync and policy failures from terminating the monitor task.
                psrl_logger.exception("elastic_rm _monitor_loop iteration failed. Retrying after sleep.")
            await asyncio.sleep(max(self.scaling_policy.monitor_interval_ms / 1000, 0.01))

    async def _fetch_request_level_snapshots(self, signals):
        """
        Fetch request queues only for policies that evaluate individual requests.
        """
        if not getattr(self.scaling_policy, "enable_request_level_candidate_evaluation", False):
            return None
        from psrl.workers.config.reward_model import resolve_active_managers
        from psrl.workers.elastic.request_snapshot import build_role_snapshot

        if any(sum(role == candidate for role, _ in self.roles) > 1 for candidate, _ in self.roles):
            raise ValueError("Request level ITL evaluation requires one model per role.")
        keys = list(self.roles)
        refs = []
        for role, model in keys:
            versions = {self.role_model_versions[(role, model)]}
            versions.update(
                self.instances_engine_stats[role][model][self.policy_identity_map.decode(signal.instance_id)[2]][
                    "elastic_model_version"
                ]
                for signal in signals
                if signal.role_name == role and signal.is_awaken
            )
            refs.append(self.coordinators[role][model].get_router_request_snapshot.remote(sorted(versions)))
        pending_results = await self._await_coordinator_refs_with_per_ref_timeout(refs, keys, "request_snapshot")
        snapshots = {}
        for (role, model), pending in zip(keys, pending_results, strict=True):
            if isinstance(pending, Exception):
                raise RuntimeError(f"Failed to fetch request queue for {role.name}/{model}.") from pending
            if role == PSRL_Role.Rollout:
                routing_cfg = self.config.psrl.rollout_coordination.routing_strategy
                rollout_cfg = self.config.gen_actor_rollout_ref.rollout
                max_len = self.config.gen_actor_rollout_ref.rollout.get("max_model_len") or (
                    self.config.data.max_prompt_length + self.config.data.max_response_length
                )
            else:
                rm_cfg = next(
                    cfg for cfg in resolve_active_managers(self.config.reward) if cfg.get("reward_model_name") == model
                )
                routing_cfg = rm_cfg.routing_strategy
                rollout_cfg = rm_cfg.rollout
                max_len = rm_cfg.rollout.max_model_len
            snapshots[role] = build_role_snapshot(
                role,
                model,
                [signal for signal in signals if signal.role_name == role],
                self.instances_engine_stats[role][model],
                pending,
                self.policy_identity_map,
                routing_cfg,
                max_len,
                rollout_cfg.tensor_model_parallel_size,
                rollout_cfg.pipeline_model_parallel_size,
                self.role_model_versions[(role, model)],
            )
            self.router_backlog_by_role[role] = {
                "count": len(pending),
                "total_tokens": sum(request["seq_len"] for request in pending),
            }
        return snapshots

    async def _scale_up_handler_loop(self):
        while not self.stop_scale_up:
            if self.scale_up_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_up = self.scale_up_task_queue.get_nowait()
            decision_id = role_need_to_scale_up.get("decision_id")
            try:
                psrl_logger.debug(
                    "elastic_rm scale_up_handler decision_id=%s begin task=%s",
                    decision_id,
                    role_need_to_scale_up,
                )
                instances_to_scaled_down = self._find_instances_to_scaled_down_for_other_roles(role_need_to_scale_up)
                if instances_to_scaled_down:
                    psrl_logger.debug(
                        "elastic_rm scale_up_handler decision_id=%s pre_sleep_other count=%s detail=%s",
                        decision_id,
                        len(instances_to_scaled_down),
                        instances_to_scaled_down,
                    )
                    operation = (
                        self._sleep_planned_instances
                        if role_need_to_scale_up.get("pre_sleep_other_preferred")
                        else self._scale_down_instances
                    )
                    await self._apply_instance_actions(operation, instances_to_scaled_down)
                    psrl_logger.debug(
                        "elastic_rm scale_up_handler decision_id=%s pre_sleep_other done",
                        decision_id,
                    )

                instances_to_scaled_up = self._find_instances_to_scaled_up(role_need_to_scale_up)
                if not instances_to_scaled_up:
                    psrl_logger.warning("No instances can be scaled up for role %s", role_need_to_scale_up)
                    continue
                psrl_logger.debug(
                    "elastic_rm scale_up_handler decision_id=%s wake_targets=%s",
                    decision_id,
                    instances_to_scaled_up,
                )
                await self._apply_instance_actions(self._scale_up_instances, instances_to_scaled_up)
                other_wakes = role_need_to_scale_up.get("pre_wake_other_preferred") or []
                if other_wakes:
                    await self._apply_instance_actions(self._scale_up_instances, other_wakes)
                psrl_logger.debug(
                    "elastic_rm scale_up_handler decision_id=%s wake_targets done. post_scale_up_abort",
                    decision_id,
                )
                if getattr(self.scaling_policy, "rebalance_after_scale_up", True):
                    if getattr(self.scaling_policy, "enable_request_level_candidate_evaluation", False):
                        moves = role_need_to_scale_up.get("planned_request_migrations") or []
                        coordinator = self.coordinators[role_need_to_scale_up["role_name"]][
                            role_need_to_scale_up["model_name"]
                        ]
                        if moves:
                            await coordinator.execute_request_migrations.remote(moves)
                    else:
                        await self._interrupt_waiting_after_scale_up(
                            role_name=role_need_to_scale_up["role_name"],
                            model_name=role_need_to_scale_up["model_name"],
                        )
                psrl_logger.debug("elastic_rm scale_up_handler decision_id=%s post_scale_up_abort done", decision_id)
            except Exception:
                psrl_logger.exception(f"Elastic scale up action failed: decision_id={decision_id!r}.")
            finally:
                self._mark_decision_action_finished(decision_id)

    async def _scale_down_handler_loop(self):
        while not self.stop_scale_down:
            if self.scale_down_task_queue.empty():
                await asyncio.sleep(0)
                continue

            role_need_to_scale_down = self.scale_down_task_queue.get_nowait()
            decision_id = role_need_to_scale_down.get("decision_id")
            try:
                psrl_logger.debug(
                    "elastic_rm scale_down_handler decision_id=%s begin task=%s",
                    decision_id,
                    role_need_to_scale_down,
                )
                instances_to_scaled_down = self._find_instances_to_scaled_down_in_role(role_need_to_scale_down)
                if not instances_to_scaled_down:
                    psrl_logger.warning("No instances can be scaled down for role %s", role_need_to_scale_down)
                    continue
                psrl_logger.debug(
                    "elastic_rm scale_down_handler decision_id=%s sleep_targets=%s",
                    decision_id,
                    instances_to_scaled_down,
                )
                await self._apply_instance_actions(self._scale_down_instances, instances_to_scaled_down)
                psrl_logger.debug("elastic_rm scale_down_handler decision_id=%s sleep_targets done", decision_id)
            except Exception:
                psrl_logger.exception(f"Elastic scale down action failed: decision_id={decision_id!r}.")
            finally:
                self._mark_decision_action_finished(decision_id)

    async def _apply_instance_actions(self, operation, instances: list[dict]) -> None:
        """
        Settle every sibling transition before releasing the decision.
        """
        groups = defaultdict(list)
        for instance in instances:
            groups[(instance["role_name"], instance["model_name"])].append(instance)
        results = await asyncio.gather(*(operation(group) for group in groups.values()), return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"Elastic instance transitions failed: {failures!r}.") from failures[0]

    def _mark_decision_action_finished(self, decision_id: int | None):
        if decision_id is None:
            return
        if decision_id not in self._decision_pending_action_counts:
            return
        remaining = self._decision_pending_action_counts[decision_id] - 1
        if remaining > 0:
            self._decision_pending_action_counts[decision_id] = remaining
            return
        self._decision_pending_action_counts.pop(decision_id, None)
        if not self._decision_pending_action_counts:
            # Failed transitions can also invalidate observations collected before the decision settled.
            self._instance_state_epoch += 1
            self._decision_execution_in_progress = False
            self._decision_settled.set()
            psrl_logger.info(f"Elastic decision execution completed: decision_id={decision_id}.")

    def _maybe_log_instance_signals(
        self,
        signals: list[InstanceSignal],
        *,
        engine_ready: bool,
        router_ready: bool,
    ):
        if not self._enable_monitor_instance_log:
            return
        now_ms = asyncio.get_running_loop().time() * 1000
        if now_ms - self._last_monitor_instance_log_ms < max(self._monitor_instance_log_interval_ms, 0):
            return
        self._last_monitor_instance_log_ms = now_ms

        rows: list[dict] = []
        for s in signals:
            role_name = getattr(s.role_name, "name", str(s.role_name))
            rows.append(
                {
                    "role": role_name,
                    "model": s.model_name,
                    "instance": str(s.instance_id),
                    "status": "AWAKEN" if bool(s.is_awaken) else "ASLEEP",
                    "running": int(s.running_queue_num),
                    "waiting": int(s.waiting_queue_num),
                    "kv_cache": float(s.kv_cache_utilization),
                    "throughput": float(s.generation_throughput),
                    "ts": s.snapshot_timestamp,
                }
            )
        rows.sort(key=lambda r: (r["status"] != "AWAKEN", r["role"], r["model"], r["instance"]))
        awake_by_role: dict[str, int] = {}
        awake_by_role_model: dict[str, int] = {}
        for row in rows:
            if row["status"] != "AWAKEN":
                continue
            role_key = str(row["role"])
            role_model_key = f"{role_key}/{row['model']}"
            awake_by_role[role_key] = awake_by_role.get(role_key, 0) + 1
            awake_by_role_model[role_model_key] = awake_by_role_model.get(role_model_key, 0) + 1
        monitor_logger.info(
            f"Monitor input snapshot: engine_ready={engine_ready!r}, router_ready={router_ready!r}, "
            f"router_backlog_by_role={self.router_backlog_by_role!r}, "
            f"execution_in_progress={self._decision_execution_in_progress!r}, instance_count={len(signals)}."
        )
        for row in rows:
            monitor_logger.info("Instance current Status: %s", row)
        monitor_logger.info("------------------------------------------------------------")
        psrl_logger.debug(
            "Awake instances summary: by_role=%s, by_role_model=%s",
            dict(sorted(awake_by_role.items())),
            dict(sorted(awake_by_role_model.items())),
        )
        monitor_logger.info(
            "Awake instances summary: by_role=%s, by_role_model=%s",
            dict(sorted(awake_by_role.items())),
            dict(sorted(awake_by_role_model.items())),
        )

    async def _await_elastic_coordinator_command(
        self,
        coordinator: ray.actor.ActorHandle,
        command: Command,
        *,
        stage: str,
    ) -> object:
        """
        Run coordinator.exec_command with optional timeout and structured tracing for elastic_rm debugging.
        """
        t0 = time.monotonic()
        psrl_logger.debug(
            "elastic_rm coordinator_cmd START stage=%s type=%s args=%s timeout_s=%s",
            stage,
            command.type.name,
            command.get_args(),
            self._coordinator_command_timeout_s,
        )
        kwargs: dict = {"blocking": True}
        if self._coordinator_command_timeout_s is not None:
            kwargs["timeout"] = self._coordinator_command_timeout_s
        try:
            result = await coordinator.exec_command.remote(command, **kwargs)
            if isinstance(result, dict) and "command_id" in result:
                command_id = result["command_id"]
                psrl_logger.warning(
                    f"Elastic command exceeded its acknowledgement timeout: stage={stage!r}, "
                    f"command_id={command_id}. Waiting for completion."
                )
                while True:
                    status = await coordinator.get_command_status.remote(command_id)
                    if status["status"] in {"SUCCEEDED", "FAILED"}:
                        result = await coordinator.synchronize_command.remote(command_id)
                        break
                    if status["status"] not in {"QUEUED", "RUNNING"}:
                        raise RuntimeError(f"Unknown elastic command lifecycle: {status!r}.")
                    psrl_logger.debug(
                        f"Elastic command is still in flight: stage={stage!r}, command_id={command_id}, "
                        f"status={status['status']!r}. Holding the decision until it settles."
                    )
                    await asyncio.sleep(1.0)
            if command.type in {CommandType.SLEEP, CommandType.WAKE_UP} and result is not True:
                raise RuntimeError(f"Elastic command did not confirm completion: stage={stage!r}, result={result!r}.")
        except Exception:
            psrl_logger.exception(
                "elastic_rm coordinator_cmd EXCEPTION stage=%s type=%s elapsed_s=%.3f",
                stage,
                command.type.name,
                time.monotonic() - t0,
            )
            raise
        elapsed = time.monotonic() - t0
        psrl_logger.debug(
            "elastic_rm coordinator_cmd END stage=%s type=%s elapsed_s=%.3f result=%r",
            stage,
            command.type.name,
            elapsed,
            result,
        )
        return result

    async def _scale_up_instances(self, instances: list[dict]) -> None:
        """
        Wake one role and model with a single coordinator command.
        """
        if not self._trainer_pool_available and any(
            self.instance_pool_ids.get((item["role_name"], item["model_name"], item["instance_id"])) == "train_pool"
            for item in instances
        ):
            raise RuntimeError("Trainer pool is reserved for training or validation.")
        await self._transition_role_instances(CommandType.WAKE_UP, instances)

    def _waiting_uids_for_abort_by_ratio(self, normalized_waiting_uids: list[str]) -> list[str]:
        """
        Take the first k waiting uids in queue order, where k = floor(n * ratio).
        """
        if not normalized_waiting_uids:
            return []
        r = self._post_scale_up_abort_waiting_ratio
        if r <= 0.0:
            return []
        if r >= 1.0:
            return list(normalized_waiting_uids)
        k = min(
            len(normalized_waiting_uids),
            int(math.floor(float(len(normalized_waiting_uids)) * r + 1e-12)),
        )
        return normalized_waiting_uids[:k]

    async def _interrupt_waiting_after_scale_up(self, role_name: PSRL_Role, model_name: str):
        if role_name == PSRL_Role.RewardModel:
            psrl_logger.warning(
                "Reward request migration requires router loopback support. Keeping active reward requests running."
            )
            return
        if not self.config.psrl.rollout_coordination.partial_rollout.enable:
            return
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        awaken_instance_ids = [
            instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN
        ]
        if not awaken_instance_ids:
            return
        instance_to_uids: dict = {}
        role_engine_stats = self.instances_engine_stats.get(role_name, {}).get(model_name, {})
        for instance_id in awaken_instance_ids:
            snapshot = role_engine_stats.get(instance_id, {})
            if not isinstance(snapshot, dict):
                continue
            scheduler_stats = snapshot.get("scheduler_stats", {})
            if not isinstance(scheduler_stats, dict):
                continue
            waiting_uids = scheduler_stats.get("req_id_in_waiting", [])
            normalized_waiting_uids = [str(uid) for uid in waiting_uids]
            selected = self._waiting_uids_for_abort_by_ratio(normalized_waiting_uids)
            if selected:
                instance_to_uids[instance_id] = selected

        coordinator = self.coordinators.get(role_name, {}).get(model_name)
        if coordinator is None:
            psrl_logger.warning(
                "Skip post-scale-up ABORT: coordinator missing for role=%s model=%s.",
                role_name,
                model_name,
            )
            return

        if not instance_to_uids:
            psrl_logger.debug(
                "Skip post-scale-up ABORT: no waiting uids selected (ratio=%.4f) role=%s model=%s.",
                self._post_scale_up_abort_waiting_ratio,
                role_name,
                model_name,
            )
            return

        try:
            interrupted_request_num = await self._await_elastic_coordinator_command(
                coordinator,
                Command(
                    type=CommandType.ABORT,
                    instance_to_uids=instance_to_uids,
                ),
                stage=(
                    f"ABORT_post_scale_up role={getattr(role_name, 'name', role_name)} model={model_name} "
                    f"instances={sorted(instance_to_uids.keys())}"
                ),
            )
            if interrupted_request_num is None:
                psrl_logger.warning(
                    "elastic_rm post-scale-up ABORT returned None (timeout?). role=%s model=%s",
                    role_name,
                    model_name,
                )
                return
            psrl_logger.debug(
                (
                    "Post-scale-up rebalancing abort executed: role=%s model=%s "
                    "awake_instances=%s uid_sources=%s interrupted=%s abort_waiting_ratio=%.4f"
                ),
                role_name,
                model_name,
                awaken_instance_ids,
                sorted(instance_to_uids.keys()),
                interrupted_request_num,
                self._post_scale_up_abort_waiting_ratio,
            )
        except Exception as exc:
            psrl_logger.warning(
                "elastic_rm post-scale-up ABORT failed for role=%s model=%s instances=%s: %s",
                role_name,
                model_name,
                awaken_instance_ids,
                exc,
            )

    async def _scale_down_instances(self, instances: list[dict]) -> None:
        """
        Validate the whole sleep batch before releasing any role devices.
        """
        role, model = instances[0]["role_name"], instances[0]["model_name"]
        minimum = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))
        statuses = self.instances_status_flags[role][model]
        awake = [iid for iid, status in statuses.items() if status is InstanceStatus.AWAKEN]
        selected = {item["instance_id"] for item in instances}
        if len(awake) - len(selected) < minimum:
            raise RuntimeError(f"Sleep batch violates minimum awake count: role={role!r}, model={model!r}.")
        if minimum == 0 and selected == set(awake):
            counts = [
                self.get_instance_running_waiting(InstanceIdentifier(role=role, model_name=model, instance_id=iid))
                for iid in selected
            ]
            if any(running > 0 or waiting > 0 for running, waiting in counts):
                psrl_logger.debug(f"Keeping the last active role instances: role={role!r}, model={model!r}.")
                return
        await self._sleep_planned_instances(instances)

    async def _sleep_planned_instances(self, instances: list[dict]) -> None:
        """
        Sleep the complete solver selected victim set in one coordinator command.
        """
        await self._transition_role_instances(CommandType.SLEEP, instances)

    async def _transition_role_instances(self, command_type: CommandType, instances: list[dict]) -> None:
        """
        Let the coordinator execute each replica concurrently within a batch.
        """
        role, model = instances[0]["role_name"], instances[0]["model_name"]
        instance_ids = list(dict.fromkeys(item["instance_id"] for item in instances))
        await self._await_elastic_coordinator_command(
            self.coordinators[role][model],
            Command(type=command_type, instance_ids=instance_ids),
            stage=f"{command_type.name} role={role.name} model={model} instances={instance_ids!r}",
        )
        self._instance_state_epoch += 1
        status = InstanceStatus.AWAKEN if command_type is CommandType.WAKE_UP else InstanceStatus.ASLEEP
        for iid in instance_ids:
            self.instances_status_flags[role][model][iid] = status
            self.topology.set_status(role, model, iid, status)
            if status is InstanceStatus.ASLEEP:
                self._reset_sleeping_snapshot(role, model, iid)

    def _reset_sleeping_snapshot(self, role, model, iid) -> None:
        """
        Clear released engine load while retaining its committed weight version.
        """
        self.instances_engine_stats[role][model][iid] = {
            **EngineStats.get_default_snapshot(),
            "elastic_model_version": self.role_model_versions[(role, model)],
        }

    def _scale_up_target_ids(self, task: dict) -> list[RolloutInstanceId]:
        """
        Resolve exact policy targets without replacing them with different replicas.
        """
        role, model = task["role_name"], task["model_name"]
        statuses = self.instances_status_flags.get(role, {}).get(model, {})
        preferred = task.get("preferred_instance_ids") or []
        target_ids = list(preferred) if preferred else sorted(statuses)
        return [iid for iid in target_ids if statuses.get(iid) is InstanceStatus.ASLEEP][
            : max(0, int(task.get("num_instances", 1)))
        ]

    def _find_instances_to_scaled_up(self, role_need_to_scale_up: dict):
        role = role_need_to_scale_up["role_name"]
        model = role_need_to_scale_up["model_name"]
        target_ids = self._scale_up_target_ids(role_need_to_scale_up)
        blocked = [iid for iid in target_ids if self.topology.has_other_role_awaken_on_shared_gpu(role, model, iid)]
        if blocked:
            raise RuntimeError(f"Scale up targets still have occupied GPUs: role={role!r}, instances={blocked!r}.")
        return [{"role_name": role, "model_name": model, "instance_id": iid} for iid in target_ids]

    def _find_instances_to_scaled_down_for_other_roles(self, role_need_to_scale_up: dict):
        """
        Release only the GPUs required by the selected wake targets.
        """
        if role_need_to_scale_up.get("pre_sleep_other_preferred"):
            return list(role_need_to_scale_up["pre_sleep_other_preferred"])
        role = role_need_to_scale_up["role_name"]
        model = role_need_to_scale_up["model_name"]
        target_ids = self._scale_up_target_ids(role_need_to_scale_up)
        target_slots = set()
        for iid in target_ids:
            slots = self.topology.get_gpu_slots(role, model, iid)
            if not slots:
                raise RuntimeError(f"Missing GPU placement for scale up target: role={role!r}, instance={iid!r}.")
            target_slots.update(slots)
        candidates = []
        minimum = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))
        for other_role, other_model in self.roles:
            if (other_role, other_model) == (role, model):
                continue
            statuses = self.instances_status_flags[other_role][other_model]
            awake = [iid for iid, status in statuses.items() if status is InstanceStatus.AWAKEN]
            conflicting = [
                iid
                for iid in awake
                if not self.topology.get_gpu_slots(other_role, other_model, iid).isdisjoint(target_slots)
            ]
            if len(awake) - len(conflicting) < minimum:
                raise RuntimeError(
                    f"Scale up conflicts with the minimum awake count: role={other_role!r}, model={other_model!r}."
                )
            candidates.extend(
                {"role_name": other_role, "model_name": other_model, "instance_id": iid} for iid in conflicting
            )
        if candidates and not getattr(self.scaling_policy, "allow_preemptive_scale_up", True):
            raise RuntimeError("Scaling policy does not allow preemptive scale up.")
        return candidates

    def _find_instances_to_scaled_down_in_role(self, role_need_to_scale_down: dict):
        role_name = role_need_to_scale_down["role_name"]
        model_name = role_need_to_scale_down["model_name"]
        num_instances = int(role_need_to_scale_down.get("num_instances", 1))
        min_awake_per_role = max(0, int(getattr(self.scaling_policy, "min_awake_per_role", 0)))
        preferred_instance_ids = list(role_need_to_scale_down.get("preferred_instance_ids", []))
        role_status = self.instances_status_flags.get(role_name, {}).get(model_name, {})
        # Use the total awake count so preferred candidates cannot reduce the removable budget.
        all_awake_ids = [instance_id for instance_id, status in role_status.items() if status == InstanceStatus.AWAKEN]
        if not all_awake_ids:
            return None
        max_scalable_down = len(all_awake_ids) - min_awake_per_role
        if max_scalable_down <= 0:
            psrl_logger.debug(
                "Skip scale down for role=%s model=%s: keep at least %d awaken instances (total awake=%d).",
                role_name,
                model_name,
                min_awake_per_role,
                len(all_awake_ids),
            )
            return None
        if preferred_instance_ids:
            if any(iid not in all_awake_ids for iid in preferred_instance_ids):
                raise RuntimeError(f"Selected sleep targets are unavailable: instances={preferred_instance_ids!r}.")
            candidate_ids = preferred_instance_ids
        else:
            candidate_ids = sorted(
                all_awake_ids,
                key=lambda instance_id: self.get_instance_kv_cache_usage(
                    InstanceIdentifier(
                        role=role_name,
                        model_name=model_name,
                        instance_id=instance_id,
                    ),
                ),
            )
        effective_num_instances = min(num_instances, max_scalable_down)
        return [
            {"role_name": role_name, "model_name": model_name, "instance_id": instance_id}
            for instance_id in candidate_ids[:effective_num_instances]
        ]

    async def _await_coordinator_refs_with_per_ref_timeout(
        self,
        refs: list,
        task_keys: list,
        op_label: str,
        *,
        trace_router_backlog: bool = False,
    ) -> list:
        """
        Await each coordinator Ray ObjectRef with its own timeout (parallel).

        Wrapping a single ``asyncio.gather`` in one ``wait_for`` lets the slowest
        RPC drop every other result for that tick. Elastic RM then misreads load.
        """
        timeout_s = self._coordinator_sync_timeout_s

        async def _one(ref, key):
            t0 = time.monotonic()
            if trace_router_backlog:
                log_elastic_rm_backlog_diag(
                    psrl_logger,
                    "stage=ElasticExecutor_wait_ref_begin op=%s key=%s timeout_s=%s",
                    op_label,
                    key,
                    timeout_s if timeout_s > 0 else "none",
                )
            try:
                if timeout_s > 0:
                    out = await asyncio.wait_for(ref, timeout=timeout_s)
                else:
                    out = await ref
                if trace_router_backlog:
                    log_elastic_rm_backlog_diag(
                        psrl_logger,
                        "stage=ElasticExecutor_wait_ref_done op=%s key=%s total_s=%.3f",
                        op_label,
                        key,
                        time.monotonic() - t0,
                    )
                return out
            except asyncio.TimeoutError:
                psrl_logger.warning(
                    "elastic_rm: %s RPC timed out after %.1fs for key=%s. Skipped for this tick. "
                    "(If PSRL_ELASTIC_RM_BACKLOG_DIAG=1, compare coordinator/router stages above.)",
                    op_label,
                    timeout_s,
                    key,
                )
                return TimeoutError(f"{op_label} key={key}")

        return await asyncio.gather(*[_one(r, k) for r, k in zip(refs, task_keys)], return_exceptions=True)

    async def _sync_engine_status_from_coordinators(self) -> bool:
        pool_epoch = self._trainer_pool_epoch
        instance_epoch = self._instance_state_epoch
        task_refs = []
        task_keys: list[tuple[PSRL_Role, str]] = []
        for role_name, model_name in self.roles:
            coordinator = self.coordinators.get(role_name, {}).get(model_name)
            if coordinator is None:
                raise RuntimeError(f"Elastic coordinator is missing for role={role_name!r}, model={model_name!r}.")
            task_refs.append(coordinator.get_instance_runtime_snapshot.remote())
            task_keys.append((role_name, model_name))

        if not task_refs:
            return False
        results = await self._await_coordinator_refs_with_per_ref_timeout(
            task_refs, task_keys, op_label="get_instance_runtime_snapshot"
        )
        staged_stats = {}
        staged_statuses = {}
        staged_versions = {}
        for (role_name, model_name), result in zip(task_keys, results):
            if isinstance(result, Exception):
                psrl_logger.warning(
                    f"Failed to fetch engine status: role={role_name!r}, model={model_name!r}: {result}."
                )
                return False
            expected = self.instances_status_flags[role_name][model_name]
            if not isinstance(result, dict) or not set(expected).issubset(result):
                psrl_logger.error(f"Incomplete elastic runtime snapshot: role={role_name!r}, model={model_name!r}.")
                return False
            role_stats = {}
            role_statuses = {}
            versions = set()
            for instance_id in expected:
                runtime = result[instance_id]
                if not isinstance(runtime, dict) or not isinstance(runtime.get("is_sleeping"), bool):
                    psrl_logger.error(f"Invalid elastic runtime state: instance={instance_id!r}, runtime={runtime!r}.")
                    return False
                snapshot = runtime.get("engine_stats")
                if (
                    not isinstance(snapshot, dict)
                    or not isinstance(runtime.get("model_version"), int)
                    or not isinstance(runtime.get("wake_model_version"), int)
                    or (not runtime["is_sleeping"] and not isinstance(snapshot.get("scheduler_stats"), dict))
                ):
                    psrl_logger.error(f"Invalid elastic version snapshot: instance={instance_id!r}.")
                    return False
                versions.add(runtime["wake_model_version"])
                role_stats[instance_id] = {
                    **(EngineStats.get_default_snapshot() if runtime["is_sleeping"] else snapshot),
                    "elastic_model_version": runtime["model_version"],
                }
                role_statuses[instance_id] = InstanceStatus.ASLEEP if runtime["is_sleeping"] else InstanceStatus.AWAKEN
                if not runtime["is_sleeping"] and ScalingPolicy._is_snapshot_staled(snapshot):
                    return False
            if len(versions) != 1:
                return False
            staged_stats[(role_name, model_name)] = role_stats
            staged_statuses[(role_name, model_name)] = role_statuses
            staged_versions[(role_name, model_name)] = versions.pop()
        # Publish all roles together so a failed observation cannot leave a partial decision cache.
        if (
            self._trainer_pool_transition
            or self._trainer_pool_epoch != pool_epoch
            or self._instance_state_epoch != instance_epoch
        ):
            return False
        for (role_name, model_name), stats in staged_stats.items():
            self.instances_engine_stats[role_name][model_name] = stats
            if not self._decision_execution_in_progress and not self._trainer_pool_transition:
                for iid, status in staged_statuses[(role_name, model_name)].items():
                    self.instances_status_flags[role_name][model_name][iid] = status
                    self.topology.set_status(role_name, model_name, iid, status)
        self.role_model_versions.update(staged_versions)
        return True

    async def _sync_router_backlog_from_coordinators(self) -> bool:
        refs = []
        task_keys: list[tuple[PSRL_Role, str]] = []
        for role_name, model_name in self.roles:
            coordinator = self.coordinators.get(role_name, {}).get(model_name)
            if coordinator is None:
                continue
            refs.append(coordinator.get_router_backlog_size.remote())
            task_keys.append((role_name, model_name))
        if not refs:
            self.router_backlog_by_role = {}
            return False
        results = await self._await_coordinator_refs_with_per_ref_timeout(
            refs,
            task_keys,
            op_label="get_router_backlog_size",
            trace_router_backlog=True,
        )
        prev_by_role = dict(self.router_backlog_by_role)
        grouped: dict[PSRL_Role, list] = defaultdict(list)
        for (role_name, model_name), result in zip(task_keys, results):
            grouped[role_name].append(result)

        role_backlog: dict[PSRL_Role, int] = {}
        complete = True
        for role_key, result_list in grouped.items():
            total = 0
            for result in result_list:
                if isinstance(result, Exception):
                    psrl_logger.warning(f"Failed to fetch router backlog: role={role_key!r}: {result}.")
                    role_backlog[role_key] = prev_by_role.get(role_key, 0)
                    complete = False
                    break
                total += max(0, int(result))
            else:
                role_backlog[role_key] = total
        self.router_backlog_by_role = role_backlog
        return complete

    async def _sync_trainer_waiting_hint(self):
        hint = {}
        if self.agent_loop_manager is not None:
            try:
                ref = self.agent_loop_manager.get_trainer_waiting_hint.remote(self._trainer_waiting_buffer_id)
                hint = (
                    await asyncio.wait_for(ref, timeout=self._coordinator_sync_timeout_s)
                    if self._coordinator_sync_timeout_s > 0
                    else await ref
                )
                if not isinstance(hint, dict):
                    raise TypeError(f"Trainer waiting hint must be a dict, got {type(hint)!r}.")
            except Exception:
                psrl_logger.exception("Failed to fetch training batch progress.")
        self.trainer_waiting_hint = {
            **hint,
            "trainer_busy": not self._trainer_pool_available,
            "waiting_buffer_id": self._trainer_waiting_buffer_id,
        }

    def get_instance_kv_cache_usage(
        self,
        instance_identifier: InstanceIdentifier,
    ) -> float:
        snapshot = (
            self.instances_engine_stats.get(instance_identifier.role, {})
            .get(instance_identifier.model_name, {})
            .get(instance_identifier.instance_id, {})
        )
        # TODO(linsh): Determine whether unavailable snapshots should default to full utilization.
        if not isinstance(snapshot, dict):
            return 1.0
        scheduler_stats = snapshot.get("scheduler_stats", {})
        return scheduler_stats.get("kv_cache_usage", 1.0)

    def get_instance_running_waiting(
        self,
        instance_identifier: InstanceIdentifier,
    ) -> tuple[int, int]:
        """
        Engine scheduler queue depth for elastic sleep gating (same keys as InstanceSignal).
        """
        snapshot = (
            self.instances_engine_stats.get(instance_identifier.role, {})
            .get(instance_identifier.model_name, {})
            .get(instance_identifier.instance_id, {})
        )
        if not isinstance(snapshot, dict):
            return (0, 0)
        scheduler_stats = snapshot.get("scheduler_stats", {})
        return (
            int(scheduler_stats.get("num_running_reqs", 0)),
            int(scheduler_stats.get("num_waiting_reqs", 0)),
        )

    def _build_instance_signals(self) -> list[InstanceSignal]:
        signals: list[InstanceSignal] = []
        for role_name, role_data in self.instances_status_flags.items():
            for model_name, instance_status in role_data.items():
                for instance_id, status in instance_status.items():
                    snapshot = (
                        self.instances_engine_stats.get(role_name, {}).get(model_name, {}).get(instance_id) or {}
                    )
                    scheduler_stats = snapshot.get("scheduler_stats", {})
                    gpu_keys = self.topology.get_instance_gpu_keys_frozenset(role_name, model_name, instance_id)
                    signal = InstanceSignal(
                        role_name=role_name,
                        model_name=model_name,
                        instance_id=instance_id,
                        is_awaken=status == InstanceStatus.AWAKEN,
                        kv_cache_utilization=float(scheduler_stats.get("kv_cache_usage", 0.0)),
                        running_queue_num=int(scheduler_stats.get("num_running_reqs", 0)),
                        waiting_queue_num=int(scheduler_stats.get("num_waiting_reqs", 0)),
                        generation_throughput=float(snapshot.get("generation_throughput", 0.0)),
                        total_token_num=int(
                            sum((scheduler_stats.get("req_id_to_prompt_token_num") or {}).values())
                            + sum((scheduler_stats.get("req_id_to_response_token_num") or {}).values())
                        ),
                        snapshot_timestamp=snapshot.get("timestamp"),
                        gpu_keys=gpu_keys,
                    )
                    signals.append(signal)
        return signals
