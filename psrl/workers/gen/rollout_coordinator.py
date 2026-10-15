# TODO(zyf): Complete and verify elastic coordination across the coordinator mixins.

import asyncio
import logging
import os
import time

import numpy as np
import ray

from psrl.utils.elastic.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
)
from psrl.utils.server.command import Command, CommandExtension, CommandType
from psrl.workers.gen.stats_collector import EngineStats

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class RolloutCoordinator(CommandExtension):
    DEFAULT_AWAIT_TIMEOUT_S = 3600

    def __init__(
        self,
        config,
        rollout_router,
        rollout_wg_list,
        validate_wg_list,
        agent_loop_workers,
        status_queues,
    ):
        """
        Initialize the RolloutCoordinator.
        Coordinates and manages rollout instances for PSRL.

        This class handles:
        - Registering and tracking rollout instances
        - Managing model version synchronization across instances
        - Handling command execution (abort, sync)
        - Collecting and distributing engine status information
        - Coordinating interruption and resumption of generation tasks

        Args:
            config: Configuration object containing PSRL settings
            rollout_router: Handle to the rollout router actor
            rollout_wg_list: List of rollout worker groups
            validate_wg_list: List of validation worker groups
            agent_loop_workers: List of agent loop worker handles
            status_queues: Queues for receiving status updates from different rollout instances
        """
        super().__init__()

        self.config = config
        self.staleness = self.config.psrl.staleness
        if self.config.psrl.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n

        self.rollout_wg_list = rollout_wg_list
        self.rollout_wg_size = len(rollout_wg_list)
        self.validate_wg_list = validate_wg_list
        self.validate_wg_size = len(validate_wg_list)

        # All rollout and validate worker groups
        self.gen_wg_list = self.rollout_wg_list + self.validate_wg_list
        self.gen_wg_size = self.rollout_wg_size + self.validate_wg_size

        self.n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        self.n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )

        assert self.rollout_wg_size == self.n_rollout_instances, (
            "The number of rollout worker groups must be the same as the number of rollout instances, "
            f"but got {self.rollout_wg_size} and {self.n_rollout_instances}."
        )
        assert self.validate_wg_size == self.n_validate_instances, (
            "The number of validate worker groups must be the same as the number of validate instances, "
            f"but got {self.validate_wg_size} and {self.n_validate_instances}."
        )
        self.agent_loop_workers = agent_loop_workers
        self.rollout_router = rollout_router

        # Stats collection
        self.status_queues = status_queues
        assert len(self.status_queues) == self.gen_wg_size, (
            "The number of status queues must be the same as the number of rollout instances, "
            f"but got {len(self.status_queues)} and {self.gen_wg_size}."
        )

        # Background event handler
        self.running_loop = None
        self.command_handler_task = None
        self._active_command_id: int | None = None
        self.sync_task = None
        self.process_status_queue_tasks = []
        self.broadcast_status_to_router_task = None
        self.disaggregated_interrupt_task = None
        self.stop_command_handler = False
        self.stop_sync_and_migrate = False
        self.stop_disaggregated_interrupt = False
        self.stop_process_status_queue = [False] * self.gen_wg_size
        self.stop_broadcast_status_to_router = False

        # Track model initialization independently for rollout and validation subsets.
        self._is_init_model_events = [asyncio.Event() for _ in range(self.gen_wg_size)]
        self._is_init_nixl_client = asyncio.Event()

        # Version tracking
        self.instance_to_latest_stale_model_version: dict[
            int, int
        ] = {}  # The latest stale model version of each instance
        self.instance_to_model_version: dict[int, int] = {}  # Track the model version of each instance
        self.ps_model_version = 0  # Current model version in the parameter server
        self.ready_buffers = set()  # The set of ready buffers

        # Engine status tracking
        self.instance_to_engine_status: dict[int, EngineStats] = {}  # Track the latest engine stats of each instance

        self.enable_elastic = self.config.psrl.deployment.elastic.enable

        # Validate the disaggregated drift experiment's mode before it can interrupt validation or reward requests.
        interrupt_config = self.config.psrl.deployment.get("disaggregated_rollout_interrupt", {})
        self._disaggregated_interrupt_enabled = bool(interrupt_config.get("enable", False))
        self._disaggregated_interrupt_interval_s = float(interrupt_config.get("interval_s", 0.0))
        deployment_mode = str(self.config.psrl.deployment.get("mode", ""))
        if self._disaggregated_interrupt_enabled:
            if deployment_mode != "disaggregated":
                raise ValueError("disaggregated_rollout_interrupt requires psrl.deployment.mode=disaggregated")
            if self._disaggregated_interrupt_interval_s <= 0:
                raise ValueError("disaggregated_rollout_interrupt.interval_s must be > 0 when enabled")
            if not self.config.psrl.partial_rollout.enable:
                raise ValueError(
                    "disaggregated_rollout_interrupt requires psrl.partial_rollout.enable=True "
                    "so interrupted requests can loop back"
                )

        # Use a free instance for swap-based weight sync when the elastic executor permits it.
        self._elastic_executor = None  # ray.actor.ActorHandle | None
        self._elastic_exec_role_name = None  # PSRL_Role for this coordinator's role
        self._elastic_exec_model_name = None  # model name string

        # Reject elastic sleep and wake commands while an instance is locked for in-place synchronization.
        self._syncing_locked_instance_ids: set[int] = set()
        self._syncing_lock_owner_by_instance: dict[int, int] = {}
        self._next_sync_operation_id = 1

        # Build logger
        self.log_prefix = "RolloutCoordinator"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    def world_size(self):
        """Get the total world size (number of rollout and validate instances)."""
        return sum([rollout_wg.world_size for rollout_wg in self.gen_wg_list])

    def resume_instances(self, instance_ids: list[int]):
        """Notify that the given instances have resumed processing.

        Args:
            instance_ids (list[int]): List of instance IDs that have resumed processing.
        """
        for instance_id in instance_ids:
            self.stop_process_status_queue[instance_id] = False

    def pause_instances(self, instance_ids: list[int]):
        """Notify that the given instances have paused processing.

        Args:
            instance_ids (list[int]): List of instance IDs that have paused processing.
        """
        for instance_id in instance_ids:
            self.stop_process_status_queue[instance_id] = True

    def _get_wgs(self, tag: str):
        """Get worker groups and their global indices based on tag.

        Args:
            tag (str): Tag to specify which instances to get ('rollout', 'validate', 'all')
        Returns:
            tuple: (worker group list, global wg indices)
        """
        if tag == "rollout":
            return self.rollout_wg_list, list(range(self.rollout_wg_size))
        elif tag == "validate":
            return self.validate_wg_list, list(range(self.rollout_wg_size, self.gen_wg_size))
        elif tag == "all":
            return self.gen_wg_list, list(range(self.gen_wg_size))
        else:
            raise ValueError(f"Unknown tag {tag} for getting worker groups")

    async def _wait_for_init_model(self, tag: str, func_name: str):
        """Wait until all worker groups under the tag finish model init."""
        _, wg_indices = self._get_wgs(tag)
        if not wg_indices:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*(self._is_init_model_events[idx].wait() for idx in wg_indices)),
                timeout=self.DEFAULT_AWAIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"[{func_name}] timed out after {self.DEFAULT_AWAIT_TIMEOUT_S}s "
                f"while waiting model init for tag={tag}, wg={wg_indices}"
            ) from e

    async def _await_futures_with_timeout(self, futures, func_name: str, tag: str, wg_indices: list[int]):
        """Await futures with a unified timeout and detailed context on timeout."""
        try:
            return await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=self.DEFAULT_AWAIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"[{func_name}] timed out after {self.DEFAULT_AWAIT_TIMEOUT_S}s "
                f"while waiting futures for tag={tag}, wg={wg_indices}"
            ) from e

    async def _wait_for_nixl_client(self, func_name: str):
        """Wait until NIXL client initialization is complete."""
        try:
            await asyncio.wait_for(
                self._is_init_nixl_client.wait(),
                timeout=self.DEFAULT_AWAIT_TIMEOUT_S,
            )
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"[{func_name}] timed out after {self.DEFAULT_AWAIT_TIMEOUT_S}s "
                "while waiting NIXL client initialization"
            ) from e

    async def init_model(self, tag: str = "rollout", init_mode: str = "full"):
        """Init the model on rollout instances and register to ps manager.

        Args:
            tag (str): Tag to specify which instances to initialize ('rollout', 'validate', 'all')
            init_mode (str): Initialization mode ('full', 'empty', etc.)
                'full' mode will load the full model weights,
                'empty' mode will load dummy model weights.
        """
        wg_list, wg_indices = self._get_wgs(tag)
        futures = []
        for i in range(len(wg_list)):
            futures.append(wg_list[i].execute_rank_zero_async("init_and_register_model", init_mode))
        await self._await_futures_with_timeout(futures, "init_model", tag, wg_indices)
        for idx in wg_indices:
            self._is_init_model_events[idx].set()

    async def init_route_strategy(self, tag: str = "rollout"):
        """Init the route strategy on rollout instances.

        This method estimates the maximum model length on each instance
        and uses it to budget the kv cache size for each instance.

        Args:
            tag (str): Tag to specify which instances to initialize ('rollout', 'validate', 'all')
        """
        assert self.rollout_router is not None, "Rollout router is not set in RolloutCoordinator"
        await self._wait_for_init_model(tag, "init_route_strategy")

        wg_list, wg_indices = self._get_wgs(tag)
        futures = []
        for i in range(len(wg_list)):
            futures.append(wg_list[i].execute_rank_zero_async("estimate_max_model_len"))
        max_model_lens = await self._await_futures_with_timeout(futures, "init_route_strategy", tag, wg_indices)
        psrl_logger.info(f"Max model lens on {tag} instances: {max_model_lens}")
        instance_to_max_model_len = {wg_idx: max(max_model_lens[j]) for j, wg_idx in enumerate(wg_indices)}
        # Use the max model len to budget the kv cache size for each instance
        await self.rollout_router.init_route_strategy.remote(instance_to_max_model_len=instance_to_max_model_len)

    async def init_nixl_client(self):
        """Init the NIXL client on rollout and validate instances."""
        await self._wait_for_init_model("all", "init_nixl_client")
        futures = []
        for i in range(self.gen_wg_size):
            futures.append(self.gen_wg_list[i].execute_rank_zero_async("init_nixl_client"))
        await self._await_futures_with_timeout(futures, "init_nixl_client", "all", self._get_wgs("all")[1])
        psrl_logger.info(f"Initialized NIXL client on all {self.gen_wg_size} instances.")
        self._is_init_nixl_client.set()

    async def nixl_protocol(self, full_tag: str = "all"):
        """Run the NIXL server protocol on rollout and validate instances.

        Args:
            full_tag (str): Tag to specify which instances to run the protocol
                            in 'full' mode ('rollout', 'validate', 'all')
        """
        await self._wait_for_nixl_client("nixl_protocol")

        if full_tag == "all":
            full_tag_list = ["full"] * self.gen_wg_size
        elif full_tag == "rollout":
            full_tag_list = ["full"] * self.rollout_wg_size + ["meta"] * self.validate_wg_size
        elif full_tag == "validate":
            full_tag_list = ["meta"] * self.rollout_wg_size + ["full"] * self.validate_wg_size
        else:
            raise ValueError(f"Unknown full_tag {full_tag} for nixl_convert_params")

        futures = []
        for i in range(self.gen_wg_size):
            futures.append(self.gen_wg_list[i].execute_rank_zero_async("nixl_protocol", full_tag_list[i]))
        await self._await_futures_with_timeout(futures, "nixl_protocol", "all", self._get_wgs("all")[1])

    async def nixl_convert_params(self):
        """Convert the model parameters to unified format on rollout and validate instances."""
        await self._wait_for_nixl_client("nixl_convert_params")
        futures = []
        for i in range(self.gen_wg_size):
            futures.append(self.gen_wg_list[i].execute_rank_zero_async("nixl_convert_params"))
        await self._await_futures_with_timeout(futures, "nixl_convert_params", "all", self._get_wgs("all")[1])

    async def initial_pull_from_ps(self, tag: str = "rollout") -> None:
        """
        Force an initial weight pull from PS to all specified gen workers via NIXL.

        This is called exactly once at initialization, after the NIXL protocol
        completes and PS buffers are populated via preload_checkpoint_to_cpu() and
        write_checkpoint_to_registered_tensors().
        Because both the PS and workers start at version 0, the version-based skip
        check inside GenWorker.sync_with_ps (curr_version >= ps_version) would
        incorrectly suppress the pull . so we call nixl_pull_model_async directly
        on each worker group instead of going through the SYNC command path.

        Args:
            tag (str): Which instances to pull into ('rollout', 'validate', 'all').
        """
        await self._wait_for_init_model(tag, "initial_pull_from_ps")
        wg_list, wg_indices = self._get_wgs(tag)
        futures = [wg_list[i].execute_rank_zero_async("nixl_pull_model_async") for i in range(len(wg_list))]
        await self._await_futures_with_timeout(futures, "initial_pull_from_ps", tag, wg_indices)
        psrl_logger.info(f"Initial PS pull complete for {len(wg_list)} {tag!r} instance(s).")

    async def sleep(self, tag: str = "all"):
        """Make rollout instances sleep and release GPU memory.

        Args:
            tag (str): Tag to specify which instances to sleep ('rollout', 'validate', 'all')
        """
        await self._wait_for_init_model(tag, "sleep")

        wg_list, wg_indices = self._get_wgs(tag)
        futures = []
        for i in range(len(wg_list)):
            futures.append(wg_list[i].execute_rank_zero_async("sleep"))
        await self._await_futures_with_timeout(futures, "sleep", tag, wg_indices)

    async def start_busy_loop(self):
        """
        Start the background event loops for command handling and status synchronization.

        This method:
        1. Starts a background task for handling commands (abort, sync, etc.).
        2. Optionally starts tasks for processing status queues of each rollout instance.
        3. Starts a task to broadcast the engine status to the agent loop workers (i.e., router).
        4. Starts a task to synchronize rollout instances with PS.
        """
        await self._wait_for_init_model("rollout", "start_busy_loop")

        if self.command_handler_task is not None and not self.command_handler_task.done():
            return

        # Start the background tasks
        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(self._on_command_handler_done)
        if self._disaggregated_interrupt_enabled:
            self.disaggregated_interrupt_task = self.running_loop.create_task(
                self._disaggregated_rollout_interrupt_loop()
            )

        # Start the status collection tasks
        if self.config.psrl.status_collection.enable:
            for instance_id in range(self.gen_wg_size):
                self.process_status_queue_tasks.append(
                    self.running_loop.create_task(self._process_status_queue(instance_id))
                )
                self.process_status_queue_tasks[instance_id].add_done_callback(
                    lambda f: f.result()
                )  # To avoid silent error in async tasks
        # Start the task to broadcast the engine status to the router
        self.broadcast_status_to_router_task = self.running_loop.create_task(self._broadcast_status_to_router())
        self.broadcast_status_to_router_task.add_done_callback(
            lambda f: f.result()
        )  # To avoid silent error in async tasks
        # Start the model synchronization and rollout migration loop
        if self.config.psrl.sync_and_mig_strategy.method == "greedy":
            self.sync_task = self.running_loop.create_task(self._greedy_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        elif self.config.psrl.sync_and_mig_strategy.method == "status_based":
            assert self.config.psrl.status_collection.enable, (
                "Status-based sync strategy is only supported when status collection is enabled"
            )
            self.sync_task = self.running_loop.create_task(self._status_based_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        else:
            raise NotImplementedError(
                f"Sync strategy {self.config.psrl.sync_and_mig_strategy.method} is not supported"
            )
        # Check if rollout migration is enabled
        if self.config.psrl.sync_and_mig_strategy.mig.enable:
            assert self.config.psrl.status_collection.enable, (
                "Rollout migration is only supported when status collection is enabled"
            )
            assert self.config.psrl.partial_rollout.enable, (
                "Rollout migration is only supported when partial rollout is enabled"
            )

    async def stop_busy_loop(self):
        """
        Stop all background tasks and clean up resources.

        This method gracefully shuts down:
        - Command handler task
        - Engine status sync task
        """
        if self.command_handler_task is None or self.command_handler_task.done():
            return

        # Stop the background tasks
        self.stop_command_handler = True
        self.stop_sync_and_migrate = True
        self.stop_disaggregated_interrupt = True
        self.stop_process_status_queue = [True] * self.gen_wg_size
        self.stop_broadcast_status_to_router = True

        if self.disaggregated_interrupt_task is not None and not self.disaggregated_interrupt_task.done():
            self.disaggregated_interrupt_task.cancel()

        tasks_to_wait = [self.command_handler_task]
        tasks_to_wait.append(self.sync_task)
        if self.process_status_queue_tasks:
            tasks_to_wait.extend(self.process_status_queue_tasks)
        tasks_to_wait.append(self.broadcast_status_to_router_task)
        if self.disaggregated_interrupt_task is not None:
            tasks_to_wait.append(self.disaggregated_interrupt_task)

        # Wait for tasks to finish with timeout
        await asyncio.gather(*tasks_to_wait, return_exceptions=True)

    async def _disaggregated_rollout_interrupt_loop(self):
        """Abort training-rollout requests periodically and let the router requeue them."""
        tick = 0
        while not self.stop_disaggregated_interrupt:
            await asyncio.sleep(self._disaggregated_interrupt_interval_s)
            if self.stop_disaggregated_interrupt:
                break

            try:
                instance_to_uids = await self.rollout_router.get_inflight_rollout_request_ids.remote()
                instance_to_uids = {
                    int(instance_id): [int(request_id) for request_id in request_ids]
                    for instance_id, request_ids in (instance_to_uids or {}).items()
                    if request_ids
                }
                request_ids = [request_id for ids in instance_to_uids.values() for request_id in ids]
                if not request_ids:
                    continue

                interrupted = await self.exec_command(
                    Command(type=CommandType.ABORT, instance_to_uids=instance_to_uids),
                    blocking=True,
                )
                tick += 1
                psrl_logger.info(
                    "[DISAGG_INTERRUPT] tick=%d interval_s=%.3f requests=%d interrupted=%s "
                    "scope=rollout_only reroute=partial_router",
                    tick,
                    self._disaggregated_interrupt_interval_s,
                    len(request_ids),
                    interrupted,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                # Keep the experiment alive after a transient Ray/engine error.
                # the next interval will take a fresh request snapshot.
                psrl_logger.exception("[DISAGG_INTERRUPT] periodic rollout interruption failed")

    async def _command_handler_loop(self):
        """
        Background loop for processing commands from the command queue.

        This method continuously processes different types of commands:
        - ABORT: Interrupt specific requests on instances
        - SYNC: Interrupt instance, pull new model weights, and resume generation

        The loop runs until stop_command_handler is set to True.
        """
        while not self.stop_command_handler:
            # Command processing
            if not self.command_queue.empty():
                command = self.command_queue.get_nowait()

                assert isinstance(command, Command), f"Expected Command type, got {type(command)}"

                # Unpack command attributes
                command_type = command.type
                command_id = command.get_kwargs()["id"]
                self._active_command_id = command_id
                self._start_command(command_id)
                command_args = command.get_args()
                psrl_logger.info(
                    f"Receive command: type = {command_type}, kwargs = {command.get_kwargs()}, args = {command_args}"
                )

                result = None
                if command_type in (CommandType.SLEEP, CommandType.WAKE_UP):
                    raw_instance_ids = command_args.get("instance_ids", [])
                    if not isinstance(raw_instance_ids, list):
                        raw_instance_ids = [raw_instance_ids]
                    lock_check_ids = list(range(self.rollout_wg_size)) if -1 in raw_instance_ids else raw_instance_ids
                    sync_operation_id = command_args.get("sync_operation_id")
                    conflicting_ids = [
                        int(instance_id)
                        for instance_id in lock_check_ids
                        if instance_id in self._syncing_locked_instance_ids
                        and self._syncing_lock_owner_by_instance.get(int(instance_id)) != sync_operation_id
                    ]
                    if conflicting_ids:
                        psrl_logger.info(
                            "Reject %s command for sync-locked instances %s (owner=%s).",
                            command_type.name,
                            conflicting_ids,
                            sync_operation_id,
                        )
                        self._complete_command(command_id, False)
                        self._active_command_id = None
                        await asyncio.sleep(0)
                        continue
                # Process the command based on its type
                if command_type == CommandType.ABORT:
                    instance_to_uids = command_args.get("instance_to_uids", None)
                    instance_ids = command_args.get("instance_ids", None)
                    request_migrations = command_args.get("request_migrations", None)
                    migration_context = command_args.get("migration_context")
                    if instance_to_uids is None and instance_ids is None and request_migrations is None:
                        raise ValueError(
                            "ABORT command must contain 'instance_to_uids', 'instance_ids', "
                            "or 'request_migrations' in args."
                        )

                    if request_migrations is not None:
                        if self.rollout_router is None:
                            instance_to_uids = {}
                        else:
                            prepared = await self.rollout_router.prepare_request_migrations.remote(
                                request_migrations,
                                migration_context,
                            )
                            instance_to_uids = prepared.get("instance_to_uids", {})
                            psrl_logger.info(
                                "Prepared rollout request migrations before ABORT: %s",
                                prepared,
                            )

                    psrl_logger.info(
                        f"Received ABORT command with instance_to_uids: "
                        f"{instance_to_uids} and instance_ids: {instance_ids}"
                    )
                    futures = []

                    if (
                        migration_context
                        and request_migrations is None
                        and instance_to_uids is not None
                        and self.rollout_router is not None
                    ):
                        await self.rollout_router.mark_migration_requests.remote(
                            instance_to_uids,
                            migration_context,
                        )

                    if instance_to_uids is not None:
                        for instance_id, uids in instance_to_uids.items():
                            if not uids:
                                continue
                            if not isinstance(uids, (list, set)):
                                uids = [uids]
                            abort_requests = set(uids)  # Ensure uniqueness
                            assert instance_id < len(self.rollout_wg_list), (
                                f"Validate instance should not be interrupted, but got instance_id {instance_id} "
                                f"which is out of rollout instance range [0, {len(self.rollout_wg_list)})."
                            )
                            futures.append(
                                self.gen_wg_list[instance_id].execute_rank_zero_async(
                                    "interrupt_requests", abort_requests
                                )
                            )
                    if instance_ids is not None:
                        for instance_id in instance_ids:
                            futures.append(
                                self.gen_wg_list[instance_id].execute_rank_zero_async("interrupt_requests", None)
                            )

                    if not futures:
                        interrupted_request_num = 0
                    else:
                        interrupted_request_nums = await asyncio.gather(*futures)
                        interrupted_request_num = np.sum(interrupted_request_nums)

                    if migration_context and self.rollout_router is not None:
                        await self.rollout_router.wait_for_exclusive_rebalance.remote()

                    result = interrupted_request_num
                    psrl_logger.info(f"Received ABORT command, interrupted {interrupted_request_num} requests")
                    # Post process the command result
                    self._complete_command(command_id, result)

                elif command_type == CommandType.SYNC:
                    # Interrupt the instance, pull the model weights from PS and resume generation.
                    instance_ids = command_args.get("instance_ids", None)
                    curr_ps_model_version = command_args.get("curr_ps_model_version", None)
                    wait_model_sync = command_args.get("wait_model_sync", False)
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if instance_ids is None or curr_ps_model_version is None:
                        raise ValueError(
                            "SYNC command must contain 'instance_ids' and 'curr_ps_model_version' in args."
                        )
                    psrl_logger.info(
                        f"Received SYNC command for instances {instance_ids} "
                        f"with PS model version {curr_ps_model_version}"
                    )

                    # Skip instances that are in vLLM sleep (no weights to pull). sync after WAKE_UP instead.
                    sleeping_flags = await asyncio.gather(
                        *[
                            self.gen_wg_list[i].execute_rank_zero_async("is_rollout_engine_sleeping")
                            for i in instance_ids
                        ]
                    )
                    sync_instance_ids: list[int] = []
                    skipped_sleeping: list[int] = []
                    for instance_id, is_sleeping in zip(instance_ids, sleeping_flags):
                        if is_sleeping:
                            skipped_sleeping.append(instance_id)
                        else:
                            sync_instance_ids.append(instance_id)
                    if skipped_sleeping:
                        psrl_logger.info(
                            "SYNC: skipping %s sleeping instance(s) %s (no interrupt/sync until wake_up)",
                            len(skipped_sleeping),
                            skipped_sleeping,
                        )

                    if not sync_instance_ids:
                        self._complete_command(command_id, [0] * len(instance_ids))
                    else:
                        # Sync with PS (interrupt, pull model, and resume generation)
                        interrupt_futures = []
                        sync_futures = []

                        for instance_id in sync_instance_ids:
                            interrupt_future = self.gen_wg_list[instance_id].execute_rank_zero_async(
                                "interrupt_generation"
                            )
                            interrupt_futures.append(interrupt_future)
                        interrupted_per_sync = await asyncio.gather(*interrupt_futures)
                        interrupted_by_id = {iid: n for iid, n in zip(sync_instance_ids, interrupted_per_sync)}
                        interrupted_request_nums = [interrupted_by_id.get(iid, 0) for iid in instance_ids]

                        for instance_id in sync_instance_ids:
                            psrl_logger.info(
                                f"Syncing with PS on instance {instance_id}, "
                                f"interrupted {interrupted_by_id[instance_id]} requests"
                            )

                        for instance_id in sync_instance_ids:
                            sync_future = self.gen_wg_list[instance_id].execute_rank_zero_async(
                                "sync_with_ps", curr_ps_model_version
                            )
                            sync_futures.append(sync_future)

                        # Post process the command result
                        if wait_model_sync:
                            await asyncio.gather(*sync_futures)
                            self._complete_command(command_id, interrupted_request_nums)
                        else:
                            # NOTE(linsh): Sometimes it's not necessary for the caller to wait for pulling from PS.
                            self._complete_command(command_id, interrupted_request_nums)
                            await asyncio.gather(*sync_futures)  # Wait for the sync to complete

                elif command_type == CommandType.SLEEP:
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_ids is None:
                        raise ValueError("SLEEP command must contain 'instance_ids' in args.")
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]
                    if -1 in instance_ids:
                        instance_ids = list(range(self.rollout_wg_size))
                    overhead_started_s = time.monotonic()
                    pause_s = 0.0
                    state_probe_s = 0.0
                    interrupt_s = 0.0
                    engine_sleep_s = 0.0
                    success = False
                    try:
                        if self.rollout_router is not None:
                            stage_started_s = time.monotonic()
                            await self.rollout_router.pause_instances.remote(instance_ids)
                            pause_s = time.monotonic() - stage_started_s
                        stage_started_s = time.monotonic()
                        sleeping_flags = await asyncio.gather(
                            *[
                                self.gen_wg_list[instance_id].execute_rank_zero_async("is_rollout_engine_sleeping")
                                for instance_id in instance_ids
                            ]
                        )
                        state_probe_s = time.monotonic() - stage_started_s
                        awake_instance_ids = [
                            instance_id
                            for instance_id, is_sleeping in zip(instance_ids, sleeping_flags)
                            if not is_sleeping
                        ]
                        stage_started_s = time.monotonic()
                        interrupted_request_nums = await asyncio.gather(
                            *[
                                self.gen_wg_list[instance_id].execute_rank_zero_async("interrupt_generation")
                                for instance_id in awake_instance_ids
                            ]
                        )
                        interrupt_s = time.monotonic() - stage_started_s
                        interrupted_request_num = np.sum(interrupted_request_nums)
                        psrl_logger.info(
                            "Received SLEEP command for instances %s, interrupted %s requests",
                            instance_ids,
                            interrupted_request_num,
                        )
                        stage_started_s = time.monotonic()
                        await asyncio.gather(
                            *[
                                self.gen_wg_list[instance_id].execute_rank_zero_async("nixl_sleep")
                                for instance_id in awake_instance_ids
                            ]
                        )
                        engine_sleep_s = time.monotonic() - stage_started_s
                    except Exception:
                        psrl_logger.exception("SLEEP command failed for instances %s", instance_ids)
                        if self.rollout_router is not None:
                            await self.rollout_router.resume_instances.remote(instance_ids)
                        self._complete_command(command_id, False)
                    else:
                        success = True
                        psrl_logger.info("SLEEP command for instances %s completed", instance_ids)
                        self._complete_command(command_id, True)
                    finally:
                        psrl_logger.info(
                            "[ELASTIC_OVERHEAD] operation=sleep role=Rollout instances=%s success=%s "
                            "router_pause_s=%.6f state_probe_s=%.6f interrupt_s=%.6f "
                            "engine_sleep_s=%.6f total_s=%.6f",
                            instance_ids,
                            success,
                            pause_s,
                            state_probe_s,
                            interrupt_s,
                            engine_sleep_s,
                            time.monotonic() - overhead_started_s,
                        )

                elif command_type == CommandType.WAKE_UP:
                    instance_ids = command_args.get("instance_ids", None)
                    if instance_ids is None:
                        raise ValueError("WAKE_UP command must contain 'instance_ids' in args.")
                    if not isinstance(instance_ids, list):
                        instance_ids = [instance_ids]

                    if -1 in instance_ids:
                        instance_ids = list(range(self.rollout_wg_size))

                    # Snapshot once. A queued command may span a PS version update.
                    # all worker and router updates in this command must agree.
                    target_ps_version = int(command_args.get("target_ps_model_version", self.ps_model_version))
                    resume_instances = bool(command_args.get("resume_instances", True))

                    overhead_started_s = time.monotonic()
                    state_probe_s = 0.0
                    engine_wakeup_s = 0.0
                    router_update_s = 0.0
                    model_sync_s = 0.0
                    router_resume_s = 0.0
                    success = False
                    try:
                        stage_started_s = time.monotonic()
                        sleeping_flags = await asyncio.gather(
                            *[
                                self.gen_wg_list[instance_id].execute_rank_zero_async("is_rollout_engine_sleeping")
                                for instance_id in instance_ids
                            ]
                        )
                        state_probe_s = time.monotonic() - stage_started_s
                        stage_started_s = time.monotonic()
                        await asyncio.gather(
                            *[
                                self.gen_wg_list[instance_id].execute_rank_zero_async("nixl_wake_up")
                                for instance_id, is_sleeping in zip(instance_ids, sleeping_flags)
                                if is_sleeping
                            ]
                        )
                        engine_wakeup_s = time.monotonic() - stage_started_s
                        psrl_logger.info("WAKE_UP command for instances %s completed", instance_ids)

                        stage_started_s = time.monotonic()
                        await self.rollout_router.update_currently_syncing_instances.remote(
                            instance_ids, target_ps_version
                        )
                        router_update_s = time.monotonic() - stage_started_s
                        psrl_logger.info(
                            "Updated currently syncing instances to %s with PS model version %d",
                            instance_ids,
                            target_ps_version,
                        )

                        stage_started_s = time.monotonic()
                        sync_futures = []
                        for instance_id in instance_ids:
                            active_tasks = await self.gen_wg_list[instance_id].execute_rank_zero_async(
                                "get_active_task_num"
                            )
                            needs_interrupt = active_tasks > 0
                            if needs_interrupt:
                                psrl_logger.warning(
                                    "WAKE_UP for instance %d found %d active tasks. interrupting before sync",
                                    instance_id,
                                    active_tasks,
                                )
                            sync_futures.append(
                                self.gen_wg_list[instance_id].execute_rank_zero_async(
                                    "sync_with_ps",
                                    ps_version=target_ps_version,
                                    interrupt_generation=needs_interrupt,
                                    sync_after_wake_up=True,
                                )
                            )
                        await asyncio.gather(*sync_futures)
                        model_sync_s = time.monotonic() - stage_started_s
                        if self.rollout_router is not None and resume_instances:
                            stage_started_s = time.monotonic()
                            await self.rollout_router.resume_instances.remote(instance_ids)
                            router_resume_s = time.monotonic() - stage_started_s
                    except Exception:
                        psrl_logger.exception(
                            "WAKE_UP command failed for instances %s at PS version %d",
                            instance_ids,
                            target_ps_version,
                        )
                        self._complete_command(command_id, False)
                    else:
                        success = True
                        psrl_logger.info(
                            "Synced with PS for instances %s with PS model version %d",
                            instance_ids,
                            target_ps_version,
                        )
                        self._complete_command(command_id, True)
                    finally:
                        psrl_logger.info(
                            "[ELASTIC_OVERHEAD] operation=wakeup role=Rollout instances=%s success=%s "
                            "state_probe_s=%.6f engine_wakeup_s=%.6f router_update_s=%.6f "
                            "model_sync_s=%.6f router_resume_s=%.6f total_s=%.6f",
                            instance_ids,
                            success,
                            state_probe_s,
                            engine_wakeup_s,
                            router_update_s,
                            model_sync_s,
                            router_resume_s,
                            time.monotonic() - overhead_started_s,
                        )
                else:
                    raise ValueError(f"Unknown command type: {command_type}")

                self._active_command_id = None
            await asyncio.sleep(0)

        psrl_logger.info("Background command handler of rollout coordinator has finished.")

    def _on_command_handler_done(self, task: asyncio.Task) -> None:
        """Fail the active command and restart the queue loop after an unexpected exception."""
        if task.cancelled():
            return
        exception = task.exception()
        if exception is None:
            return
        psrl_logger.error(
            "Rollout command handler crashed. active_command_id=%s. Restarting queue loop.",
            self._active_command_id,
            exc_info=(type(exception), exception, exception.__traceback__),
        )
        if self._active_command_id is not None:
            self._complete_command(self._active_command_id, False)
            self._active_command_id = None
        if not self.stop_command_handler and self.running_loop is not None:
            self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
            self.command_handler_task.add_done_callback(self._on_command_handler_done)

    async def _process_status_queue(self, instance_id: int):
        psrl_logger.info(f"Starting to process status queue for instance {instance_id}")
        while not self.stop_process_status_queue[instance_id]:
            # TODO(lhy): Add timeout handling for future fault tolerance of rollout instances.
            recv_stats = await self.status_queues[instance_id].get_async(block=True, timeout=None)
            self.instance_to_engine_status[instance_id] = recv_stats
            psrl_logger.debug(
                f"Updated engine status for instance "
                f"{recv_stats.instance_id}: {self.instance_to_engine_status[instance_id]}"
            )

    def get_instance_engine_status_snapshot(self) -> dict[int, dict]:
        """
        Return a lightweight snapshot map for elastic scaling decisions.
        """
        result: dict[int, dict] = {}
        for instance_id, engine_stats in self.instance_to_engine_status.items():
            result[int(instance_id)] = {
                "instance_id": int(engine_stats.instance_id),
                "model_version": int(engine_stats.model_version),
                "timestamp": engine_stats.snapshot.get("timestamp"),
                "scheduler_stats": engine_stats.snapshot.get("scheduler_stats", {}),
                "generation_throughput": float(engine_stats.snapshot.get("generation_throughput", 0.0)),
            }
        return result

    async def get_router_backlog_size(self) -> int:
        """Return pending request count in rollout router queue."""
        t_enter = time.monotonic()
        model_tag = str(self.config.gen_actor_rollout_ref.model.path).rstrip("/").split("/")[-1]
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_enter model=%s elapsed_since_entry_s=0.000",
            model_tag,
        )
        if self.rollout_router is None:
            return 0
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_before_router_rpc model=%s since_enter_s=%.3f",
            model_tag,
            time.monotonic() - t_enter,
        )
        t_rpc = time.monotonic()
        pending = int(await self.rollout_router.get_pending_request_count.remote())
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_after_router_rpc model=%s pending=%d router_rpc_s=%.3f since_enter_s=%.3f",
            model_tag,
            pending,
            time.monotonic() - t_rpc,
            time.monotonic() - t_enter,
        )
        return pending

    async def get_router_backlog_summary(self, top_t: int | None = None) -> dict[str, int]:
        """Return selected rollout router waiting queue load for elastic scaling."""
        t_enter = time.monotonic()
        model_tag = str(self.config.gen_actor_rollout_ref.model.path).rstrip("/").split("/")[-1]
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_summary_enter model=%s elapsed_since_entry_s=0.000",
            model_tag,
        )
        if self.rollout_router is None:
            return {"pending": 0, "count": 0, "total_tokens": 0}
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_summary_before_router_rpc model=%s since_enter_s=%.3f",
            model_tag,
            time.monotonic() - t_enter,
        )
        t_rpc = time.monotonic()
        summary = await self.rollout_router.get_pending_request_summary.remote(top_t)
        log_elastic_rm_backlog_diag(
            psrl_logger,
            (
                "stage=RolloutCoordinator_summary_after_router_rpc model=%s pending=%d count=%d "
                "total_tokens=%d router_rpc_s=%.3f since_enter_s=%.3f"
            ),
            model_tag,
            int(summary.get("pending", 0)),
            int(summary.get("count", 0)),
            int(summary.get("total_tokens", 0)),
            time.monotonic() - t_rpc,
            time.monotonic() - t_enter,
        )
        return summary

    async def get_candidate_evaluation_snapshot(self, top_t: int | None = None) -> dict:
        """Proxy the compact request-level snapshot from the rollout router."""
        if self.rollout_router is None:
            raise RuntimeError("rollout router is not initialized")
        return await self.rollout_router.get_candidate_evaluation_snapshot.remote(top_t)

    async def _broadcast_status_to_router(self):
        """
        Broadcast the engine status to the router.
        """
        assert self.rollout_router is not None, "Rollout router is not set in RolloutCoordinator"

        while not self.stop_broadcast_status_to_router:
            # Broadcast the engine status to the router every coordinator sync interval
            await asyncio.sleep(self.config.psrl.status_collection.coordinator_sync_interval_in_ms / 1000)
            await self.rollout_router.update_instance_status.remote(self.instance_to_engine_status)

    async def _is_routing(self) -> bool:
        """Check if any agent loop worker is currently routing requests
        (i.e., the router is currently routing requests).

        Returns:
            bool: True if any agent loop worker is currently routing requests,
                False otherwise.
        """
        is_routing = await self.rollout_router.is_routing.remote()
        return is_routing

    async def _greedy_sync_and_migrate_loop(self):
        """
        Background loop to synchronize with PS based on the greedy algorithm.

        This method:
        1. Greedily synchronize with PS for rollout that lags behind PS version.
        2. Check whether the instance has no active tasks if forbid partial rollout.
        3. Check if any instance is starving and do migration if necessary.
        """
        psrl_logger.info("Starting greedy model synchronization and rollout migration loop")

        while not self.stop_sync_and_migrate:
            # Sleep for a period of time
            await asyncio.sleep(self.config.psrl.sync_and_mig_strategy.check_interval_in_ms / 1000)

            have_syncing_instance = False
            sync_instance_ids = []
            for instance_id in range(self.rollout_wg_size):
                # Skip instances currently under a sync_with_ps call (swap or in-place).
                if self.enable_elastic and instance_id in self._syncing_locked_instance_ids:
                    have_syncing_instance = True
                    continue
                # Check whether engine status is stale (the instance is currently being synchronized with PS)
                if self.instance_to_model_version.get(
                    instance_id, 0
                ) <= self.instance_to_latest_stale_model_version.get(instance_id, -1):
                    have_syncing_instance = True
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) == self.ps_model_version:
                    continue
                # Check whether current instance workload is empty if forbid partial rollout
                if not self.config.psrl.partial_rollout.enable:
                    if not await self.check_no_activate_tasks(instance_id):
                        continue
                # Check whether the training side can seamlessly continue to train after the synchronization
                if self.config.psrl.sync_and_mig_strategy.sync.seamless_train_version >= self.ps_model_version:
                    if self.ps_model_version not in self.ready_buffers:
                        continue
                # Add the instance to the sync list
                sync_instance_ids.append(instance_id)

            if sync_instance_ids:
                await self.sync_with_ps(sync_instance_ids)
            elif not have_syncing_instance and self.config.psrl.sync_and_mig_strategy.mig.enable:
                # No instance is syncing with PS, check if migration is needed
                await self.check_and_migrate()

        psrl_logger.info("Greedy model synchronization and rollout migration loop stopped.")

    async def _status_based_sync_and_migrate_loop(self):
        """
        Background loop to collect engine status and decide whether to synchronize with PS based on the engine status.

        This method:
        1. Analyze the instance status (engine waiting & running request counts, etc.).
        2. Decide whether to synchronize with PS for each instance.
        """
        psrl_logger.info("Starting status based model synchronization and rollout migration loop")

        while not self.stop_sync_and_migrate:
            # Sleep for a period of time and analyze the instance status
            await asyncio.sleep(self.config.psrl.sync_and_mig_strategy.check_interval_in_ms / 1000)

            have_syncing_instance = False
            sync_instance_ids = []
            for instance_id, engine_stats in self.instance_to_engine_status.items():
                # Ignore validate instances for weight synchronization
                if instance_id >= self.rollout_wg_size:
                    continue
                # Skip instances currently under a sync_with_ps call (swap or in-place).
                if self.enable_elastic and instance_id in self._syncing_locked_instance_ids:
                    have_syncing_instance = True
                    continue
                # Check whether engine status is stale (the instance is currently being synchronized with PS)
                if engine_stats.model_version <= self.instance_to_latest_stale_model_version.get(instance_id, -1):
                    have_syncing_instance = True
                    continue
                # We do not synchronize with PS if the router is currently routing requests
                if await self._is_routing():
                    # psrl_logger.info(f"Skipping synchronization with PS for instance
                    # {instance_id} because the router is currently routing requests")
                    continue
                # Check whether instance version lags behind PS version
                if self.instance_to_model_version.get(instance_id, 0) == self.ps_model_version:
                    continue
                # Check whether current instance workload is empty (forbid partial rollout)
                # or satisfies the partial rollout policy
                if self.config.psrl.partial_rollout.enable:
                    if not (await self.check_should_sync(instance_id)):
                        continue
                else:
                    if engine_stats.get_waiting_and_running_queue_size() > 0:
                        continue
                # Check whether the training side can seamlessly continue to train after the synchronization
                if self.config.psrl.sync_and_mig_strategy.sync.seamless_train_version >= self.ps_model_version:
                    if self.ps_model_version not in self.ready_buffers:
                        continue
                # Add the instance to the sync list
                sync_instance_ids.append(instance_id)
                """
                # NOTE(lhy): currently, we only synchronize with PS for one instance at a time
                # But the model pulling time can be overlapped
                break
                """

            if sync_instance_ids:
                await self.sync_with_ps(sync_instance_ids)
            elif not have_syncing_instance and self.config.psrl.sync_and_mig_strategy.mig.enable:
                # No instance is syncing with PS, check if migration is needed
                await self.check_and_migrate()

        psrl_logger.info("Status based model synchronization and rollout migration loop stopped.")

    # ------- FUNCTIONS FOR MODEL SYNCING -------

    # This is called by the PS manager to update the PS model version after pushing
    def set_elastic_executor(
        self,
        elastic_executor,
        role_name,
        model_name: str,
    ) -> None:
        """Inject the ElasticExecutor handle so sync_with_ps can attempt swap-based sync.

        Args:
            elastic_executor: Ray actor handle for the ElasticExecutor.
            role_name: PSRL_Role for this coordinator's role (e.g. PSRL_Role.Rollout).
            model_name: Model name string used to look up this coordinator in ElasticExecutor.
        """
        self._elastic_executor = elastic_executor
        self._elastic_exec_role_name = role_name
        self._elastic_exec_model_name = model_name
        psrl_logger.info(
            "RolloutCoordinator: ElasticExecutor registered for swap-based sync (role=%s model=%s).",
            getattr(role_name, "name", role_name),
            model_name,
        )

    def set_ps_model_version(self, version: int):
        """
        Set the current PS model version.

        This method updates the internal PS model version.

        Args:
            version (int): The new PS model version to set.
        """
        self.ps_model_version = version
        assert self.ps_model_version > 0, "PS model version must be greater than 0"
        assert (self.ps_model_version - 1) in self.ready_buffers, (
            "PS model version must be greater than the ready buffers"
        )
        self.ready_buffers.remove(self.ps_model_version - 1)
        psrl_logger.info(f"Updated PS model version to {version}")

    # This is called by the PS manager to update the rollout instance model version after pulling
    def set_rollout_instance_model_version(self, rollout_instance_id: int, version_tag: int):
        """
        Set the model version for a specific rollout instance.

        Args:
            rollout_instance_id (int): The ID of the rollout instance.
            version_tag (int): The model version tag to set for the instance.
        """
        old_version = self.instance_to_model_version.get(rollout_instance_id, None)
        self.instance_to_model_version[rollout_instance_id] = version_tag
        psrl_logger.info(
            f"Updated rollout instance {rollout_instance_id} model version: {old_version} -> {version_tag}"
        )

    def update_ready_buffer(self, ready_buffer: int):
        """
        Update the ready buffer.
        """
        self.ready_buffers.add(ready_buffer)
        psrl_logger.info(f"Updated ready buffers to: {self.ready_buffers}")

    async def _sync_instances_directly(
        self,
        instance_ids: list[int],
        *,
        wait_interrupted_partial_requests_loop_back: bool,
    ) -> None:
        """
        Synchronize instances without elastic swap or scaling operations.

        Dedicated validation instances are globally indexed after rollout
        instances but are not registered with ElasticExecutor. They must use
        this direct synchronization path.
        """
        with log_dual_events(
            f"Synchronize generation instances {instance_ids} with PS "
            "(model pull and request loopback are blocking for the coordinator)",
            psrl_logger,
            level=logging.INFO,
            event_type=EventType.OTHER,
        ):
            for instance_id in instance_ids:
                self.instance_to_latest_stale_model_version[instance_id] = self.instance_to_model_version.get(
                    instance_id, 0
                )
            target_ps_version = int(self.ps_model_version)
            transition_id = await self.rollout_router.begin_instance_transition.remote(instance_ids)
            await self.rollout_router.pause_routing.remote()
            psrl_logger.info("Paused routing for synchronization transition %d", transition_id)
            try:
                await self.rollout_router.update_currently_syncing_instances.remote(
                    instance_ids,
                    target_ps_version,
                )
                result = await self.exec_command(
                    Command(
                        type=CommandType.SYNC,
                        instance_ids=instance_ids,
                        curr_ps_model_version=target_ps_version,
                        wait_model_sync=True,
                    ),
                    blocking=True,
                )
                if result is None or result is False:
                    raise RuntimeError(f"SYNC command failed for instances {instance_ids}")
                if wait_interrupted_partial_requests_loop_back and self.config.psrl.partial_rollout.enable:
                    await self.rollout_router.wait_instance_transition.remote(transition_id)
            finally:
                await self.rollout_router.finish_instance_transition.remote(transition_id, resume=True)
                await self.rollout_router.resume_routing.remote()
                psrl_logger.info("Resumed routing after synchronization transition %d", transition_id)

    async def sync_with_ps(
        self,
        instance_ids: list[int],
        wait_model_sync: bool = False,
        wait_interrupted_partial_requests_loop_back: bool = True,
    ):
        """
        Synchronize with PS for the given instance IDs.
        """
        invalid_instance_ids = [
            instance_id for instance_id in instance_ids if instance_id < 0 or instance_id >= self.gen_wg_size
        ]
        if invalid_instance_ids:
            raise ValueError(
                f"Generation instance IDs must be in [0, {self.gen_wg_size}), got: {invalid_instance_ids!r}."
            )

        validation_sync_ids = [instance_id for instance_id in instance_ids if instance_id >= self.rollout_wg_size]
        if validation_sync_ids:
            await self._sync_instances_directly(
                validation_sync_ids,
                wait_interrupted_partial_requests_loop_back=wait_interrupted_partial_requests_loop_back,
            )

        instance_ids = [instance_id for instance_id in instance_ids if instance_id < self.rollout_wg_size]
        if not instance_ids:
            return

        if not self.enable_elastic:
            await self._sync_instances_directly(
                instance_ids,
                wait_interrupted_partial_requests_loop_back=wait_interrupted_partial_requests_loop_back,
            )
            return

        """
        If an ElasticExecutor is registered (via set_elastic_executor), the method first
        checks whether there is a free (asleep, non-conflicting) rollout instance available.

        - **Free instance found**: a swap is performed — the free instance is woken up
          (which automatically pulls the latest model weights inside the WAKE_UP handler),
          and the original syncing instances are then put to sleep.  No in-place weight
          pull is needed.

        - **No free instance**: falls back to the traditional in-place sync.  During the
          sync the affected instances are added to ``_syncing_locked_instance_ids`` so that
          concurrent elastic SLEEP/WAKE_UP commands from ElasticExecutor are rejected.
          After sync completes, the ElasticExecutor is notified to grant each instance an
          immunity window (preventing an immediate re-sleep by the scaling policy).

        Concurrent-call guard:
          The coordinator's asyncio event loop yields during ``await perform_sync_swap``
          and ``await exec_command``, which allows the sync_and_migrate loop to be
          rescheduled.  Without a guard, it would call sync_with_ps again for the same
          instances, producing duplicate WAKE_UP commands that trigger an AssertionError
          in gen_worker (active_tasks != 0 when interrupt_generation=False).  We prevent
          this by locking instance IDs at the top of this method for both swap and
          in-place paths.
        """
        # Only rollout instances (index < rollout_wg_size) participate in elastic swap.
        rollout_sync_ids = [i for i in instance_ids if i < self.rollout_wg_size]

        # Drop any instances already locked by a concurrent sync_with_ps call.
        already_locked = [i for i in rollout_sync_ids if i in self._syncing_locked_instance_ids]
        if already_locked:
            psrl_logger.info(
                "sync_with_ps: skipping instances %s already under sync-lock (concurrent call suppressed).",
                already_locked,
            )
            rollout_sync_ids = [i for i in rollout_sync_ids if i not in self._syncing_locked_instance_ids]
            if not rollout_sync_ids:
                return

        # Acquire sync-lock for ALL paths (swap and in-place) immediately.
        sync_operation_id = self._next_sync_operation_id
        self._next_sync_operation_id += 1
        for instance_id in rollout_sync_ids:
            self._syncing_locked_instance_ids.add(instance_id)
            self._syncing_lock_owner_by_instance[instance_id] = sync_operation_id
            psrl_logger.info(
                "sync_with_ps: acquired sync-lock for instance %d (operation=%d).",
                instance_id,
                sync_operation_id,
            )

        _swap_done = False
        sync_policy_gate_token = None
        try:
            if self._elastic_executor is not None:
                sync_policy_gate_token = await self._elastic_executor.enter_inplace_sync_policy_gate.remote(
                    self._elastic_exec_role_name,
                    self._elastic_exec_model_name,
                    rollout_sync_ids,
                )
            # Wake a free instance with current weights before sleeping the stale instance.
            if self._elastic_executor is not None and rollout_sync_ids:
                free_ids: list[int] = await self._elastic_executor.get_free_instances.remote(
                    self._elastic_exec_role_name,
                    self._elastic_exec_model_name,
                    rollout_sync_ids,  # exclude the instances being synced
                )
                if len(free_ids) >= len(rollout_sync_ids):
                    psrl_logger.info(
                        "sync_with_ps: swap-based sync . waking free instances %s, "
                        "sleeping syncing instances %s (PS version %d).",
                        free_ids[: len(rollout_sync_ids)],
                        rollout_sync_ids,
                        self.ps_model_version,
                    )
                    swap_pairs = list(zip(free_ids[: len(rollout_sync_ids)], rollout_sync_ids))
                    swap_all_succeeded = await self._elastic_executor.perform_sync_swaps.remote(
                        self._elastic_exec_role_name,
                        self._elastic_exec_model_name,
                        swap_pairs,
                        sync_operation_id,
                        sync_policy_gate_token,
                    )
                    if not swap_all_succeeded:
                        psrl_logger.info(
                            "sync_with_ps: swap-based sync not fully applied. continue with in-place sync path."
                        )
                    else:
                        psrl_logger.info(
                            "sync_with_ps: swap-based sync completed for instances %s.",
                            rollout_sync_ids,
                        )
                        _swap_done = True
                        return  # finally block runs, releases locks. immunity set by _scale_up_instance

                psrl_logger.info(
                    "sync_with_ps: no free instances available for swap (free=%s). "
                    "falling back to in-place sync for %s.",
                    free_ids,
                    rollout_sync_ids,
                )

            # The worker updates its version on pull. Update the stale-version boundary here to reject old snapshots.
            with log_dual_events(
                f"Synchronize rollout instances {instance_ids} with PS "
                f"(model pull is {'non-blocking' if not wait_model_sync else 'blocking'} "
                f"for the coordinator)",
                psrl_logger,
                level=logging.INFO,
                event_type=EventType.OTHER,
            ):
                for instance_id in instance_ids:
                    self.instance_to_latest_stale_model_version[instance_id] = self.instance_to_model_version.get(
                        instance_id, 0
                    )

                # For in-place sync fallback: sleep -> wake_up only for target instances.
                # Skip instances already in vLLM sleep mode.
                sleeping_flags = await asyncio.gather(
                    *[self.gen_wg_list[i].execute_rank_zero_async("is_rollout_engine_sleeping") for i in instance_ids]
                )
                in_place_sync_ids: list[int] = []
                skipped_sleeping: list[int] = []
                for instance_id, is_sleeping in zip(instance_ids, sleeping_flags):
                    if is_sleeping:
                        skipped_sleeping.append(instance_id)
                    else:
                        in_place_sync_ids.append(instance_id)

                if skipped_sleeping:
                    psrl_logger.info(
                        "In-place sync fallback: skip %s sleeping instance(s) %s.",
                        len(skipped_sleeping),
                        skipped_sleeping,
                    )

                if in_place_sync_ids:
                    wake_up_completed = False
                    transition_id: int | None = None
                    try:
                        # Recheck sleep status after waiting on the policy gate to avoid racing another sleep
                        # transition.
                        sleeping_flags_2 = await asyncio.gather(
                            *[
                                self.gen_wg_list[i].execute_rank_zero_async("is_rollout_engine_sleeping")
                                for i in in_place_sync_ids
                            ]
                        )
                        skipped_asleep_double: list[int] = []
                        still_awake: list[int] = []
                        for instance_id, is_sleeping in zip(in_place_sync_ids, sleeping_flags_2):
                            if is_sleeping:
                                skipped_asleep_double.append(instance_id)
                            else:
                                still_awake.append(instance_id)

                        if skipped_asleep_double:
                            psrl_logger.info(
                                "In-place sync fallback: double-check skip %s instance(s) now sleeping %s.",
                                len(skipped_asleep_double),
                                skipped_asleep_double,
                            )
                            for iid in skipped_asleep_double:
                                await self.rollout_router.update_currently_syncing_instances.remote(
                                    [iid],
                                    self.instance_to_model_version.get(iid, 0),
                                )

                        in_place_sync_ids = still_awake

                        if in_place_sync_ids:
                            transition_id = await self.rollout_router.begin_instance_transition.remote(
                                in_place_sync_ids
                            )
                            sleep_result = await self.exec_command(
                                Command(
                                    type=CommandType.SLEEP,
                                    instance_ids=in_place_sync_ids,
                                    sync_operation_id=sync_operation_id,
                                ),
                                blocking=True,
                            )
                            if sleep_result is not True:
                                raise RuntimeError(
                                    f"In-place sync SLEEP failed for {in_place_sync_ids}: {sleep_result!r}"
                                )
                            psrl_logger.info(
                                "Executed SLEEP command for in-place sync fallback: %s",
                                in_place_sync_ids,
                            )

                            target_ps_version = int(self.ps_model_version)
                            wake_result = await self.exec_command(
                                Command(
                                    type=CommandType.WAKE_UP,
                                    instance_ids=in_place_sync_ids,
                                    target_ps_model_version=target_ps_version,
                                    resume_instances=False,
                                    sync_operation_id=sync_operation_id,
                                ),
                                blocking=True,
                            )
                            if wake_result is not True:
                                raise RuntimeError(
                                    f"In-place sync WAKE_UP failed for {in_place_sync_ids}: {wake_result!r}"
                                )
                            wake_up_completed = True
                            psrl_logger.info(
                                "Executed WAKE_UP command for in-place sync fallback: %s",
                                in_place_sync_ids,
                            )

                            if wait_interrupted_partial_requests_loop_back and self.config.psrl.partial_rollout.enable:
                                psrl_logger.info(
                                    "Waiting for interrupted requests in transition %d to loop back",
                                    transition_id,
                                )
                                await self.rollout_router.wait_instance_transition.remote(transition_id)
                                psrl_logger.info(
                                    "Interrupted requests in transition %d have looped back",
                                    transition_id,
                                )
                        else:
                            psrl_logger.info(
                                "In-place sync fallback: no runnable instances after double-check "
                                "(all targets asleep)."
                            )
                    finally:
                        if transition_id is not None:
                            await self.rollout_router.finish_instance_transition.remote(
                                transition_id,
                                resume=True,
                                resume_instance_ids=in_place_sync_ids,
                            )
                            if not wake_up_completed:
                                psrl_logger.warning(
                                    "In-place sync fallback exited early. released routing transition for: %s",
                                    in_place_sync_ids,
                                )
                else:
                    psrl_logger.info("In-place sync fallback: no runnable instances to sleep/wake_up.")

        finally:
            # Release sync-lock for all paths (swap and in-place).
            for instance_id in rollout_sync_ids:
                self._syncing_locked_instance_ids.discard(instance_id)
                self._syncing_lock_owner_by_instance.pop(instance_id, None)
                psrl_logger.info("sync_with_ps: released sync-lock for instance %d.", instance_id)
            try:
                # Grant immunity only for in-place sync. swap path grants immunity via
                # _set_instance_immunity inside _scale_up_instance for the new instance.
                if not _swap_done and self._elastic_executor is not None:
                    await asyncio.gather(
                        *[
                            self._elastic_executor.set_instance_immunity.remote(
                                self._elastic_exec_role_name,
                                self._elastic_exec_model_name,
                                instance_id,
                            )
                            for instance_id in rollout_sync_ids
                        ]
                    )
            finally:
                if sync_policy_gate_token is not None:
                    await self._elastic_executor.leave_inplace_sync_policy_gate.remote(
                        self._elastic_exec_role_name,
                        self._elastic_exec_model_name,
                        rollout_sync_ids,
                        sync_policy_gate_token,
                    )

    async def check_no_activate_tasks(self, instance_id: int) -> bool:
        """
        Check whether the instance has no active tasks.
        """
        active_task_num = await self.gen_wg_list[instance_id].execute_rank_zero_async("get_active_task_num")
        return active_task_num == 0

    async def check_should_sync(self, instance_id: int) -> bool:
        """
        Check whether to synchronize with PS for the instance.
        """
        assert self.config.psrl.sync_and_mig_strategy.method == "status_based", (
            "Partial rollout is only supported for status-based sync strategy"
        )
        assert self.config.psrl.status_collection.enable, (
            "Partial rollout is only supported when status collection is enabled"
        )
        assert self.config.psrl.partial_rollout.enable, "Partial rollout is not enabled"
        # TODO(lhy): Route through a global router.
        return await self.rollout_router.check_should_sync.remote(instance_id, self.ps_model_version)

    async def check_and_migrate(self, wait_interrupted_partial_requests_loop_back: bool = True):
        """
        Check if any instance is starving and do migration if necessary.
        """
        assert self.config.psrl.sync_and_mig_strategy.mig.enable, "Rollout migration is not enabled"
        assert self.config.psrl.status_collection.enable, (
            "Rollout migration is only supported when status collection is enabled"
        )
        # psrl_logger.info("Checking if any instance is starving and doing migration if necessary")
        migrate_instance_ids = await self.rollout_router.check_should_migrate.remote()
        if migrate_instance_ids:
            with log_dual_events(
                f"Migrating instances {migrate_instance_ids}",
                psrl_logger,
                level=logging.INFO,
                event_type=EventType.OTHER,
            ):
                transition_id = await self.rollout_router.begin_instance_transition.remote(migrate_instance_ids)
                await self.rollout_router.pause_routing.remote()
                psrl_logger.info("Interrupted routing for migration transition %d", transition_id)
                try:
                    await self.exec_command(
                        Command(
                            type=CommandType.ABORT,
                            instance_ids=migrate_instance_ids,
                        ),
                        blocking=True,
                    )
                    if wait_interrupted_partial_requests_loop_back:
                        await self.rollout_router.wait_instance_transition.remote(transition_id)
                finally:
                    await self.rollout_router.finish_instance_transition.remote(transition_id, resume=True)
                    await self.rollout_router.resume_routing.remote()
                    psrl_logger.info("Resumed routing after migration transition %d", transition_id)
