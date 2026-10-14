"""RolloutCoordinator composes loop mixins into the public coordinator API."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time

import aiohttp
import ray
from ray.actor import ActorHandle

from psrl.utils.common.http_utils import find_available_port, get_host_info
from psrl.utils.elastic.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.logger import DualOutputHandler
from psrl.utils.server.command import CommandExtension
from psrl.workers.gen.smg_adapter import build_weight_version_updates
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.gen.utils import DEFAULT_MAX_CONNECTIONS, DEFAULT_TIMEOUT, RolloutInstanceId
from psrl.workers.gen.zmq_queue import ZMQPullQueue

from . import base as base_module
from . import command_loop as command_loop_module
from . import status_loop as status_loop_module
from .base import CoordinatorBase
from .command_loop import CommandHandlerMixin
from .session import thunder_agent as thunder_agent_module
from .session.thunder_agent import ThunderAgentSessionMixin
from .status_loop import StatusMixin
from .sync_and_migrate import GreedySyncMixin, StatusBasedSyncMixin, SyncAndMigrateMixin
from .sync_and_migrate import greedy as greedy_module
from .sync_and_migrate import status_based as status_based_module
from .sync_and_migrate import sync_and_migrate_mixin as sync_and_migrate_module

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Loop modules use distinct loggers. Modules targeting one file must share a
# handler because concurrent write-mode handlers would truncate each other.
_LOG_FILE_TO_MODULE_LOGGERS = {
    "RolloutCoordinator": [command_loop_module.psrl_logger, base_module.psrl_logger],
    "SyncAndMigStrategy": [
        sync_and_migrate_module.psrl_logger,
        greedy_module.psrl_logger,
        status_based_module.psrl_logger,
    ],
    "Status": [status_loop_module.psrl_logger],
    "SessionStrategy": [thunder_agent_module.psrl_logger],
}


class RolloutCoordinator(
    CommandHandlerMixin,
    StatusMixin,
    SyncAndMigrateMixin,
    GreedySyncMixin,
    StatusBasedSyncMixin,
    ThunderAgentSessionMixin,
    CoordinatorBase,
    CommandExtension,
):
    def __init__(
        self,
        config,
        ps_manager: ray.actor.ActorHandle,
        rollout_gateway_url: str,
        session_router_url: str | None = None,
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
        - Session hang/continue scheduling (ThunderAgent port)

        Args:
            config: Configuration object containing PSRL settings
            rollout_gateway_url: HTTP base URL of the SMG rollout gateway.
            session_router_url: HTTP base URL of the SessionRouter (for session
                hang/continue control). None disables the thunder_agent loop.
        """
        super().__init__()

        self.config = config
        self.staleness = self.config.psrl.staleness
        self.ps_manager = ps_manager

        # Rollout replica tracking
        self.rollout_replicas = {}
        self.server_handles = {}
        self.replica_operation_locks: dict[str, asyncio.Lock] = {}
        self.routing_operation_lock = asyncio.Lock()
        self.replica_ids = set()
        self.instance_ids = set()
        # For convenience, maintain separate lists for rollout and validate instances
        self.tag_to_replica_ids = {"rollout": set(), "validate": set()}

        if not rollout_gateway_url:
            raise ValueError("Rollout gateway URL must not be empty.")
        self.rollout_gateway_url = rollout_gateway_url.rstrip("/")
        connector = aiohttp.TCPConnector(
            limit=DEFAULT_MAX_CONNECTIONS,
            limit_per_host=DEFAULT_MAX_CONNECTIONS,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(total=DEFAULT_TIMEOUT)
        self.gateway_client = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
        )

        # Stats collection
        status_host = ray.util.get_node_ip_address().strip("[]")
        status_port = find_available_port(base_port=28000)
        self.status_sink_endpoint = f"tcp://{status_host}:{status_port}"
        self.status_queue = ZMQPullQueue(endpoint=self.status_sink_endpoint)
        self.replica_idx_to_replica_id: dict[int, str] = {}

        # Background event handler
        self.running_loop = None
        self.command_handler_task = None
        self.sync_task = None
        self.model_sync_tasks: set[asyncio.Task] = set()
        self.replica_sync_tasks: dict[str, asyncio.Task] = {}
        self.process_status_queue_task = None
        self.sync_status_to_router_task = None
        self.stats_recorder_task = None
        self.stop_command_handler = False
        self.stop_sync_and_migrate = False
        self.stop_process_status_queue = False
        self.stop_sync_status_to_router = False
        self.stop_stats_recorder = False

        # Asyncio event loop order control
        self._is_init_nixl_client = asyncio.Event()

        # Version tracking
        # Track the model version of each instance
        self.instance_to_model_version: dict[RolloutInstanceId, int] = {}
        # Tracks model version each instance will have after its next sync.
        self.instance_to_version_after_sync: dict[RolloutInstanceId, int] = {}

        self.ps_model_version = 0  # Current model version in the parameter server
        self.ready_buffers = set()  # The set of ready buffers

        # Engine status tracking
        # Track the latest engine stats of each instance
        self.instance_to_engine_status: dict[RolloutInstanceId, EngineStats] = {}
        self.sleeping_replica_ids: set[str] = set()
        # Absolute KV-cache token capacity per instance (max_concurrency * max_model_len,
        # obtained via get_total_kv_cache_tokens). Used by the hang/continue scheduler.
        self.instance_to_total_kv_tokens: dict[RolloutInstanceId, int] = {}

        # --- Session hang/continue scheduling (ThunderAgent port) ---
        self.session_router_url = session_router_url.rstrip("/") if session_router_url else None
        self._thunder_agent_cfg = self.config.psrl.rollout_coordination.session_strategy.thunder_agent
        self._thunder_agent_enabled = bool(self._thunder_agent_cfg.enable and self.session_router_url is not None)
        self.thunder_agent_task = None
        self.stop_thunder_agent = False
        self._thunder_scheduler = None
        self._session_client: aiohttp.ClientSession | None = None

        # LMCache P2P Controller subprocess handle (started by init_lmcache_p2p, if enabled).
        self._lmcache_controller_proc: subprocess.Popen | None = None
        self._lmcache_controller_url: str | None = None

        # Route each loop logger to its configured file.
        self.log_prefix = "RolloutCoordinator"
        self._attach_loop_log_handlers()
        psrl_logger.info("Initialized RolloutCoordinator")

        # Stats recorder (opt-in)
        self._stats_recorder = None
        if self.config.psrl.status_collection.stats_recorder.enable:
            from psrl.workers.gen.stats_recorder import StatsRecorder

            self._stats_recorder = StatsRecorder(
                self.config.psrl,
                os.path.expanduser(self.config.psrl.logging_path),
            )
            self._stats_recorder.write_config(
                routing_strategy=self.config.psrl.rollout_coordination.routing_strategy.method,
                partial_rollout=self.config.psrl.rollout_coordination.partial_rollout.enable,
            )

    def _attach_loop_log_handlers(self) -> None:
        """
        Route each loop module logger to its configured file.

        Loggers targeting one file share a `DualOutputHandler` to prevent
        write-mode handles from truncating each other.
        """
        logging_path = self.config.psrl.logging_path

        # coordinator.py's own logger shares RolloutCoordinator.log with the
        # command/base loggers, so create that file's handler first and reuse it.
        coordinator_handler = self._ensure_dual_handler(psrl_logger, logging_path, "RolloutCoordinator")
        for log_prefix, module_loggers in _LOG_FILE_TO_MODULE_LOGGERS.items():
            shared_handler = coordinator_handler if log_prefix == "RolloutCoordinator" else None
            for module_logger in module_loggers:
                shared_handler = self._ensure_dual_handler(
                    module_logger, logging_path, log_prefix, handler=shared_handler
                )

    @staticmethod
    def _ensure_dual_handler(target_logger, logging_path, log_prefix, handler=None):
        """Attach ``handler`` (or a fresh DualOutputHandler for ``log_prefix``) to
        ``target_logger`` unless a DualOutputHandler is already present, and return
        the handler so callers can reuse it across loggers sharing one file."""
        existing = next((h for h in target_logger.handlers if isinstance(h, DualOutputHandler)), None)
        if existing is not None:
            return existing
        if handler is None:
            handler = DualOutputHandler(logging_path, log_prefix)
        target_logger.addHandler(handler)
        return handler

    def add_worker(
        self,
        rollout_replica,
        server_handle,
        replica_id: str,
        dp_size: int,
        is_validate: bool = False,
        model_version: int = 0,
    ):
        """Add a rollout replica to the coordinator.

        Args:
            rollout_replica: Rollout replica object
            server_handle: Handle to the rollout replica actor
            replica_id (str): ID of the rollout replica
            dp_size (int): Number of instances in the rollout replica
        """
        self.rollout_replicas[replica_id] = rollout_replica
        self.server_handles[replica_id] = server_handle
        self.replica_ids.add(replica_id)
        self.instance_ids.update([(replica_id, i) for i in range(dp_size)])
        self.replica_idx_to_replica_id[rollout_replica.replica_rank] = replica_id

        tag = "validate" if is_validate else "rollout"
        self.tag_to_replica_ids[tag].add(replica_id)

        # Initialize version_after_sync for newly registered instances
        for i in range(dp_size):
            instance_id = (replica_id, i)
            self.instance_to_model_version[instance_id] = model_version
            self.instance_to_version_after_sync[instance_id] = model_version
            self.instance_to_engine_status[instance_id] = EngineStats(
                replica_idx=rollout_replica.replica_rank,
                data_parallel_rank=i,
                model_version=model_version,
                snapshot=EngineStats.get_default_snapshot(),
            )

    def get_status_sink_endpoint(self) -> str:
        return self.status_sink_endpoint

    def get_all_instance_ids(self) -> list:
        """Return all registered (replica_id, dp_rank) instance IDs, sorted for determinism."""
        return sorted(self.instance_ids)

    def _get_sleep_level(self) -> int:
        """Sleep level for server.sleep(). Rollout uses level=2 (full GPU release). Subclasses may override."""
        return 2

    async def _do_sleep_instance(self, replica_id: str) -> None:
        """Execute sleep on one replica. Rollout path uses nixl_sleep. Subclasses may override."""
        await self.server_handles[replica_id].nixl_sleep.remote(level=self._get_sleep_level())

    async def _do_wake_up_instance(self, replica_id: str) -> None:
        """Restore memory and committed weights before publishing the replica version."""
        server = self.server_handles[replica_id]
        await server.nixl_wake_up.remote()
        actual_version = await server.pull_model_for_sync.remote(self.ps_model_version)
        instance_ids = sorted(instance_id for instance_id in self.instance_ids if instance_id[0] == replica_id)
        await self._publish_weight_version_updates(build_weight_version_updates(instance_ids, actual_version))
        for instance_id in instance_ids:
            self.set_rollout_instance_model_version(instance_id, actual_version)
            self.instance_to_version_after_sync[instance_id] = actual_version

    def world_size(self):
        """Get the total world size (number of rollout and validate instances)."""
        return sum([rollout_replica.world_size for rollout_replica in self.rollout_replicas.values()])

    def _tag_to_server(self, tag: str):
        """Get the rollout server handles for a given tag.

        Args:
            tag (str): Tag to specify which instances to get ('rollout', 'validate', 'all')
        Returns:
            list: List of worker group handles
        """
        if tag in ["rollout", "validate"]:
            replica_ids = self.tag_to_replica_ids[tag]
        elif tag == "all":
            replica_ids = self.replica_ids
        else:
            raise ValueError(f"Unknown tag {tag} for getting server handles and number")
        server_handles = [self.server_handles[replica_id] for replica_id in replica_ids]
        return server_handles

    def _get_ordered_server_items(self, tag: str = "all") -> list[tuple[int, str, ActorHandle]]:
        """Return `(replica_idx, replica_id, server_handle)` sorted by replica index."""
        if tag in ["rollout", "validate"]:
            replica_ids = self.tag_to_replica_ids[tag]
        elif tag == "all":
            replica_ids = self.replica_ids
        else:
            raise ValueError(f"Unknown tag {tag} for getting server handles")
        return [
            (self.rollout_replicas[replica_id].replica_rank, replica_id, self.server_handles[replica_id])
            for replica_id in sorted(replica_ids, key=lambda rid: self.rollout_replicas[rid].replica_rank)
        ]

    async def init_nixl_client(self):
        """Init the NIXL client on rollout and validate instances."""
        futures = []
        for server_handle in self.server_handles.values():
            futures.append(server_handle.init_nixl_client.remote())
        await asyncio.gather(*futures)
        psrl_logger.info(f"Initialized NIXL clients: replica_count={len(self.server_handles)}.")
        self._is_init_nixl_client.set()

    async def nixl_protocol(self, full_tag: str = "all"):
        """Run the NIXL server protocol on rollout and validate instances.

        Args:
            full_tag (str): Tag to specify which instances to run the protocol
                            in 'full' mode ('rollout', 'validate', 'all')
        """
        await self._is_init_nixl_client.wait()

        if full_tag == "all":
            rollout_tag = "full"
            validate_tag = "full"
        elif full_tag == "rollout":
            rollout_tag = "full"
            validate_tag = "meta"
        elif full_tag == "meta":
            rollout_tag = "meta"
            validate_tag = "meta"
        elif full_tag == "validate":
            rollout_tag = "meta"
            validate_tag = "full"
        else:
            raise ValueError(f"Unknown full_tag {full_tag} for nixl_protocol")

        futures = []
        for replica_id in self.tag_to_replica_ids["rollout"]:
            futures.append(self.server_handles[replica_id].nixl_protocol.remote(rollout_tag))
        for replica_id in self.tag_to_replica_ids["validate"]:
            futures.append(self.server_handles[replica_id].nixl_protocol.remote(validate_tag))
        await asyncio.gather(*futures)

    async def nixl_convert_params(self):
        """Convert the model parameters to unified format on rollout and validate instances."""
        await self._is_init_nixl_client.wait()
        futures = []
        for server_handle in self.server_handles.values():
            futures.append(server_handle.nixl_convert_params.remote())
        await asyncio.gather(*futures)

    async def initial_pull_from_ps(self, tag: str = "rollout") -> None:
        futures = [server_handle.pull_model.remote() for server_handle in self._tag_to_server(tag)]
        await asyncio.gather(*futures)
        psrl_logger.info(f"Initial parameter server pull complete: replicas={len(futures)}, tag={tag!r}.")
        # Pulled replicas must advertise the parameter server version to the gateway.
        pulled_replica_ids = self.replica_ids if tag == "all" else self.tag_to_replica_ids[tag]
        pulled_instance_ids = [inst_id for inst_id in self.instance_ids if inst_id[0] in pulled_replica_ids]
        for instance_id in pulled_instance_ids:
            self.instance_to_model_version[instance_id] = self.ps_model_version
            self.instance_to_version_after_sync[instance_id] = self.ps_model_version
        if pulled_instance_ids:
            updates = build_weight_version_updates(pulled_instance_ids, self.ps_model_version)
            await self._publish_weight_version_updates(updates)
            psrl_logger.info(
                f"Published initial weight version: version={self.ps_model_version}, "
                f"instances={len(pulled_instance_ids)}, tag={tag!r}."
            )

    async def sleep(self, tag: str = "all"):
        """Make rollout instances sleep and release GPU memory.

        Args:
            tag (str): Tag to specify which instances to sleep ('rollout', 'validate', 'all')
        """
        replica_ids = self.replica_ids if tag == "all" else self.tag_to_replica_ids[tag]
        await self._sleep_replicas([iid for iid in self.instance_ids if iid[0] in replica_ids])

    async def start_busy_loop(self):
        """
        Start the background event loops for command handling and status synchronization.

        This method:
        1. Starts a background task for handling commands (abort, sync, etc.).
        2. Optionally starts tasks for processing status queues of each rollout instance.
        3. Starts a task to broadcast the engine status to the agent loop workers (i.e., router).
        4. Starts a task to synchronize rollout instances with PS.
        """
        if self.command_handler_task is not None and not self.command_handler_task.done():
            return

        # Start the background tasks
        self.running_loop = asyncio.get_running_loop()
        self.command_handler_task = self.running_loop.create_task(self._command_handler_loop())
        self.command_handler_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

        # Start the status collection tasks
        if self.config.psrl.status_collection.enable:
            self.process_status_queue_task = self.running_loop.create_task(self._process_status_queue())
            self.process_status_queue_task.add_done_callback(lambda f: f.result())
        # Start the task to broadcast the engine status to the router
        self.sync_status_to_router_task = self.running_loop.create_task(self._sync_status_to_router())
        self.sync_status_to_router_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        # Start the model synchronization and rollout migration loop
        if self.config.psrl.rollout_coordination.sync_and_mig_strategy.method == "greedy":
            self.sync_task = self.running_loop.create_task(self._greedy_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        elif self.config.psrl.rollout_coordination.sync_and_mig_strategy.method == "status_based":
            assert self.config.psrl.status_collection.enable, (
                "Status-based sync strategy is only supported when status collection is enabled"
            )
            self.sync_task = self.running_loop.create_task(self._status_based_sync_and_migrate_loop())
            self.sync_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks
        else:
            raise NotImplementedError(
                f"Sync strategy {self.config.psrl.rollout_coordination.sync_and_mig_strategy.method} is not supported"
            )
        # Check if rollout migration is enabled
        if self.config.psrl.rollout_coordination.sync_and_mig_strategy.mig.enable:
            assert self.config.psrl.status_collection.enable, (
                "Rollout migration is only supported when status collection is enabled"
            )
            assert self.config.psrl.rollout_coordination.partial_rollout.enable, (
                "Rollout migration is only supported when partial rollout is enabled"
            )

        # Start the stats recorder loop (opt-in)
        if self.config.psrl.status_collection.stats_recorder.enable:
            self.stats_recorder_task = self.running_loop.create_task(self._stats_recorder_loop())
            self.stats_recorder_task.add_done_callback(lambda f: f.result())

        # Start the session hang/continue scheduling loop (opt-in, ThunderAgent port)
        if self._thunder_agent_enabled:
            self.thunder_agent_task = self.running_loop.create_task(self._thunder_agent_loop())
            self.thunder_agent_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        """
        Stop all background tasks and clean up resources.

        This method gracefully shuts down:
        - Command handler task
        - Engine status sync task
        - Stats recorder task (if enabled)
        """
        # Stop the background tasks
        self.stop_command_handler = True
        self.stop_sync_and_migrate = True
        self.stop_sync_status_to_router = True
        self.stop_process_status_queue = True
        self.stop_stats_recorder = True
        self.stop_thunder_agent = True

        tasks = [
            self.command_handler_task,
            self.process_status_queue_task,
            self.sync_status_to_router_task,
            self.sync_task,
            self.stats_recorder_task,
            self.thunder_agent_task,
            *self.model_sync_tasks,
        ]
        results = await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
        if self._stats_recorder is not None:
            self._stats_recorder.close()
        psrl_logger.info("All background tasks have been stopped.")
        self.status_queue.close()
        if not self.gateway_client.closed:
            await self.gateway_client.close()
        if self._session_client is not None and not self._session_client.closed:
            await self._session_client.close()
        psrl_logger.info("Cleaned up resources in RolloutCoordinator.")
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise RuntimeError(f"Coordinator background tasks failed during shutdown: {failures!r}.") from failures[0]

    async def get_router_backlog_size(self) -> int:
        """
        Return queued training demand for elastic rollout capacity.
        """
        t_enter = time.monotonic()
        model_tag = str(self.config.gen_actor_rollout_ref.model.path).rstrip("/").split("/")[-1]
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_enter model=%s elapsed_since_entry_s=0.000",
            model_tag,
        )
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_before_router_rpc model=%s since_enter_s=%.3f",
            model_tag,
            time.monotonic() - t_enter,
        )
        t_rpc = time.monotonic()
        pending = len(await self._get_training_request_snapshot())
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RolloutCoordinator_after_router_rpc model=%s pending=%d router_rpc_s=%.3f since_enter_s=%.3f",
            model_tag,
            pending,
            time.monotonic() - t_rpc,
            time.monotonic() - t_enter,
        )
        return pending

    # This is called by the PS manager to update the PS model version after pushing
    def set_ps_model_version(self, version: int):
        """
        Set the current PS model version.

        This method updates the internal PS model version.

        Args:
            version (int): The new PS model version to set.
        """
        self.ps_model_version = version
        assert self.ps_model_version > 0, "PS model version must be greater than 0."
        # NOTE(claude): Skip the ready-buffer check until resume consumes its first buffer.
        if self.ready_buffers:
            assert (self.ps_model_version - 1) in self.ready_buffers, (
                "PS model version must be greater than the ready buffers."
            )
            self.ready_buffers.remove(self.ps_model_version - 1)
        psrl_logger.info(f"Updated PS model version to {version}")

    def init_ps_model_version(self, version: int):
        """
        Initialize the PS model version during resume without the ready_buffers assertion.

        Unlike set_ps_model_version (which expects the previous buffer to be consumed),
        this method is used during checkpoint resume where no buffers have been consumed yet.

        Args:
            version (int): The PS model version to initialize to.
        """
        self.ps_model_version = version
        psrl_logger.info(f"Initialized resume PS model version: version={version}.")

    # This is called by the PS manager to update the rollout instance model version after pulling
    def set_rollout_instance_model_version(self, rollout_instance_id: RolloutInstanceId, version_tag: int):
        """
        Set the model version for a specific rollout instance.

        Args:
            rollout_instance_id (int): The ID of the rollout instance.
            version_tag (int): The model version tag to set for the instance.
        """
        old_version = self.instance_to_model_version.get(rollout_instance_id, None)
        self.instance_to_model_version[rollout_instance_id] = version_tag
        if rollout_instance_id in self.instance_to_engine_status:
            self.instance_to_engine_status[rollout_instance_id].model_version = version_tag
        psrl_logger.info(
            f"Updated rollout model version: instance_id={rollout_instance_id!r}, "
            f"old_version={old_version!r}, new_version={version_tag}."
        )

    def update_ready_buffer(self, ready_buffer: int):
        """
        Update the ready buffer.
        """
        self.ready_buffers.add(ready_buffer)
        psrl_logger.info(f"Updated ready buffers to: {self.ready_buffers}")

    # ------- FUNCTIONS FOR LMCACHE P2P -------

    async def init_lmcache_p2p(self) -> None:
        """
        Broadcast the Controller URL to all rollout server actors.

        Requires start_lmcache_controller() to have been called first.
        Must be called after init_model() has completed on all instances.
        """
        assert self._lmcache_controller_url is not None, (
            "start_lmcache_controller() must be called before init_lmcache_p2p()"
        )
        controller_url = self._lmcache_controller_url

        server_items = self._get_ordered_server_items("all")
        futures = [
            server_handle.set_lmcache_controller_url.remote(controller_url) for _, _, server_handle in server_items
        ]
        await asyncio.gather(*futures)
        psrl_logger.info(f"Broadcast LMCache P2P controller: url={controller_url!r}, replicas={len(server_items)}.")

        # After Controller URL is set and Workers are registered, broadcast the
        # peer registry so server actors can bypass the Controller for KV transfers.
        await self._broadcast_peer_registry()

    async def _broadcast_peer_registry(self) -> None:
        """
        Query the Controller for each replica's per-rank peer_init_url and worker ZMQ
        URLs, then broadcast to all server actors for direct transfer bypass.

        This enables `kv_transfer_direct()` on each server actor to send MoveWorkerMsg
        directly to each local LMCacheWorker via ZMQ, eliminating the Controller as
        a centralized bottleneck under burst transfer scenarios. KV is sharded per
        rank (TP heads, PP layers), so the registry maps instance_id → rank-sorted
        list of peer_init_url, and each replica is handed its own rank-sorted list of
        local worker ZMQ URLs.
        """
        from psrl.utils.common.http_utils import init_http_client, post

        controller_url = self._lmcache_controller_url
        assert controller_url, "start_lmcache_controller() must be called first."

        # Ensure HTTP client is initialized in the Coordinator process.
        init_http_client(server_concurrency=4, rollout_engine_num=1)

        server_items = self._get_ordered_server_items("all")

        # Registry list indices must match global worker ranks.
        peer_registry: dict[str, list[str]] = {}
        per_replica_worker_zmq_urls: dict[int, list[str]] = {}

        for replica_idx, _, _ in server_items:
            instance_id = f"psrl_instance_{replica_idx}"
            resp = await post(
                f"{controller_url}/query_worker_info",
                {"instance_id": instance_id},
                max_retries=3,
            )
            assert resp and "worker_infos" in resp and resp["worker_infos"], (
                f"[LMCache] query_worker_info returned empty for {instance_id!r}. "
                f"Workers may not have registered yet. resp={resp!r}"
            )
            # Empty peer URLs mark ranks without a P2P endpoint.
            infos = sorted(resp["worker_infos"], key=lambda wi: wi.get("worker_id", 0))
            peer_registry[instance_id] = [wi.get("peer_init_url", "") for wi in infos]
            per_replica_worker_zmq_urls[replica_idx] = [f"{wi.get('ip', '')}:{wi.get('port', 0)}" for wi in infos]

        assert peer_registry, (
            "[LMCache] Peer registry is empty after querying Controller. "
            "No instances have peer_init_url. P2P is not configured correctly."
        )

        # Broadcast to each server actor with the full registry plus that replica's
        # own rank-sorted worker ZMQ URL list.
        futures = []
        for replica_idx, _, server_handle in server_items:
            zmq_urls = per_replica_worker_zmq_urls.get(replica_idx, [])
            futures.append(server_handle.kv_set_peer_registry.remote(peer_registry, zmq_urls))
        await asyncio.gather(*futures)

        total_ranks = sum(len(v) for v in per_replica_worker_zmq_urls.values())
        psrl_logger.info(
            f"[LMCache] Peer registry broadcast: replicas={len(server_items)}, "
            f"ranks={total_ranks}, peer_instances={len(peer_registry)}."
        )

    def start_lmcache_controller(self) -> str:
        """
        Start the LMCache Controller subprocess (synchronous, non-async).

        Must be called BEFORE init_model() so the Controller is already listening
        when LMCache workers inside EngineCore try to register at startup.

        Returns:
            str: Base URL of the Controller, e.g. `"http://10.0.0.1:9042"`.
        """
        self._lmcache_controller_url = self._start_lmcache_controller()
        return self._lmcache_controller_url

    def _start_lmcache_controller(self) -> str:
        """
        Start the `lmcache_controller` subprocess and poll until healthy.

        The controller must become healthy within the configured timeout. Its
        output is retained in `lmcache_controller.log` for failure diagnosis.

        Returns:
            str: Base URL of the Controller, e.g. `"http://10.0.0.1:9042"`.

        Raises:
            RuntimeError: If the Controller exits early or does not become healthy
                within the configured timeout.
        """
        import requests as _requests

        _, host = get_host_info()
        port = find_available_port(self.config.psrl.lmcache.controller_base_port)
        cmd = [
            "lmcache_controller",
            "--host",
            host,
            "--port",
            str(port),
        ]

        # Redirect the controller's stdout/stderr to a log file so failures are
        # visible after the fact (Popen without a pipe would otherwise drop them).
        log_dir = os.path.expanduser(self.config.psrl.logging_path)
        os.makedirs(log_dir, exist_ok=True)
        controller_log_path = os.path.join(log_dir, "lmcache_controller.log")
        self._lmcache_controller_log_path = controller_log_path
        controller_log_file = open(controller_log_path, "w")

        psrl_logger.info(f"[LMCache] Starting shared Controller: {' '.join(cmd)}. Logging to {controller_log_path!r}.")
        self._lmcache_controller_proc = subprocess.Popen(cmd, stdout=controller_log_file, stderr=subprocess.STDOUT)
        controller_url = f"http://{host}:{port}"

        def _tail_controller_log(num_lines: int = 40) -> str:
            try:
                with open(controller_log_path) as f:
                    lines = f.readlines()
                return "".join(lines[-num_lines:])
            except Exception as e:
                return f"<failed to read controller log {controller_log_path!r}: {e}>"

        # Poll GET /openapi.json (FastAPI built-in, always available once uvicorn binds).
        # LMCache's /health is a POST endpoint requiring a body, so we use openapi.json instead.
        health_url = f"{controller_url}/openapi.json"
        max_attempts = int(self.config.psrl.lmcache.controller_health_timeout_s)
        for attempt in range(max_attempts):
            # A terminated process cannot become healthy within this attempt.
            returncode = self._lmcache_controller_proc.poll()
            if returncode is not None:
                self._lmcache_controller_proc = None
                raise RuntimeError(
                    f"LMCache Controller process exited early with code {returncode} "
                    f"before becoming healthy at {health_url!r}. "
                    f"Controller log tail:\n{_tail_controller_log()}"
                )
            try:
                resp = _requests.get(health_url, timeout=2)
                if resp.status_code == 200:
                    psrl_logger.info(
                        f"[LMCache] Controller healthy: url={controller_url!r}, attempt={attempt + 1}/{max_attempts}."
                    )
                    return controller_url
            except Exception:
                pass
            time.sleep(1)

        self._lmcache_controller_proc.kill()
        self._lmcache_controller_proc = None
        raise RuntimeError(
            f"LMCache Controller did not become healthy at {health_url!r} within "
            f"{max_attempts} s. Controller log tail:\n{_tail_controller_log()}"
        )
