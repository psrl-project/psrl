import asyncio
import logging
import os
from collections import Counter, OrderedDict

import ray
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import TensorDict
from transfer_queue import KVBatchMeta
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig

from psrl.utils.common.http_utils import init_distributed_post_pool
from psrl.utils.dataset import DatasetType
from psrl.utils.logger import (
    DualOutputHandler,
    EventType,
    log_dual_events,
    log_single_event,
)
from psrl.utils.ray import AsyncBusyPollingRayLock
from psrl.utils.transferqueue_utils import (
    PayloadState,
    async_clear_payload,
    clear_payload,
    request_payload_keys,
    validate_ready_payload,
)
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import RolloutInstanceId
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus
from psrl.workers.ps.staleness_controller import EntryInfo

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class BoundedIdSet:
    """Insertion-ordered id set that evicts the oldest entry once full.

    Used for the training failed-group record, which has no safe clear-point. Train
    prompt ids never repeat within a realistic run, so nothing requires the record to
    be cleared, and no step boundary is a valid eviction point either: with
    `staleness > 0` an in-flight prompt outlives several buffers. Bounding by size is
    what keeps a long run from accumulating entries forever.

    Eviction follows insertion order rather than access order on purpose. Membership is
    tested for every late arrival, so refreshing on read would tie an entry's lifetime
    to straggler traffic instead of to its age.
    """

    def __init__(self, capacity: int) -> None:
        """Build an empty set.

        Args:
            capacity (int): Maximum retained ids. Must be positive.
        """
        if capacity <= 0:
            raise ValueError(f"BoundedIdSet capacity must be positive, got {capacity}.")
        self.capacity = capacity
        self._ids: OrderedDict[int, None] = OrderedDict()

    def add(self, item: int) -> None:
        """Record `item`, evicting the oldest id when already at capacity."""
        if item in self._ids:
            return
        self._ids[item] = None
        if len(self._ids) > self.capacity:
            self._ids.popitem(last=False)

    def __contains__(self, item: object) -> bool:
        return item in self._ids

    def __len__(self) -> int:
        return len(self._ids)


class PSRL_AgentLoopManager:
    def __init__(
        self,
        config: DictConfig,
        data_queue_size: int,
        agent_loop_workers: list[ray.actor.ActorHandle],
        ps_manager_handle: ray.actor.ActorHandle,
        data_processor: ray.actor.ActorHandle,
        group_post_process_fn=None,
        buffer_post_process_fn=None,
    ):
        """Initialize agent loop manager.
        Agent loop manager that manages a group of agent loop workers.
        Handles data distribution, versioning, and coordination between workers.

        Args:
            config (DictConfig): Configuration containing training and rollout settings.
            data_queue_size (int): Size of the data queue.
            agent_loop_workers (list[ray.actor.ActorHandle]): List of agent loop worker instances.
            ps_manager_handle (ray.actor.ActorHandle): Handle to the parameter server manager.
            data_processor (ray.actor.ActorHandle): Handle to the data processor.
            group_post_process_fn (Optional[callable]): Optional function to post-process
                grouped entry data before occupying the buffer
            buffer_post_process_fn (Optional[callable]): Optional function to post-process
                ready buffer data
        """
        self.config = config
        model_config = config.gen_actor_rollout_ref.model
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        # TransferQueue bootstrap.
        tq.init()

        self.staleness = self.config.psrl.staleness
        self.group_post_process_fn = group_post_process_fn
        self.buffer_post_process_fn = buffer_post_process_fn
        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
            self.alg_rollout_n = self.config.psrl.rollout_coordination.redundant_rollout.alg_rollout_n
        else:
            self.rollout_n = self.config.gen_actor_rollout_ref.rollout.n
            self.alg_rollout_n = self.rollout_n
        self.val_rollout_n = self.config.train_actor_rollout_ref.rollout.val_kwargs.n

        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            self.entries_per_buffer = (
                self.config.psrl.rollout_coordination.redundant_rollout.redundant_global_batch_size
            )
            self.ready_entries_per_buffer = (
                self.config.psrl.rollout_coordination.redundant_rollout.alg_global_batch_size
            )
        else:
            self.entries_per_buffer = self.config.psrl.staleness_buffer_entries
            self.ready_entries_per_buffer = self.config.psrl.staleness_buffer_entries

        self.train_data_queue: asyncio.Queue = asyncio.Queue(maxsize=data_queue_size)
        self.val_data_queue: asyncio.Queue = asyncio.Queue(maxsize=data_queue_size)
        self.result_queue = asyncio.Queue()
        self.agent_loop_workers = agent_loop_workers
        self.ps_manager_handle = ps_manager_handle
        self.data_processor = data_processor
        self.reward_manager = None
        self.distributed_post_actors: list[ray.actor.ActorHandle] = []

        self._request_counter = 0
        self._dispatch_idx = 0
        self._val_buffer_id = 0
        self.running_loop: asyncio.AbstractEventLoop | None = None
        self.train_dispatch_task: asyncio.Task | None = None
        self.val_dispatch_task: asyncio.Task | None = None
        self.stop_train_dispatch_task = False
        self.stop_val_dispatch_task = False
        self.stop_collect_task = False

        self.curr_ps_version_tag = 0
        self.initial_ps_version = 0  # Set during resume to offset version calculations

        # Accumulated EntryInfo buffers (train path).
        self.train_data_buffers: dict[int, KVBatchMeta] = {}  # metadata of READY buffer in ps manager
        self.train_accumulated_buffers: dict[
            int, dict[int, list[EntryInfo]]
        ] = {}  # Maps buffer_id to dict of model_version to READY entry_info list
        self.train_accumulated_buffer_size: dict[int, int] = {}  # Maps buffer id to current accumulated size

        # Accumulated EntryInfo buffers (val path).
        self.val_data_buffers: dict[int, KVBatchMeta] = {}  # metadata of READY buffer in ps manager
        self.val_accumulated_buffers: dict[
            int, dict[int, list[EntryInfo]]
        ] = {}  # Maps buffer_id to dict of model_version to READY entry_info list
        self.val_accumulated_buffer_size: dict[int, int] = {}  # Maps buffer id to current accumulated size
        self.val_buffer_size: int | None = None  # Set by main trainer when starting validation

        # Set of buffer ids that have been logged as ready, to avoid duplicate logging
        self.logged_ready_train_buffer_ids: set[int] = set()
        self.logged_ready_val_buffer_ids: set[int] = set()

        # Waiting lists for training batches
        self._train_buffer_waiters: dict[
            int, list[asyncio.Future]
        ] = {}  # Maps buffer IDs to a set of futures waiting for that buffer
        self._val_buffer_waiters: dict[
            int, list[asyncio.Future]
        ] = {}  # Maps buffer IDs to a set of futures waiting for that buffer

        # Chunk size is measured in prompt groups. `None` preserves full-batch behavior.
        self.train_chunk_size: int | None = None
        # Maps buffer_id -> number of prompt-groups already handed out as chunks.
        self._train_chunk_consumed: dict[int, int] = {}
        # Entry identities already emitted for each buffer. The version buckets
        # may be reordered by late arrivals, so a numeric offset is insufficient.
        self._train_chunk_emitted_entry_ids: dict[int, set[int]] = {}
        # Maps (buffer_id, chunk_index) -> list of asyncio.Future waiting for that chunk.
        self._train_chunk_waiters: dict[tuple[int, int], list] = {}
        # Durable store for emitted chunks: (buffer_id, chunk_index) -> (KVBatchMeta, is_last).
        # Kept separate from waiters so results survive flush / waiter-bookkeeping churn.
        self._resolved_train_chunks: dict[tuple[int, int], tuple] = {}

        # Track finished child requests for Group Sampling
        self.rollout_request_tracker: dict[
            str | int, list[EntryInfo]
        ] = {}  # Maps parent request ids to "occupied" child entries

        # Groups whose failure was already processed, so concurrent sibling failures are not
        # handled twice. Train and validation keep separate records because their lifetimes differ.
        #
        # The train record is bounded rather than cleared, since train prompt ids never repeat
        # and no step boundary is safe to clear on once staleness keeps prompts in flight.
        failed_id_capacity = max(4 * self.entries_per_buffer * (self.staleness + 1), 65536)
        self._failed_train_group_ids: BoundedIdSet = BoundedIdSet(failed_id_capacity)
        self._failed_val_group_ids: set[int] = set()

        # Preserve an all-failed validation result for waiters that register late.
        self._val_round_all_failed: bool = False

        # Refill breaker state (train only). Groups that fail before being replaced are counted
        # consecutively and cleared by any occupied group, so sporadic failures never trip it.
        self.refill_failure_threshold = self.config.psrl.agentic_rl.get("refill_failure_threshold", 32)
        self._consecutive_group_failures = 0
        self._group_failure_reasons: Counter = Counter()
        # Set once the breaker trips. `wait_for_training_batch` turns it into the
        # exception that ends the run.
        self._refill_breaker_diagnosis: str | None = None

        # Build logger
        self.log_prefix = "AgentLoopManager"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    # AGENT(VERL): `generate_sequences`, `_run_agent_loop` are moved to agent loop workers.
    # The manager only handles data distribution and coordination.

    def set_chunk_size(self, chunk_size: int | None) -> None:
        """Set the number of prompt-groups per chunk for fine_grain_overlap.

        Call once from the trainer before starting the training loop.
        chunk_size=None disables chunk-yielding (default full-batch behavior).
        """
        self.train_chunk_size = chunk_size
        psrl_logger.info("AgentLoopManager: train_chunk_size set to %s", chunk_size)

    async def _init_distributed_post_pool(self) -> None:
        if not self.config.psrl.rollout_gateway.use_distributed_post or self.distributed_post_actors:
            return

        n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )
        n_active_instance = n_rollout_instances + n_validate_instances

        total_concurrency = self.config.psrl.rollout_gateway.server_max_concurrency * n_active_instance
        post_actor_num_per_node = self.config.psrl.rollout_gateway.get("post_actor_num_per_node", 1)
        self.distributed_post_actors = init_distributed_post_pool(
            total_concurrency=total_concurrency,
            post_actor_num_per_node=post_actor_num_per_node,
        )
        await asyncio.gather(
            *[
                worker.set_distributed_post_actors.remote(
                    self.distributed_post_actors,
                    True,
                    worker_index,
                )
                for worker_index, worker in enumerate(self.agent_loop_workers)
            ]
        )
        psrl_logger.info(
            "Distributed POST pool started: actors=%d actors_per_node=%d total_concurrency=%d "
            "server_max_concurrency=%d engines=%d.",
            len(self.distributed_post_actors),
            post_actor_num_per_node,
            total_concurrency,
            self.config.psrl.rollout_gateway.server_max_concurrency,
            n_active_instance,
        )

    async def _shutdown_distributed_post_pool(self) -> None:
        if not self.distributed_post_actors:
            return
        await asyncio.gather(
            *[worker.set_distributed_post_actors.remote(None, False, 0) for worker in self.agent_loop_workers],
            return_exceptions=True,
        )
        await asyncio.gather(
            *[actor.aclose.remote() for actor in self.distributed_post_actors],
            return_exceptions=True,
        )
        self.distributed_post_actors = []

    def set_val_buffer_size(self, val_buffer_size: int):
        """Set the validation buffer size and reset per-round validation state.

        Args:
            val_buffer_size (int): Number of prompt groups this validation round expects.
        """
        self.val_buffer_size = val_buffer_size
        # Validation-only: prompt ids repeat across rounds, so a stale record would drop
        # a fresh group as already-failed. The train record is deliberately untouched.
        self._failed_val_group_ids.clear()
        self._val_round_all_failed = False

    def set_reward_manager(self, reward_manager: ray.actor.ActorHandle):
        """Set the reward manager for awaiting async reward completion."""
        self.reward_manager = reward_manager

    async def start_busy_loop(self):
        """Start the busy loop for continuous data processing from the queue."""
        if (
            self.train_dispatch_task is not None
            and not self.train_dispatch_task.done()
            or self.val_dispatch_task is not None
            and not self.val_dispatch_task.done()
        ):
            return

        # Start the busy loop of agent loop workers.
        await self._init_distributed_post_pool()
        await asyncio.gather(*[worker.start_busy_loop.remote() for worker in self.agent_loop_workers])

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.train_dispatch_task = self.running_loop.create_task(self._train_dispatch_data())
        self.train_dispatch_task.add_done_callback(lambda f: f.result())
        self.val_dispatch_task = self.running_loop.create_task(self._val_dispatch_data())
        self.val_dispatch_task.add_done_callback(lambda f: f.result())
        self.collect_task = self.running_loop.create_task(self._collect_results())
        self.collect_task.add_done_callback(lambda f: f.result())

    async def stop_busy_loop(self):
        """Stop the busy loop and wait for all tasks to complete."""
        if (
            (not self.train_dispatch_task or self.train_dispatch_task.done())
            and (not self.val_dispatch_task or self.val_dispatch_task.done())
            and (not self.collect_task or self.collect_task.done())
        ):
            return

        self.stop_train_dispatch_task = True
        self.stop_val_dispatch_task = True
        self.stop_collect_task = True
        await asyncio.gather(self.train_dispatch_task, self.val_dispatch_task, self.collect_task)

        await asyncio.gather(*[worker.stop_busy_loop.remote() for worker in self.agent_loop_workers])
        await self._shutdown_distributed_post_pool()

    async def put_data(self, batch: TensorDict, is_validate: bool = False):
        """Put objectref of data into the manager's data queue."""
        queue = self.val_data_queue if is_validate else self.train_data_queue
        await queue.put(batch)

    async def put_result(self, result: dict):
        """Put result data into the manager's result queue."""
        await self.result_queue.put(result)

    async def _collect_results(self):
        """Main collection loop that gathers results from workers.

        Drains all available results from the queue and processes them.
        Validation results are batched through ``occupy_requests`` to avoid
        per-request lock acquisition overhead. Training results still go
        through the per-request path because group sampling can trigger retry
        and abort side effects per prompt.
        """
        while not self.stop_collect_task:
            # Drain all available results from the queue, separating train/val.
            train_results: list[dict] = []
            val_results: list[dict] = []
            while not self.result_queue.empty():
                result = self.result_queue.get_nowait()
                if result.get("is_validate", False):
                    val_results.append(result)
                else:
                    train_results.append(result)

            if val_results:
                await self.occupy_requests(
                    request_id=[r["request_id"] for r in val_results],
                    prompt_id=[r["prompt_id"] for r in val_results],
                    rollout_instance_id=[r["rollout_instance_id"] for r in val_results],
                    version_tag=[r["version_tag"] for r in val_results],
                    n_trajectory=[r.get("n_trajectory", 1) for r in val_results],
                    is_validate=True,
                )

            # Process training results one-by-one (group sampling requires it).
            for result in train_results:
                dispositions = await self.occupy_requests(**result)
                assert len(dispositions) == 1, (
                    f"Expected one disposition for scalar training result, got {dispositions}."
                )
                psrl_logger.debug(
                    "Training result request_id=%s entered payload state %s.",
                    result["request_id"],
                    dispositions[0].value,
                )

            if not train_results and not val_results:
                await asyncio.sleep(0)  # Yield control to the event loop only when idle
        psrl_logger.info("Stop collecting results.")

    async def _train_dispatch_data(self):
        """Main dispatch loop that processes data from the queue and routes to workers."""
        while not self.stop_train_dispatch_task:
            if not self.train_data_queue.empty():
                data: TensorDict | None = self.train_data_queue.get_nowait()
            else:
                await asyncio.sleep(0)
                continue

            # Receive END signal to stop processing data queue
            if data is None:
                psrl_logger.info(
                    "Received END signal, stopping train dispatch. request_counter=%d, result_queue=%d.",
                    self._request_counter,
                    self.result_queue.qsize(),
                )
                self.stop_train_dispatch_task = True
                continue

            # Wait for version update in ps
            # NOTE(lhy): we restrict the extra dispatched data to be no more than (staleness + 1) * buffer_size
            expected_ps_version = self._get_expected_ps_version()
            if expected_ps_version > self.curr_ps_version_tag:
                psrl_logger.debug(f"Waiting for ps model version: {expected_ps_version}")
                # Busy polling until the PS worker has the needed model version
                while (
                    await self.ps_manager_handle.get_ps_model_version.remote(debug_info="agent_loop_manager")
                ) < expected_ps_version:
                    await asyncio.sleep(0.1)
                self.curr_ps_version_tag = expected_ps_version
                psrl_logger.info(f"ps model version updated to {self.curr_ps_version_tag}, continue to dispatch")

            # Initialize the version tag to -1 for all requests
            tu.assign_non_tensor_stack(data, "version_tag", [-1] * len(data))

            # Dispatch data to agent loop workers
            await self._inner_dispatch_data(data, is_validate=False)
            # Increment counter after dispatch so _get_expected_ps_version reflects the number
            # of requests that have actually been sent out.
            self._request_counter += len(data)
            await asyncio.sleep(0)  # Yield control to the event loop

    async def _val_dispatch_data(self):
        """Main dispatch loop that processes data from the queue and routes to workers."""
        while not self.stop_val_dispatch_task:
            if not self.val_data_queue.empty():
                data: TensorDict | None = self.val_data_queue.get_nowait()
            else:
                await asyncio.sleep(0)
                continue

            # Receive END signal to stop processing data queue
            if data is None:
                psrl_logger.info("Received END signal, stopping agent loop manager validation dispatch task.")
                self.stop_val_dispatch_task = True
                continue

            # Validation samples all share the current PS version.
            tu.assign_non_tensor_stack(data, "version_tag", [self.curr_ps_version_tag] * len(data))

            # Dispatch data to agent loop workers
            await self._inner_dispatch_data(data, is_validate=True)
            await asyncio.sleep(0)  # Yield control to the event loop

        psrl_logger.info("Agent loop manager validation dispatch task stopped.")

    async def _retry_data(self, n_prompts: int = 1) -> int:
        """Retry exactly `n_prompts` prompts by sampling fresh data on demand.

        Dispatches directly to workers via `_inner_dispatch_data` (NOT through
        `train_data_queue`), so it is independent of the dispatch loop and its END
        (None) signal. It deliberately does NOT check `stop_train_dispatch_task`:
        a group that failed AFTER the dispatch loop stopped (the DataProcessor has
        sent all planned prompts) must STILL be compensated by one fresh prompt,
        otherwise the final buffer is left permanently short and the trainer hangs.
        The collect task stays alive post-END, so the refill's results are still
        accumulated into the waiting buffer.

        Args:
            n_prompts (int): Number of prompts to retry. Each prompt expands
                to `rollout_n` children. Defaults to 1.

        Returns:
            int: Number of requests actually dispatched. `0` if the agent
            loop is not running, `n_prompts <= 0`, or the dataloader was
            exhausted.
        """
        if self.running_loop is None:
            psrl_logger.warning("Agent loop manager has no running loop, the retry operation will be skipped.")
            return 0
        if n_prompts <= 0:
            return 0

        rollout_n = (
            self.config.psrl.rollout_coordination.redundant_rollout.redundant_rollout_n
            if self.config.psrl.rollout_coordination.redundant_rollout.enable
            else self.rollout_n
        )

        data: TensorDict | None = await self.data_processor.sample_train_prompts.remote(
            n_prompts=n_prompts,
        )
        if data is None or len(data) == 0:
            psrl_logger.warning(f"Retry skipped: DataProcessor returned no data for n_prompts={n_prompts}.")
            return 0

        tu.assign_non_tensor_stack(data, "version_tag", [-1] * len(data))
        psrl_logger.info(f"Retry dispatch: requests={len(data)}, prompts={len(data) // rollout_n}.")
        await self._inner_dispatch_data(data, is_validate=False)

        # Account retry dispatches in the staleness throttle so that
        # `_train_dispatch_data` doesn't over-dispatch on top of retry bursts.
        self._request_counter += len(data)
        return len(data)

    def _reset_group_failure_streak(self) -> None:
        """Clear the refill breaker's failure streak after a group is occupied.

        One occupied group proves the rollout pipeline can still produce trainable
        data, so the preceding failures were sporadic rather than deterministic.
        Resetting here is what keeps a long run from tripping on accumulated noise.
        """
        if self._consecutive_group_failures:
            self._consecutive_group_failures = 0
            self._group_failure_reasons.clear()

    def _record_group_failure(self, terminate_reason: TerminateReason | None) -> None:
        """Account one failed training group against the refill breaker.

        Args:
            terminate_reason (TerminateReason | None): Why the rollout produced no
                data. Recorded so the breaker can name the dominant cause instead of
                reporting a bare count.
        """
        self._consecutive_group_failures += 1
        reason = terminate_reason.value if terminate_reason is not None else TerminateReason.UNKNOWN.value
        self._group_failure_reasons[reason] += 1

        if self._consecutive_group_failures >= self.refill_failure_threshold:
            breakdown = ", ".join(f"{reason}={count}" for reason, count in self._group_failure_reasons.most_common())
            self._trip_refill_breaker(
                f"{self._consecutive_group_failures} consecutive rollout group failures with no group "
                f"succeeding in between (threshold={self.refill_failure_threshold}). "
                f"Terminate reasons: {breakdown}. Every failed group was replaced by a fresh prompt, so "
                f"the buffer never filled and training made no progress. This is an environment or task "
                f"harness fault rather than a transient error. Raise "
                f"psrl.agentic_rl.refill_failure_threshold only if these failures are genuinely sporadic."
            )

    def _trip_refill_breaker(self, diagnosis: str) -> None:
        """Latch a refill failure and fail every training waiter with it.

        Raising from the caller cannot stop the run: `notify_group_failed` is invoked
        as a fire-and-forget task in the agent loop worker, whose done-callback logs
        exceptions and discards them. The driver's only blocking calls into this actor
        are `wait_for_training_batch` and `wait_for_training_chunk`, so the diagnosis is
        latched here and converted into an exception there. Both waiter maps are failed
        directly as well, because the trainer usually blocks before the breaker trips.

        Args:
            diagnosis (str): Human-readable explanation carried to the driver.
        """
        if self._refill_breaker_diagnosis is not None:
            return
        self._refill_breaker_diagnosis = diagnosis
        psrl_logger.error(f"Refill breaker tripped: {diagnosis}")

        # A fresh exception per future: Ray serializes each one separately, and sharing
        # a single instance makes every waiter report the same reused traceback.
        for buffer_id in list(self._train_buffer_waiters.keys()):
            for fut in self._train_buffer_waiters[buffer_id]:
                if not fut.done():
                    fut.set_exception(self._refill_breaker_error())
            del self._train_buffer_waiters[buffer_id]

        # The chunk path is a separate waiter map. `fine_grain_overlap` blocks only
        # here, so skipping it would leave that strategy livelocked.
        for chunk_key in list(self._train_chunk_waiters.keys()):
            for fut in self._train_chunk_waiters[chunk_key]:
                if not fut.done():
                    fut.set_exception(self._refill_breaker_error())
            del self._train_chunk_waiters[chunk_key]
        self._resolved_train_chunks.clear()

    def _refill_breaker_error(self) -> RuntimeError:
        """Build the exception that reports a tripped refill breaker."""
        return RuntimeError(f"Training aborted, rollout groups cannot be refilled. {self._refill_breaker_diagnosis}")

    def _raise_if_refill_breaker_tripped(self) -> None:
        """Convert a latched refill failure into an exception for the driver.

        Raises:
            RuntimeError: When `_trip_refill_breaker` has latched a diagnosis.
        """
        if self._refill_breaker_diagnosis is not None:
            raise self._refill_breaker_error()

    async def _refill_failed_group(self, n_prompts: int, context: str) -> int:
        """Replace `n_prompts` lost training slots and treat a zero refill as fatal.

        Every caller reaches this only after the affected entries were already purged
        and their siblings aborted, so there is no state left to retry from. A refill
        that dispatches nothing therefore leaves the buffer permanently short, which
        the trainer experiences as an unexplained hang.

        Args:
            n_prompts (int): Number of prompts to dispatch as replacements.
            context (str): Which recovery path is refilling, used in the diagnosis.

        Returns:
            int: Number of requests dispatched.
        """
        dispatched = await self._retry_data(n_prompts=n_prompts)
        if dispatched == 0:
            self._trip_refill_breaker(
                f"{context} could not dispatch a replacement for {n_prompts} lost buffer slot(s). "
                "The failed entries were already purged, so the buffer is permanently short and no "
                "further rollout can complete it. Check that the training dataloader still yields "
                "data and that the agent loop manager is running."
            )
        return dispatched

    def _entry_info_tq_keys(self, entry_info: EntryInfo, rollout_n: int) -> list[str]:
        request_idxs = entry_info.request_idx if isinstance(entry_info.request_idx, list) else [entry_info.request_idx]
        n_trajectories = (
            entry_info.n_trajectory
            if isinstance(entry_info.n_trajectory, list)
            else [entry_info.n_trajectory] * len(request_idxs)
        )
        keys: list[str] = []
        for request_idx, n_trajectory in zip(request_idxs, n_trajectories):
            request_id = entry_info.prompt_id * rollout_n + request_idx
            keys.extend(request_payload_keys(request_id, n_trajectory))
        return keys

    async def _clear_payload_if_present(
        self,
        keys: list[str],
        partition_id: str,
        state: PayloadState,
    ) -> None:
        """Clear the currently present subset of a terminal payload."""
        await async_clear_payload(
            keys=keys,
            partition_id=partition_id,
            state=state,
        )

    async def _purge_tracker_group(
        self,
        parent_id: int,
        rollout_n: int,
        is_validate: bool,
    ) -> list[EntryInfo]:
        """Drop partially accumulated tracker entries and their TQ payloads."""
        entries = self.rollout_request_tracker.pop(parent_id, [])
        if not entries:
            return []

        keys: list[str] = []
        for entry_info in entries:
            keys.extend(self._entry_info_tq_keys(entry_info, rollout_n))
        if keys:
            await self._clear_payload_if_present(
                keys=keys,
                partition_id="val" if is_validate else "train",
                state=PayloadState.DROPPED,
            )
        psrl_logger.info(
            "_purge_tracker_group: removed %d partial entries (%d TQ keys) for parent_id=%s.",
            len(entries),
            len(keys),
            parent_id,
        )
        return entries

    async def _flush_ready_buffer(self, buffer_id: int, is_validate: bool) -> bool:
        """Assemble and publish an accumulated buffer that reached its target."""
        accumulated_buffers = self.val_accumulated_buffers if is_validate else self.train_accumulated_buffers
        accumulated_buffer_size = (
            self.val_accumulated_buffer_size if is_validate else self.train_accumulated_buffer_size
        )
        if buffer_id not in accumulated_buffers:
            return False

        prompt_entry_infos: list[EntryInfo] = []
        for model_version in sorted(list(accumulated_buffers[buffer_id].keys())):
            prompt_entry_infos.extend(accumulated_buffers[buffer_id][model_version])
        # Sort by `prompt_id` to match the buffer prompt order.
        prompt_entry_infos.sort(key=lambda ei: ei.prompt_id)

        batch = self.entry_infos_to_kv_batch_meta(prompt_entry_infos, is_validate)
        partitions = await tq.async_kv_list(partition_id=batch.partition_id)
        validate_ready_payload(batch.keys, batch.partition_id, partitions)

        # The chunk path bypasses full-batch waiters to avoid duplicate publication
        # and retained `train_data_buffers` entries.
        if not is_validate and self.train_chunk_size is not None:
            psrl_logger.info(
                "Training buffer %d is READY with %d entries (chunk path).",
                buffer_id,
                len(batch),
            )
            self.log_ready_buffer(buffer_id, is_validate=False)
            await self.ps_manager_handle.handle_ready_buffer.remote(buffer_id)
            self._emit_pending_chunks(buffer_id)
            # Resolved chunks remain available for late waiters.
            self._train_chunk_consumed.pop(buffer_id, None)
            self._train_chunk_emitted_entry_ids.pop(buffer_id, None)
            accumulated_buffers.pop(buffer_id, None)
            accumulated_buffer_size.pop(buffer_id, None)
            return True

        add_buffer = self.maybe_add_buffer(buffer_id, batch, is_validate)
        if add_buffer:
            psrl_logger.info(
                "%s buffer %d is READY with %d entries.",
                "Validation" if is_validate else "Training",
                buffer_id,
                len(batch),
            )
            await self.handle_ready_buffer(buffer_id, is_validate)
            # Flush any tail chunk (remainder that didn't fill a full chunk_size).
            if not is_validate:
                self._emit_pending_chunks(buffer_id)
            accumulated_buffers.pop(buffer_id, None)
            accumulated_buffer_size.pop(buffer_id, None)
        return add_buffer

    async def notify_group_failed(
        self,
        parent_id: int,
        failed_uid: int,
        is_validate: bool,
        terminate_reason: TerminateReason | None = None,
        failure_summary: str | None = None,
    ):
        """Recover a rollout group after one child fails without producing data.

        Args:
            parent_id (int): Prompt id of the group that lost a child.
            failed_uid (int): Child request id that failed.
            is_validate (bool): Whether the group belongs to a validation round.
            terminate_reason (TerminateReason | None): Why the child produced no data.
                Recorded on the train path so the refill breaker can name the cause.
            failure_summary (str | None): Worker-captured root cause text for the log.
        """
        if failure_summary:
            psrl_logger.error(
                "notify_group_failed: root cause for parent_id=%s failed_uid=%s:\n%s",
                parent_id,
                failed_uid,
                failure_summary,
            )
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            rollout_n = self.val_rollout_n if is_validate else self.rollout_n
            failed_group_ids = self._failed_val_group_ids if is_validate else self._failed_train_group_ids
            if parent_id in failed_group_ids:
                psrl_logger.warning(
                    f"Duplicate group failure: parent_id={parent_id}, "
                    f"is_validate={is_validate}, failed_uid={failed_uid}. Skipping."
                )
                return
            failed_group_ids.add(parent_id)

            all_child_uids = [parent_id * rollout_n + i for i in range(rollout_n)]
            sibling_uids = [uid for uid in all_child_uids if uid != failed_uid]
            if sibling_uids:
                psrl_logger.info(
                    "notify_group_failed: aborting %d sibling uids=%s for parent_id=%s.",
                    len(sibling_uids),
                    sibling_uids,
                    parent_id,
                )
                await self.ps_manager_handle.abort_requests.remote(sibling_uids, blocking=False)

            await self._purge_tracker_group(parent_id, rollout_n, is_validate)

            if is_validate:
                if self.val_buffer_size is None or self.val_buffer_size <= 0:
                    psrl_logger.warning(
                        "notify_group_failed: val_buffer_size=%s, cannot shrink for parent_id=%s.",
                        self.val_buffer_size,
                        parent_id,
                    )
                    return
                self.val_buffer_size -= 1
                psrl_logger.warning(
                    "notify_group_failed: val_buffer_size decremented to %d for parent_id=%s.",
                    self.val_buffer_size,
                    parent_id,
                )
                for buffer_id, accumulated_size in list(self.val_accumulated_buffer_size.items()):
                    if accumulated_size == self.val_buffer_size and buffer_id not in self.val_data_buffers:
                        psrl_logger.info(
                            "notify_group_failed (val): buffer_id=%d now meets "
                            "adjusted val_buffer_size=%d, assembling and firing.",
                            buffer_id,
                            self.val_buffer_size,
                        )
                        await self._flush_ready_buffer(buffer_id, is_validate=True)

                # Wake pending waiters with an empty batch when every validation group
                # fails, preventing a permanent wait.
                if self.val_buffer_size <= 0:
                    # Latch the all-failed state so a waiter registering after this
                    # last failure (race) still observes it instead of blocking.
                    self._val_round_all_failed = True
                    empty_batch = KVBatchMeta(keys=[], tags=[], partition_id="val")
                    for waiter_buffer_id in list(self._val_buffer_waiters.keys()):
                        psrl_logger.warning(
                            "notify_group_failed (val): all groups failed (val_buffer_size=0). "
                            "waking waiter for buffer_id=%d with an empty batch to avoid deadlock.",
                            waiter_buffer_id,
                        )
                        for fut in self._val_buffer_waiters[waiter_buffer_id]:
                            if not fut.done():
                                fut.set_result(empty_batch)
                        del self._val_buffer_waiters[waiter_buffer_id]
            else:
                # Replace each failed training group directly so the target success
                # count remains unchanged. Dataset sampling cycles until shutdown.
                self._record_group_failure(terminate_reason)
                dispatched = await self._refill_failed_group(
                    n_prompts=1,
                    context="notify_group_failed (train)",
                )
                psrl_logger.info(
                    "notify_group_failed (train): dispatched %d fresh replacement request(s) for parent_id=%s "
                    "(consecutive failures=%d).",
                    dispatched,
                    parent_id,
                    self._consecutive_group_failures,
                )

    def _get_expected_ps_version(self):
        """
        Get the expected PS version tag based on the current staleness and request counter.
        """
        if self.config.psrl.rollout_coordination.redundant_rollout.enable:
            buffer_size = (
                self.config.psrl.rollout_coordination.redundant_rollout.redundant_global_batch_size * self.rollout_n
            )
        else:
            buffer_size = self.config.psrl.staleness_buffer_entries * self.rollout_n

        max_dispatch_ahead = self.config.psrl.get("max_dispatch_ahead", 5)
        effective_staleness = min(self.staleness, max_dispatch_ahead)
        expected_ps_version = (
            self.initial_ps_version + max(self._request_counter - effective_staleness * buffer_size, 0) // buffer_size
        )
        return expected_ps_version

    def set_initial_ps_version(self, version: int):
        """
        Set the initial PS version for resume. This offsets the expected version
        calculation and initializes curr_ps_version_tag.

        Args:
            version (int): The initial PS model version (= checkpoint global_step).
        """
        self.initial_ps_version = version
        self.curr_ps_version_tag = version
        psrl_logger.info(f"Initialized resume PS version: version={version}.")

    async def _inner_dispatch_data(self, data: TensorDict, is_validate: bool = False):
        """Update request status to RUNNING in PSManager, then fan out to workers."""
        # Rows are ordered as contiguous groups of `rollout_n` children per parent.
        uids = tu.get(data, "uid")
        versions = tu.get(data, "version_tag")

        # Update request status from PENDING to RUNNING
        update_status_success = await self.ps_manager_handle.update_request_status.remote(
            uids,
            PSRL_RequestStatus.RUNNING,
            model_version=versions,
            is_validate=is_validate,
        )
        if not update_status_success:
            return

        dispatch_plan = self.get_dispatch_plan(data, is_validate=is_validate)
        for worker_index, batch in dispatch_plan.items():
            self.agent_loop_workers[worker_index].add_agent_program.remote(batch)

    def get_dispatch_plan(self, data: TensorDict, is_validate: bool = False) -> dict[int, TensorDict]:
        """Round-robin dispatch plan keyed by worker index, co-locating siblings.

        Children sharing a ``parent_id`` (group sampling) land on the same worker.
        """
        keys_by_worker: dict[int, list[str]] = {}
        prompt_to_worker: dict[int, int] = {}
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        prompt_ids = tu.get(data, "parent_id") if rollout_n > 1 else tu.get(data, "uid")

        # Round-robin dispatching
        for i, prompt_id in enumerate(prompt_ids):
            if prompt_id in prompt_to_worker:
                worker_index = prompt_to_worker[prompt_id]
            else:
                worker_index = (self._dispatch_idx + len(prompt_to_worker)) % len(self.agent_loop_workers)
                prompt_to_worker[prompt_id] = worker_index
            keys_by_worker.setdefault(worker_index, []).append(i)

        self._dispatch_idx = (self._dispatch_idx + len(prompt_to_worker)) % len(self.agent_loop_workers)
        return {worker_index: data[keys] if keys else None for worker_index, keys in keys_by_worker.items()}

    async def occupy_requests(
        self,
        request_id: int | list[int],
        prompt_id: int | list[int],
        rollout_instance_id: RolloutInstanceId | tuple | list,
        version_tag: int | list[int],
        n_trajectory: int | list[int] = 1,
        is_validate: bool = False,
    ) -> list[PayloadState]:
        """Flat-arg RPC invoked by rollout workers once a request finishes.

        The rollout worker has already written the per-sample TensorDict to TQ
        under keys reconstructed from ``request_id`` and ``n_trajectory``. This
        method accepts either a single request via scalar arguments or a batch
        via list arguments (all list args must share the same length). It
        appends the corresponding ``EntryInfo`` objects into the manager's
        trackers and triggers PSManager occupation, running group/buffer
        post-processing on KVBatchMeta slices (never on tensor payload).

        Returns:
            One payload lifecycle state per input request. ``PENDING_GROUP`` is
            accepted ownership, not a failure and never authorizes payload
            cleanup.
        """
        # Normalize scalar inputs to batch form.
        if isinstance(request_id, list):
            request_ids, prompt_ids, rollout_instance_ids, version_tags = (
                request_id,
                prompt_id,
                rollout_instance_id,
                version_tag,
            )
            n_trajectories = n_trajectory if isinstance(n_trajectory, list) else [n_trajectory] * len(request_ids)
        else:
            request_ids, prompt_ids, rollout_instance_ids, version_tags, n_trajectories = (
                [request_id],
                [prompt_id],
                [rollout_instance_id],
                [version_tag],
                [n_trajectory],
            )

        if not request_ids:
            return []

        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            rollout_n = self.val_rollout_n if is_validate else self.rollout_n
            alg_rollout_n = self.val_rollout_n if is_validate else self.alg_rollout_n
            partition_id = "val" if is_validate else "train"
            failed_group_ids = self._failed_val_group_ids if is_validate else self._failed_train_group_ids

            ready_buffer_ids: set[int] = set()
            dispositions = [PayloadState.PENDING_GROUP for _ in request_ids]
            request_positions = {request_id: i for i, request_id in enumerate(request_ids)}
            occupy_jobs: list[tuple[list[int], object, list[str]]] = []
            dropped_payload_keys: list[str] = []

            # 1. Judge whether to abort requests and occupy requests in the PS worker
            for position, (request_id, prompt_id, rollout_instance_id, version_tag, n_trajectory) in enumerate(
                zip(request_ids, prompt_ids, rollout_instance_ids, version_tags, n_trajectories)
            ):
                if prompt_id in failed_group_ids:
                    psrl_logger.warning(
                        "occupy_requests: discarding late arrival request_id=%s from failed "
                        "group parent_id=%s is_validate=%s.",
                        request_id,
                        prompt_id,
                        is_validate,
                    )
                    dispositions[position] = PayloadState.DROPPED
                    await async_clear_payload(
                        keys=request_payload_keys(request_id, n_trajectory),
                        partition_id=partition_id,
                        state=PayloadState.DROPPED,
                    )
                    continue

                # Update n_trajectory in PSManager (moved from worker to avoid
                # an extra per-request PSManager RPC on the worker's critical path).
                if n_trajectory > 1:
                    await self.ps_manager_handle.update_request_n_trajectory.remote(
                        request_id=request_id,
                        n_trajectory=n_trajectory,
                        is_validate=is_validate,
                    )

                if rollout_n > 1:
                    entry_info = EntryInfo(
                        rollout_instance_id=rollout_instance_id,
                        request_idx=request_id % rollout_n,
                        prompt_id=prompt_id,
                        model_version=version_tag,
                        n_trajectory=n_trajectory,
                        is_validate=is_validate,
                    )
                    self.rollout_request_tracker.setdefault(prompt_id, []).append(entry_info)
                    psrl_logger.debug(
                        f"Stored rollout entry: prompt_id={prompt_id}, entry={entry_info!r}, "
                        f"count={len(self.rollout_request_tracker[prompt_id])}."
                    )

                    if len(self.rollout_request_tracker[prompt_id]) >= alg_rollout_n:
                        psrl_logger.debug(
                            f"Reached/Required: "
                            f"({len(self.rollout_request_tracker[prompt_id])}/{alg_rollout_n}) "
                            f"samples for prompt {prompt_id}"
                        )
                        entry_infos = self.rollout_request_tracker.pop(prompt_id)
                        psrl_logger.debug(
                            f"Popped entry_infos from rollout_request_tracker for prompt_id {prompt_id}, "
                            f"entry count: {len(entry_infos)}"
                        )

                        all_child_idxs = set(range(rollout_n))
                        stored_child_idxs = set()
                        for entry_info in entry_infos:
                            assert isinstance(entry_info.request_idx, int), (
                                f"entry_info.request_idx should be int, but got {type(entry_info.request_idx)}"
                            )
                            request_idx: int = entry_info.request_idx
                            stored_child_idxs.add(request_idx)
                        abort_child_idxs = all_child_idxs - stored_child_idxs
                        abort_child_ids = [prompt_id * rollout_n + idx for idx in abort_child_idxs]
                        psrl_logger.debug(
                            f"Stored child IDs: "
                            f"{[prompt_id * rollout_n + idx for idx in stored_child_idxs]}, "
                            f"Abort child IDs: {abort_child_ids}"
                        )

                        # Notify the request status manager to abort the child requests
                        if abort_child_ids:
                            assert not is_validate, "Abort child requests should not happen in validation."
                            psrl_logger.info(
                                f"Aborting child requests: request_ids={abort_child_ids!r}, prompt_id={prompt_id}."
                            )
                            with log_dual_events(
                                f"Abort {len(abort_child_ids)} requests",
                                psrl_logger,
                                level=logging.INFO,
                                event_type=EventType.OTHER,
                            ):
                                await self.ps_manager_handle.abort_requests.remote(
                                    list(abort_child_ids), blocking=False
                                )

                        # Abort the extra finished entries beyond alg_rollout_n
                        extra_entry_infos = entry_infos[alg_rollout_n:]
                        for extra_entry_info in extra_entry_infos:
                            extra_request_id = prompt_id * rollout_n + extra_entry_info.request_idx
                            dropped_payload_keys.extend(
                                request_payload_keys(extra_request_id, extra_entry_info.n_trajectory)
                            )
                            if extra_request_id in request_positions:
                                dispositions[request_positions[extra_request_id]] = PayloadState.DROPPED

                        alg_entry_infos = entry_infos[:alg_rollout_n]
                        add_data = True
                        # Perform group post-processing for training data only.
                        if self.group_post_process_fn and not is_validate:
                            add_data = await self._group_post_process(alg_entry_infos)

                        if not add_data:
                            for dropped_entry_info in alg_entry_infos:
                                dropped_request_id = prompt_id * rollout_n + dropped_entry_info.request_idx
                                if dropped_request_id in request_positions:
                                    dispositions[request_positions[dropped_request_id]] = PayloadState.DROPPED
                            # Retry immediately and no occupation.
                            # NOTE(linsh): data has been cleared in `_group_post_process`.
                            psrl_logger.info(
                                f"Post-processing function returned empty data for "
                                f"prompt {prompt_id}. Retrying immediately."
                            )
                            # Clear the reserved entries for the group entry.
                            await self.ps_manager_handle.clear_reserved_entries.remote(prompt_id, is_validate)
                            # Notify the manager to retry new requests. Deliberate filtering is not
                            # counted against the refill breaker, since it is not a rollout fault.
                            await self._refill_failed_group(
                                n_prompts=1,
                                context="Group post-processing filter",
                            )
                        else:
                            child_request_ids = [
                                prompt_id * rollout_n + entry_info.request_idx for entry_info in alg_entry_infos
                            ]
                            job_positions = [
                                request_positions[child_request_id]
                                for child_request_id in child_request_ids
                                if child_request_id in request_positions
                            ]
                            job_keys = []
                            for child_request_id, child_entry_info in zip(child_request_ids, alg_entry_infos):
                                job_keys.extend(request_payload_keys(child_request_id, child_entry_info.n_trajectory))
                            occupy_jobs.append(
                                (
                                    job_positions,
                                    self.ps_manager_handle.occupy_rollout_instance_request.remote(
                                        prompt_id=prompt_id,
                                        request_ids=child_request_ids,
                                        is_validate=is_validate,
                                    ),
                                    job_keys,
                                )
                            )
                else:
                    # Without group sampling (e.g., PPO).
                    # Group post processing is not used and every data will be added.
                    occupy_jobs.append(
                        (
                            [position],
                            self.ps_manager_handle.occupy_rollout_instance_request.remote(
                                prompt_id=request_id,
                                is_validate=is_validate,
                            ),
                            request_payload_keys(request_id, n_trajectory),
                        )
                    )

            # 2. Occupy requests in the PS worker
            if occupy_jobs:
                with log_dual_events(
                    "Occupy requests",
                    psrl_logger,
                    level=logging.DEBUG,
                    event_type=EventType.OTHER,
                ):
                    results = await asyncio.gather(*(job[1] for job in occupy_jobs))
            else:
                results = []

            # 3. Handle the occupied results to accumulate data
            for (job_positions, _, job_keys), result in zip(occupy_jobs, results):
                buffer_id, occupy_num, prompt_entry_info = result
                if buffer_id is None:
                    for job_position in job_positions:
                        dispositions[job_position] = PayloadState.DROPPED
                    await self._clear_payload_if_present(
                        keys=job_keys,
                        partition_id=partition_id,
                        state=PayloadState.DROPPED,
                    )
                    continue
                for job_position in job_positions:
                    dispositions[job_position] = PayloadState.OCCUPIED
                if not is_validate:
                    # A group reaching OCCUPIED is the only proof that rollouts can
                    # still produce trainable data, so it is the breaker's reset point.
                    self._reset_group_failure_streak()

                psrl_logger.debug(
                    f"Occupied prompt: entry={prompt_entry_info!r}, buffer_id={buffer_id}, occupy_num={occupy_num}."
                )

                # Accumulate data
                accumulated_buffers = self.val_accumulated_buffers if is_validate else self.train_accumulated_buffers
                accumulated_buffer_size = (
                    self.val_accumulated_buffer_size if is_validate else self.train_accumulated_buffer_size
                )
                expected_buffer_size = self.val_buffer_size if is_validate else self.ready_entries_per_buffer

                if buffer_id not in accumulated_buffers:
                    accumulated_buffers[buffer_id] = {}
                    accumulated_buffer_size[buffer_id] = 0
                model_version = prompt_entry_info.get_entry_version()
                accumulated_buffers[buffer_id].setdefault(model_version, []).append(prompt_entry_info)
                accumulated_buffer_size[buffer_id] += 1
                psrl_logger.info(
                    f"Accumulated buffer: buffer_id={buffer_id}, "
                    f"size={accumulated_buffer_size[buffer_id]}/{expected_buffer_size}."
                )
                # Emit pending chunks if chunk-yielding is active (train path only).
                if not is_validate:
                    self._emit_pending_chunks(buffer_id)

                # Check if the buffer is the earliest waiting buffer
                # If so, handle the waiting buffer using the abort and truncate strategy
                if not is_validate and self._train_buffer_waiters:
                    min_waiter_buffer_id = min(self._train_buffer_waiters.keys())
                    if min_waiter_buffer_id == buffer_id:
                        await self.handle_waiting_buffer(buffer_id)

                # Check for READY buffers
                if accumulated_buffer_size[buffer_id] == expected_buffer_size and buffer_id not in ready_buffer_ids:
                    psrl_logger.info(f"Adding ready buffer: buffer_id={buffer_id}.")
                    ready_buffer_ids.add(buffer_id)

            # 4. Release TQ state for explicitly dropped entries beyond alg_rollout_n.
            if dropped_payload_keys:
                await async_clear_payload(
                    keys=dropped_payload_keys,
                    partition_id=partition_id,
                    state=PayloadState.DROPPED,
                )

            # 5. Process READY buffers
            for buffer_id in sorted(list(ready_buffer_ids)):
                await self._flush_ready_buffer(buffer_id, is_validate)
            return dispositions

    def maybe_add_buffer(self, buffer_id: int, batch: KVBatchMeta, is_validate: bool = False) -> bool:
        """
        Apply buffer post-processing function if defined and add the buffer to data_buffers.

        Args:
            buffer_id (int): The ID of the buffer to be added.
            batch (KVBatchMeta): The data buffer to be potentially post-processed and added.
            is_validate (bool): Whether the buffer is for validation.
        Returns:
            bool: whether the buffer was added to data_buffers.
        """
        if is_validate:
            self.val_data_buffers[buffer_id] = batch
            psrl_logger.debug(f"Added validation buffer without post-processing: buffer_id={buffer_id}.")
            return True

        add_buffer = True
        if self.buffer_post_process_fn:
            add_buffer, batch = self._buffer_post_process(buffer_id, batch)

        if add_buffer:
            self.train_data_buffers[buffer_id] = batch
            psrl_logger.debug(f"Added post-processed training buffer: buffer_id={buffer_id}.")
        return add_buffer

    def entry_infos_to_kv_batch_meta(
        self,
        entry_infos: list[EntryInfo],
        is_validate: bool = False,
    ) -> KVBatchMeta:
        """Build a ``KVBatchMeta`` for the given EntryInfos."""
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        partition = "val" if is_validate else "train"
        keys: list[str] = []
        tags: list[dict] = []
        for entry_info in entry_infos:
            request_idxs = (
                entry_info.request_idx if isinstance(entry_info.request_idx, list) else [entry_info.request_idx]
            )
            n_trajectories = (
                entry_info.n_trajectory
                if isinstance(entry_info.n_trajectory, list)
                else [entry_info.n_trajectory] * len(request_idxs)
            )
            model_versions = (
                entry_info.model_version if isinstance(entry_info.model_version, list) else [entry_info.model_version]
            )
            # Arrival-correlated fields must be reordered together by `request_idx`
            # to preserve each request's model version.
            if len(model_versions) == len(request_idxs):
                request_idxs, n_trajectories, model_versions = (
                    list(t)
                    for t in zip(*sorted(zip(request_idxs, n_trajectories, model_versions), key=lambda x: x[0]))
                )
            else:
                # A scalar `model_version` covers every request in the entry, so there is
                # nothing to reorder and the `model_versions[-1]` fallback below applies.
                request_idxs, n_trajectories = (
                    list(t) for t in zip(*sorted(zip(request_idxs, n_trajectories), key=lambda x: x[0]))
                )

            for j, (request_idx, n_trajectory) in enumerate(zip(request_idxs, n_trajectories)):
                if n_trajectory == 1:
                    keys.append(f"{entry_info.prompt_id * rollout_n + request_idx}")
                    tags.append(
                        {
                            "uid": entry_info.prompt_id * rollout_n + request_idx,
                            "parent_id": entry_info.prompt_id,
                            "version_tag": (model_versions[j] if j < len(model_versions) else model_versions[-1]),
                            "rollout_instance_id": entry_info.rollout_instance_id,
                        }
                    )
                else:
                    request_id = entry_info.prompt_id * rollout_n + request_idx
                    for trajectory_index in range(n_trajectory):
                        keys.append(f"{request_id}_{trajectory_index}")
                        tags.append(
                            {
                                "uid": request_id,
                                "parent_id": entry_info.prompt_id,
                                "version_tag": (model_versions[j] if j < len(model_versions) else model_versions[-1]),
                                "rollout_instance_id": entry_info.rollout_instance_id,
                            }
                        )
        if (
            not is_validate
            and rollout_n > 1
            and self.group_post_process_fn
            and self.config.reward.launch_reward_fn_async
        ):
            for tag in tags:
                tag["reward_ready"] = True
        return KVBatchMeta(keys=keys, tags=tags, partition_id=partition)

    async def _group_post_process(self, entry_infos: list[EntryInfo]) -> bool:
        """Apply post-processing function to a group of entry infos.

        This method retrieves data from the data pool for each entry, applies
        the group post-processing function, and stores the processed data back.

        Args:
            entry_infos (List[EntryInfo]): List of entry info objects to process

        Returns:
            bool: whether the group data is reserved
        """
        assert self.group_post_process_fn is not None, "Group post-processing function is not set."
        assert all(not entry_info.is_validate for entry_info in entry_infos), (
            "Group post-processing should not be applied to validation data."
        )

        keys = []
        for entry_info in entry_infos:
            if entry_info.n_trajectory == 1:
                keys.append(f"{entry_info.prompt_id * self.rollout_n + entry_info.request_idx}")
            else:
                for i in range(entry_info.n_trajectory):
                    keys.append(f"{entry_info.prompt_id * self.rollout_n + entry_info.request_idx}_{i}")

        # Async reward fields must be present in TQ before filtering.
        if self.config.reward.launch_reward_fn_async:
            await self.reward_manager.wait_for_reward_ready.remote(keys)

        # TODO(linsh): Fetch only columns required for post-processing.
        meta = KVBatchMeta(
            keys=keys,
            tags=[{} for _ in keys],
            partition_id="train",
            fields=None,
        )
        data = await tq.async_kv_batch_get_by_meta(meta)

        processed_data = self.group_post_process_fn(data)
        if processed_data is None:
            await async_clear_payload(
                keys=keys,
                partition_id="train",
                state=PayloadState.DROPPED,
            )
            return False

        # Mutation path: re-upsert the processed TensorDict under the same keys.
        await tq.async_kv_batch_put(keys=keys, partition_id="train", fields=processed_data)
        return True

    def _buffer_post_process(self, buffer_id: int, batch_meta: KVBatchMeta) -> tuple[bool, KVBatchMeta | None]:
        """Apply post-processing function to a full buffer of data.

        This method applies the buffer post-processing function to the data
        in the buffer and stores the processed data if valid.
        Note that the buffer post process only targets training data.

        Args:
            buffer_id (int): The ID of the buffer to process
            buffer_data (KVBatchMeta): The data in the buffer to be processed
        Returns:
            Tuple[bool, Optional[KVBatchMeta]]: A tuple where the first element indicates
            whether to add the buffer to data_buffers, and the second element is the
            processed KVBatchMeta or None if not added.
        """
        assert self.buffer_post_process_fn is not None, "Buffer post-processing function is not set."

        original_keys = batch_meta.keys
        # TODO(linsh): Fetch only columns required for post-processing.
        data = tq.kv_batch_get_by_meta(batch_meta)
        processed_data = self.buffer_post_process_fn(data)

        original_size = len(batch_meta)
        processed_size = 0 if processed_data is None else len(processed_data)

        if processed_data is not None and processed_size == original_size:
            # Just write mutations and keep the original meta.
            tq.kv_batch_put(keys=original_keys, partition_id=batch_meta.partition_id, fields=processed_data)
            return True, batch_meta

        # Clear entries from accumulated_data_buffer
        self.train_accumulated_buffers.pop(buffer_id, None)
        self.train_accumulated_buffer_size.pop(buffer_id, None)
        self.train_accumulated_buffers[buffer_id] = {}
        self.train_accumulated_buffer_size[buffer_id] = 0

        if processed_data is None or processed_size == 0:
            clear_payload(
                keys=original_keys,
                partition_id=batch_meta.partition_id,
                state=PayloadState.DROPPED,
            )
            return False, None

        # Partial clear: recover kept keys from the processor's uid column and
        # rebuild EntryInfo inventory + a tighter KVBatchMeta.
        request_ids = tu.get_non_tensor_data(processed_data, "uid")
        trajectory_indexs = tu.get_non_tensor_data(processed_data, "trajectory_index")
        trajectory_nums = tu.get_non_tensor_data(processed_data, "trajectory_num")

        kept_keys = []
        for request_id, trajectory_index, trajectory_num in zip(request_ids, trajectory_indexs, trajectory_nums):
            if trajectory_num == 1:
                kept_keys.append(f"{request_id}")
            else:
                kept_keys.append(f"{request_id}_{trajectory_index}")
        dropped_keys = [k for k in original_keys if k not in set(kept_keys)]
        if dropped_keys:
            clear_payload(
                keys=dropped_keys,
                partition_id=batch_meta.partition_id,
                state=PayloadState.DROPPED,
            )
        tq.kv_batch_put(keys=kept_keys, partition_id=batch_meta.partition_id, fields=processed_data)

        prompt_entry_infos = self.extract_entry_infos_from_td(processed_data)
        for entry_info in prompt_entry_infos:
            model_version = (
                min(entry_info.model_version)
                if isinstance(entry_info.model_version, list)
                else entry_info.model_version
            )
            self.train_accumulated_buffers[buffer_id].setdefault(model_version, []).append(entry_info)
            self.train_accumulated_buffer_size[buffer_id] += 1

        tags = []
        version_tags = tu.get_non_tensor_data(processed_data, "version_tag")
        rollout_instance_ids = tu.get_non_tensor_data(processed_data, "rollout_instance_id")
        for request_id, version_tag, rollout_instance_id in zip(request_ids, version_tags, rollout_instance_ids):
            tags.append(
                {
                    "uid": request_id,
                    "version_tag": version_tag,
                    "rollout_instance_id": rollout_instance_id,
                }
            )
        if self.rollout_n > 1:
            parent_ids = tu.get_non_tensor_data(processed_data, "parent_id")
            for parent_id, tag in zip(parent_ids, tags):
                tag["parent_id"] = parent_id
        if self.rollout_n > 1 and self.group_post_process_fn and self.config.reward.launch_reward_fn_async:
            for tag in tags:
                tag["reward_ready"] = True

        new_meta = KVBatchMeta(
            keys=kept_keys,
            tags=tags,
            partition_id=batch_meta.partition_id,
            fields=None,
        )

        return False, new_meta

    def extract_entry_infos_from_td(self, data: TensorDict) -> list[EntryInfo]:
        """Extract EntryInfo objects from TensorDict.

        This method extracts EntryInfo objects from the non-tensor batch
        information in the provided TensorDict.

        Args:
            data (TensorDict): The data from which to extract EntryInfo objects
        Returns:
            List[EntryInfo]: List of extracted EntryInfo objects
        """
        _validate_raw = tu.get(data, "validate", False)
        is_validate = bool(_validate_raw[0]) if isinstance(_validate_raw, (list, tuple)) else bool(_validate_raw)
        rollout_n = self.val_rollout_n if is_validate else self.rollout_n
        entry_infos_map: dict[int, EntryInfo] = {}
        if rollout_n > 1:
            parent_ids = tu.get(data, "parent_id")
            rollout_instance_ids = tu.get(data, "rollout_instance_id")
            request_ids = tu.get(data, "uid")
            model_versions = tu.get(data, "version_tag")
            trajectory_nums = tu.get(data, "trajectory_num")
            for parent_id, rollout_instance_id, request_id, model_version, trajectory_num in zip(
                parent_ids, rollout_instance_ids, request_ids, model_versions, trajectory_nums
            ):
                if parent_id in entry_infos_map:
                    entry_info = entry_infos_map[parent_id]
                    if isinstance(entry_info.request_idx, list):
                        entry_info.request_idx.append(request_id % rollout_n)
                    else:
                        entry_info.request_idx = [
                            entry_info.request_idx,
                            request_id % rollout_n,
                        ]
                    if isinstance(entry_info.model_version, list):
                        entry_info.model_version.append(model_version)
                    else:
                        entry_info.model_version = [
                            entry_info.model_version,
                            model_version,
                        ]
                    if isinstance(entry_info.rollout_instance_id, list):
                        entry_info.rollout_instance_id.append(rollout_instance_id)
                    else:
                        entry_info.rollout_instance_id = [
                            entry_info.rollout_instance_id,
                            rollout_instance_id,
                        ]
                    if isinstance(entry_info.n_trajectory, list):
                        entry_info.n_trajectory.append(trajectory_num)
                    else:
                        entry_info.n_trajectory = [
                            entry_info.n_trajectory,
                            trajectory_num,
                        ]
                else:
                    entry_info = EntryInfo(
                        rollout_instance_id=rollout_instance_id,
                        request_idx=request_id % rollout_n,
                        prompt_id=parent_id,
                        model_version=model_version,
                        n_trajectory=trajectory_num,
                        is_validate=is_validate,
                    )
                    entry_infos_map[parent_id] = entry_info
        else:
            request_ids = tu.get_non_tensor_data(data, "uid")
            model_versions = tu.get_non_tensor_data(data, "version_tag")
            rollout_instance_ids = tu.get_non_tensor_data(data, "rollout_instance_id")
            trajectory_nums = tu.get_non_tensor_data(data, "trajectory_num")
            for request_id, model_version, rollout_instance_id, trajectory_num in zip(
                request_ids, model_versions, rollout_instance_ids, trajectory_nums
            ):
                entry_info = EntryInfo(
                    rollout_instance_id=rollout_instance_id,
                    request_idx=0,
                    prompt_id=request_id,
                    model_version=model_version,
                    n_trajectory=trajectory_num,
                    is_validate=is_validate,
                )
                entry_infos_map[request_id] = entry_info
        return list(entry_infos_map.values())

    def log_ready_buffer(self, buffer_id: int, is_validate: bool = False):
        """Log the ready buffer.

        Args:
            buffer_id (int): The ID of the buffer that is ready.
            is_validate (bool): Whether the buffer is for validation data.
        """
        logged_ready_buffer_ids = (
            self.logged_ready_val_buffer_ids if is_validate else self.logged_ready_train_buffer_ids
        )
        if buffer_id not in logged_ready_buffer_ids:
            log_single_event(
                f"{'Train' if not is_validate else 'Validate'} Buffer {buffer_id} is ready",
                psrl_logger,
                event_type=EventType.BUFFER_READY,
            )
            logged_ready_buffer_ids.add(buffer_id)

    async def handle_ready_buffer(self, buffer_id: int, is_validate: bool = False):
        """
        Handle the ready buffer.

        Args:
            buffer_id (int): The ID of the buffer that is ready.
            is_validate (bool): Whether the buffer is for validation data.
        """
        # Check whether there exists ready buffer for training
        self.log_ready_buffer(buffer_id, is_validate)

        psrl_logger.info(f"Checking staleness and aborting requests for buffer {buffer_id}.")
        if not is_validate:
            await self.ps_manager_handle.handle_ready_buffer.remote(buffer_id)
            # The PS manager clears aborted request data from TQ.

        data_buffers = self.val_data_buffers if is_validate else self.train_data_buffers
        _buffer_waiters = self._val_buffer_waiters if is_validate else self._train_buffer_waiters
        if not data_buffers:
            return
        min_ready_buffer_id = min(data_buffers.keys())

        # Wake all Futures waiting for this buffer
        if min_ready_buffer_id in _buffer_waiters:
            batch: KVBatchMeta = self.consume_buffer(min_ready_buffer_id, is_validate=is_validate)
            assert len(_buffer_waiters[min_ready_buffer_id]) == 1, (
                f"Expected only one waiter for buffer {min_ready_buffer_id}, "
                f"but found {len(_buffer_waiters[min_ready_buffer_id])}."
            )
            # Set the result for all futures
            for fut in _buffer_waiters[min_ready_buffer_id]:
                if not fut.done():
                    fut.set_result(batch)
            # Remove the key after waking all waiters
            del _buffer_waiters[min_ready_buffer_id]

            if is_validate:
                await self.ps_manager_handle.maybe_delete_buffer.remote(min_ready_buffer_id, is_validate)
        else:
            psrl_logger.warning(f"No waiter found: buffer_id={buffer_id}.")

    async def handle_waiting_buffer(self, buffer_id: int):
        """Handle the waiting buffer."""
        # WIP(lhy): Implement the retry and truncate strategy
        if self.config.psrl.rollout_coordination.proactive_filter_strategy.method is None:
            return
        if self.config.psrl.rollout_coordination.proactive_filter_strategy.method == "retry":
            gap = self.ready_entries_per_buffer - self.train_accumulated_buffer_size[buffer_id]
            if gap == 0:
                return
            assert gap > 0, f"Gap should be greater than 0, but got {gap}"
            if gap <= self.config.psrl.rollout_coordination.proactive_filter_strategy.threshold:
                psrl_logger.info(f"Preparing buffer recovery: buffer_id={buffer_id}, gap={gap}.")
                # Guarantee other buffers have enough entries to make it ready
                total_available_entries = 0
                for other_buffer_id in sorted(list(self.train_accumulated_buffers.keys()), reverse=True):
                    if other_buffer_id == buffer_id:
                        break
                    total_available_entries += sum(
                        len(self.train_accumulated_buffers[other_buffer_id][model_version])
                        for model_version in self.train_accumulated_buffers[other_buffer_id].keys()
                    )
                if total_available_entries < gap:
                    psrl_logger.info(
                        f"Insufficient recovery entries: buffer_id={buffer_id}, "
                        f"gap={gap}, available={total_available_entries}."
                    )
                    if not self.stop_train_dispatch_task:
                        # More data may still fill the gap.
                        return
                    # Dispatch has stopped, so force readiness below.
                else:
                    psrl_logger.info(f"Recovering buffer with occupied entries: buffer_id={buffer_id}, gap={gap}.")
                    aborted_entry_num, _ = await self.ps_manager_handle.abort_reserved_requests.remote(buffer_id)
                    total_moved_entries = 0
                    moved_occupied_entry_infos: list[EntryInfo] = []
                    for other_buffer_id in sorted(list(self.train_accumulated_buffers.keys()), reverse=True):
                        if other_buffer_id == buffer_id:
                            break
                        for model_version in sorted(list(self.train_accumulated_buffers[other_buffer_id].keys())):
                            moved_entry_infos: list[EntryInfo] = []
                            for entry_info in self.train_accumulated_buffers[other_buffer_id][model_version]:
                                moved_entry_infos.append(entry_info)
                                total_moved_entries += 1
                                if total_moved_entries == gap:
                                    break
                            self.train_accumulated_buffers[buffer_id].setdefault(model_version, [])
                            for moved_entry_info in moved_entry_infos:
                                moved_occupied_entry_infos.append(moved_entry_info)
                                self.train_accumulated_buffers[buffer_id][model_version].append(moved_entry_info)
                                self.train_accumulated_buffer_size[buffer_id] += 1
                                self.train_accumulated_buffers[other_buffer_id][model_version].remove(moved_entry_info)
                                self.train_accumulated_buffer_size[other_buffer_id] -= 1
                            if total_moved_entries == gap:
                                break
                        for model_version in list(self.train_accumulated_buffers[other_buffer_id].keys()):
                            if len(self.train_accumulated_buffers[other_buffer_id][model_version]) == 0:
                                self.train_accumulated_buffers[other_buffer_id].pop(model_version)
                        if total_moved_entries == gap:
                            break
                    for other_buffer_id in list(self.train_accumulated_buffers.keys()):
                        if len(self.train_accumulated_buffers[other_buffer_id]) == 0:
                            self.train_accumulated_buffers.pop(other_buffer_id)
                            self.train_accumulated_buffer_size.pop(other_buffer_id)
                    # Finally, notify the PS manager to move the occupied entries to the buffer
                    await self.ps_manager_handle.move_occupied_entries.remote(moved_occupied_entry_infos, buffer_id)
                    psrl_logger.info(
                        f"Moved occupied entries: count={total_moved_entries}, "
                        f"gap={gap}, destination_buffer_id={buffer_id}."
                    )
                    if aborted_entry_num > 0:
                        await self._refill_failed_group(
                            n_prompts=aborted_entry_num,
                            context="Buffer recovery compensation",
                        )

                    return  # Normal accumulation completes the recovered buffer.

            # A partial buffer must be forced ready after dispatch stops or waiters block.
            remaining_gap = self.ready_entries_per_buffer - self.train_accumulated_buffer_size[buffer_id]
            if self.stop_train_dispatch_task and remaining_gap > 0:
                psrl_logger.warning(
                    f"Forcing partial buffer ready: buffer_id={buffer_id}, "
                    f"size={self.train_accumulated_buffer_size[buffer_id]}/{self.ready_entries_per_buffer}, "
                    f"gap={remaining_gap}."
                )
                await self.ps_manager_handle.abort_reserved_requests.remote(buffer_id)
                # Override the counter so the caller's equality check sees the buffer as full.
                self.train_accumulated_buffer_size[buffer_id] = self.ready_entries_per_buffer
        elif self.config.psrl.rollout_coordination.proactive_filter_strategy.method == "truncate":
            raise NotImplementedError("Truncate strategy is not implemented yet.")

    def _emit_pending_chunks(self, buffer_id: int) -> None:
        """
        Resolve pending chunk waiters from accumulated data.

        Chunks preserve accumulation order across emissions and sort by `prompt_id`
        only within each chunk. The final chunk may be smaller.
        """
        if self.train_chunk_size is None:
            return
        if buffer_id not in self.train_accumulated_buffers:
            return

        # Flat list in stable insertion order (version then append order).
        accumulated_buffers = self.train_accumulated_buffers[buffer_id]
        all_entry_infos: list = []
        for model_version in sorted(accumulated_buffers.keys()):
            all_entry_infos.extend(accumulated_buffers[model_version])

        emitted_entry_ids = self._train_chunk_emitted_entry_ids.setdefault(buffer_id, set())
        pending_entry_infos = [entry for entry in all_entry_infos if id(entry) not in emitted_entry_ids]
        consumed = self._train_chunk_consumed.get(buffer_id, 0)
        ready_total = self.ready_entries_per_buffer
        accumulated_size = self.train_accumulated_buffer_size.get(buffer_id, len(all_entry_infos))
        is_buffer_complete = accumulated_size >= ready_total

        while True:
            available = len(pending_entry_infos)
            chunk_idx = consumed // self.train_chunk_size
            key = (buffer_id, chunk_idx)

            # Determine if we can emit this chunk.
            is_last = False
            if is_buffer_complete and available > 0 and available < self.train_chunk_size:
                # Tail chunk (remainder after last full chunk).
                emit_count = available
                is_last = True
            elif available >= self.train_chunk_size:
                emit_count = self.train_chunk_size
                # is_last if this chunk reaches the total.
                is_last = consumed + emit_count >= ready_total
            else:
                break

            # Slice by accumulation order, then sort within the chunk for
            # deterministic GRPO group layout without disturbing the consumed prefix.
            chunk_entries = pending_entry_infos[:emit_count]
            del pending_entry_infos[:emit_count]
            chunk_entries = sorted(chunk_entries, key=lambda ei: ei.prompt_id)
            chunk_batch = self.entry_infos_to_kv_batch_meta(chunk_entries, is_validate=False)

            result = (chunk_batch, is_last)
            # WARNING so this shows under default PSRL_LOGGING_LEVEL=WARN.
            psrl_logger.warning(
                "Emitted train chunk buffer=%d idx=%d size=%d is_last=%s (consumed %d -> %d / %d)",
                buffer_id,
                chunk_idx,
                len(chunk_batch),
                is_last,
                consumed,
                consumed + emit_count,
                ready_total,
            )

            # Keep a resolved result for late waiters. Remove it after waking
            # registered waiters to prevent duplicate delivery.
            if key in self._train_chunk_waiters:
                for fut in self._train_chunk_waiters[key]:
                    if not fut.done():
                        fut.set_result(result)
                del self._train_chunk_waiters[key]
                self._resolved_train_chunks.pop(key, None)
            else:
                self._resolved_train_chunks[key] = result

            emitted_entry_ids.update(id(entry) for entry in chunk_entries)
            self._train_chunk_consumed[buffer_id] = consumed + emit_count
            consumed += emit_count

            if is_last:
                break

    async def wait_for_training_batch(self, buffer_id: int) -> KVBatchMeta:
        """Await a training batch for a specific buffer ID.

        Raises:
            RuntimeError: When the refill breaker has tripped. This is the driver's
                only blocking call into the manager, so it is where an unfillable
                buffer becomes a visible failure instead of a hang.
        """
        self._raise_if_refill_breaker_tripped()

        await self.ps_manager_handle.ensure_train_buffer_exists.remote(buffer_id)

        if buffer_id in self.train_data_buffers:
            # If the buffer is ready, return immediately
            psrl_logger.info(f"Training buffer ready: buffer_id={buffer_id}.")
            return self.consume_buffer(buffer_id)

        # TODO(lhy): Support additional stuck-buffer consumption strategies.
        if buffer_id in self.train_accumulated_buffers:
            async with AsyncBusyPollingRayLock(self.ps_manager_handle):
                await self.handle_waiting_buffer(buffer_id)

                if self.train_accumulated_buffer_size[buffer_id] == self.ready_entries_per_buffer:
                    # Share the single publish path so READY validation, chunk
                    # emission, and accumulator cleanup stay consistent.
                    add_buffer = await self._flush_ready_buffer(buffer_id, is_validate=False)
                    if add_buffer:
                        psrl_logger.info(f"Recovered training buffer: buffer_id={buffer_id}.")
                        return self.consume_buffer(buffer_id)

        # If the buffer is still not ready after the abort and truncate strategy, wait for it to be ready
        psrl_logger.info(f"Waiting for training buffer: buffer_id={buffer_id}.")
        fut = asyncio.get_event_loop().create_future()
        self._train_buffer_waiters.setdefault(buffer_id, []).append(fut)
        batch_meta = await fut
        return batch_meta

    async def wait_for_training_chunk(self, buffer_id: int, chunk_index: int) -> tuple["KVBatchMeta", bool]:
        """
        Await a specific indexed chunk for a buffer.

        The returned boolean is true only for the final chunk.

        Raises:
            RuntimeError: When the refill breaker has tripped. `fine_grain_overlap`
                blocks only here, so this guard is what ends the run under that
                strategy.
        """
        self._raise_if_refill_breaker_tripped()

        key = (buffer_id, chunk_index)
        if key in self._resolved_train_chunks:
            result = self._resolved_train_chunks.pop(key)
            psrl_logger.warning(
                "wait_for_training_chunk: buffer=%d idx=%d ready immediately (is_last=%s, size=%d)",
                buffer_id,
                chunk_index,
                result[1],
                len(result[0]),
            )
            return result

        fut = asyncio.get_running_loop().create_future()
        self._train_chunk_waiters.setdefault(key, []).append(fut)

        # Re-check after registration in case emit raced between the miss and append.
        if key in self._resolved_train_chunks:
            result = self._resolved_train_chunks.pop(key)
            waiters = self._train_chunk_waiters.get(key, [])
            if fut in waiters:
                waiters.remove(fut)
                if not waiters:
                    del self._train_chunk_waiters[key]
            if not fut.done():
                fut.set_result(result)
            psrl_logger.warning(
                "wait_for_training_chunk: buffer=%d idx=%d resolved via race re-check (is_last=%s)",
                buffer_id,
                chunk_index,
                result[1],
            )
            return result

        psrl_logger.warning(
            "wait_for_training_chunk: buffer=%d idx=%d waiting for emit...",
            buffer_id,
            chunk_index,
        )
        return await fut

    async def wait_for_validation_batch(self, buffer_id: int) -> KVBatchMeta:
        """Await a validation batch, returning a ``KVBatchMeta``."""
        async with AsyncBusyPollingRayLock(self.ps_manager_handle):
            await self.ps_manager_handle.ensure_validate_buffer_exists.remote()

        if buffer_id in self.val_data_buffers:
            # If the buffer is ready, return immediately
            psrl_logger.info(f"Validation buffer ready: buffer_id={buffer_id}.")
            return self.consume_buffer(buffer_id, is_validate=True)

        # An all-failed validation round cannot assemble a buffer, so return an
        # empty batch for late waiters.
        if self._val_round_all_failed:
            psrl_logger.warning(
                "Validate buffer %d: all groups in this round already failed. "
                "returning an empty batch to avoid deadlock.",
                buffer_id,
            )
            return KVBatchMeta(keys=[], tags=[], partition_id="val")

        # TODO(lhy): Support partial, truncate, and relocation strategies.

        psrl_logger.info(f"Waiting for validation buffer: buffer_id={buffer_id}.")
        fut: asyncio.Future = asyncio.get_event_loop().create_future()
        self._val_buffer_waiters.setdefault(buffer_id, []).append(fut)
        return await fut

    async def generate_validate_sequences(self) -> int:
        """Dispatch a validation batch and return its buffer ID."""
        test_batch: TensorDict = await self.data_processor.get_single_controller_batch.remote(
            DatasetType.val, return_meta=False
        )
        prompt_num = len(test_batch) // self.val_rollout_n
        self.set_val_buffer_size(prompt_num)
        await self.ps_manager_handle.set_val_staleness_inventory_capacity.remote(prompt_num)

        # Batch dispatch: register all request IDs and send the full batch in one
        # call for maximal dispatch throughput and vLLM batching efficiency.
        all_request_ids = tu.get(test_batch, "uid")
        await self.ps_manager_handle.add_request.remote(all_request_ids, is_validate=True)
        await self.put_data(test_batch, is_validate=True)
        self._val_buffer_id += 1

        return self._val_buffer_id - 1

    def log_buffer(self, buffer_id: int, is_validate: bool = False):
        """Log a histogram of ``version_tag`` values for the given buffer."""
        data_buffer = self.val_data_buffers if is_validate else self.train_data_buffers
        assert buffer_id in data_buffer, (
            f"Missing buffer: buffer_id={buffer_id}, partition={('val' if is_validate else 'train')!r}."
        )

        version_tags = [tag.get("version_tag", -1) for tag in data_buffer[buffer_id].tags]

        # Count different version_tags
        version_tag_counts = Counter(version_tags)
        total_count = len(version_tags)

        # Calculate staleness for each version_tag
        staleness_dict = {
            version_tag: (None if is_validate else buffer_id - version_tag)
            for version_tag in version_tag_counts.keys()
        }

        psrl_logger.info(
            f"Buffer version distribution: partition={('validation' if is_validate else 'training')!r}, "
            f"buffer_id={buffer_id}."
        )
        for version_tag in sorted(version_tag_counts.keys()):
            count = version_tag_counts[version_tag]
            percentage = (count / total_count) * 100
            staleness = staleness_dict[version_tag]
            psrl_logger.info(f"version_tag={version_tag}: count={count} ({percentage:.2f}%), staleness={staleness}")

    def consume_buffer(self, buffer_id: int, is_validate: bool = False) -> KVBatchMeta:
        """
        Consume (retrieve and remove) all data from the specified buffer.

        Args:
            buffer_id (int): The ID of the buffer to consume.
            is_validate (bool): Whether the buffer is for validation data.
        Returns:
            KVBatchMeta: The concatenated data from the buffer.
        Raises:
            AssertionError: If the buffer is not in READY state.
        """

        self.log_buffer(buffer_id, is_validate)
        buffer = (
            self.val_data_buffers.pop(buffer_id, None) if is_validate else self.train_data_buffers.pop(buffer_id, None)
        )
        assert buffer is not None, f"Missing or consumed buffer: buffer_id={buffer_id}."
        # NOTE(linsh): Delay PS-side deletion because remaining in-flight requests stay trainable.
        if not is_validate:
            # Clear chunk-yielding bookkeeping for this buffer.
            self._train_chunk_consumed.pop(buffer_id, None)
            stale_waiter_keys = [k for k in self._train_chunk_waiters if k[0] == buffer_id]
            for k in stale_waiter_keys:
                del self._train_chunk_waiters[k]
            stale_resolved_keys = [k for k in self._resolved_train_chunks if k[0] == buffer_id]
            for k in stale_resolved_keys:
                del self._resolved_train_chunks[k]
        return buffer
