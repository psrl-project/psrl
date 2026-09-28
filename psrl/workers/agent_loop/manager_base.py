import asyncio
import logging
import os

import ray
from omegaconf import DictConfig
from tensordict import TensorDict
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig

from psrl.utils.common.http_utils import init_distributed_post_pool
from psrl.utils.logger import DualOutputHandler

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class AgentLoopManagerBase:
    """Feed prompts to a pool of agent loop workers, independent of what collects results.

    Owns the concerns shared by every caller that drives agent loops: the bounded
    prompt queue that provides backpressure, round-robin dispatch that co-locates
    the children of one prompt on one worker, the worker busy-loop lifecycle, and
    the distributed HTTP POST actor pool.

    Result collection is left to subclasses. RL accumulates trajectories into
    version-bucketed staleness buffers and drives the parameter server, while batch
    rollout writes each one to disk as it lands. Subclasses customize two hooks:

    - `_before_dispatch`: gate or stamp a batch before it is fanned out.
    - `_on_dispatch`: react to a batch once its destination workers are chosen.
    """

    def __init__(
        self,
        config: DictConfig,
        data_queue_size: int,
        agent_loop_workers: list[ray.actor.ActorHandle],
        log_prefix: str = "AgentLoopManager",
    ):
        """Initialize the dispatch half of an agent loop manager.

        Args:
            config (DictConfig): Configuration containing rollout settings.
            data_queue_size (int): Capacity of the prompt queue. This is the
                backpressure bound: a full queue blocks the producer.
            agent_loop_workers (list[ray.actor.ActorHandle]): Worker actors to
                dispatch to.
            log_prefix (str): Stem of this manager's log file.
        """
        self.config = config
        model_config = config.gen_actor_rollout_ref.model
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        self._init_data_plane()

        self.train_data_queue: asyncio.Queue = asyncio.Queue(maxsize=data_queue_size)
        self.agent_loop_workers = agent_loop_workers
        self.distributed_post_actors: list[ray.actor.ActorHandle] = []

        self._request_counter = 0
        self._dispatch_idx = 0
        self.running_loop: asyncio.AbstractEventLoop | None = None
        self.train_dispatch_task: asyncio.Task | None = None
        self.stop_train_dispatch_task = False
        self._extra_tasks: list[asyncio.Task] = []

        self.log_prefix = log_prefix
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))

    ###### Subclass hooks ######

    def _init_data_plane(self) -> None:
        """Attach to the transport that carries trajectory payloads.

        Called once during construction. The base dispatcher needs no transport.
        """

    async def _before_dispatch(self, data: TensorDict) -> None:
        """Gate or stamp one batch before it is fanned out to workers.

        Runs after the batch leaves the queue and before any worker sees it, so an
        implementation may block here to throttle dispatch.

        Args:
            data (TensorDict): The batch about to be dispatched.
        """

    async def _on_dispatch(self, data: TensorDict, is_validate: bool = False) -> bool:
        """React to one batch once its destination workers are known.

        Args:
            data (TensorDict): The batch being dispatched.
            is_validate (bool): Whether this batch belongs to a validation round.

        Returns:
            bool: Whether to proceed with the dispatch.
        """
        return True

    ###### Rollout concurrency ######

    @property
    def n_active_rollout_instances(self) -> int:
        """Count the rollout engines this manager's HTTP concurrency is sized against."""
        n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )
        return n_rollout_instances + n_validate_instances

    async def _init_distributed_post_pool(self) -> None:
        if not self.config.psrl.rollout_gateway.use_distributed_post or self.distributed_post_actors:
            return

        n_active_instance = self.n_active_rollout_instances

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

    ###### Prompt queue ######

    async def put_data(self, batch: TensorDict):
        """Put a batch of prompts into the manager's queue.

        Blocks while the queue is at capacity, which is what bounds the number of
        in-flight episodes.
        """
        await self.train_data_queue.put(batch)

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
                    "Received END signal, stopping train dispatch. request_counter=%d.",
                    self._request_counter,
                )
                self.stop_train_dispatch_task = True
                continue

            await self._before_dispatch(data)

            # Dispatch data to agent loop workers
            await self._inner_dispatch_data(data)
            # Increment counter after dispatch so dispatch throttling reflects the
            # number of requests that have actually been sent out.
            self._request_counter += len(data)
            await asyncio.sleep(0)  # Yield control to the event loop

    async def _inner_dispatch_data(self, data: TensorDict, is_validate: bool = False):
        """Fan one batch out to the workers chosen by the dispatch plan."""
        if not await self._on_dispatch(data, is_validate=is_validate):
            return

        dispatch_plan = self.get_dispatch_plan(data, is_validate=is_validate)
        for worker_index, batch in dispatch_plan.items():
            self.agent_loop_workers[worker_index].add_agent_program.remote(batch)

    def get_dispatch_plan(self, data: TensorDict, is_validate: bool = False) -> dict[int, TensorDict]:
        """Round-robin dispatch plan keyed by worker index, co-locating siblings.

        Children sharing a ``parent_id`` (group sampling) land on the same worker.
        """
        from verl.utils import tensordict_utils as tu

        keys_by_worker: dict[int, list[str]] = {}
        prompt_to_worker: dict[int, int] = {}
        rollout_n = self.dispatch_rollout_n(is_validate)
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

    def dispatch_rollout_n(self, is_validate: bool = False) -> int:
        """Return the sibling count used to group prompt children onto one worker."""
        return self.rollout_n

    ###### Lifecycle ######

    def _spawn_extra_tasks(self) -> list[asyncio.Task]:
        """Spawn background tasks this manager needs beyond train dispatch.

        Returns:
            list[asyncio.Task]: Tasks `stop_busy_loop` must await before teardown.
        """
        return []

    def _request_extra_task_stop(self) -> None:
        """Signal the tasks from `_spawn_extra_tasks` to finish."""

    def _has_running_tasks(self) -> bool:
        """Return whether any background task of this manager is still running."""
        return self.train_dispatch_task is not None and not self.train_dispatch_task.done()

    async def start_busy_loop(self):
        """Start the dispatch loop and the workers' own busy loops."""
        if self._has_running_tasks():
            return

        # Start the busy loop of agent loop workers.
        await self._init_distributed_post_pool()
        await asyncio.gather(*[worker.start_busy_loop.remote() for worker in self.agent_loop_workers])

        # Start the background task to process data
        self.running_loop = asyncio.get_running_loop()
        self.train_dispatch_task = self.running_loop.create_task(self._train_dispatch_data())
        self.train_dispatch_task.add_done_callback(lambda f: f.result())
        self._extra_tasks = self._spawn_extra_tasks()

    async def stop_busy_loop(self):
        """Stop the dispatch loop and wait for the workers to drain."""
        if not self._has_running_tasks():
            return

        self.stop_train_dispatch_task = True
        self._request_extra_task_stop()
        await asyncio.gather(self.train_dispatch_task, *getattr(self, "_extra_tasks", []))

        await asyncio.gather(*[worker.stop_busy_loop.remote() for worker in self.agent_loop_workers])
        await self._shutdown_distributed_post_pool()
