import asyncio
import atexit
import logging
import os
import traceback
import uuid
from collections import deque

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.workers.config.model import HFModelConfig

from psrl.utils.common.chat_template import resolve_chat_template_value
from psrl.utils.common.docker_utils import (
    force_remove_containers_by_label,
    spawn_actor_reaper,
)
from psrl.utils.common.http_io_thread import init_http_io_thread
from psrl.utils.common.http_utils import configure_distributed_post, init_http_client
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events
from psrl.utils.rollout.rollout_trace import RolloutTraceConfig, rollout_trace_attr
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.utils import AGENT_LOOP_REGISTRY, DictConfigWrap, TerminateReason
from psrl.workers.gen.utils import TokenOutput

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class AgentLoopWorkerBase:
    """Host agent loops in one actor process, independent of what consumes their output.

    Owns everything that running an agent loop requires regardless of the caller's
    purpose: the Docker container reaper scoped to this actor, the HTTP client and
    I/O thread, the agent-loop registry, the chat template, the rollout trace config,
    the pending-program queue, and the retry loop around each episode.

    What happens to a finished trajectory is left to subclasses. RL commits it to
    TransferQueue and advances the request through PSManager, while batch rollout
    serializes it to disk. Subclasses customize three hooks:

    - `_init_data_plane`: attach to whatever transport carries the payload.
    - `_handle_output`: take ownership of a completed trajectory.
    - `_handle_failure`: take ownership of an episode that produced nothing.
    """

    def __init__(
        self,
        config: DictConfig,
        rollout_gateway_url: str,
        session_router_url: str,
        worker_id: int = 0,
        worker_num: int = 1,
        log_prefix: str = "AgentLoopWorker",
    ):
        """Initialize the agent loop host.

        Args:
            config (DictConfig): Configuration containing model and rollout settings.
            rollout_gateway_url (str): HTTP base URL of the SMG rollout gateway.
            session_router_url (str): URL of the session router.
            worker_id (int): Unique identifier for this worker instance.
            worker_num (int): Total number of worker instances.
            log_prefix (str): Stem of this worker's log file, without the worker index.
        """

        # Actor-scoped labels let the reaper reclaim only containers owned by this
        # process after abnormal termination.
        self._actor_id = f"w{worker_id}-{os.uname().nodename}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        os.environ["PSRL_ACTOR_ID"] = self._actor_id
        # Use the config parameter directly (self.config is set below) so the
        # reaper log lands next to the AgentLoopWorker_N.log files.
        _reaper_log_dir = getattr(getattr(config, "psrl", None), "logging_path", None)
        self._reaper_proc = spawn_actor_reaper(
            self._actor_id,
            log_dir=_reaper_log_dir,
        )
        # Graceful shutdown reaps containers before stopping the sidecar.
        atexit.register(self._terminate_reaper)
        psrl_logger.info(
            f"{type(self).__name__} {worker_id}: actor_id={self._actor_id!r}, "
            f"reaper pid={self._reaper_proc.pid}, "
            f"reaper log_dir={_reaper_log_dir!r}."
        )

        self.config = config
        model_config = config.gen_actor_rollout_ref.model
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.overlong_filtering = bool(config.gen_actor_rollout_ref.rollout.agent.get("overlong_filtering", False))
        if self.overlong_filtering:
            psrl_logger.warning("Overlong filtering enabled: budget-truncated trajectories contribute no gradient.")

        self._init_data_plane()

        self.dataset_cls = get_dataset_class(config.data)

        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        self.rollout_gateway_url = rollout_gateway_url
        self.session_router_url = session_router_url

        n_rollout_instances = self.config.psrl.deployment.n_rollout_instances
        n_validate_instances = (
            self.config.psrl.deployment.n_validate_instances if self.config.psrl.colocate_validate_and_train else 0
        )
        server_max_concurrency = self.config.psrl.rollout_gateway.server_max_concurrency

        init_http_client(
            server_concurrency=server_max_concurrency,
            rollout_engine_num=n_rollout_instances + n_validate_instances,
            producer_count=worker_num,
            producer_index=worker_id,
        )

        # Dedicated HTTP I/O thread (event loop isolation).
        init_http_io_thread(
            server_concurrency=server_max_concurrency,
            rollout_engine_num=n_rollout_instances + n_validate_instances,
            producer_count=worker_num,
            producer_index=worker_id,
        )

        self.agent_programs = set()
        self.put_tasks = set()
        self.pending_program_queue = deque()
        self.running_loop = None
        self.busy_loop_task = None
        self.stop_busy_loop_task = False

        # Register agent loop configs from file
        agent_loop_config_path = config.gen_actor_rollout_ref.rollout.agent.agent_loop_config_path
        if agent_loop_config_path:
            agent_loop_configs = OmegaConf.load(agent_loop_config_path)
            for agent_loop_config in agent_loop_configs:
                AGENT_LOOP_REGISTRY[agent_loop_config.name] = agent_loop_config
        custom_template_value = config.gen_actor_rollout_ref.model.get("custom_chat_template", None)
        resolved_template = resolve_chat_template_value(custom_template_value)
        if resolved_template is not None:
            if self.model_config.processor is not None:
                self.model_config.processor.chat_template = resolved_template
            self.model_config.tokenizer.chat_template = resolved_template
            psrl_logger.info(f"Applied custom chat template: source={custom_template_value!r}.")

        # Initialize rollout trace config
        trace_config = self.config.gen_actor_rollout_ref.rollout.get("trace", {})
        RolloutTraceConfig.init(
            self.config.trainer.project_name,
            self.config.trainer.experiment_name,
            trace_config.get("backend"),
            trace_config.get("token2text", False),
            trace_config.get("max_samples_per_step_per_worker", None),
        )

        # Build logger
        self.log_prefix = f"{log_prefix}_I{worker_id}"
        handler = DualOutputHandler(self.config.psrl.logging_path, self.log_prefix)
        logging.getLogger("psrl").addHandler(handler)
        psrl_logger.addHandler(handler)

    ###### Subclass hooks ######

    def _init_data_plane(self) -> None:
        """Attach to the transport that carries trajectory payloads.

        Called once during construction, before any agent loop runs. The base host
        needs no transport of its own.
        """

    def _build_agent_loop_context(self) -> AgentLoopContext:
        """Bundle the framework dependencies handed to every agent loop.

        Subclasses that own a reward manager or a parameter server supply them here.
        """
        return AgentLoopContext(
            config=self.config,
            rollout_gateway_url=self.rollout_gateway_url,
            session_router_url=self.session_router_url,
            reward_manager=None,
            ps_manager_handle=None,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )

    async def _handle_output(
        self,
        output: "TokenOutput | list[TokenOutput]",
        batch: TensorDict,
        terminate_reason: TerminateReason,
    ) -> None:
        """Take ownership of a completed trajectory.

        Args:
            output (TokenOutput | list[TokenOutput]): Trajectories the loop produced.
            batch (TensorDict): The originating single-row prompt batch.
            terminate_reason (TerminateReason): Why the episode stopped.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _handle_output().")

    async def _handle_failure(
        self,
        batch: TensorDict,
        terminate_reason: TerminateReason,
    ) -> None:
        """Take ownership of an episode that produced no trajectory.

        Args:
            batch (TensorDict): The originating single-row prompt batch.
            terminate_reason (TerminateReason): Why the episode produced nothing.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement _handle_failure().")

    ###### Container reaping ######

    def _terminate_reaper(self) -> None:
        """Belt-and-suspenders cleanup on graceful actor shutdown.

        Belt: synchronously force-remove our actor's containers from the
              actor process itself. Takes ~5-30 s for hundreds of containers,
              well within Ray's SIGTERM grace period. This is the fast path
              that wins the race against the bash sidecar.
        Suspenders: also signal the bash sidecar to terminate so it does not
                    run a redundant (and harmless) post-mortem sweep after we
                    already cleaned up here.
        """
        try:
            force_remove_containers_by_label("psrl.actor_id", self._actor_id)
        except Exception as e:
            psrl_logger.debug(f"Synchronous atexit reap failed: {e}.")
        proc = getattr(self, "_reaper_proc", None)
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception as e:
            psrl_logger.debug(f"Failed to terminate reaper sidecar: {e}.")

    ###### Program admission ######

    def set_distributed_post_actors(
        self,
        actors: list[ray.actor.ActorHandle] | None,
        enabled: bool,
        producer_index: int = 0,
    ):
        """Install distributed HTTP POST actors for this worker process."""
        configure_distributed_post(
            actors,
            enabled=enabled,
            start_index=producer_index,
        )

    def add_agent_program(self, batch: TensorDict | None):
        """Add a new agent program to the pending queue for processing.

        Args:
            batch (TensorDict or None): Data to process, or None to signal termination.
        """
        if batch is None:
            self.pending_program_queue.append(None)
            return

        n = len(batch)
        if n == 0:
            return
        requests = batch.chunk(n)
        for request in requests:
            self.pending_program_queue.append(request)

    def start_busy_loop(self):
        """Start the busy loop to continuously process agent programs from the queue."""
        if self.busy_loop_task is not None and not self.busy_loop_task.done():
            return

        # Start the background task to process data
        self.stop_busy_loop_task = False
        self.running_loop = asyncio.get_running_loop()
        self.busy_loop_task = self.running_loop.create_task(self._launch_agent_loop())
        self.busy_loop_task.add_done_callback(lambda f: f.result())  # To avoid silent error in async tasks

    async def stop_busy_loop(self):
        """Stop the busy loop and wait for the current task to complete."""
        if not self.busy_loop_task or self.busy_loop_task.done():
            return

        self.stop_busy_loop_task = True
        # Wait for the background task to finish
        # Note: This is now async-safe and won't deadlock when called from Ray actors
        try:
            await asyncio.wait_for(self.busy_loop_task, timeout=10.0)
        except asyncio.TimeoutError:
            psrl_logger.warning("Timeout waiting for busy loop task to complete")
            self.busy_loop_task.cancel()

    async def _launch_agent_loop(self):
        """Main loop that processes agent programs from the pending queue."""
        while not self.stop_busy_loop_task:
            if len(self.pending_program_queue) > 0:
                program = self.pending_program_queue.popleft()
                if program is None:
                    self.stop_busy_loop_task = True
                    continue
                await self.generate_trajectory(program)
            await asyncio.sleep(0)

    def in_flight(self) -> int:
        """Count episodes this worker is still running or has yet to start.

        A caller with no other completion signal, such as an offline collection
        driver, uses this to tell "still working" from "finished".
        """
        return len(self.agent_programs) + len(self.pending_program_queue)

    def _create_task_done_callback(self, task):
        """Create a callback function to handle task completion."""

        def task_done_callback(future):
            try:
                future.result()  # This will raise an exception if the task failed
            except Exception as e:
                tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                psrl_logger.error(f"Task failed: task={task!r}, error={e!r}.\nTraceback:\n{tb_str}")
            finally:
                self.agent_programs.discard(task)

        return task_done_callback

    async def generate_trajectory(self, batch: TensorDict):
        """Generate trajectories using the specified agent type based on configuration.

        This method only create the task (agent_loop) and add the task to the agent_programs set.
        But the task is not await here so different agent_loop can be run in parallel.

        Args:
            batch (TensorDict): Input batch metadata containing prompts and metadata.
        """
        assert len(batch) == 1, "Only support single request for generation"

        default_agent_name = self.config.gen_actor_rollout_ref.rollout.agent.default_agent_loop
        agent_name = tu.get(batch, "agent_name", [default_agent_name])[0]
        task = asyncio.create_task(self._run_agent_loop(agent_name, batch))
        task.add_done_callback(self._create_task_done_callback(task))
        self.agent_programs.add(task)

    ###### Episode execution ######

    async def _run_agent_loop(
        self,
        agent_name: str,
        batch: TensorDict,
    ):
        """Execute the specified agent loop on the given requests.

        This method instantiates the agent loop based on the registered configuration
        and runs it with the provided requests. It handles retries based on termination reasons.

        Args:
            agent_name (str): Name of the agent loop to run.
            batch (TensorDict): Input batch metadata containing prompts and metadata.
        """
        assert len(batch) == 1, "Only support single request for generation"

        if "parent_id" in batch:
            prompt_index = tu.get(batch, "parent_id")[0]
            request_index = tu.get(batch, "uid")[0]
        else:
            prompt_index = tu.get(batch, "uid")[0]
            request_index = tu.get(batch, "uid")[0]

        try:
            await self._run_agent_loop_inner(agent_name, batch, prompt_index, request_index)
        except Exception as e:
            tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            psrl_logger.error(
                f"Agent loop failed: name={agent_name!r}, request_id={request_index}, "
                f"prompt_id={prompt_index}, error={e!r}.\n"
                f"Full traceback:\n{tb_str}"
            )
            raise

    async def _run_agent_loop_inner(
        self,
        agent_name: str,
        batch: TensorDict,
        prompt_index,
        request_index,
    ):
        """Inner implementation of the agent loop execution."""
        request_ids = tu.get(batch, "uid")

        global_steps = tu.get(batch, "global_steps", -1)
        validate = tu.get(batch, "validate", False)[0]

        with rollout_trace_attr(
            prompt_index=prompt_index,
            request_index=request_index,
            step=global_steps,
            name=agent_name,
            validate=validate,
        ):
            assert agent_name in AGENT_LOOP_REGISTRY, (
                f"Unregistered agent loop: name={agent_name!r}, available={AGENT_LOOP_REGISTRY.keys()!r}."
            )
            agent_loop_config = AGENT_LOOP_REGISTRY[agent_name]

            context = self._build_agent_loop_context()
            # Keep framework objects out of Hydra's dataclass conversion path.
            agent_loop_factory = hydra.utils.instantiate(
                config=agent_loop_config,
                _partial_=True,
            )
            agent_loop = agent_loop_factory(context=context)

            with log_dual_events(
                f"Agent loop with requests {request_ids}",
                psrl_logger,
                level=logging.DEBUG,
                event_type=EventType.GEN,
            ):
                retry_limit = self.config.gen_actor_rollout_ref.rollout.agent.retry_limit
                raised_error = None
                for retry_attempt in range(1, retry_limit + 1):
                    raise_on_error = (
                        retry_attempt == retry_limit
                    ) and self.config.gen_actor_rollout_ref.rollout.agent.raise_on_error
                    try:
                        output, terminate_reason = await agent_loop.run_with_termination_handling(
                            batch, raise_on_error=raise_on_error
                        )
                    except Exception as e:
                        # Log the traceback before cleanup and propagation.
                        tb_str = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                        psrl_logger.error(
                            f"Agent loop failed before cleanup: request_ids={request_ids!r}.\nTraceback:\n{tb_str}"
                        )
                        raised_error = e
                        terminate_reason = TerminateReason.ROLLOUT_ERROR
                        output = None

                    if not terminate_reason.needs_worker_retry():
                        break

                    # Retry if applicable
                    if retry_attempt < retry_limit:
                        psrl_logger.warning(
                            f"Retrying agent loop: request_ids={request_ids!r}, "
                            f"reason={terminate_reason.value!r}, "
                            f"attempt={retry_attempt}/{retry_limit}."
                        )
                        continue

                if terminate_reason.needs_worker_retry() or terminate_reason.is_aborted:
                    psrl_logger.warning(
                        f"Agent loop exhausted retries: request_ids={request_ids!r}, "
                        f"reason={terminate_reason.value!r}, attempts={retry_limit}."
                    )
                    output = None

                if not terminate_reason.needs_manager_retry():
                    psrl_logger.debug(
                        f"Agent loop terminated: request_ids={request_ids!r}, reason={terminate_reason.value!r}."
                    )

            # NOTE(claude): Every manager-retry reason is also a worker-retry reason,
            # which the branch above nulls the output for, so the hooks stay exclusive.
            if output is not None:
                await self._handle_output(output, batch, terminate_reason)
            else:
                await self._handle_failure(batch, terminate_reason)

            # After ALL cleanup is complete, re-raise the original error so it
            # propagates to the task callback.
            if raised_error is not None:
                raise raised_error
