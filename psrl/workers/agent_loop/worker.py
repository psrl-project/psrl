import asyncio
import logging
import os
import traceback
import uuid
from collections import deque

import hydra
import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from verl.trainer.distillation import is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.dataset.rl_dataset import get_dataset_class
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
)
from verl.workers.config.model import HFModelConfig

from psrl.sandbox.config import build_sandbox_manager
from psrl.utils.common.chat_template import resolve_chat_template_value
from psrl.utils.common.http_io_thread import init_http_io_thread
from psrl.utils.common.http_utils import configure_distributed_post, init_http_client
from psrl.utils.logger import DualOutputHandler, EventType, log_dual_events
from psrl.utils.rollout.rollout_trace import RolloutTraceConfig, rollout_trace_attr
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.utils import AGENT_LOOP_REGISTRY, DictConfigWrap, TerminateReason
from psrl.workers.gen.utils import TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class PSRL_AgentLoopWorker:
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle: ray.actor.ActorHandle,
        rollout_gateway_url: str,
        session_router_url: str,
        worker_id: int = 0,
        worker_num: int = 1,
        capacity_coordinator: ray.actor.ActorHandle | None = None,
    ):
        """Initialize agent loop worker.

        Args:
            config (DictConfig): Configuration containing model and rollout settings.
            ps_manager_handle (ray.actor.ActorHandle): Handle to the parameter server manager.
            rollout_gateway_url (str): HTTP base URL of the SMG rollout gateway.
            session_router_url (str): URL of the session router.
            worker_id (int): Unique identifier for this worker instance.
            worker_num (int): Total number of worker instances.
            capacity_coordinator: Node-local sandbox capacity coordinator.
        """

        # The sandbox lease uses this id to attribute containers to this worker.
        self._actor_id = f"w{worker_id}-{os.uname().nodename}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        os.environ["PSRL_ACTOR_ID"] = self._actor_id

        self.config = config
        model_config = config.gen_actor_rollout_ref.model
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config)
        self.overlong_filtering = bool(config.gen_actor_rollout_ref.rollout.agent.get("overlong_filtering", False))
        if self.overlong_filtering:
            psrl_logger.warning("Overlong filtering enabled: budget-truncated trajectories contribute no gradient.")

        # TransferQueue bootstrap (connects to controller/storage spun up by the driver).
        tq.init()

        self.distillation_config = config.get("distillation", None)
        self.distillation_enabled = is_distillation_enabled(self.distillation_config)
        if self.distillation_enabled:
            raise NotImplementedError("Distillation is not supported in PSRL yet.")

        self.dataset_cls = get_dataset_class(config.data)

        self.tokenizer = self.model_config.tokenizer
        self.processor = self.model_config.processor

        self.rollout_gateway_url = rollout_gateway_url
        self.session_router_url = session_router_url
        self.ps_manager_handle = ps_manager_handle
        self.agent_loop_manager = None
        self.reward_manager = None
        sandbox_config = config.gen_actor_rollout_ref.rollout.agent.sandbox
        self.sandbox_manager = build_sandbox_manager(
            sandbox_config,
            capacity_coordinator=capacity_coordinator,
            owner_id=self._actor_id,
        )

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
        self.log_prefix = f"AgentLoopWorker_I{worker_id}"
        handler = DualOutputHandler(self.config.psrl.logging_path, self.log_prefix)
        logging.getLogger("psrl").addHandler(handler)
        psrl_logger.addHandler(handler)

    def set_agent_loop_manager(self, agent_loop_manager: ray.actor.ActorHandle):
        """Set the agent loop manager handle for communication.

        Args:
            agent_loop_manager: Handle to the agent loop manager actor.
        """
        self.agent_loop_manager = agent_loop_manager

    def set_reward_manager(self, reward_manager: ray.actor.ActorHandle):
        """Set the reward manager handle for sending processed data.

        Args:
            reward_manager: Handle to the reward manager actor.
        """
        self.reward_manager = reward_manager

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
        if self.busy_loop_task and not self.busy_loop_task.done():
            self.stop_busy_loop_task = True
            # Wait for the background task to finish. This is async-safe and
            # will not deadlock when called from Ray actors.
            try:
                await asyncio.wait_for(self.busy_loop_task, timeout=10.0)
            except asyncio.TimeoutError:
                psrl_logger.warning("Timeout waiting for busy loop task to complete")
                self.busy_loop_task.cancel()
        if self.agent_programs:
            await asyncio.gather(*self.agent_programs, return_exceptions=True)
        metrics = {name: snapshot.as_dict() for name, snapshot in self.sandbox_manager.metrics_snapshot().items()}
        psrl_logger.info("Final sandbox lifecycle metrics: %s.", metrics)
        await self.sandbox_manager.shutdown()

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

    @staticmethod
    def _failure_diagnostics(agent_loop) -> str:
        """Extract the original loop failure retained across retry handling."""
        details = getattr(agent_loop, "last_error_traceback", "")
        return details or "<no underlying exception was captured>"

    async def generate_trajectory(self, batch: TensorDict):
        """Generate trajectories using the specified agent type based on configuration.

        This method only create the task (agent_loop) and add the task to the agent_programs set.
        But the task is not await here so different agent_loop can be run in parallel.

        Args:
            batch (TensorDict): Input batch metadata containing prompts and metadata.
        """
        assert len(batch) == 1, "Only support single request for generation"

        agent_name = "mini_swe_claude_code"
        task = asyncio.create_task(self._run_agent_loop(agent_name, batch))
        task.add_done_callback(self._create_task_done_callback(task))
        self.agent_programs.add(task)

    def _create_agent_loop(self, agent_name: str):
        """Instantiate the registered agent loop with this worker's shared context."""
        assert agent_name in AGENT_LOOP_REGISTRY, (
            f"Agent loop {agent_name} not registered, registered agent loops: {AGENT_LOOP_REGISTRY.keys()}"
        )
        agent_loop_config = AGENT_LOOP_REGISTRY[agent_name]

        context = AgentLoopContext(
            config=self.config,
            rollout_gateway_url=self.rollout_gateway_url,
            session_router_url=self.session_router_url,
            reward_manager=self.reward_manager,
            ps_manager_handle=self.ps_manager_handle,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
            sandbox_manager=self.sandbox_manager,
        )
        # Keep framework objects out of Hydra's dataclass conversion path.
        agent_loop_factory = hydra.utils.instantiate(
            config=agent_loop_config,
            _partial_=True,
        )
        return agent_loop_factory(context=context)

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
            agent_loop = self._create_agent_loop(agent_name)

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
                        if getattr(agent_loop, "last_error_traceback", ""):
                            psrl_logger.error(
                                "Agent loop request %s attempt %d/%d root cause:\n%s",
                                request_ids,
                                retry_attempt,
                                retry_limit,
                                self._failure_diagnostics(agent_loop),
                            )
                        psrl_logger.warning(
                            f"Retrying agent loop: request_ids={request_ids!r}, "
                            f"reason={terminate_reason.value!r}, "
                            f"attempt={retry_attempt}/{retry_limit}."
                        )
                        continue

                if terminate_reason.needs_worker_retry() or terminate_reason.is_aborted:
                    if getattr(agent_loop, "last_error_traceback", ""):
                        psrl_logger.error(
                            "Agent loop requests %s exhausted with terminate_reason=%s. Root cause:\n%s",
                            request_ids,
                            terminate_reason.value,
                            self._failure_diagnostics(agent_loop),
                        )
                    psrl_logger.warning(
                        f"Agent loop exhausted retries: request_ids={request_ids!r}, "
                        f"reason={terminate_reason.value!r}, attempts={retry_limit}."
                    )
                    output = None

                # Only the manager can purge the partial group and dispatch a replacement
                # prompt, and it owns the breaker that ends a deterministically failing run.
                if terminate_reason.needs_manager_retry():
                    # `validate` must remain scalar or training failures enter the
                    # validation recovery branch.
                    failed_uid = tu.get(batch, "uid")[0]
                    parent_id = tu.get(batch, "parent_id")[0] if "parent_id" in batch else failed_uid
                    if self.config.psrl.agentic_rl.get("manager_retry_on_error", True):
                        psrl_logger.warning(
                            "Group slot lost for uid=%s parent_id=%s "
                            "(terminate_reason=%s, validate=%s), notifying manager.",
                            failed_uid,
                            parent_id,
                            terminate_reason.value,
                            validate,
                        )
                        await self.agent_loop_manager.notify_group_failed.remote(
                            parent_id=parent_id,
                            failed_uid=failed_uid,
                            is_validate=validate,
                            terminate_reason=terminate_reason,
                            failure_summary=self._failure_diagnostics(agent_loop),
                        )
                    else:
                        raise RuntimeError(
                            f"Agent loop for uid={request_ids} "
                            f"failed with terminate_reason={terminate_reason.value} "
                            f"after {retry_limit} attempt(s). "
                            "Set psrl.agentic_rl.manager_retry_on_error=True to recover silently."
                        )
                else:
                    psrl_logger.debug(
                        f"Agent loop terminated: request_ids={request_ids!r}, reason={terminate_reason.value!r}."
                    )

            # Put the output into the TransferQueue and notify PSManager
            # + AgentLoopManager via metadata-only RPCs.
            if output is not None:
                with log_dual_events(
                    "Update request status",
                    psrl_logger,
                    level=logging.DEBUG,
                    event_type=EventType.OTHER,
                ):
                    update_status_success = await self.ps_manager_handle.update_request_status.remote(
                        request_ids,
                        PSRL_RequestStatus.COMPLETED,
                        is_validate=validate,
                    )

                if update_status_success:
                    with log_dual_events(
                        f"Put requests {request_ids} into TransferQueue",
                        psrl_logger,
                        level=logging.DEBUG,
                        event_type=EventType.OTHER,
                    ):
                        await self.postprocess_output(output, batch, terminate_reason)
            elif terminate_reason != TerminateReason.ABORTED:
                # Abort the reserved inventory entry after an unreported generation
                # failure so the buffer can progress.
                psrl_logger.warning(
                    f"Aborting failed generation: request_ids={request_ids!r}, reason={terminate_reason.value!r}."
                )
                await self.ps_manager_handle.abort_requests.remote(request_ids)

            # After ALL cleanup is complete (notify_group_failed + abort_requests),
            # re-raise the original error so it propagates to the task callback.
            if raised_error is not None:
                raise raised_error

    async def postprocess_output(
        self,
        output: TokenOutput | list[TokenOutput],
        batch: TensorDict,
        terminate_reason: TerminateReason = TerminateReason.FINISHED,
    ):
        """Commit generation output to TQ and notify the manager.

        Tensor payloads stay in TQ. Only compact request metadata is sent to the
        manager for group occupation.

        Args:
            output (TokenOutput | list[TokenOutput]): Trajectories to commit.
            batch (TensorDict): The originating prompt batch.
            terminate_reason (TerminateReason): Why the episode stopped. Carried into
                the committed fields so the trainer can drop budget-truncated
                trajectories from the loss.
        """
        uid = tu.get(batch, "uid")[0]
        is_validate = tu.get(batch, "validate")[0]
        version_tag = tu.get(batch, "version_tag")[0]
        partition_id = "val" if tu.get(batch, "validate")[0] else "train"
        prompt_id = tu.get(batch, "parent_id")[0] if "parent_id" in batch else tu.get(batch, "uid")[0]

        outputs = output if isinstance(output, list) else [output]

        keys, fields = self._build_output_fields(outputs, batch, uid, version_tag, terminate_reason)

        await tq.async_kv_batch_put(
            keys=keys,
            partition_id=partition_id,
            fields=list_of_dict_to_tensordict(fields),
            tags=[{"status": "success"}] * len(keys),
        )

        # Notify manager with metadata only after output commit.
        await self.agent_loop_manager.put_result.remote(
            {
                "request_id": uid,
                "prompt_id": prompt_id,
                "rollout_instance_id": outputs[0].rollout_instance_id,
                "version_tag": version_tag,
                "n_trajectory": len(outputs),
                "is_validate": is_validate,
            }
        )

    def _build_output_fields(
        self,
        outputs: list,
        batch: TensorDict,
        uid: int,
        version_tag: int,
        terminate_reason: TerminateReason = TerminateReason.FINISHED,
    ) -> tuple[list[str], list[dict]]:
        """Build output keys and field dicts with tensor operations.

        Designed for ``run_in_executor`` so that CPU-bound torch operations
        (tensor creation, concatenation, position-ID computation) do not block
        the asyncio event loop.
        """
        keys, fields = [], []
        for i, out in enumerate(outputs):
            prompts = torch.tensor(out.prompt_ids, dtype=torch.int64)
            responses = torch.tensor(out.response_ids, dtype=torch.int64)
            input_ids = torch.cat([prompts, responses], dim=0)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            multi_modal_inputs = self._compute_multi_modal_inputs(out, input_ids)
            position_ids = self._compute_position_ids(
                input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
            ).squeeze(0)
            # `images_seqlens` is training metadata, so retain only its top-level copy.
            images_seqlens = multi_modal_inputs.pop("images_seqlens", None)
            if images_seqlens is None:
                images_seqlens = torch.empty(0, dtype=torch.int64)
            multi_modal_inputs.pop("mm_token_type_ids", None)

            if len(outputs) > 1:
                keys.append(f"{uid}_{i}")
            else:
                keys.append(str(uid))

            field = batch[0].to_dict()
            field.update(out.as_dict())
            # do not store raw image/video
            field.pop("multi_modal_data", None)
            field = {k: v for k, v in field.items() if v is not None}
            # NOTE(lhy): DAPO overlong filtering. A truncated reward reports the cutoff
            # and an ungraded one was never measured, so neither may steer the policy.
            # Zero only the VALUES: the mask carries the nested per-row length contract.
            if self.overlong_filtering and (terminate_reason.is_budget_truncated or terminate_reason.is_ungraded):
                field["response_mask"] = torch.zeros_like(field["response_mask"])
            field["loss_mask"] = field["response_mask"]
            field["input_ids"] = input_ids
            field["position_ids"] = position_ids
            field["multi_modal_inputs"] = multi_modal_inputs
            field["images_seqlens"] = images_seqlens
            prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
            # Merged batch fields can carry a stale `response_mask`, so recheck the
            # response-length invariant after assembly.
            mask_len = field["response_mask"].size(0)
            if mask_len != response_len:
                raise AssertionError(
                    f"[uid={uid} trajectory={i}/{len(outputs)}] responses has {response_len} "
                    f"tokens but response_mask has {mask_len} after merging the batch fields "
                    f"with this trajectory's output (prompt_len={prompt_len}). The merge picked "
                    "up a mask that does not belong to this trajectory."
                )
            field["seq_len"] = prompt_len + response_len
            field["prompt_len"] = prompt_len
            field["response_len"] = response_len
            field["uid"] = uid
            field.setdefault("version_tag", version_tag)
            if "parent_id" in batch:
                field["parent_id"] = tu.get(batch, "parent_id")[0]
            field["trajectory_index"] = i
            field["trajectory_num"] = len(outputs)
            # Carried for metrics so the trainer can report reward and gradient share per
            # termination without re-deriving them from the rollout logs.
            field["terminate_reason"] = terminate_reason.value
            fields.append(field)
        return keys, fields

    def _compute_multi_modal_inputs(self, output, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute multi-modal inputs with image and video."""
        multi_modal_inputs = {}
        if self.processor is None:
            return multi_modal_inputs

        multi_modal_data = output.multi_modal_data or {}
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        current_text = self.tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)

        multi_modal_inputs = build_multimodal_processor_inputs(
            self.processor,
            text=[current_text],
            images=images,
            videos=videos,
            audio=audios,
            mm_processor_kwargs=getattr(output, "mm_processor_kwargs", None),
        )
        multi_modal_inputs.pop("input_ids", None)
        multi_modal_inputs.pop("attention_mask", None)

        # We must use dict(multi_modal_inputs) to convert BatchFeature values to a new dict
        # because np.array() only keeps the keys for BatchFeature.
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
            multi_modal_inputs["images_seqlens"] = images_seqlens
        return multi_modal_inputs

    def _compute_position_ids(
        self,
        input_ids,
        attention_mask,
        multi_modal_inputs,
    ) -> torch.Tensor:
        """Compute position ids for multi-modal inputs."""
        if self.processor is None or not hasattr(self.processor, "get_rope_index"):
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0, mm_token_type_ids is only used to calculate position ids.
        if multi_modal_inputs.pop("mm_token_type_ids", None) is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            image_token_id = get_processor_token_id(self.processor, "image")
            video_token_id = get_processor_token_id(self.processor, "video")
            if image_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == image_token_id] = 1
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # Model's get_rope_index has been dynamically bind to the processor.
        vision_position_ids, _ = self.processor.get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq_len) => (1, 3, seq_len)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids), dim=1)  # (1, 4, seq_length)
        return position_ids
