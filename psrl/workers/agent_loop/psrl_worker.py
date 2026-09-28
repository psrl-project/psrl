import logging
import os

import ray
import torch
import transfer_queue as tq
from omegaconf import DictConfig
from tensordict import TensorDict
from verl.trainer.distillation import is_distillation_enabled
from verl.utils import tensordict_utils as tu
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tensordict_utils import list_of_dict_to_tensordict
from verl.utils.tokenizer import (
    build_multimodal_processor_inputs,
    get_processor_token_id,
)

from psrl.utils.logger import EventType, log_dual_events
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.utils import DictConfigWrap, TerminateReason
from psrl.workers.agent_loop.worker_base import AgentLoopWorkerBase
from psrl.workers.gen.utils import TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@ray.remote
class PSRL_AgentLoopWorker(AgentLoopWorkerBase):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        ps_manager_handle: ray.actor.ActorHandle,
        rollout_gateway_url: str,
        session_router_url: str,
        worker_id: int = 0,
        worker_num: int = 1,
    ):
        """Initialize agent loop worker.

        Args:
            config (DictConfig): Configuration containing model and rollout settings.
            ps_manager_handle (ray.actor.ActorHandle): Handle to the parameter server manager.
            rollout_gateway_url (str): HTTP base URL of the SMG rollout gateway.
            session_router_url (str): URL of the session router.
            worker_id (int): Unique identifier for this worker instance.
            worker_num (int): Total number of worker instances.
        """
        self.ps_manager_handle = ps_manager_handle
        self.agent_loop_manager = None
        self.reward_manager = None

        super().__init__(
            config=config,
            rollout_gateway_url=rollout_gateway_url,
            session_router_url=session_router_url,
            worker_id=worker_id,
            worker_num=worker_num,
        )

        self.distillation_config = config.get("distillation", None)
        self.distillation_enabled = is_distillation_enabled(self.distillation_config)
        if self.distillation_enabled:
            raise NotImplementedError("Distillation is not supported in PSRL yet.")

    def _init_data_plane(self) -> None:
        """Connect to the TransferQueue controller/storage spun up by the driver."""
        tq.init()

    def _build_agent_loop_context(self) -> AgentLoopContext:
        """Bundle the framework dependencies, including the RL-only manager handles."""
        return AgentLoopContext(
            config=self.config,
            rollout_gateway_url=self.rollout_gateway_url,
            session_router_url=self.session_router_url,
            reward_manager=self.reward_manager,
            ps_manager_handle=self.ps_manager_handle,
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=DictConfigWrap(self.config.data),
        )

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

    async def _handle_output(
        self,
        output: TokenOutput | list[TokenOutput],
        batch: TensorDict,
        terminate_reason: TerminateReason,
    ) -> None:
        """Advance the request through PSManager, then commit the payload to TransferQueue."""
        request_ids = tu.get(batch, "uid")
        validate = tu.get(batch, "validate", False)[0]

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

    async def _handle_failure(
        self,
        batch: TensorDict,
        terminate_reason: TerminateReason,
    ) -> None:
        """Refill the lost group slot and release the reserved inventory entry.

        Only the manager can purge the partial group and dispatch a replacement
        prompt, and it owns the breaker that ends a deterministically failing run.
        """
        request_ids = tu.get(batch, "uid")
        validate = tu.get(batch, "validate", False)[0]

        if terminate_reason.needs_manager_retry():
            # `validate` must remain scalar or training failures enter the
            # validation recovery branch.
            failed_uid = tu.get(batch, "uid")[0]
            parent_id = tu.get(batch, "parent_id")[0] if "parent_id" in batch else failed_uid
            psrl_logger.warning(
                f"Group slot lost for uid={failed_uid} parent_id={parent_id} "
                f"(terminate_reason={terminate_reason.value}, validate={validate}), notifying manager."
            )
            await self.agent_loop_manager.notify_group_failed.remote(
                parent_id=parent_id,
                failed_uid=failed_uid,
                is_validate=validate,
                terminate_reason=terminate_reason,
            )

        if terminate_reason != TerminateReason.ABORTED:
            # Abort the reserved inventory entry after an unreported generation
            # failure so the buffer can progress.
            psrl_logger.warning(
                f"Aborting failed generation: request_ids={request_ids!r}, reason={terminate_reason.value!r}."
            )
            await self.ps_manager_handle.abort_requests.remote(request_ids)

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
