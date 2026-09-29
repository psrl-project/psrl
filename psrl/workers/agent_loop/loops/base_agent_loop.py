import asyncio
import base64
import io
import json
import logging
import os
import traceback
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from PIL import Image
from tensordict import NonTensorData, NonTensorStack, TensorDict
from verl.utils import tensordict_utils as tu
from verl.utils.chat_template import apply_chat_template, initialize_system_prompt
from verl.utils.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

from psrl.utils.common.http_io_thread import get_http_io_thread
from psrl.utils.common.http_utils import (
    RequestAbortedByGatewayError,
    is_distributed_post_enabled,
    request_json_maybe_distributed,
)
from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.rollout.gateway_multimodal import GatewayMultimodalPayloadBuilder
from psrl.utils.rollout.loop_timer import LoopTimer
from psrl.utils.rollout.trajectory_writer import TrajectoryWriter
from psrl.utils.rollout.vision_utils import (
    messages_contain_images,
    resolve_message_image_refs,
)
from psrl.workers.agent_loop.context import AgentLoopContext
from psrl.workers.agent_loop.loops.utils import TerminateReason
from psrl.workers.gen.utils import TokenInput, TokenOutput
from psrl.workers.ps.request_status_tracker import PSRL_RequestStatus

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class AgentLoopBase(ABC):
    def __init__(
        self,
        context: AgentLoopContext,
    ):
        """Initialize an agent loop from its framework context.

        Args:
            context (AgentLoopContext): Framework dependencies shared by the loop.
        """
        self.config = context.config
        self.model_config = self.config.gen_actor_rollout_ref.model
        self.rollout_config = self.config.gen_actor_rollout_ref.rollout
        self.rollout_gateway_url = context.rollout_gateway_url.rstrip("/")

        self.reward_manager = context.reward_manager
        self.ps_manager_handle = context.ps_manager_handle
        self.tokenizer = context.tokenizer
        self.processor = context.processor
        self.traj_writer = TrajectoryWriter.from_config(self.config)
        self.timer = LoopTimer()
        self.dataset_cls = context.dataset_cls
        self.data_config = context.data_config.config
        self.sandbox_manager = context.sandbox_manager
        self.apply_chat_template_kwargs = self.data_config.get("apply_chat_template_kwargs", {})
        self.mm_processor_kwargs = dict(self.data_config.get("mm_processor_kwargs", {}))
        self.system_prompt = initialize_system_prompt(self.tokenizer, **self.apply_chat_template_kwargs)
        self.loop = asyncio.get_running_loop()
        self.response_length = self.rollout_config.response_length
        self.prompt_length = self.rollout_config.prompt_length
        self.output_in_tq = False
        # Retry handling may convert an exception into a TerminateReason, so keep the original
        # failure for the final worker and manager diagnostics.
        self.last_error: BaseException | None = None
        self.last_error_traceback = ""
        gateway_config = self.config.psrl.rollout_gateway
        self.gateway_multimodal = GatewayMultimodalPayloadBuilder(
            gateway_config.get("multimodal_preprocessing", "rust"),
            self.processor,
            self.tokenizer,
            self.loop.run_in_executor,
        )

    async def process_multi_modal_info(self, messages: list[dict]) -> dict:
        """Extract images, videos and audios from messages.

        Args:
            messages (list[dict]): Input messages.

        Returns:
            dict: Multi-modal data with keys like "images", "videos" and "audios".
        """
        multi_modal_data = {}
        if self.processor is not None:
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", 14)
            if hasattr(self.dataset_cls, "process_multi_modal_info"):
                images, videos, audios = await self.dataset_cls.process_multi_modal_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
            else:
                images, videos = await self.dataset_cls.process_vision_info(
                    messages, image_patch_size=image_patch_size, config=self.data_config
                )
                audios = None
            if images is not None:
                multi_modal_data["images"] = images
            if videos is not None:
                multi_modal_data["videos"] = videos
            if audios is not None:
                multi_modal_data["audios"] = audios

        return multi_modal_data

    async def apply_chat_template(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        images: list[Image.Image] | None = None,
        videos: list[tuple[torch.Tensor, dict]] | None = None,
        audios: list[Any] | None = None,
        remove_system_prompt: bool = False,
        expand_multimodal_tokens: bool = True,
    ):
        """Apply chat template to messages with optional tools, images, and videos.

        Args:
            messages (list[dict]): Input messages.
            tools (list[dict], optional): Tools schemas. Defaults to None.
            images (list[Image.Image], optional): Input images. Defaults to None.
            videos (list[tuple[torch.Tensor, dict]], optional): Input videos. Defaults to None.
            remove_system_prompt (bool, optional): Whether to remove system prompt. Defaults to False.
            expand_multimodal_tokens (bool, optional): Whether the Python
                processor expands multimodal anchors. Rust gateway mode sets
                this to false so SMG performs the expansion exactly once.

        Returns:
            list[int]: Prompt token ids.
        """
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.processor,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )

            if expand_multimodal_tokens:
                model_inputs = build_multimodal_processor_inputs(
                    self.processor,
                    text=[raw_prompt],
                    images=images,
                    videos=videos,
                    audio=audios,
                    mm_processor_kwargs=self.mm_processor_kwargs,
                )
                prompt_ids = normalize_token_ids(model_inputs.pop("input_ids"))
            else:
                tokenized_prompt = await self.loop.run_in_executor(
                    None,
                    lambda: self.tokenizer.encode(raw_prompt, add_special_tokens=False),
                )
                prompt_ids = normalize_token_ids(tokenized_prompt)
        else:
            tokenized_prompt = await self.loop.run_in_executor(
                None,
                lambda: apply_chat_template(
                    self.tokenizer,
                    messages,
                    tools=tools,
                    add_generation_prompt=True,
                    tokenize=True,
                    **self.apply_chat_template_kwargs,
                ),
            )
            prompt_ids = normalize_token_ids(tokenized_prompt)

        if remove_system_prompt:
            prompt_ids = prompt_ids[len(self.system_prompt) :]

        # Prompts must fit `rollout.prompt_length`. Multimodal prompts cannot be
        # sliced without corrupting placeholder-to-feature alignment.
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            if images or videos or audios:
                raise ValueError(
                    f"Multimodal prompt produced {len(prompt_ids)} tokens, exceeding "
                    f"rollout.prompt_length={prompt_length}. Truncating multimodal token "
                    f"sequences corrupts vision/audio feature alignment, so this is treated "
                    f"as a configuration error. Reduce the multimodal input size "
                    f"(e.g. ``total_pixels`` / ``max_pixels`` / fps / number of frames) or "
                    f"increase ``rollout.prompt_length``."
                )
            psrl_logger.warning(
                "Prompt of %d tokens exceeds rollout.prompt_length=%d. Left-truncating.",
                len(prompt_ids),
                prompt_length,
            )
            prompt_ids = prompt_ids[-prompt_length:]

        return prompt_ids

    async def compute_reward_score(
        self,
        outputs: TokenOutput | list[TokenOutput],
        **kwargs,
    ) -> TokenOutput | list[TokenOutput] | None:
        """Compute reward score for the generated response and merge it into the output.

        This function sends the generated response to the reward manager and waits for the
        computed reward score. If reward computation succeeds, it merges the score into the
        output. If the request is aborted during reward computation, it returns ``None``.

        Args:
            data (TokenOutput): The output data structure containing the generated response and associated metadata.
        Returns:
            TokenOutput | None: The output with reward score, or None if the request was aborted.
        """
        if not isinstance(outputs, list):
            outputs = [outputs]
        # NOTE(linsh): Only compute reward for the last trajectory.
        final_output = outputs[-1]

        # Build TensorDict: tensors in batch, metadata in non_tensor_batch/meta_info
        tensor_dict = {
            "prompts": torch.tensor(final_output.prompt_ids, dtype=torch.int64).unsqueeze(0),
            "responses": torch.tensor(final_output.response_ids, dtype=torch.int64).unsqueeze(0),
            "multi_modal_data": np.array([final_output.multi_modal_data], dtype=object),
            "num_turns": np.array([final_output.num_turns]),
            "tool_extra_fields": np.array([final_output.extra_fields], dtype=object),
            "uid": np.array([kwargs.get("uid")]),
            "n_trajectory": np.array([len(outputs)]),
            "data_source": np.array([kwargs.get("data_source", "unknown")]),
            "reward_model": np.array([kwargs.get("reward_model", {})], dtype=object),
            "extra_info": np.array([kwargs.get("extra_info", {})], dtype=object),
            "agent_reward_info": np.array([final_output.agent_reward_info], dtype=object),
            "reward_model_dicts": np.array([kwargs.get("reward_model_dicts", [])], dtype=object),
        }
        if kwargs.get("parent_id") is not None:
            tensor_dict["parent_id"] = np.array([kwargs.get("parent_id")])

        reward_requests = tu.get_tensordict(
            tensor_dict=tensor_dict,
            non_tensor_dict={
                "validate": kwargs.get("validate", False),
            },
        )

        reward_result = await self.reward_manager.compute_score.remote(reward_requests)
        if not self.config.reward.launch_reward_fn_async:
            if not reward_result:
                return None
            # Broadcast to all trajectories
            for output in outputs:
                output.reward_score = reward_result["reward_score"]
                output.extra_fields["reward_extra_info"] = reward_result["reward_extra_info"]
                output.extra_fields["reward_metrics"] = reward_result.get("reward_metrics", {})

        if len(outputs) == 1:
            return outputs[0]
        return outputs

    async def generate_sequence(self, request: dict, is_sticky_session: bool = False) -> "TokenOutput":
        request_input: TokenInput = await self.pre_process_inputs(request)
        sampling_params = self._get_sampling_params(request_input)
        if request_input.stop_token_ids:
            sampling_params["stop_token_ids"] = list(
                set((sampling_params.get("stop_token_ids") or []) + request_input.stop_token_ids)
            )
        with self.timer.generation():
            if not self.rollout_gateway_url:
                raise RuntimeError("Rollout gateway address is empty.")
            # `/generate` accepts pretokenized text and multimodal inputs.
            return await self._generate_via_generate_endpoint(request_input, sampling_params, is_sticky_session)

    async def _generate_via_generate_endpoint(
        self,
        request_input: "TokenInput",
        sampling_params: dict,
        is_sticky_session: bool,
    ) -> "TokenOutput":
        """Call SMG /generate and consume its token-native response."""
        request_url = f"{self.rollout_gateway_url}/generate"

        payload_sampling_params = dict(sampling_params)
        if hasattr(payload_sampling_params.get("output_kind"), "value"):
            payload_sampling_params["output_kind"] = payload_sampling_params["output_kind"].value

        req_headers = {
            "x-request-id": str(request_input.request_id),
            "x-prompt-id": str(request_input.prompt_id),
            "x-version-tag": str(request_input.version_tag),
            "x-is-validate": str(request_input.is_validate).lower(),
        }
        if request_input.rollout_instance_id is not None:
            replica_id, dp_rank = request_input.rollout_instance_id
            req_headers["x-base-worker-id"] = str(replica_id)
            req_headers["x-target-dp-rank"] = str(dp_rank)

        if is_sticky_session:
            req_headers["x-is-sticky"] = "true"

        payload = {
            "model": self.model_config.path,
            "request_id": str(request_input.request_id),
            "input_ids": request_input.input_ids,
            "sampling_params": payload_sampling_params,
            "stream": False,
            "return_logprob": sampling_params.get("logprobs") is not None,
        }

        # Multimodal payload.
        mm_data = request_input.multi_modal_data or {}
        videos = mm_data.get("videos")
        audios = mm_data.get("audios")
        if videos or audios:
            raise NotImplementedError(
                "SMG /generate currently supports image_data only; "
                "video/audio payloads cannot be forwarded on this endpoint."
            )

        fallback_images = mm_data.get("images")
        image_refs = []
        if request_input.raw_prompt is not None:
            image_refs = resolve_message_image_refs(request_input.raw_prompt, fallback_images)
        if not image_refs and fallback_images is not None:
            image_refs = list(fallback_images)

        if image_refs:
            payload.update(await self.gateway_multimodal.build(request_input, image_refs, mm_data))
        expects_gateway_prompt_ids = bool(payload.get("return_prompt_token_ids"))

        # Call SMG /generate directly via aiohttp so we can read both the
        # response body (a JSON array) and the worker-instance headers in one pass.
        gen_responses, base_worker_id, target_dp_rank = await self._post_generate(request_url, payload, req_headers)

        if not gen_responses:
            psrl_logger.error(
                "Gateway /generate returned empty response for request_id=%s",
                request_input.request_id,
            )
            return None

        first = gen_responses[0]
        meta_info = first.get("meta_info", {})

        prompt_ids = request_input.input_ids
        if expects_gateway_prompt_ids:
            prompt_ids = meta_info.get("prompt_token_ids")
            if not isinstance(prompt_ids, list):
                raise RuntimeError(
                    "SMG did not return meta_info.prompt_token_ids for a Rust-preprocessed "
                    "multimodal /generate request. PSRL cannot safely align training tokens "
                    "with the prompt dispatched to vLLM."
                )
            reported_prompt_tokens = meta_info.get("prompt_tokens")
            if reported_prompt_tokens is not None and reported_prompt_tokens != len(prompt_ids):
                raise RuntimeError(
                    "SMG returned inconsistent multimodal prompt metadata: "
                    f"prompt_tokens={reported_prompt_tokens}, "
                    f"len(prompt_token_ids)={len(prompt_ids)}."
                )
            self._validate_multimodal_prompt_length(prompt_ids)
        # rollout instance id
        replica_id = base_worker_id
        rollout_instance_id = (replica_id, int(target_dp_rank) if target_dp_rank is not None else 0)

        # token ids
        token_ids = first["output_ids"]

        # Each SMG output position contains top-k log probabilities in descending order.
        log_probs = None
        if sampling_params.get("logprobs") is not None:
            raw_logprobs = meta_info.get("output_token_logprobs")
            if raw_logprobs is not None:
                log_probs = [next((lp for lp in per_pos if lp is not None), 0.0) for per_pos in raw_logprobs]

        # finish_reason: SMG returns {"type": "stop"} or {"type": "length", "length": N}
        finish_reason_raw = meta_info.get("finish_reason", {})
        if isinstance(finish_reason_raw, dict):
            finish_reason = finish_reason_raw.get("type", "stop")
        else:
            finish_reason = str(finish_reason_raw)

        # Determine interrupted based on finish_reason
        interrupted = finish_reason == "abort"

        # SMG aligns routed experts to prompt and completion token positions.
        # Partial rollout loopback is merged by the gateway.
        routed_experts = None
        if self.rollout_config.enable_rollout_routing_replay:
            routed_experts = self._decode_routed_experts_payload(meta_info.get("routed_experts"))

        return TokenOutput(
            prompt_ids=prompt_ids,
            response_ids=token_ids,
            response_mask=[1] * len(token_ids),
            response_log_probs=log_probs,
            routed_experts=routed_experts,
            multi_modal_data=request_input.multi_modal_data,
            mm_processor_kwargs=request_input.mm_processor_kwargs,
            stop_reason=finish_reason,
            interrupted=interrupted,
            update_status=PSRL_RequestStatus.ROLLOUT_COMPLETED,
            rollout_instance_id=rollout_instance_id,
        )

    async def pre_process_inputs(self, request: dict) -> TokenInput:
        version_tag = request["version_tag"]
        is_validate = request.get("validate", False)
        prompt_id = request.get("parent_id", request["uid"])
        rollout_instance_id = request.get("rollout_instance_id", None)

        multi_modal_data = None
        messages = None
        raw_prompt = request.get("raw_prompt")
        raw_prompt_has_images = bool(raw_prompt and messages_contain_images(raw_prompt))
        expected_multimodal_token_mode = (
            "unexpanded"
            if self.gateway_multimodal.uses_rust_preprocessing and raw_prompt_has_images
            else "preexpanded"
        )
        rebuild_unexpanded_multimodal_ids = bool(
            raw_prompt_has_images
            and request.get("_raw_prompt_multimodal_token_mode") != expected_multimodal_token_mode
        )
        if "raw_prompt_ids" not in request or rebuild_unexpanded_multimodal_ids:
            if request.get("input_ids", None) is not None and not rebuild_unexpanded_multimodal_ids:
                input_ids = request["input_ids"]
                raw_prompt_ids = _pre_process_inputs(self.tokenizer.pad_token_id, input_ids)
            elif raw_prompt is not None:
                messages = list(raw_prompt)

                # 1. extract multimodal payloads from messages
                multi_modal_data = await self.process_multi_modal_info(messages)
                images = multi_modal_data.get("images")
                videos = multi_modal_data.get("videos")
                audios = multi_modal_data.get("audios")
                if not multi_modal_data:
                    multi_modal_data = None

                # 2. apply chat template and tokenize
                raw_prompt_ids = await self.apply_chat_template(
                    messages,
                    images=images,
                    videos=videos,
                    audios=audios,
                    expand_multimodal_tokens=not (
                        raw_prompt_has_images and self.gateway_multimodal.uses_rust_preprocessing
                    ),
                )
                request["raw_prompt_ids"] = np.array(raw_prompt_ids)
                if raw_prompt_has_images:
                    request["_raw_prompt_multimodal_token_mode"] = expected_multimodal_token_mode
                if multi_modal_data:
                    request["multi_modal_data"] = multi_modal_data
            else:
                raise ValueError(
                    "Request must contain 'raw_prompt_ids', 'raw_prompt', or 'input_ids' "
                    "to build generation input. Got keys: "
                    f"{request.keys()}"
                )
        else:
            raw_prompt_ids = request["raw_prompt_ids"]
            if isinstance(raw_prompt_ids, np.ndarray):
                raw_prompt_ids = raw_prompt_ids.tolist()
            if "raw_prompt" in request:
                messages = list(request["raw_prompt"])

        # Recover multi-modal data stored by prepare_generation_request().
        if multi_modal_data is None:
            multi_modal_data = request.get("multi_modal_data", None)

        raw_response_ids = request.get("raw_response_ids", [])
        if isinstance(raw_response_ids, np.ndarray):
            raw_response_ids = raw_response_ids.tolist()

        raw_prompt_ids.extend(raw_response_ids)

        return TokenInput(
            input_ids=raw_prompt_ids,
            request_id=request["uid"],
            prompt_id=prompt_id,
            rollout_instance_id=rollout_instance_id,
            version_tag=version_tag,
            cu_response_len=len(raw_response_ids),
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=getattr(self, "mm_processor_kwargs", None) if multi_modal_data else None,
            raw_prompt=messages,
            is_validate=is_validate,
            stop_token_ids=request.get("stop_token_ids", None),
        )

    def _get_sampling_params(self, request: TokenInput):
        is_validate = request.is_validate
        input_length = len(request.input_ids)

        # Rust multimodal inputs contain unexpanded anchors, so reserve the full
        # prompt budget until SMG returns expanded IDs.
        mm_data = request.multi_modal_data or {}
        has_images = bool(mm_data.get("images")) or bool(
            request.raw_prompt and messages_contain_images(request.raw_prompt)
        )
        if has_images and self.gateway_multimodal.uses_rust_preprocessing:
            input_length = self.rollout_config.prompt_length

        # When max_model_len is not configured (None), fall back to prompt_length + response_length
        max_model_len = self.rollout_config.max_model_len
        if max_model_len is None:
            max_model_len = self.rollout_config.prompt_length + self.rollout_config.response_length
        max_possible_tokens = max_model_len - input_length
        if max_possible_tokens < 0:
            raise ValueError(f"Input length {input_length} exceeds the maximum model length {max_model_len}")

        max_tokens = self.rollout_config.response_length + self.rollout_config.prompt_length - input_length
        max_tokens = max(0, min(max_tokens, max_possible_tokens))
        assert max_tokens <= max_possible_tokens, (
            f"max_tokens={max_tokens} exceeds available context space={max_possible_tokens}."
        )

        # `top_k=-1` is the only disabled value accepted by both generation endpoints.
        top_k = int(self.rollout_config.top_k)

        sampling_params = dict(
            n=1,
            logprobs=0,  # return sampled-token logprob for importance-sampling weight computation
            temperature=float(self.rollout_config.temperature),
            top_p=float(self.rollout_config.top_p),
            top_k=top_k,
            repetition_penalty=float(self.rollout_config.get("repetition_penalty", 1.0)),
            ignore_eos=self.rollout_config.get("ignore_eos", False),
            detokenize=False,
            max_new_tokens=max_tokens,
        )

        # override sampling params for validation
        if is_validate:
            val_config = self.config.train_actor_rollout_ref.rollout.val_kwargs
            sampling_params["top_k"] = int(val_config.top_k)
            sampling_params["top_p"] = float(val_config.top_p)
            sampling_params["temperature"] = float(val_config.temperature)

        return sampling_params

    def _validate_multimodal_prompt_length(self, prompt_ids: list[int]) -> None:
        prompt_length = self.rollout_config.prompt_length
        if len(prompt_ids) > prompt_length:
            raise ValueError(
                f"Multimodal prompt produced {len(prompt_ids)} tokens, exceeding "
                f"rollout.prompt_length={prompt_length}. Truncating multimodal token "
                "sequences corrupts vision feature alignment, so this is treated as a "
                "configuration error. Reduce the multimodal input size (e.g. "
                "`total_pixels` / `max_pixels`) or increase `rollout.prompt_length`."
            )

    @staticmethod
    def _decode_routed_experts_payload(routed_experts_b64: str | None) -> np.ndarray | None:
        """
        Decode SMG's routed-expert payload into a NumPy array.

        The result has shape `[num_tokens, num_layers, top_k]` and uses
        an unsigned 8-bit or 16-bit integer dtype.
        """
        if not routed_experts_b64:
            return None

        raw_bytes = base64.b64decode(routed_experts_b64)
        return np.load(io.BytesIO(raw_bytes)).copy()

    async def _post_generate(
        self,
        url: str,
        payload: dict,
        headers: dict[str, str],
        max_retries: int = 1,
    ) -> tuple[list[dict], str | None, str | None]:
        """
        POST to SMG `/generate` and return responses with routing headers.

        HTTP I/O runs in a dedicated thread because socket callbacks must not
        contend with the Ray actor event loop.

        Returns:
            tuple: Response dictionaries, base worker ID, and target DP rank.
        """
        if is_distributed_post_enabled():
            response = await request_json_maybe_distributed(
                "POST",
                url,
                payload=payload,
                headers=headers,
                max_retries=max_retries + 1,
            )
        else:
            io_thread = get_http_io_thread()
            response = await io_thread.request_json(
                "POST",
                url,
                payload=payload,
                headers=headers,
                max_retries=max_retries + 1,
            )
        base_worker_id = response.headers.get("x-base-worker-id", None)
        target_dp_rank = response.headers.get("x-target-dp-rank", None)

        responses = response.data
        if isinstance(responses, dict):
            responses = [responses]
        elif not isinstance(responses, list):
            psrl_logger.error(
                "_post_generate: unexpected response type %s from %s.",
                type(responses),
                url,
            )
            responses = []

        return responses, base_worker_id, target_dp_rank

    @abstractmethod
    async def run(self, request: dict) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Execute the agent loop for the given request.

        Args:
            request (dict): Input request to process.

        Returns:
            Tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
                A tuple containing the output data (if any) and the termination reason.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_generate_fields(self) -> list[str]:
        """Determine which fields to select from the key-value store for generation.

        Returns:
            list[str] | None: List of field names to select from the key-value store,
                              or None to fetch all fields.
        """
        fields = [
            # dataset metadata
            "data_source",
            "reward_model",
            "extra_info",
            "reward_model_dicts",
            # request metadata
            "uid",
            "parent_id",
            "validate",
            "rollout_instance_id",
            # prompt
            "raw_prompt_ids",
            "raw_prompt",
            "input_ids",
            "raw_response_ids",
            # multi-modal data
            "multi_modal_data",
        ]

        return fields

    async def run_with_termination_handling(
        self, request: TensorDict, raise_on_error: bool = True
    ) -> tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
        """Run the agent loop with termination event handling.

        This method wraps the run method to catch termination events and handle them appropriately.
        It enables timeouts and error handling based on the provided configuration.

        Args:
            request (TensorDict): Input request to process.
            raise_on_error (bool): Whether to raise exceptions on errors.

        Returns:
            Tuple[TokenOutput | list[TokenOutput] | None, TerminateReason]:
                A tuple containing the output data (if any) and the termination reason.
        """
        request_ids = tu.get(request, "uid", "N/A")
        self.last_error = None
        self.last_error_traceback = ""
        try:
            prompt = {}
            for k, v in request.items():
                if isinstance(v, torch.Tensor):
                    prompt[k] = v[0]
                elif isinstance(v, NonTensorStack):
                    prompt[k] = v[0].data
                elif isinstance(v, NonTensorData):
                    prompt[k] = v.data
                else:
                    psrl_logger.exception(f"Unsupported value: type={type(v)!r}, key={k!r}.")

            timeout = self.config.gen_actor_rollout_ref.rollout.agent.trajectory_timeout
            output, terminate_reason = await asyncio.wait_for(
                self.run(prompt),
                timeout=timeout,
            )
            if output is not None:
                await self._resolve_version_for_dump(output, prompt)
                self._attach_loop_timing(output)
                self._dump_trajectory_text(prompt, output, terminate_reason)
                return output, terminate_reason
            elif output is None and terminate_reason.is_aborted:
                return None, terminate_reason
            elif output is None and terminate_reason.needs_worker_retry():
                # Error-class terminate reasons: respect raise_on_error so that
                # errors are surfaced instead of being silently swallowed.
                if terminate_reason.is_error:
                    if raise_on_error:
                        raise RuntimeError(
                            f"Agent loop run for request {request_ids} "
                            f"terminated with error: {terminate_reason.value}."
                        ) from self.last_error
                    psrl_logger.error(
                        "Agent loop run for request %s terminated with "
                        "error: %s (raise_on_error=False, returning for retry/abort).\n"
                        "Underlying failure:\n%s",
                        request_ids,
                        terminate_reason.value,
                        self.last_error_traceback or "<no underlying exception was captured>",
                    )
                return None, terminate_reason
            elif not raise_on_error:
                return None, TerminateReason.UNKNOWN
            else:
                raise RuntimeError("Agent loop run did not return a valid TokenOutput output.")
        except RequestAbortedByGatewayError as e:
            # PS Manager has already taken ownership of cleaning the request's data
            # flow (TQ entry cleared, staleness inventory updated).
            psrl_logger.info(
                "Request %s aborted by PS Manager because the gateway returned request_aborted. "
                "Ending data flow without retry.",
                e.request_id or "N/A",
            )
            return None, TerminateReason.ABORTED
        except asyncio.TimeoutError:
            psrl_logger.error(
                "Timeout in agent_loop.run for request %s (this can come from downstream calls, not only trajectory_timeout)",  # noqa: E501
                request_ids,
                exc_info=True,
            )
            return None, TerminateReason.TRAJECTORY_TIMEOUT
        except Exception as exc:
            self.last_error = exc
            self.last_error_traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if not raise_on_error:
                psrl_logger.error(
                    "Exception in agent_loop.run for request %s.\nUnderlying failure:\n%s",
                    request_ids,
                    self.last_error_traceback,
                    exc_info=True,
                )
                return None, TerminateReason.ROLLOUT_ERROR
            raise

    def _attach_loop_timing(self, output: "TokenOutput | list[TokenOutput]") -> None:
        """Stamp the per-trajectory wall-clock timing onto each output.

        Read directly from ``self.timer`` (one loop instance == one trajectory).
        Stored under ``extra_fields['loop_timing']`` so it survives the trip back
        to ``_dump_trajectory_text``. Never raises.
        """
        try:
            timing = self.timer.as_dict()
            for out in output if isinstance(output, list) else [output]:
                out.extra_fields.setdefault("loop_timing", timing)
        except Exception:
            psrl_logger.debug("Failed to attach loop timing.", exc_info=True)

    async def _resolve_version_for_dump(self, output: "TokenOutput | list[TokenOutput]", prompt: dict) -> None:
        """
        Resolve the served model version for trajectory bucketing.

        A dispatched `version_tag` of `-1` requires a manager lookup. Failures
        fall back to the current parameter server version and then zero.
        """
        prompt_version = prompt.get("version_tag", 0)
        if prompt_version not in (None, -1):
            return
        if self.ps_manager_handle is None:
            return
        outs = output if isinstance(output, list) else [output]
        for out in outs:
            resolved: int | None = None
            try:
                if out.rollout_instance_id is not None:
                    resolved = await self.ps_manager_handle.get_rollout_instance_model_version.remote(
                        out.rollout_instance_id
                    )
                else:
                    resolved = await self.ps_manager_handle.get_ps_model_version.remote(debug_info="trajectory_dump")
            except Exception:
                psrl_logger.debug(
                    "Failed to resolve served version for uid=%s. Falling back to 0.",
                    prompt.get("uid", "N/A"),
                    exc_info=True,
                )
            if resolved is not None:
                out.extra_fields["resolved_version"] = int(resolved)

    def _build_summary_text(
        self,
        out: "TokenOutput",
        terminate_reason: "TerminateReason",
    ) -> str:
        """Build the ``=== Submission ===`` / ``=== Summary ===`` trailer for one trajectory."""
        info = out.agent_reward_info or {}
        patch = info.get("patch")

        n_prompt = len(out.prompt_ids)
        n_assistant = out.response_mask.count(1) if out.response_mask else 0
        n_env = out.response_mask.count(0) if out.response_mask else 0
        total_tokens = n_prompt + n_assistant + n_env

        turns = info.get("num_turns")
        if turns is None:
            turns = out.num_turns if out.num_turns is not None else 0

        # Prefer the loop's wall-clock timing. Mini-SWE carries finer timing under
        # agent_reward_info['timing'] (assistant/env/prep/grading) measured in the runner.
        loop_timing = (out.extra_fields or {}).get("loop_timing", {})
        runner_timing = info.get("timing", {}) or {}
        generation_s = runner_timing.get("assistant_s", loop_timing.get("generation_s", 0.0))
        env_s = runner_timing.get("env_s", loop_timing.get("env_s", 0.0))
        elapsed_s = runner_timing.get("elapsed_s", loop_timing.get("elapsed_s", 0.0))

        breakdown = [f"generation: {generation_s:.1f}s", f"env: {env_s:.1f}s"]
        if runner_timing.get("grading_s"):
            breakdown.append(f"grading: {runner_timing['grading_s']:.1f}s")
        if runner_timing.get("prep_s"):
            # Fine-grained prep breakdown to locate bottlenecks in task prep, sandbox cold
            # start, clean snapshot, sandbox init, git sanitization, and harness preparation.
            prep_parts = [f"total={runner_timing['prep_s']:.1f}s"]
            for key, label in (
                ("task_prepare_s", "task"),
                ("sandbox_create_s", "sandbox"),
                ("snapshot_s", "snapshot"),
                ("sandbox_init_s", "sandbox_init"),
                ("git_probe_s", "git_probe"),
                ("git_purge_s", "git_purge"),
                ("harness_prepare_s", "harness_prepare"),
            ):
                value = runner_timing.get(key)
                if value:
                    prep_parts.append(f"{label}={value:.1f}s")
            breakdown.append("prep: " + " | ".join(prep_parts))

        text = ""
        if patch:
            text += f"=== Submission ===\n{patch}\n\n"
        text += (
            "=== Summary ===\n"
            f"turns: {turns}, patch: {'yes' if patch else 'no'}, "
            f"stop: {terminate_reason.value}, elapsed: {elapsed_s:.1f}s\n"
            f"[Token Counts] prompt: {n_prompt} | assistant: {n_assistant} | "
            f"env: {n_env} | total: {total_tokens}\n"
            f"[Time Breakdown] {' | '.join(breakdown)}\n"
        )
        compaction_window = info.get("compaction_context_window_tokens")
        compaction_limit = info.get("compaction_token_limit")
        if compaction_window is not None or compaction_limit is not None:
            text += (
                "[Compaction] context_window_tokens: "
                f"{compaction_window if compaction_window is not None else 'disabled'} | "
                "trigger_tokens: "
                f"{compaction_limit if compaction_limit is not None else 'disabled'} | "
                f"trajectory_count: {info.get('trajectory_count', 1)}\n"
            )
        leaf = (out.extra_fields or {}).get("tito_leaf")
        if isinstance(leaf, dict):
            text += (
                f"[TITO Leaf] trajectory_id={leaf.get('trajectory_id')} | "
                f"node_id={leaf.get('node_id')} | parent={leaf.get('parent')} | "
                f"finish_reason={leaf.get('finish_reason')} | truncated={leaf.get('truncated')} | "
                f"num_tokens={leaf.get('num_tokens')} | path={leaf.get('path_node_ids')}\n"
            )
        return text

    def _dump_trajectory_text(
        self,
        prompt: dict,
        output: "TokenOutput | list[TokenOutput]",
        terminate_reason: "TerminateReason",
    ) -> None:
        """Write per-trajectory text for any agent loop via the shared writer.

        Renders text uniformly from the returned ``TokenOutput`` (prompt + the
        assistant/observation segments of ``response_ids``, split by
        ``response_mask``), then appends a ``=== Summary ===`` block (turns, stop
        reason, token counts, wall-clock timing) and, when a patch is present, a
        ``=== Submission ===`` block. This is the single chokepoint all loops pass
        through, so no per-loop wiring is needed. Never raises: a dump failure must
        not break a rollout.

        Args:
            prompt (dict): The unwrapped request dict (carries ``uid`` and
                ``version_tag``).
            output (TokenOutput | list[TokenOutput]): The loop's generation
                output(s).
            terminate_reason (TerminateReason): Final termination classification,
                surfaced as the ``stop:`` field of the summary.
        """
        if not getattr(self, "traj_writer", None) or not self.traj_writer.enable:
            return
        try:
            outs = output if isinstance(output, list) else [output]
            uid = prompt.get("uid", "N/A")
            for idx, out in enumerate(outs):
                version = int((out.extra_fields or {}).get("resolved_version", prompt.get("version_tag", 0)) or 0)
                prompt_text = self.tokenizer.decode(out.prompt_ids, skip_special_tokens=False)
                parts = [f"=== Prompt ===\n{prompt_text}\n\n"]
                turn = 0
                for role, text in self._segment_by_mask(out.response_ids, out.response_mask):
                    if role == "assistant":
                        turn += 1
                        parts.append(f"=== Turn {turn} (assistant) ===\n{text}\n\n")
                    else:
                        parts.append(f"--- observation ---\n{text}\n\n")
                parts.append(self._build_summary_text(out, terminate_reason))
                traj_id = str(uid) if len(outs) == 1 else f"{uid}_{idx}"
                self.traj_writer.write(version, traj_id, "".join(parts))
                # Persist the raw SMG TITO prefix-tree snapshot once per request for offline analysis.
                if idx == 0:
                    tree = (out.extra_fields or {}).get("tito_tree")
                    if isinstance(tree, dict):
                        self.traj_writer.write(
                            version,
                            f"{uid}.tree",
                            json.dumps(tree, ensure_ascii=False, sort_keys=True, indent=1),
                            suffix=".json",
                        )
        except Exception:
            psrl_logger.warning(
                "Failed to dump trajectory text for uid=%s.",
                prompt.get("uid", "N/A"),
                exc_info=True,
            )

    def _segment_by_mask(
        self,
        response_ids: list[int],
        response_mask: list[int],
    ) -> list[tuple[str, str]]:
        """
        Split `response_ids` into ordered role and text runs by `response_mask`.

        A mask of `1` marks assistant tokens, while `0` marks observation or tool
        tokens. Each contiguous run is decoded separately.

        Args:
            response_ids (list[int]): Response token ids.
            response_mask (list[int]): Parallel mask (1=assistant, 0=observation).

        Returns:
            list[tuple[str, str]]: Ordered (role, decoded_text) segments.
        """
        segments: list[tuple[str, str]] = []
        if not response_ids:
            return segments
        if not response_mask or len(response_mask) != len(response_ids):
            # No usable mask: emit the whole response as a single assistant run.
            return [("assistant", self.tokenizer.decode(response_ids, skip_special_tokens=False))]
        run_ids: list[int] = []
        run_mask: int | None = None
        for tok, mask in zip(response_ids, response_mask, strict=False):
            mask = int(bool(mask))
            if run_mask is None:
                run_mask = mask
            if mask != run_mask:
                role = "assistant" if run_mask == 1 else "observation"
                segments.append((role, self.tokenizer.decode(run_ids, skip_special_tokens=False)))
                run_ids = []
                run_mask = mask
            run_ids.append(tok)
        if run_ids:
            role = "assistant" if run_mask == 1 else "observation"
            segments.append((role, self.tokenizer.decode(run_ids, skip_special_tokens=False)))
        return segments
