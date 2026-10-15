import logging
import os
import time

import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import DualOutputHandler
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.reward_loop.registry import register
from psrl.workers.reward.reward_model.manager import RewardModelManager

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("opd")
class OPDRewardLoop(RewardLoopBase):
    """
    Reward loop for online policy distillation.

    This manager handles:
    1. Preprocessing `input_ids` and `attention_mask` for teacher models
    2. Sending prompts to RewardModelManager (via router or direct replica access)
    3. Returning logprobs from teacher models
    """

    def __init__(
        self,
        config,
        tokenizer,
        reward_model_manager: RewardModelManager | None = None,
        **reward_kwargs,
    ):
        """
        Initialize OPDRewardLoop.

        Args:
            config: PSRL config.
            tokenizer: Policy tokenizer (for decoding agent responses).
            reward_model_manager: RewardModelManager instance (provides gateway URL or replica handles).
            reward_kwargs: Additional keyword arguments for compute_score.
        """
        super().__init__(config, tokenizer)
        assert reward_model_manager is not None, "reward_model_manager must be provided for OPD reward loop."

        self.reward_model_manager = reward_model_manager
        self.reward_model_tokenizer = self.reward_model_manager.get_reward_model_tokenizer()
        self.gateway_url = self.reward_model_manager.get_gateway_url()
        # SMG replaced the old RM router process. Prefer replica servers when present.
        self.router_process = None
        self.replica_handles = [
            replica.servers[0]
            for replica in getattr(reward_model_manager, "replicas", [])
            if getattr(replica, "servers", None)
        ]
        self._replica_rr_index = 0
        self.teacher_key = reward_kwargs.get("teacher_key", "default")
        self.reward_kwargs = reward_kwargs
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, "opd_reward_loop"))
        psrl_logger.info("Initialized OPDRewardLoop.")

    async def run_single(self, data: DataProto) -> dict:
        """
        Query the teacher model for response token log probabilities.

        Args:
            data: DataProto containing a single agent response.

        Returns:
            dict with keys:
                - teacher_logprobs: Logprobs from teacher models
                - reward_extra_info: Additional metrics (e.g., accuracy, RM raw output)
        """
        assert len(data) == 1, "Only support single data item in run_single."
        data_item = data[0]
        request_uid = self._format_request_uid(data_item.non_tensor_batch.get("uid"))
        prompt_token_ids, response_token_ids = self._extract_prompt_response_token_ids(data_item)

        psrl_logger.info(
            "OPD reward loop received uid=%s prompt_len=%d response_len=%d.",
            request_uid,
            len(prompt_token_ids),
            len(response_token_ids),
        )

        teacher_data_proto = self._build_teacher_data_proto(
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            request_uid=request_uid,
        )

        teacher_output_dict = await self._query_teacher_model(
            teacher_data_proto=teacher_data_proto,
            request_uid=request_uid,
            response_len=len(response_token_ids),
        )
        teacher_logprobs = teacher_output_dict["teacher_logprobs"]
        reward_metrics = teacher_output_dict.get("reward_metrics", {})
        if not isinstance(reward_metrics, dict):
            reward_metrics = {}

        reward_extra_info = {
            "teacher_key": self.teacher_key,
            "teacher_input_len": len(prompt_token_ids) + len(response_token_ids),
            "teacher_response_len": len(response_token_ids),
            "teacher_logprob_len": teacher_logprobs.shape[0],
        }
        if "teacher_prefill_elapsed_s" in teacher_output_dict:
            reward_extra_info["teacher_prefill_elapsed_s"] = teacher_output_dict["teacher_prefill_elapsed_s"]
        if "teacher_vllm_prefill_s" in teacher_output_dict:
            reward_extra_info["teacher_vllm_prefill_s"] = teacher_output_dict["teacher_vllm_prefill_s"]

        result = {
            "teacher_logprobs": teacher_logprobs,
            "reward_extra_info": reward_extra_info,
            "reward_metrics": reward_metrics,
        }
        if "teacher_ids" in teacher_output_dict:
            result["teacher_ids"] = teacher_output_dict["teacher_ids"]
        return result

    def _extract_prompt_response_token_ids(self, data_item: DataProto) -> tuple[list[int], list[int]]:
        """
        Remove padding from prompt and response tensors.

        Args:
            data_item (DataProto): A single preprocessed rollout item.

        Returns:
            tuple[list[int], list[int]]: Unpadded prompt IDs and response IDs.
        """
        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]
        prompt_length = prompt_ids.shape[-1]
        response_length = response_ids.shape[-1]
        attention_mask = data_item.batch.get("attention_mask", None)

        if attention_mask is not None:
            prompt_attention_mask = attention_mask[:prompt_length].bool()
            response_attention_mask = attention_mask[prompt_length : prompt_length + response_length].bool()
            prompt_token_ids = prompt_ids[prompt_attention_mask].tolist()
            response_token_ids = response_ids[response_attention_mask].tolist()
        else:
            prompt_pad_token_id = self._get_pad_token_id()
            response_pad_token_id = self._get_pad_token_id()
            prompt_token_ids = _pre_process_inputs(prompt_pad_token_id, prompt_ids)
            response_token_ids = response_ids[response_ids != response_pad_token_id].tolist()

        return self._normalize_token_list(prompt_token_ids), self._normalize_token_list(response_token_ids)

    def _build_teacher_data_proto(
        self,
        prompt_token_ids: list[int],
        response_token_ids: list[int],
        request_uid: str | None,
    ) -> DataProto:
        """
        Build teacher-model input from unpadded prompt and response IDs.

        Args:
            prompt_token_ids (list[int]): Unpadded prompt token IDs.
            response_token_ids (list[int]): Unpadded response token IDs to score.
            request_uid (str | None): Request ID for routing and metrics.

        Returns:
            DataProto: Input payload accepted by vLLM reward-model workers.
        """
        teacher_input_ids = prompt_token_ids + response_token_ids
        input_ids = torch.tensor([teacher_input_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            batch_size=1,
        )

        raw_prompt_ids = np.empty(1, dtype=object)
        raw_response_ids = np.empty(1, dtype=object)
        raw_prompt_ids[0] = teacher_input_ids
        raw_response_ids[0] = []

        topk = self._get_teacher_topk()
        sampling_params = {
            "detokenize": False,
            "max_tokens": 1,
            "prompt_logprobs": topk,
        }
        sampling_params.update(self.reward_kwargs.get("sampling_params", {}))

        non_tensor_batch = {
            "uid": np.array([request_uid if request_uid is not None else "unknown"], dtype=object),
            "raw_prompt_ids": raw_prompt_ids,
            "raw_response_ids": raw_response_ids,
            "response_unpadded_len": np.array([len(response_token_ids)], dtype=int),
            "sampling_params": np.array([sampling_params], dtype=object),
        }

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    async def _query_teacher_model(
        self,
        teacher_data_proto: DataProto,
        request_uid: str | None,
        response_len: int,
    ) -> dict:
        """
        Send the OPD prefill request to the teacher model.

        Args:
            teacher_data_proto (DataProto): Teacher model request.
            request_uid (str | None): Request ID for logging.
            response_len (int): Number of response tokens that need teacher logprobs.

        Returns:
            dict: Teacher log probabilities and vLLM metrics.
        """
        teacher_input_len = self._get_teacher_input_len(teacher_data_proto)
        psrl_logger.info(
            "[opd] Teacher prefill request uid=%s input_len=%d response_len=%d.",
            request_uid,
            teacher_input_len,
            response_len,
        )

        request_start_time = time.monotonic()
        if self.router_process is not None:
            psrl_logger.info("Routing OPD teacher request uid=%s through router.", request_uid)
            teacher_outputs = await self.router_process.generate_async.remote(teacher_data_proto)
        else:
            replica_idx, replica_handle = self._get_next_replica_handle()
            psrl_logger.info("Sending OPD teacher request uid=%s to replica_%d.", request_uid, replica_idx)
            if hasattr(replica_handle.generate_async, "remote"):
                teacher_outputs = await replica_handle.generate_async.remote(teacher_data_proto)
            else:
                teacher_outputs = await replica_handle.generate_async(teacher_data_proto)
        teacher_prefill_elapsed_s = time.monotonic() - request_start_time

        result = {
            "teacher_logprobs": torch.empty(0, dtype=torch.float32),
            "reward_metrics": {},
            "teacher_prefill_elapsed_s": teacher_prefill_elapsed_s,
        }
        if teacher_outputs is None or len(teacher_outputs) == 0:
            psrl_logger.warning("Teacher model returned empty output for uid=%s.", request_uid)
            psrl_logger.info(
                "[opd] Teacher prefill finished uid=%s input_len=%d response_len=%d elapsed_s=%.6f.",
                request_uid,
                teacher_input_len,
                response_len,
                teacher_prefill_elapsed_s,
            )
            return result

        reward_metrics = teacher_outputs.meta_info.pop("vllm_metrics", None)
        if reward_metrics is not None:
            result["reward_metrics"] = reward_metrics[0]
        teacher_vllm_prefill_s = self._get_metric_value(result["reward_metrics"], "prefill_time")
        if teacher_vllm_prefill_s is not None:
            result["teacher_vllm_prefill_s"] = teacher_vllm_prefill_s

        teacher_logprobs = self._extract_teacher_logprobs(teacher_outputs)
        if response_len > 0 and len(teacher_logprobs) > response_len:
            teacher_logprobs = teacher_logprobs[:response_len]
        if response_len > 0 and len(teacher_logprobs) != response_len:
            psrl_logger.warning(
                "Teacher logprob length mismatch for uid=%s: expected=%d, got=%d.",
                request_uid,
                response_len,
                len(teacher_logprobs),
            )
        result["teacher_logprobs"] = teacher_logprobs
        self._log_teacher_logprob_preview(request_uid, "teacher_logprobs", teacher_logprobs)
        teacher_ids = self._extract_teacher_ids(teacher_outputs)
        if teacher_ids is not None:
            if response_len > 0 and len(teacher_ids) > response_len:
                teacher_ids = teacher_ids[:response_len]
            if response_len > 0 and len(teacher_ids) != response_len:
                psrl_logger.warning(
                    "Teacher id length mismatch for uid=%s: expected=%d, got=%d.",
                    request_uid,
                    response_len,
                    len(teacher_ids),
                )
            result["teacher_ids"] = teacher_ids
            self._log_teacher_logprob_preview(request_uid, "teacher_ids", teacher_ids)
        vllm_prefill_msg = "none" if teacher_vllm_prefill_s is None else f"{teacher_vllm_prefill_s:.6f}"
        psrl_logger.info(
            "[opd] Teacher prefill finished uid=%s input_len=%d response_len=%d logprob_shape=%s "
            "elapsed_s=%.6f vllm_prefill_s=%s.",
            request_uid,
            teacher_input_len,
            response_len,
            tuple(teacher_logprobs.shape),
            teacher_prefill_elapsed_s,
            vllm_prefill_msg,
        )
        return result

    def _extract_teacher_logprobs(self, teacher_outputs: DataProto) -> torch.Tensor:
        """
        Extract teacher log probabilities from vLLM post-processed output.

        Args:
            teacher_outputs (DataProto): Output returned by the teacher model worker.

        Returns:
            torch.Tensor: Log probabilities for response tokens.
        """
        non_tensor_batch = teacher_outputs.non_tensor_batch
        logprobs = non_tensor_batch.get("teacher_log_probs", [])
        if len(logprobs) == 0:
            return torch.empty(0, dtype=torch.float32)
        return torch.tensor(logprobs[0], dtype=torch.float32)

    def _extract_teacher_ids(self, teacher_outputs: DataProto) -> torch.Tensor | None:
        non_tensor_batch = teacher_outputs.non_tensor_batch
        teacher_ids = non_tensor_batch.get("teacher_ids", [])
        if len(teacher_ids) == 0:
            return None
        return torch.tensor(teacher_ids[0], dtype=torch.long)

    @staticmethod
    def _get_teacher_input_len(teacher_data_proto: DataProto) -> int:
        input_ids = teacher_data_proto.batch.get("input_ids", None)
        if input_ids is None:
            return 0
        return int(input_ids.shape[-1])

    @staticmethod
    def _get_metric_value(metrics, name: str) -> float | None:
        if metrics is None:
            return None
        if isinstance(metrics, dict):
            value = metrics.get(name, None)
        else:
            value = getattr(metrics, name, None)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_tensor_preview(tensor: torch.Tensor, max_elements: int = 8) -> list:
        if tensor.numel() == 0:
            return []
        return tensor.detach().cpu().reshape(-1)[:max_elements].tolist()

    def _log_teacher_logprob_preview(self, request_uid: str | None, name: str, tensor: torch.Tensor) -> None:
        psrl_logger.debug(
            "[opd] %s preview uid=%s shape=%s dtype=%s first_values=%s.",
            name,
            request_uid,
            tuple(tensor.shape),
            tensor.dtype,
            self._format_tensor_preview(tensor),
        )

    def _get_teacher_topk(self) -> int:
        distillation_config = self.config.get("distillation", None)
        if distillation_config is None or not distillation_config.get("enabled", False):
            return int(self.reward_kwargs.get("topk", 0) or 0)
        loss_config = distillation_config.distillation_loss
        loss_mode = loss_config.get("loss_mode", "k3")
        if loss_mode == "forward_kl_topk":
            return int(loss_config.get("topk", 0) or 0)
        return 0

    @staticmethod
    def _normalize_token_list(token_ids) -> list[int]:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return [int(token_id) for token_id in token_ids]

    def _get_pad_token_id(self) -> int:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0
        return int(pad_token_id)

    def _get_next_replica_handle(self):
        if not self.replica_handles:
            raise RuntimeError("Replica handles are not available for teacher model inference")
        handle = self.replica_handles[self._replica_rr_index]
        replica_idx = self._replica_rr_index
        self._replica_rr_index = (self._replica_rr_index + 1) % len(self.replica_handles)
        return replica_idx, handle

    @staticmethod
    def _get_uid_list(uid_value) -> list:
        if uid_value is None:
            return []
        if hasattr(uid_value, "tolist"):
            uid_value = uid_value.tolist()
        if isinstance(uid_value, (list, tuple)):
            return list(uid_value)
        return [uid_value]

    @classmethod
    def _format_request_uid(cls, uid_value) -> str:
        uid_list = cls._get_uid_list(uid_value)
        if not uid_list:
            return "unknown"
        if len(uid_list) == 1:
            return str(uid_list[0])
        return ",".join(str(uid) for uid in uid_list)
