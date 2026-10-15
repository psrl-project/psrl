import asyncio
import logging
import os
import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from transformers import AutoModelForCausalLM
from transformers.distributed import DistributedConfig
from verl import DataProto

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.workers.config import HFModelConfig, RolloutConfig

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class PSRL_TransformersRollout:
    """
    Transformers rollout backend for OPD teacher logprob scoring.

    This backend intentionally implements only the teacher prefill path used by
    online policy distillation. It does not provide vLLM scheduling, abort, cache,
    or parameter-server weight update semantics.
    """

    def __init__(
        self,
        psrl_config: DictConfig,
        config: RolloutConfig,
        model_config: HFModelConfig,
        is_reward_model: bool = False,
        is_teacher_model: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.psrl_config = psrl_config
        self.config = config
        self.model_config = model_config
        self.is_reward_model = is_reward_model
        self.is_teacher_model = is_teacher_model
        self.is_validate = kwargs.get("is_validate", False)
        self.reward_model_name = kwargs.get("reward_model_name") if is_reward_model else None
        self.stat_collector = None
        self.inference_engine = None
        self.sampling_params: dict[str, Any] = {
            "detokenize": False,
            "max_tokens": 1,
        }

        if not self.is_reward_model or not self.is_teacher_model:
            raise NotImplementedError("Transformers rollout only supports OPD teacher reward-model scoring.")

        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.tp_size = int(config.get("tensor_model_parallel_size", 1))
        self.pp_size = int(config.get("pipeline_model_parallel_size", 1))
        self.ep_size = int(config.get("expert_parallel_size", 1))
        if self.pp_size != 1:
            raise NotImplementedError("Transformers rollout does not support pipeline parallelism yet.")
        if self.world_size != self.tp_size * self.pp_size:
            raise ValueError(
                "Transformers rollout expects world_size == tensor_model_parallel_size * "
                f"pipeline_model_parallel_size, got {self.world_size=} {self.tp_size=} {self.pp_size=}."
            )
        if self.ep_size > 1 and self.ep_size != self.tp_size:
            raise NotImplementedError(
                "Transformers rollout currently uses a 1D mesh for EP and requires "
                f"expert_parallel_size == tensor_model_parallel_size, got {self.ep_size=} {self.tp_size=}."
            )

        self.device = self._select_device()
        self.tokenizer = model_config.tokenizer
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer is not None else None
        self.dtype = self._resolve_dtype(config.get("dtype", "bfloat16"))
        self.use_cache = not bool(config.get("disable_kv_cache", True))
        self.device_mesh = init_device_mesh("cuda", (self.world_size,)) if self.world_size > 1 else None

        self.model = self._load_model()
        self.model.config.use_cache = self.use_cache
        self.model.eval()

    def _select_device(self) -> torch.device:
        if not torch.cuda.is_available():
            return torch.device("cpu")

        bundle_indices = os.environ.get("VLLM_RAY_BUNDLE_INDICES")
        if bundle_indices:
            local_ids = [int(idx) for idx in bundle_indices.split(",") if idx != ""]
            local_rank = local_ids[self.rank % len(local_ids)] if local_ids else self.rank
        else:
            local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RAY_LOCAL_RANK", self.rank)))
        local_rank = local_rank % max(torch.cuda.device_count(), 1)
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        dtype = str(dtype).lower()
        if dtype in ("bf16", "bfloat16"):
            return torch.bfloat16
        if dtype in ("fp16", "float16", "half"):
            return torch.float16
        if dtype in ("fp32", "float32"):
            return torch.float32
        if dtype == "auto":
            return torch.bfloat16
        raise ValueError(f"Unsupported transformers rollout dtype: {dtype!r}.")

    def _load_model(self):
        kwargs = {
            "torch_dtype": self.dtype,
            "trust_remote_code": self.model_config.trust_remote_code,
            "attn_implementation": getattr(self.model_config.hf_config, "_attn_implementation", None),
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}

        if self.world_size > 1:
            kwargs["device_mesh"] = self.device_mesh
            kwargs["tp_size"] = self.world_size
            if self.ep_size > 1:
                if not getattr(self.model_config.hf_config, "base_model_ep_plan", None):
                    raise NotImplementedError(
                        "expert_parallel_size > 1 requires a Transformers model config with base_model_ep_plan."
                    )
                kwargs["distributed_config"] = DistributedConfig(enable_expert_parallel=True)
            else:
                kwargs["tp_plan"] = "auto"

        model = AutoModelForCausalLM.from_pretrained(self.model_config.local_path, **kwargs)
        if self.world_size == 1:
            model.to(self.device)
        return model

    def _get_input_ids(self, prompts: DataProto, idx: int) -> list[int]:
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" in non_tensor_batch:
            raw_prompt_ids = non_tensor_batch["raw_prompt_ids"][idx]
            raw_response_ids = non_tensor_batch.get("raw_response_ids", [[] for _ in range(len(prompts))])[idx]
            return list(raw_prompt_ids) + list(raw_response_ids)

        if self.pad_token_id is None:
            return prompts.batch["input_ids"][idx].detach().cpu().tolist()
        return _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][idx])

    @staticmethod
    def _object_array(values: list[Any]) -> np.ndarray:
        out = np.empty(len(values), dtype=object)
        for idx, value in enumerate(values):
            out[idx] = value
        return out

    @torch.no_grad()
    async def generate_sequences_async(self, prompts: DataProto, sampling_params: dict[str, Any]) -> DataProto:
        return await asyncio.to_thread(self._score_teacher, prompts, sampling_params)

    def _score_teacher(self, prompts: DataProto, sampling_params: dict[str, Any]) -> DataProto:
        batch_size = len(prompts)
        non_tensor_batch = dict(prompts.non_tensor_batch)
        response_lens = non_tensor_batch.get("response_unpadded_len", np.array([0] * batch_size, dtype=int))
        teacher_topk = int(sampling_params.get("prompt_logprobs", 0) or 0)

        teacher_log_probs: list[Any] = []
        teacher_ids: list[Any] = []
        interrupted: list[bool] = []
        interrupted_by_scheduler: list[bool] = []
        metrics: list[dict[str, float]] = []

        for idx in range(batch_size):
            input_ids_list = self._get_input_ids(prompts, idx)
            response_len = int(response_lens[idx])
            start_time = time.monotonic()
            log_probs, top_ids = self._score_single(input_ids_list, response_len, teacher_topk)
            prefill_time = time.monotonic() - start_time

            teacher_log_probs.append(log_probs)
            teacher_ids.append(top_ids)
            interrupted.append(False)
            interrupted_by_scheduler.append(False)
            metrics.append({"prefill_time": prefill_time, "backend": "transformers"})

        non_tensor_batch["teacher_log_probs"] = self._object_array(teacher_log_probs)
        if teacher_topk > 0:
            non_tensor_batch["teacher_ids"] = self._object_array(teacher_ids)
        non_tensor_batch["interrupted"] = np.array(interrupted, dtype=bool)
        non_tensor_batch["interrupted_by_scheduler"] = np.array(interrupted_by_scheduler, dtype=bool)

        meta_info = dict(prompts.meta_info)
        meta_info["vllm_metrics"] = np.array(metrics, dtype=object)
        batch = TensorDict({"input_ids": prompts.batch["input_ids"]}, batch_size=batch_size)
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

    def _score_single(self, input_ids_list: list[int], response_len: int, teacher_topk: int) -> tuple[Any, Any]:
        if response_len <= 0:
            return [], []
        if response_len >= len(input_ids_list):
            raise ValueError(
                f"response_unpadded_len must be smaller than input length, got {response_len=} "
                f"and input_len={len(input_ids_list)}."
            )

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids, device=self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=self.use_cache)
        logits = outputs.logits
        if hasattr(logits, "to_local"):
            logits = logits.to_local()
        logits = logits[0]

        response_start = len(input_ids_list) - response_len
        target_logits = logits[response_start - 1 : len(input_ids_list) - 1]
        target_ids = input_ids[0, response_start:]
        log_probs = torch.log_softmax(target_logits.float(), dim=-1)

        if teacher_topk > 0:
            top_log_probs, top_ids = torch.topk(log_probs, k=teacher_topk, dim=-1)
            return top_log_probs.cpu().tolist(), top_ids.cpu().tolist()

        gathered = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
        return gathered.cpu().tolist(), []

    async def raw_generate_sequences_async(self, prompts: DataProto, sampling_params: dict[str, Any]):
        raise NotImplementedError("Transformers rollout only supports OPD teacher generate_sequences_async.")

    def post_process_outputs(self, prompts: DataProto, outputs: Any) -> DataProto:
        raise NotImplementedError("Transformers rollout returns already post-processed DataProto outputs.")

    async def interrupt_all_requests_async(self) -> int:
        return 0

    async def interrupt_requests_async(self, request_ids):
        return 0
