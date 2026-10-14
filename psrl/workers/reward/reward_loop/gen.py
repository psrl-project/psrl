"""
Route generative and pooling reward requests through the SMG gateway.

Endpoint mapping from `third_party/smg/model_gateway/src/server.rs`:
`generate/generate` -> `POST /v1/completions` -> `choices[0]["text"]`
`pooling/classify` -> `POST /v1/classify` -> `data[0]["embedding"][0]`
`pooling/embed*` -> `POST /v1/embeddings` -> `data[0]["embedding"]`
"""

import inspect
import logging
import os
from collections.abc import Callable

import aiohttp
import numpy as np
import torch
from tensordict import TensorDict
from verl import DataProto
from verl.utils import tensordict_utils as tu

from psrl.utils.dataset.utils import _pre_process_inputs
from psrl.utils.logger import DualOutputHandler
from psrl.workers.reward.gen_reward_function import DefaultGenRewardFunction, GenRewardFunctionBase
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.reward_loop.registry import register
from psrl.workers.reward.reward_protocol import RewardModelRuntimeInfo

psrl_logger = logging.getLogger(__name__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


@register("gen")
class GenRewardLoop(RewardLoopBase):
    """Access generative or pooling reward models through the SMG gateway."""

    def __init__(
        self,
        config,
        tokenizer,
        reward_model_manager: RewardModelRuntimeInfo | None = None,
        reward_function: GenRewardFunctionBase | None = None,
        **reward_kwargs,
    ):
        super().__init__(config, tokenizer, compute_score=None)

        if reward_function is None:
            reward_function = DefaultGenRewardFunction()
        self.reward_function = reward_function
        self.is_async_reward_score = inspect.iscoroutinefunction(self.reward_function.compute_score)

        assert reward_model_manager is not None, "GenRewardLoop requires a reward_model_manager."
        self.reward_model_manager = reward_model_manager
        self.reward_model_tokenizer = reward_model_manager.get_reward_model_tokenizer()

        rm_cfg = reward_model_manager.reward_model_config
        self._model_name = rm_cfg.reward_model_name
        runner = rm_cfg.rollout.runner
        task = rm_cfg.rollout.task
        self._sampling_config = dict(rm_cfg.get("sampling_config", {}))

        self.gateway_url: str = reward_model_manager.get_gateway_url()
        self._smg_endpoint, self._parse_response = self._resolve_endpoint_and_parser(runner, task)
        self._rm_response_length = int(rm_cfg.rollout.response_length)
        self._rm_max_model_len = int(rm_cfg.rollout.max_model_len)

        self._http_client: aiohttp.ClientSession | None = None
        self.reward_kwargs = reward_kwargs

        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, "gen_reward_loop"))
        psrl_logger.info(
            "GenRewardLoop initialized. gateway=%s endpoint=%s",
            self.gateway_url,
            self._smg_endpoint,
        )

    # --- Endpoint Dispatch ---

    def _resolve_endpoint_and_parser(self, runner: str, task: str) -> tuple[str, Callable]:
        """
        Map (runner, task) to the smg gateway endpoint and its response parser.

        Reference: `third_party/smg/model_gateway/src/server.rs`.
        """
        if runner == "generate":
            return "/v1/completions", self._parse_completions_response
        if runner == "pooling":
            if task == "classify":
                return "/v1/classify", self._parse_classify_response
            if task in ("embed", "embedding", "embeddings"):
                return "/v1/embeddings", self._parse_embeddings_response
        raise ValueError(
            f"Unsupported reward model runner={runner!r} / task={task!r}. "
            f"Supported: (generate, generate) | (pooling, classify) | (pooling, embed)"
        )

    # --- Request Payload Builders ---

    def _build_request_payload(self, prompt_ids: list[int]) -> dict:
        """Build the smg-compatible JSON payload for the given token IDs."""
        if self._smg_endpoint == "/v1/completions":
            top_p = self._sampling_config.get("top_p", -1)
            return {
                "model": self._model_name,
                "prompt": prompt_ids,
                "max_tokens": self._sampling_config.get("max_tokens", self._rm_response_length),
                "temperature": self._sampling_config.get("temperature", 1.0),
                "top_p": float(top_p) if isinstance(top_p, (int, float)) and top_p > 0 else 1.0,
                "top_k": self._sampling_config.get("top_k", -1),
            }
        return {"model": self._model_name, "input": prompt_ids}

    # --- Response Parsers ---

    def _parse_completions_response(self, data: dict, request_uid: str) -> dict:
        """
        Parse `/v1/completions` into `rm_output_str`.

        The SMG response is already decoded.
        """
        choices = data.get("choices", [])
        if not choices:
            return {"rm_output_str": "", "rm_output_value": None, "rm_output_len": 0, "reward_metrics": {}}
        generated_str = choices[0].get("text", "")
        rm_output_len = data.get("usage", {}).get("completion_tokens", len(generated_str.split()))
        return {
            "rm_output_str": generated_str,
            "rm_output_value": None,
            "rm_output_len": rm_output_len,
            "reward_metrics": {},
        }

    def _parse_classify_response(self, data: dict, request_uid: str) -> dict:
        """
        Parse `/v1/classify` into a scalar reward score.

        For a single-logit classifier, returns the scalar directly.
        For multi-class output, returns the full list for `compute_score`.
        """
        entries = data.get("data", [])
        if not entries:
            return {"rm_output_str": "", "rm_output_value": None, "reward_metrics": {}}
        embedding = entries[0].get("embedding", [])
        score = embedding[0] if len(embedding) == 1 else embedding
        return {
            "rm_output_str": "",
            "rm_output_value": float(score) if isinstance(score, (int, float)) else score,
            "reward_metrics": {},
        }

    def _parse_embeddings_response(self, data: dict, request_uid: str) -> dict:
        """
        Parse `/v1/embeddings` into an embedding vector.

        Store the full vector in `rm_output_value`. `compute_score` converts it to a scalar reward.
        """
        entries = data.get("data", [])
        if not entries:
            return {"rm_output_str": "", "rm_output_value": None, "reward_metrics": {}}
        return {
            "rm_output_str": "",
            "rm_output_value": entries[0].get("embedding", []),
            "reward_metrics": {},
        }

    # --- HTTP Query ---

    async def _query_reward_model(self, rm_data_proto: DataProto, request_uid: str) -> dict:
        """
        Send one request to the smg gateway and return the parsed result dict.

        Keys returned: rm_output_str, rm_output_value, rm_output_len (optional),
                       reward_metrics.
        """
        raw_prompt_ids = rm_data_proto.non_tensor_batch["raw_prompt_ids"][0]
        if hasattr(raw_prompt_ids, "tolist"):
            raw_prompt_ids = raw_prompt_ids.tolist()

        payload = self._build_request_payload(raw_prompt_ids)
        output_budget = int(payload["max_tokens"]) if self._smg_endpoint == "/v1/completions" else 0
        prompt_budget = self._rm_max_model_len - output_budget
        if prompt_budget <= 0:
            raise ValueError(
                f"Reward output budget leaves no prompt capacity: max_model_len={self._rm_max_model_len}, "
                f"max_tokens={output_budget}."
            )
        truncated_tokens = max(0, len(raw_prompt_ids) - prompt_budget)
        if truncated_tokens:
            # Keep the response tail and scoring instruction in the RM tokenizer's token space.
            raw_prompt_ids = raw_prompt_ids[-prompt_budget:]
            payload = self._build_request_payload(raw_prompt_ids)
            psrl_logger.warning(
                f"Truncated reward prompt: uid={request_uid!r}, removed_tokens={truncated_tokens}, "
                f"prompt_tokens={len(raw_prompt_ids)}, max_model_len={self._rm_max_model_len}."
            )
        url = f"{self.gateway_url.rstrip('/')}{self._smg_endpoint}"
        if self._http_client is None or self._http_client.closed:
            self._http_client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=256, enable_cleanup_closed=True),
                timeout=aiohttp.ClientTimeout(total=None),
            )

        psrl_logger.info(
            f"Querying reward model: uid={request_uid!r}, endpoint={self._smg_endpoint!r}, "
            f"payload_tokens={len(raw_prompt_ids)}."
        )

        interrupted_count = 0
        while True:
            async with self._http_client.post(url, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json(content_type=None)
            if self._smg_endpoint != "/v1/completions" or not any(
                choice.get("finish_reason") == "abort" for choice in data.get("choices", [])
            ):
                break
            # RM requests restart as a whole because interrupted scores are not valid rewards.
            interrupted_count += 1
            psrl_logger.info(f"Requeueing interrupted reward request: uid={request_uid!r}.")

        result = self._parse_response(data, request_uid)
        result["reward_metrics"]["rm_prompt_truncated_tokens"] = truncated_tokens
        result["reward_metrics"]["rm_request_interruptions"] = interrupted_count
        psrl_logger.info(
            f"Reward model response: uid={request_uid!r}, str_len={len(result.get('rm_output_str', ''))}, "
            f"value={result.get('rm_output_value')!r}."
        )
        return result

    # --- Main Entry Point ---

    async def run_single(self, data: TensorDict) -> dict:
        """Process one data item through the reward model."""
        assert len(data) == 1, "Only single data items supported in run_single"
        data_item = data[0]
        request_uid = self._format_request_uid(tu.get(data_item, "uid"))

        prompt_ids = data_item["prompts"]
        prompt_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(prompt_ids, skip_special_tokens=True),
        )
        response_ids = data_item["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        data_source = tu.get(data_item, "data_source", "unknown")
        reward_model_info = tu.get(data_item, "reward_model", {})
        ground_truth = reward_model_info.get("ground_truth", "") if isinstance(reward_model_info, dict) else ""
        extra_info = self.merge_extra_info(data_item)

        rm_prompt = self.reward_function.prompt_constructor(prompt_str=prompt_str, response_str=response_str)
        using_sys_prompt = self.reward_function.using_sys_prompt
        rm_inputs = await self.loop.run_in_executor(
            None,
            lambda: self.reward_model_tokenizer.apply_chat_template(
                rm_prompt,
                tokenize=True,
                add_generation_prompt=using_sys_prompt,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ),
        )
        if isinstance(rm_inputs, torch.Tensor):
            rm_inputs = {"input_ids": rm_inputs}

        rm_input_len = None
        input_ids_tensor = rm_inputs.get("input_ids")
        attn_mask = rm_inputs.get("attention_mask")
        if isinstance(attn_mask, torch.Tensor):
            rm_input_len = int(attn_mask[0].sum().item() if attn_mask.dim() == 2 else attn_mask.sum().item())
        elif isinstance(input_ids_tensor, torch.Tensor):
            rm_input_len = int(
                input_ids_tensor[0].numel() if input_ids_tensor.dim() == 2 else input_ids_tensor.numel()
            )

        rm_data_proto = self._build_rm_data_proto(rm_inputs, request_uid)

        rm_output_dict = await self._query_reward_model(rm_data_proto, request_uid)
        rm_output_str = rm_output_dict.get("rm_output_str", "")
        rm_output_value = rm_output_dict.get("rm_output_value")
        reward_metrics = rm_output_dict.get("reward_metrics", {})
        if not isinstance(reward_metrics, dict):
            reward_metrics = {}

        if self.is_async_reward_score:
            result = await self.reward_function.compute_score(
                data_source=data_source,
                solution_str=response_str,
                rm_output=rm_output_str,
                rm_output_value=rm_output_value,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **self.reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.reward_function.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    rm_output=rm_output_str,
                    rm_output_value=rm_output_value,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **self.reward_kwargs,
                ),
            )

        reward_extra_info: dict = {}
        if isinstance(result, dict):
            score = result["score"]
            reward_extra_info.update(result)
        else:
            score = result
            reward_extra_info["acc"] = score
        reward_extra_info["rm_output"] = rm_output_str
        if rm_output_value is not None:
            reward_extra_info["rm_output_value"] = rm_output_value
        reward_extra_info["agent_response"] = response_str
        if rm_input_len is not None:
            reward_extra_info["rm_input_len"] = rm_input_len
        if rm_output_dict.get("rm_output_len") is not None:
            reward_extra_info["rm_output_len"] = rm_output_dict["rm_output_len"]

        psrl_logger.info("Reward computed uid=%s score=%.4f source=%s", request_uid, score, data_source)
        return {"reward_score": score, "reward_extra_info": reward_extra_info, "reward_metrics": reward_metrics}

    # --- Helpers ---

    def _build_rm_data_proto(self, rm_inputs: dict, request_uid: str | None) -> DataProto:
        """Build a minimal DataProto for RM HTTP inference (raw token IDs only)."""
        input_ids = rm_inputs["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        else:
            input_ids = torch.tensor(input_ids)
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)

        attention_mask = rm_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif isinstance(attention_mask, torch.Tensor):
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
        else:
            attention_mask = torch.tensor(attention_mask)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        batch_size = input_ids.shape[0]
        batch = TensorDict(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            batch_size=batch_size,
        )

        pad_token_id = self.reward_model_tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.reward_model_tokenizer.eos_token_id or 0
        raw_prompt_ids = [_pre_process_inputs(pad_token_id, input_ids[i]) for i in range(batch_size)]

        raw_prompt_ids_arr = np.empty(len(raw_prompt_ids), dtype=object)
        for i, ids in enumerate(raw_prompt_ids):
            raw_prompt_ids_arr[i] = ids.tolist() if isinstance(ids, np.ndarray) else list(ids)

        raw_response_ids_arr = np.empty(batch_size, dtype=object)
        for i in range(batch_size):
            raw_response_ids_arr[i] = []

        uid_value = request_uid if request_uid is not None else "unknown"
        non_tensor_batch = {
            "uid": np.array([uid_value], dtype=object),
            "raw_prompt_ids": raw_prompt_ids_arr,
            "raw_response_ids": raw_response_ids_arr,
        }
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)

    def _get_next_replica_handle(self):
        raise RuntimeError(
            "Direct replica handle access is not available in server-based GenRewardLoop. "
            "All requests go through the smg gateway."
        )

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
        return ",".join(str(u) for u in uid_list)
