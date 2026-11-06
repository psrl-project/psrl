"""Score complete trajectories through a reward model's SMG gateway."""

import math
from typing import Any

import aiohttp
import torch
from tensordict import TensorDict
from verl.trainer.distillation.losses import get_distillation_loss_settings
from verl.utils import tensordict_utils as tu

from psrl.utils.rollout.vision_utils import serialize_image_inputs
from psrl.workers.reward.reward_loop.base import RewardLoopBase
from psrl.workers.reward.reward_loop.registry import register
from psrl.workers.reward.reward_protocol import RewardModelRuntimeInfo


def _logprob_pair(pair: list, vocab_size: int) -> tuple[float, int]:
    if not isinstance(pair, list) or len(pair) != 2:
        raise ValueError(f"Invalid teacher logprob pair {pair!r}.")
    probability, token = pair
    if isinstance(token, float) and token.is_integer():
        token = int(token)
    if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token < vocab_size:
        raise ValueError(f"Invalid teacher token ID {token!r} for vocabulary size {vocab_size!r}.")
    if isinstance(probability, bool) or not isinstance(probability, (float, int)):
        raise ValueError(f"Invalid teacher logprob {probability!r}.")
    if not math.isfinite(probability) or probability > 0:
        raise ValueError(f"Invalid teacher logprob {probability!r}.")
    return float(probability), token


def parse_teacher_response(
    result: dict,
    input_ids: list[int],
    prompt_length: int,
    topk: int,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    """
    Validate raw SMG prompt probabilities and select response targets.

    The response starts at input position `prompt_length`, including masked tool tokens.
    """
    if not isinstance(result, dict) or not 0 < prompt_length < len(input_ids):
        raise ValueError("Teacher scoring requires a response object and a nonempty prompt and response.")
    meta = result.get("meta_info", {})
    if not isinstance(meta, dict):
        raise ValueError("Teacher response requires meta_info.")
    if meta.get("prompt_token_ids") != input_ids:
        raise ValueError("Teacher prompt_token_ids do not match the complete input token IDs.")
    logprobs = meta.get("input_token_logprobs")
    if not isinstance(logprobs, list) or len(logprobs) != len(input_ids):
        raise ValueError("Teacher input_token_logprobs must cover every input position.")
    if logprobs[0] != [None, input_ids[0]]:
        raise ValueError("Teacher first input token must have a null logprob and its raw token ID.")
    for position, pair in enumerate(logprobs[1:], 1):
        _, token = _logprob_pair(pair, vocab_size)
        if token != input_ids[position]:
            raise ValueError(f"Teacher token alignment failed at input position {position!r}.")
    if not topk:
        return {"teacher_logprobs": torch.tensor(logprobs[prompt_length:], dtype=torch.float32)[:, :1]}

    alternatives = meta.get("input_top_logprobs")
    if not isinstance(alternatives, list) or len(alternatives) != len(input_ids) or alternatives[0] != []:
        raise ValueError("Teacher input_top_logprobs must cover the input with an empty first position.")
    selected = []
    for position, pairs in enumerate(alternatives[1:], 1):
        if not isinstance(pairs, list):
            raise ValueError(f"Invalid teacher Top-k alternatives at input position {position!r}.")
        values = [_logprob_pair(pair, vocab_size) for pair in pairs]
        if len(values) < topk or len({token for _, token in values}) != len(values):
            raise ValueError(f"Teacher Top-k entries are incomplete or duplicated at position {position!r}.")
        values.sort(key=lambda pair: (-pair[0], pair[1]))
        if position >= prompt_length:
            selected.append(values[:topk])
    return {
        "teacher_logprobs": torch.tensor([[p for p, _ in row] for row in selected], dtype=torch.float32),
        "teacher_ids": torch.tensor([[t for _, t in row] for row in selected], dtype=torch.int64),
    }


@register("opd")
class OPDRewardLoop(RewardLoopBase):
    """Return response aligned teacher targets through the reward model gateway."""

    def __init__(
        self,
        config,
        tokenizer,
        reward_model_manager: RewardModelRuntimeInfo | None = None,
        **reward_kwargs: Any,
    ):
        super().__init__(config, tokenizer, compute_score=None)
        if reward_kwargs:
            raise ValueError("Configure OPD losses under distillation, without reward_kwargs.")
        if reward_model_manager is None:
            raise ValueError("OPD requires reward model runtime information.")
        teacher_tokenizer = reward_model_manager.get_reward_model_tokenizer()
        if tokenizer.get_vocab() != teacher_tokenizer.get_vocab():
            raise ValueError("OPD student and teacher tokenizers must have identical token ID vocabularies.")
        self.vocab_size = len(tokenizer)
        loss = config.distillation.distillation_loss
        self.topk = loss.topk if get_distillation_loss_settings(loss.loss_mode).use_topk else 0
        if (
            isinstance(self.topk, bool)
            or not isinstance(self.topk, int)
            or not 0 <= self.topk <= self.vocab_size
            or (get_distillation_loss_settings(loss.loss_mode).use_topk and not self.topk)
        ):
            raise ValueError(f"Invalid teacher Top-k width {self.topk!r}.")
        rm_config = reward_model_manager.reward_model_config
        if rm_config.rollout.runner != "generate":
            raise ValueError("OPD teachers require a generative reward model.")
        self.max_model_len = int(rm_config.rollout.max_model_len)
        self.model_name = rm_config.reward_model_name
        self.gateway_url = reward_model_manager.get_gateway_url().rstrip("/")

    async def run_single(self, data: TensorDict) -> dict:
        if len(data) != 1:
            raise ValueError("OPD scoring requires exactly one trajectory.")
        multimodal = tu.get(data, "multi_modal_data", [None])[0] or {}
        if multimodal.get("videos") or multimodal.get("audios"):
            raise ValueError("OPD through SMG supports image trajectories. Video and audio are unsupported.")
        prompts, responses = data["prompts"][0], data["responses"][0]
        mask = data["attention_mask"][0].bool()
        prompt_ids = prompts[mask[: len(prompts)]].tolist()
        response_ids = responses[mask[len(prompts) :]].tolist()
        if not prompt_ids or not response_ids:
            raise ValueError("OPD requires a nonempty prompt and response.")
        input_ids = prompt_ids + response_ids
        if any(not 0 <= token < self.vocab_size for token in input_ids):
            raise ValueError("OPD input contains a token outside the teacher vocabulary.")
        if len(input_ids) + 1 > self.max_model_len:
            raise ValueError(f"Teacher input requires {len(input_ids) + 1} tokens, capacity is {self.max_model_len}.")
        payload = {
            "model": self.model_name,
            "input_ids": input_ids,
            "sampling_params": {"temperature": 1.0, "max_new_tokens": 1},
            "return_logprob": True,
            "return_prompt_token_ids": True,
            "logprob_start_len": 0,
            "top_logprobs_num": self.topk,
            "return_text_in_logprobs": False,
            "stream": False,
        }
        images = multimodal.get("images")
        if images:
            payload.update(
                image_data=await serialize_image_inputs(images),
                multimodal_token_mode="preexpanded",
                modalities=["multi-images" if len(images) > 1 else "image"],
            )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as client:
            async with client.post(f"{self.gateway_url}/generate", json=payload) as response:
                if response.status != 200:
                    raise RuntimeError(f"Teacher scoring failed with HTTP {response.status}: {await response.text()}.")
                result = await response.json()
        if not isinstance(result, list) or len(result) != 1:
            raise ValueError("Teacher scoring must return exactly one SMG response in an array.")
        targets = parse_teacher_response(result[0], input_ids, len(prompt_ids), self.topk, self.vocab_size)
        return {"reward_score": 0.0, "reward_extra_info": {}, "reward_metrics": {}, **targets}
