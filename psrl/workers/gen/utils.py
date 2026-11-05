from dataclasses import dataclass, field
from typing import Any

import torch

from psrl.utils.routed_experts import canonicalize_routed_experts
from psrl.utils.sampler_distribution import SamplerCapture, SamplerDistribution

# (worker_id, data_parallel_rank)
RolloutInstanceId = tuple[str, int]
INVALID_ROLLOUT_INSTANCE_ID: RolloutInstanceId = ("", -1)

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_CONNECTIONS = 2000


def rollout_token_budget(rollout_config: Any) -> int:
    """
    Return the trainable context budget shared by prompt and response tokens.

    A multi-turn trajectory is one sample whose real limit is the total context,
    so prompt and response share one budget. Capping the response at
    `rollout.response_length` alone would drop the final turns of a long episode.

    Args:
        rollout_config (Any): A rollout config exposing `prompt_length` and
            `response_length`, either a structured config or an OmegaConf node.

    Returns:
        int: The prompt plus response token budget.
    """
    return int(rollout_config.prompt_length) + int(rollout_config.response_length)


@dataclass
class TokenInput:
    input_ids: list[int]
    """input token ids"""
    request_id: int
    """the unique request id"""
    prompt_id: int
    """the unique prompt id"""
    raw_prompt: list[dict] | None = None
    """original chat messages, preserved for SMG multimodal chat-completion calls"""
    rollout_instance_id: RolloutInstanceId | None = None
    """the rollout instance id assigned for this generation"""
    version_tag: int = -1
    """the version tag of the model used for this generation"""
    cu_response_len: int | None = None
    """the current response length for this generation"""
    multi_modal_data: dict | None = None
    """the multi-modal data for this generation, e.g., image/video features or metadata"""
    mm_processor_kwargs: dict[str, Any] | None = None
    """kwargs required to reproduce the rollout's local multimodal tensors"""
    is_validate: bool = False
    """whether this request is for validation purpose"""
    stop_token_ids: list[int] | None = None
    """extra stop token ids required by model-specific tool parsers"""
    priority: int = 0
    """vLLM scheduling priority, lower values are scheduled first"""


@dataclass
class TokenOutput:
    prompt_ids: list[int]
    """input token ids"""
    response_ids: list[int]
    """response token ids"""
    response_mask: list[int]
    """Response mask, 1 for LLM generated token, 0 for tool response token."""
    response_log_probs: list[float] | None = None
    """logprobs of response token ids"""
    routed_experts: Any | None = None
    """routed experts of response token ids"""
    sampler_distribution: SamplerDistribution | None = None
    """
    Compact sampler rows aligned with the response, including empty environment rows.
    """
    sampler_capture: SamplerCapture = field(default_factory=SamplerCapture)
    """Resolved capture config (mode, budget, compact transport) that produced ``sampler_distribution``."""
    # NOTE(linsh): Pooling models return embedding or classification tensors in
    # `pooling_output`, which remains `None` for generative models.
    pooling_output: Any | None = None
    """pooling output tensor for pooling/reward models (torch.Tensor or None)"""
    multi_modal_data: dict | None = None
    """the multi-modal data for this generation, e.g., image/video features or metadata"""
    mm_processor_kwargs: dict[str, Any] | None = None
    """kwargs required to reproduce the rollout's local multimodal tensors"""
    reward_score: float | None = None
    """Reward score for the trajectory."""
    stop_reason: str | None = None
    """stop reason: 'completed', 'aborted', or None for unknown"""
    num_preempted: int | None = None
    """number of preempted times for metric calculation"""
    interrupted: bool = False
    """whether the generation is interrupted"""
    interrupted_by_scheduler: bool = False
    """whether the generation is interrupted by scheduler (preempted)"""
    num_turns: int | None = None
    """number of turns for multi-turn generation"""
    update_status: Any | None = None
    """status of the request"""
    rollout_instance_id: RolloutInstanceId | None = None
    """the rollout instance id used for this generation"""
    agent_reward_info: dict[str, Any] = field(default_factory=dict)
    """Reward metadata produced by the agent loop"""
    extra_fields: dict[str, Any] = field(default_factory=dict)
    """Extra fields for dynamic addition."""

    def as_dict(self) -> dict[str, Any]:
        """Convert the TokenOutput dataclass to a dictionary."""
        output = {
            "prompt_ids": self.prompt_ids,
            "response_ids": self.response_ids,
            "response_mask": self.response_mask,
            "response_log_probs": self.response_log_probs,
            "routed_experts": self.routed_experts,
            "pooling_output": self.pooling_output,
            "multi_modal_data": self.multi_modal_data,
            "reward_score": self.reward_score,
            "stop_reason": self.stop_reason,
            "num_preempted": self.num_preempted,
            "interrupted": self.interrupted,
            "interrupted_by_scheduler": self.interrupted_by_scheduler,
            "num_turns": self.num_turns,
            "update_status": self.update_status,
            "rollout_instance_id": self.rollout_instance_id,
            "agent_reward_info": self.agent_reward_info,
            **self.extra_fields,  # Include any extra fields in the dictionary
        }
        # A row is its own session primary unless a multi-trajectory loop says otherwise.
        # The trainer reads these fields for every rollout, so they must always be
        # present; a loop that sets them to None would otherwise drop the field from
        # the TransferQueue row and fail the batch-wide field read.
        if output.get("session_primary") is None:
            output["session_primary"] = True
        if output.get("loss_weight") is None:
            output["loss_weight"] = 1.0

        output["prompts"] = torch.tensor(output.pop("prompt_ids"), dtype=torch.int64)
        output["responses"] = torch.tensor(output.pop("response_ids"), dtype=torch.int64)
        output["response_mask"] = torch.tensor(output.pop("response_mask"), dtype=torch.int64)

        # Response tensors and masks must describe identical token positions or
        # downstream PPO shapes diverge.
        if output["responses"].size(0) != output["response_mask"].size(0):
            raise AssertionError(
                f"TokenOutput.as_dict: responses has {output['responses'].size(0)} tokens but "
                f"response_mask has {output['response_mask'].size(0)} "
                f"(stop_reason={self.stop_reason}, num_turns={self.num_turns}). "
                "These are built together and must match."
            )

        response_logprobs = output.pop("response_log_probs", None)
        if response_logprobs is not None:
            output["rollout_log_probs"] = torch.tensor(response_logprobs, dtype=torch.float32)
            if output["rollout_log_probs"].size(0) != output["responses"].size(0):
                raise AssertionError(
                    f"TokenOutput.as_dict: rollout_log_probs has "
                    f"{output['rollout_log_probs'].size(0)} entries but responses has "
                    f"{output['responses'].size(0)} tokens "
                    f"(stop_reason={self.stop_reason}, num_turns={self.num_turns}). "
                    "Importance sampling needs one logprob per response token."
                )

        routed_experts = canonicalize_routed_experts(
            output.pop("routed_experts", None),
            prompt_length=len(self.prompt_ids),
            response_length=len(self.response_ids),
        )
        if routed_experts is not None:
            output["routed_experts"] = routed_experts

        if self.sampler_distribution is not None:
            from psrl.utils.sampler_adapter import to_compact_support, to_dense_support, to_dense_topk

            capture = self.sampler_capture
            self.sampler_distribution.validate_capture(capture, self.response_mask)
            if capture.mode == "topk":
                output.update(
                    to_dense_topk(
                        self.sampler_distribution,
                        self.response_mask,
                        prompt_length=len(self.prompt_ids),
                    )
                )
            elif capture.mode == "support":
                adapter = to_compact_support if capture.compact_support else to_dense_support
                output.update(
                    adapter(
                        self.sampler_distribution,
                        self.response_mask,
                        width=capture.width,
                        prompt_length=len(self.prompt_ids),
                    )
                )
            else:
                raise ValueError(f"sampler_distribution present with unsupported capture mode {capture.mode!r}.")

        reward_score = output.get("reward_score", None)
        if reward_score is not None:
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        return output
