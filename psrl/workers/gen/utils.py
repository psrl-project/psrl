from dataclasses import dataclass, field
from typing import Any

# (worker_id, data_parallel_rank)
RolloutInstanceId = tuple[str, int]
INVALID_ROLLOUT_INSTANCE_ID: RolloutInstanceId = ("", -1)

# Default configuration constants
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_CONNECTIONS = 2000


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
        import torch

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

        routed_experts = output.pop("routed_experts", None)
        if routed_experts is not None:
            output["routed_experts"] = torch.tensor(routed_experts, dtype=torch.int64)

        reward_score = output.get("reward_score", None)
        if reward_score is not None:
            reward_score = float(reward_score)
            output["reward_score"] = reward_score
            rm_scores = torch.zeros_like(output["response_mask"], dtype=torch.float32)
            rm_scores[-1] = reward_score
            output["rm_scores"] = rm_scores

        return output
