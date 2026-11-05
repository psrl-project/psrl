import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu

from psrl.utils.dataset.utils import normalize_reward_model_dicts


def ensure_reward_attention_mask(inputs: TensorDict) -> TensorDict:
    """Add the all-valid mask expected by reward loops when it is absent.

    Agent-loop reward requests contain unpadded prompt and response tensors, so
    every token is valid. Callers may provide an existing mask for padded inputs.
    """
    if tu.get(inputs, "attention_mask", default=None) is not None:
        return inputs

    prompts = tu.get(inputs, "prompts")
    responses = tu.get(inputs, "responses")
    total_length = prompts.shape[-1] + responses.shape[-1]
    inputs["attention_mask"] = prompts.new_ones(
        (*prompts.shape[:-1], total_length),
        dtype=torch.int64,
    )
    return inputs


def attach_reward_trajectories(inputs: TensorDict, outputs: list) -> None:
    """Carry every training trajectory to the reward worker before TransferQueue publication."""
    specs = normalize_reward_model_dicts(tu.get(inputs, "reward_model_dicts")[0])
    if not any(spec.get("reward_loop_type") == "opd" for spec in specs):
        return
    tu.assign_non_tensor_stack(
        inputs,
        "reward_trajectories",
        [
            [
                {
                    "prompts": output.prompt_ids,
                    "responses": output.response_ids,
                    "multi_modal_data": output.multi_modal_data,
                }
                for output in outputs
            ]
        ],
    )
