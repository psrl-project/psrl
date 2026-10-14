import torch
from tensordict import TensorDict
from verl.utils import tensordict_utils as tu


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
