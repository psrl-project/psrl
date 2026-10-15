import pytest
import torch
from psrl.workers.reward.utils import ensure_reward_attention_mask
from tensordict import TensorDict

pytestmark = pytest.mark.cpu_test


def test_ensure_reward_attention_mask_adds_all_valid_mask():
    inputs = TensorDict(
        {
            "prompts": torch.tensor([[1, 2, 3]], dtype=torch.int64),
            "responses": torch.tensor([[4, 5]], dtype=torch.int64),
        },
        batch_size=[1],
    )

    result = ensure_reward_attention_mask(inputs)

    assert result is inputs
    assert torch.equal(result["attention_mask"], torch.ones((1, 5), dtype=torch.int64))
    assert "position_ids" not in result


def test_ensure_reward_attention_mask_preserves_existing_mask():
    attention_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.int64)
    inputs = TensorDict(
        {
            "prompts": torch.tensor([[1, 2]], dtype=torch.int64),
            "responses": torch.tensor([[3, 0]], dtype=torch.int64),
            "attention_mask": attention_mask,
        },
        batch_size=[1],
    )

    result = ensure_reward_attention_mask(inputs)

    assert result["attention_mask"] is attention_mask
