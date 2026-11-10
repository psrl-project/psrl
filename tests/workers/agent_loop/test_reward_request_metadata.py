"""
Test validation metadata and reward results through the shared reward client.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from psrl.workers.agent_loop.loops.generate_agent_loop import GenerateAgentLoop
from psrl.workers.gen.utils import TokenOutput

pytestmark = pytest.mark.cpu_test


@pytest.mark.asyncio
@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize("reward_score", [1.25, np.float32(1.25), np.float64(1.25), np.float32(0), np.float32(-0.5)])
async def test_reward_request_preserves_validation_metadata(validate, reward_score):
    loop = object.__new__(GenerateAgentLoop)
    loop.config = SimpleNamespace(reward=SimpleNamespace(launch_reward_fn_async=False))
    loop.reward_manager = SimpleNamespace(
        compute_score=AsyncMock(return_value={"reward_score": reward_score, "reward_extra_info": {}}),
    )
    output = TokenOutput(prompt_ids=[1], response_ids=[2, 3], response_mask=[1, 1])
    result = await loop.compute_reward_score(output, uid=7, validate=validate)
    request = loop.reward_manager.compute_score.call_args.args[0]
    from verl.utils import tensordict_utils as tu

    assert tu.get(request, "validate") == validate, "Validation metadata must reach the reward manager."
    assert result is output, "Reward computation must retain the output."
    assert result.reward_score == reward_score, "The reward result must be merged."
    fields = result.as_dict()
    assert type(fields["reward_score"]) is float, "Serialized rewards must be Python floats."
    torch.testing.assert_close(fields["rm_scores"], torch.tensor([0.0, float(reward_score)], dtype=torch.float32))


@pytest.mark.asyncio
async def test_aborted_reward_returns_none():
    loop = object.__new__(GenerateAgentLoop)
    loop.config = SimpleNamespace(reward=SimpleNamespace(launch_reward_fn_async=False))
    loop.reward_manager = SimpleNamespace(compute_score=AsyncMock(return_value=None))
    output = TokenOutput(prompt_ids=[1], response_ids=[2], response_mask=[1])
    assert await loop.compute_reward_score(output, uid=7) is None, "An aborted reward must abort the output."


def test_unscored_output_does_not_create_token_rewards():
    output = TokenOutput(prompt_ids=[1], response_ids=[2], response_mask=[1])
    fields = output.as_dict()
    assert fields["reward_score"] is None, "Deferred rewards must remain unset."
    assert "rm_scores" not in fields, "An unscored rollout must not publish a zero reward."
