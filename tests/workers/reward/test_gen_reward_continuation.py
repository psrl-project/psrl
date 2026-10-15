"""
Check metadata and completion handling at the reward gateway boundary.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psrl.workers.reward.reward_loop.gen import GenRewardLoop

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("is_validate", [False, True])
async def test_reward_query_delegates_continuation_to_native_gateway(is_validate):
    class Response:
        headers = {"x-smg-partial-rollout-count": "2"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        async def json(self, **kwargs):
            return {
                "choices": [{"text": "complete judgement", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 17},
            }

    loop = object.__new__(GenRewardLoop)
    loop._model_name = "reward-model"
    loop._smg_endpoint = "/v1/completions"
    loop._rm_response_length = 32
    loop._rm_max_model_len = 128
    loop._sampling_config = {}
    loop.gateway_url = "http://reward-gateway"
    loop._parse_response = loop._parse_completions_response
    loop._http_client = SimpleNamespace(closed=False, post=Mock(return_value=Response()))
    result = await loop._query_reward_model(
        SimpleNamespace(non_tensor_batch={"raw_prompt_ids": [[1, 2, 3]]}),
        "7",
        is_validate=is_validate,
    )
    assert loop._http_client.post.call_count == 1, "A logical RM request must use one gateway call."
    assert loop._http_client.post.call_args.kwargs["headers"] == {
        "x-request-id": "7",
        "x-prompt-id": "7",
        "x-version-tag": "0",
        "x-is-validate": str(is_validate).lower(),
    }, "Reward requests must carry scheduling metadata."
    assert result["rm_output_str"] == "complete judgement", "The complete native result must reach reward scoring."
    assert result["rm_output_len"] == 17, "Reward metrics must use the merged token count."
    assert result["reward_metrics"]["rm_request_interruptions"] == 2, "Native interruptions must remain observable."


async def test_reward_query_rejects_unfinished_gateway_response():
    class Response:
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        async def json(self, **kwargs):
            return {"choices": [{"text": "partial", "finish_reason": "abort"}]}

    loop = object.__new__(GenRewardLoop)
    loop._model_name = "reward-model"
    loop._smg_endpoint = "/v1/completions"
    loop._rm_response_length = 32
    loop._rm_max_model_len = 128
    loop._sampling_config = {}
    loop.gateway_url = "http://reward-gateway"
    loop._http_client = SimpleNamespace(closed=False, post=Mock(return_value=Response()))
    with pytest.raises(RuntimeError, match="unfinished request"):
        await loop._query_reward_model(SimpleNamespace(non_tensor_batch={"raw_prompt_ids": [[1]]}), "7")
    assert loop._http_client.post.call_count == 1, "An unfinished request must not restart from its original prompt."
