"""
Exercise reward context limits and interruptions at the gateway boundary.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from psrl.workers.gen.vllm_async_server import PSRL_vLLMHttpServer
from psrl.workers.reward.reward_loop.gen import GenRewardLoop

pytestmark = pytest.mark.cpu_test


class Response:
    def __init__(self, body):
        self.body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def raise_for_status(self):
        pass

    async def json(self, **kwargs):
        return self.body


def _loop(endpoint, responses, *, max_tokens=512):
    loop = object.__new__(GenRewardLoop)
    loop._model_name = "reward-model"
    loop._smg_endpoint = endpoint
    loop._rm_response_length = 512
    loop._rm_max_model_len = 40960
    loop._sampling_config = {"max_tokens": max_tokens}
    loop.gateway_url = "http://reward-gateway"
    loop._parse_response = (
        loop._parse_completions_response if endpoint == "/v1/completions" else loop._parse_classify_response
    )
    loop._http_client = SimpleNamespace(closed=False, post=Mock(side_effect=[Response(body) for body in responses]))
    return loop


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/classify"])
def test_reward_tokens_are_limited_after_rm_tokenization(endpoint):
    loop = _loop(endpoint, [{"choices": [{"text": "1", "finish_reason": "stop"}], "data": [{"embedding": [1]}]}])
    tokens = list(range(41249))
    data = SimpleNamespace(non_tensor_batch={"raw_prompt_ids": [tokens]})
    result = asyncio.run(loop._query_reward_model(data, "7060"))
    payload = loop._http_client.post.call_args.kwargs["json"]
    actual = payload["prompt"] if endpoint == "/v1/completions" else payload["input"]
    output = payload.get("max_tokens", 0)
    assert len(actual) + output <= 40960, "The RM context must include its requested generation budget."
    assert actual[-1] == tokens[-1], "Truncation must preserve the trailing scoring instruction."
    assert len(tokens) == 41249, "Truncation must not mutate the trajectory."
    assert result["reward_metrics"]["rm_prompt_truncated_tokens"] == 41249 - len(actual), (
        "Reward metrics must expose truncation."
    )


def test_reward_generation_budget_cannot_consume_entire_context():
    loop = _loop("/v1/completions", [], max_tokens=40960)
    with pytest.raises(ValueError, match="no prompt capacity"):
        asyncio.run(loop._query_reward_model(SimpleNamespace(non_tensor_batch={"raw_prompt_ids": [[1]]}), "1"))
    loop._http_client.post.assert_not_called()


def test_interrupted_reward_is_requeued_and_scored_only_after_completion():
    loop = _loop(
        "/v1/completions",
        [
            {"choices": [{"text": "partial score", "finish_reason": "abort"}]},
            {"choices": [{"text": "1", "finish_reason": "stop"}]},
        ],
    )
    result = asyncio.run(loop._query_reward_model(SimpleNamespace(non_tensor_batch={"raw_prompt_ids": [[1, 2]]}), "9"))
    assert result["rm_output_str"] == "1", "Interrupted text must never become a reward."
    assert loop._http_client.post.call_count == 2, "Reclaim must preserve the reward request."
    assert loop._http_client.post.call_args_list[0] == loop._http_client.post.call_args_list[1], (
        "The requeued reward must use the complete original prompt."
    )
    assert result["reward_metrics"]["rm_request_interruptions"] == 1, "Interruptions must remain observable."


def test_reward_sleep_interrupts_instead_of_waiting_for_long_decode():
    server = object.__new__(PSRL_vLLMHttpServer)
    server._is_reward_model = Mock(return_value=True)
    server.close_grpc_generate_admission = AsyncMock()
    server.engine = SimpleNamespace(sleep=AsyncMock())
    server.psrl_config = SimpleNamespace(tms=SimpleNamespace(range="none"))
    asyncio.run(server.sleep(2))
    server.close_grpc_generate_admission.assert_awaited_once()
    server.engine.sleep.assert_awaited_once_with(2, mode="abort")
