from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psrl.workers.reward.gen_reward_function.default_gen_rm import DefaultGenRewardFunction
from psrl.workers.reward.reward_loop.gen import GenRewardLoop
from transformers import AutoTokenizer

pytestmark = pytest.mark.cpu_test

_WORKSPACE = Path(__file__).resolve().parents[3]
_QWEN_PATH = _WORKSPACE / "models" / "Qwen2.5-1.5B"
_GLM_PATH = _WORKSPACE / "models" / "GLM-Z1-9B-0414"
_RM_PROMPT_LENGTH = 1024 * 11
_ACTOR_RESPONSE_LENGTH = 1024 * 8


def _n_tokens(tokenized) -> int:
    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        input_ids = tokenized["input_ids"]
    else:
        input_ids = tokenized
    if hasattr(input_ids, "shape"):
        return int(input_ids.shape[-1])
    if input_ids and isinstance(input_ids[0], list):
        return len(input_ids[0])
    return len(input_ids)


def _decode_tail(tokenizer, tokenized, n_tokens: int = 80) -> str:
    if hasattr(tokenized, "keys") and "input_ids" in tokenized:
        input_ids = tokenized["input_ids"]
    else:
        input_ids = tokenized
    if hasattr(input_ids, "tolist"):
        ids = input_ids[0].tolist() if getattr(input_ids, "dim", lambda: 1)() == 2 else input_ids.tolist()
    else:
        ids = input_ids
    return tokenizer.decode(ids[-n_tokens:], skip_special_tokens=True)


@pytest.fixture(scope="module")
def tokenizers():
    if not _QWEN_PATH.exists() or not _GLM_PATH.exists():
        pytest.skip("Qwen/GLM tokenizer checkpoints are not available.")
    qwen = AutoTokenizer.from_pretrained(_QWEN_PATH, trust_remote_code=True)
    glm = AutoTokenizer.from_pretrained(_GLM_PATH, trust_remote_code=True)
    return qwen, glm


def _fill_qwen_response(qwen, text: str, n_tokens: int) -> str:
    token_ids = qwen.encode(text, add_special_tokens=False)
    assert token_ids, "seed text must produce at least one Qwen token."
    repeats = (n_tokens + len(token_ids) - 1) // len(token_ids)
    return qwen.decode((token_ids * repeats)[:n_tokens], skip_special_tokens=True)


def test_qwen_trajectory_does_not_inflate_to_30k_glm_tokens(tokenizers):
    qwen, glm = tokenizers
    reward_fn = DefaultGenRewardFunction()
    problem = "Find the minimum of a/(b+c)+b/(c+a)+c/(a+b) given a+b+c=3."
    solution = _fill_qwen_response(
        qwen,
        "By AM-GM $\\frac{a}{b+c}\\ge \\frac{3}{2}$ and Cauchy-Schwarz. ",
        _ACTOR_RESPONSE_LENGTH,
    )
    messages = reward_fn.prompt_constructor(prompt_str=problem, response_str=solution)
    glm_ids = glm.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    glm_len = _n_tokens(glm_ids)

    # Cross-tokenizer inflation on math CoT is ~1x. 8192 Qwen tokens stay well
    # under both RM prompt_length (11k) and max_model_len (30k).
    assert glm_len < _RM_PROMPT_LENGTH, (
        f"Unexpected Qwen-to-GLM inflation: 8192 Qwen response tokens became {glm_len} GLM tokens."
    )
    assert glm.model_max_length >= 128000, (
        f"GLM model_max_length should be the unclamped HF default, got: {glm.model_max_length}."
    )


def test_apply_chat_template_without_max_length_forwards_overlong_prompt(tokenizers):
    _, glm = tokenizers
    reward_fn = DefaultGenRewardFunction()
    messages = reward_fn.prompt_constructor(
        prompt_str="short problem",
        response_str=("reason step. " * 20000),
    )
    unclamped = glm.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    unclamped_len = _n_tokens(unclamped)
    assert unclamped_len > _RM_PROMPT_LENGTH, (
        f"Expected an overlong RM prompt without max_length, got {unclamped_len}."
    )

    import asyncio

    class Response:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            pass

        async def json(self, **kwargs):
            return {"choices": [{"text": "1", "finish_reason": "stop"}]}

    loop = object.__new__(GenRewardLoop)
    loop._model_name = "reward-model"
    loop._smg_endpoint = "/v1/completions"
    loop._rm_response_length = 128
    loop._rm_max_model_len = _RM_PROMPT_LENGTH + 128
    loop._sampling_config = {}
    loop.gateway_url = "http://reward-gateway"
    loop._parse_response = loop._parse_completions_response
    loop._http_client = SimpleNamespace(closed=False, post=Mock(return_value=Response()))
    ids = unclamped["input_ids"] if hasattr(unclamped, "keys") else unclamped
    ids = ids[0].tolist()
    data = SimpleNamespace(non_tensor_batch={"raw_prompt_ids": [ids]})
    asyncio.run(loop._query_reward_model(data, "request"))
    clamped = loop._http_client.post.call_args.kwargs["json"]["prompt"]
    assert len(clamped) == _RM_PROMPT_LENGTH, "The gateway prompt must fit the model context."
    assert "boxed" in glm.decode(clamped[-80:], skip_special_tokens=True).lower(), (
        "Left truncation must retain the scoring instruction."
    )
    assert glm.truncation_side == "right", "Gateway truncation must not modify tokenizer state."
