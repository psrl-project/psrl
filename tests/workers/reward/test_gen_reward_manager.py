from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from psrl.workers.reward.gen_reward_function.default_gen_rm import DefaultGenRewardFunction
from psrl.workers.reward.reward_loop.gen import GenRewardLoop

pytestmark = pytest.mark.cpu_test


def _make_manager(runner: str, task: str):
    """Construct a GenRewardLoop with a mock reward_model_manager."""
    cfg = OmegaConf.create(
        {
            "psrl": {"logging_path": "/tmp"},
            "gen_actor_rollout_ref": {"rollout": {"prompt_length": 512, "response_length": 512}},
        }
    )

    rm_mgr = MagicMock()
    rm_mgr.get_gateway_url.return_value = "http://127.0.0.1:8200"
    rm_mgr.get_reward_model_tokenizer.return_value = MagicMock()
    rm_mgr.reward_model_config = OmegaConf.create(
        {
            "rollout": {"runner": runner, "task": task, "response_length": 512},
            "sampling_config": {"temperature": 1.0, "top_p": -1},
        }
    )

    manager = GenRewardLoop.__new__(GenRewardLoop)
    manager.config = cfg
    manager.tokenizer = MagicMock()
    manager.reward_model_manager = rm_mgr
    manager.reward_model_tokenizer = MagicMock()
    manager._model_name = "TestRM"
    manager.reward_function = DefaultGenRewardFunction()
    manager._sampling_config = {"temperature": 1.0, "top_p": -1}
    manager._http_client = None
    manager._rm_response_length = 512
    manager._rm_max_model_len = 40960
    manager.reward_kwargs = {}
    manager._smg_endpoint, manager._parse_response = manager._resolve_endpoint_and_parser(runner, task)
    return manager


def test_endpoint_dispatch_generate():
    mgr = _make_manager("generate", "generate")
    assert mgr._smg_endpoint == "/v1/completions"


def test_endpoint_dispatch_classify():
    mgr = _make_manager("pooling", "classify")
    assert mgr._smg_endpoint == "/v1/classify"


def test_endpoint_dispatch_embeddings():
    mgr = _make_manager("pooling", "embed")
    assert mgr._smg_endpoint == "/v1/embeddings"


def test_endpoint_dispatch_unknown_raises():
    mgr = GenRewardLoop.__new__(GenRewardLoop)
    with pytest.raises(ValueError, match="Unsupported"):
        mgr._resolve_endpoint_and_parser("unknown", "task")


def test_parse_completions_response():
    mgr = _make_manager("generate", "generate")
    data = {
        "choices": [{"text": "yes"}],
        "usage": {"completion_tokens": 1},
    }
    result = mgr._parse_completions_response(data, "uid-1")
    assert result["rm_output_str"] == "yes"
    assert result["rm_output_value"] is None
    assert result["rm_output_len"] == 1


def test_parse_classify_response_scalar():
    mgr = _make_manager("pooling", "classify")
    data = {"data": [{"embedding": [0.87]}]}
    result = mgr._parse_classify_response(data, "uid-2")
    assert abs(result["rm_output_value"] - 0.87) < 1e-6
    assert result["rm_output_str"] == ""


def test_parse_classify_response_empty():
    mgr = _make_manager("pooling", "classify")
    result = mgr._parse_classify_response({}, "uid-3")
    assert result["rm_output_value"] is None


def test_parse_embeddings_response():
    mgr = _make_manager("pooling", "embed")
    vec = [0.1, 0.2, 0.3]
    data = {"data": [{"embedding": vec}]}
    result = mgr._parse_embeddings_response(data, "uid-4")
    assert result["rm_output_value"] == vec


def test_build_request_payload_completions():
    mgr = _make_manager("generate", "generate")
    payload = mgr._build_request_payload([1, 2, 3])
    assert payload["prompt"] == [1, 2, 3]
    assert "max_tokens" in payload
    assert "temperature" in payload


def test_build_request_payload_classify():
    mgr = _make_manager("pooling", "classify")
    payload = mgr._build_request_payload([10, 20])
    assert payload == {"model": "TestRM", "input": [10, 20]}


def test_runtime_info_preserves_rm_sampling_and_length():
    from psrl.workers.reward.reward_protocol import RewardModelRuntimeInfo

    rm_config = OmegaConf.create(
        {
            "reward_model_name": "GLM-Z1-9B-0414",
            "rollout": {
                "runner": "generate",
                "task": "generate",
                "response_length": 20480,
                "max_model_len": 40960,
            },
            "sampling_config": {"temperature": 0.8, "top_p": 0.7, "top_k": -1},
        }
    )
    runtime = RewardModelRuntimeInfo("http://127.0.0.1:8200", MagicMock(), rm_config)
    loop = GenRewardLoop(OmegaConf.create({"psrl": {"logging_path": "/tmp"}}), MagicMock(), runtime)
    assert loop.compute_score is None
    payload = loop._build_request_payload([1, 2, 3])
    assert payload["model"] == "GLM-Z1-9B-0414"
    assert payload["max_tokens"] == 20480
    assert payload["temperature"] == 0.8
    assert payload["top_p"] == 0.7
    assert payload["top_k"] == -1
