"""
Check reward routing configuration and tokenizer readiness through SMG APIs.
"""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import psrl.workers.reward.reward_model.gateway as gateway_module
import pytest
from omegaconf import OmegaConf
from psrl.workers.gen.smg_adapter import build_reward_router_args

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def rm_config():
    return OmegaConf.create(
        {
            "reward_model_name": "reward-model",
            "model": {"path": "models/reward-model"},
            "rollout": {"max_model_len": 40960},
            "max_concurrent_requests_per_instance": 128,
            "routing_strategy": {
                "method": "itl",
                "cost_model_path": "cost-model.json",
                "max_concurrent_seqs_per_instance": 512,
                "max_num_waiting_reqs_after_preemption": 3,
                "request_budget": 1024,
            },
        }
    )


@pytest.mark.parametrize("method", ["itl", "throughput_optimal", "round_robin"])
def test_reward_router_arguments_use_native_policy_and_limits(rm_config, method):
    rm_config["routing_strategy"]["method"] = method
    args = build_reward_router_args(OmegaConf.create({}), "127.0.0.1", 8300, None, rm_config=rm_config)
    assert args.policy == method
    assert args.connection_mode == "grpc"
    assert args.enable_routing_loop is True
    assert args.max_concurrent_seqs_per_instance == 128
    assert args.max_num_waiting_reqs_after_preemption == 3
    assert args.request_budget == 1024
    assert args.model_path == "models/reward-model"
    assert args.tokenizer_path == "models/reward-model"
    assert args.disable_tokenizer_autoload is True


@pytest.mark.parametrize("method", ["itl", "throughput_optimal", "throughput_optimal_with_budget"])
def test_reward_router_requires_cost_model_for_cost_based_policy(rm_config, method):
    rm_config["routing_strategy"] = {"method": method}
    with pytest.raises(ValueError, match="requires routing_strategy.cost_model_path"):
        build_reward_router_args(OmegaConf.create({}), "127.0.0.1", 8300, None, rm_config=rm_config)


def test_reward_router_resolves_cost_model_directory(rm_config):
    rm_config["model"]["path"] = "Qwen/Qwen3-8B"
    rm_config["reward_model_name"] = "Qwen3-8B"
    rm_config["routing_strategy"]["cost_model_path"] = "psrl/trainer/config/cost_model"
    args = build_reward_router_args(OmegaConf.create({}), "127.0.0.1", 8300, None, rm_config=rm_config)
    assert args.cost_model_path == "psrl/trainer/config/cost_model/qwen3_8b.json"


def test_reward_router_requires_model_path(rm_config):
    rm_config["model"] = {}
    with pytest.raises(ValueError, match="requires model.path"):
        build_reward_router_args(OmegaConf.create({}), "127.0.0.1", 8300, None, rm_config=rm_config)


def test_reward_router_rejects_zero_request_capacity(rm_config):
    rm_config["max_concurrent_requests_per_instance"] = 0
    with pytest.raises(ValueError, match="concurrency must be positive"):
        build_reward_router_args(OmegaConf.create({}), "127.0.0.1", 8300, None, rm_config=rm_config)


def test_reward_router_respects_explicit_tokenizer_path(rm_config):
    rm_config["model"]["tokenizer_path"] = "models/reward-tokenizer"
    args = build_reward_router_args(OmegaConf.create({}), "127.0.0.1", 8300, None, rm_config=rm_config)
    assert args.tokenizer_path == "models/reward-tokenizer"


@pytest.fixture
def gateway(monkeypatch):
    cls = gateway_module.RewardModelGateway.__ray_metadata__.modified_class
    gateway = object.__new__(cls)
    gateway.config = OmegaConf.create({})
    gateway.model_name = "reward-model"
    gateway.smg_ip = None
    gateway.smg_port = None
    gateway.smg_url = None
    gateway.router_process = None
    gateway._init_router_args = Mock(
        return_value=SimpleNamespace(
            tokenizer_path="models/reward-tokenizer",
            chat_template=None,
        )
    )
    process = Mock()
    process.is_alive.return_value = True
    monkeypatch.setattr(gateway_module.multiprocessing, "Process", Mock(return_value=process))
    monkeypatch.setattr(gateway_module.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(gateway_module, "find_available_port", lambda **kwargs: 8300)
    monkeypatch.setattr(gateway_module.time, "sleep", lambda _: None)
    return gateway


def _mock_tokenizer_api(monkeypatch, gateway, *, status="completed", tokenizer=None):
    requests = []
    default_tokenizer = {"name": gateway.model_name, "source": "models/reward-tokenizer", "vocab_size": 100}

    def respond(request):
        requests.append(request)
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/routing_loop/status":
            return httpx.Response(200, json={"paused": False})
        if request.url.path == "/routing_loop/snapshot":
            return httpx.Response(200, json=[])
        if request.url.path == "/routing_loop/migration":
            return httpx.Response(200, content=b"null")
        if request.url.path == "/v1/tokenizers" and request.method == "POST":
            return httpx.Response(200, json={"id": "tokenizer-1"})
        if request.url.path == "/v1/tokenizers/tokenizer-1/status":
            return httpx.Response(200, json={"status": status, "message": "Tokenizer loading failed."})
        if request.url.path == "/v1/tokenizers/tokenizer-1":
            return httpx.Response(200, json=tokenizer or default_tokenizer)
        raise AssertionError(f"Unexpected HTTP request: {request!r}.")

    client_cls = httpx.Client
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(gateway_module.httpx, "Client", lambda **kwargs: client_cls(transport=transport, **kwargs))
    return requests


def test_reward_gateway_registers_named_tokenizer_before_publishing_url(gateway, monkeypatch):
    requests = _mock_tokenizer_api(monkeypatch, gateway)
    assert gateway.launch_router() == "http://127.0.0.1:8300"
    registration = next(request for request in requests if request.method == "POST")
    assert json.loads(registration.content) == {
        "name": "reward-model",
        "source": "models/reward-tokenizer",
        "chat_template_path": None,
    }
    gateway.router_process.start.assert_called_once_with()
    gateway.router_process.terminate.assert_not_called()
    request_count = len(requests)
    assert gateway.launch_router() == gateway.smg_url
    assert len(requests) == request_count


def test_reward_gateway_checks_elastic_apis_before_tokenizer_loading(gateway, monkeypatch):
    gateway.config = OmegaConf.create(
        {
            "psrl": {
                "deployment": {
                    "elastic": {
                        "throughput_optimal": {
                            "enable_request_level_candidate_evaluation": True,
                        }
                    }
                }
            },
        }
    )
    requests = _mock_tokenizer_api(monkeypatch, gateway)
    gateway.launch_router()
    assert [request.url.path for request in requests[:4]] == [
        "/health",
        "/routing_loop/status",
        "/routing_loop/snapshot",
        "/routing_loop/migration",
    ], "Native elastic APIs must be ready before reward model registration."


def test_reward_gateway_stops_if_native_control_apis_are_missing(gateway, monkeypatch):
    monkeypatch.setattr(gateway_module, "wait_for_router_ready", Mock(side_effect=RuntimeError("API unavailable.")))
    gateway._initialize_tokenizer = Mock()
    with pytest.raises(RuntimeError, match="API unavailable"):
        gateway.launch_router()
    gateway._initialize_tokenizer.assert_not_called()
    assert gateway.smg_url is None, "An unavailable gateway must not publish a URL."
    gateway.router_process.terminate.assert_called_once_with()
    gateway.router_process.join.assert_called_once_with()


def test_failed_tokenizer_job_stops_gateway_before_accepting_requests(gateway, monkeypatch):
    _mock_tokenizer_api(monkeypatch, gateway, status="failed")
    with pytest.raises(RuntimeError, match="failed to load"):
        gateway.launch_router()
    assert gateway.smg_url is None
    gateway.router_process.terminate.assert_called_once_with()
    gateway.router_process.join.assert_called_once_with()


@pytest.mark.parametrize("field,value", [("name", "wrong-model"), ("source", "wrong-source"), ("vocab_size", 0)])
def test_wrong_registered_tokenizer_stops_gateway(gateway, monkeypatch, field, value):
    tokenizer = {"name": gateway.model_name, "source": "models/reward-tokenizer", "vocab_size": 100}
    tokenizer[field] = value
    _mock_tokenizer_api(monkeypatch, gateway, tokenizer=tokenizer)
    with pytest.raises(RuntimeError, match="unexpected tokenizer"):
        gateway.launch_router()
    assert gateway.smg_url is None
    gateway.router_process.terminate.assert_called_once_with()
    gateway.router_process.join.assert_called_once_with()


def test_exited_gateway_process_fails_tokenizer_readiness(gateway, monkeypatch):
    _mock_tokenizer_api(monkeypatch, gateway)
    gateway_module.multiprocessing.Process.return_value.is_alive.return_value = False
    with pytest.raises(RuntimeError, match="exited during initialization"):
        gateway.launch_router()
    assert gateway.smg_url is None


def test_tokenizer_startup_timeout_stops_gateway(gateway, monkeypatch):
    _mock_tokenizer_api(monkeypatch, gateway, status="pending")
    times = iter([0.0, 0.0, 0.0, 121.0])
    monkeypatch.setattr(gateway_module.time, "monotonic", lambda: next(times))
    with pytest.raises(TimeoutError, match="initialization timed out"):
        gateway.launch_router()
    assert gateway.smg_url is None
    gateway.router_process.terminate.assert_called_once_with()
