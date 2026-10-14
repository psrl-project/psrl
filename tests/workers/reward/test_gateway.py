import json
from unittest.mock import MagicMock

import httpx
import psrl.workers.reward.reward_model.gateway as _gateway_mod
import pytest
from omegaconf import OmegaConf
from psrl.workers.reward.reward_model.gateway import RewardModelGateway

pytestmark = pytest.mark.cpu_test

RewardModelGateway = RewardModelGateway.__ray_metadata__.modified_class


def _make_config():
    return OmegaConf.create(
        {
            "psrl": {
                "logging_path": "/tmp/psrl_test_logs",
            }
        }
    )


def _make_rm_config():
    return OmegaConf.create(
        {
            "reward_model_name": "TestRM",
            "model": {"path": "models/TestRM"},
            "max_concurrent_requests_per_instance": 128,
            "rollout": {"max_model_len": 40960},
            "routing_strategy": {
                "method": "itl",
                "cost_model_path": "model.json",
                "max_concurrent_seqs_per_instance": 256,
                "delta_throughput_threshold": 0.005,
                "request_budget": 2048,
                "max_num_waiting_reqs_after_preemption": 3,
            },
        }
    )


def test_gateway_init():
    """RewardModelGateway retains its model configuration."""
    cfg = _make_config()
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(cfg, _make_rm_config())
    assert gw.model_name == "TestRM"
    assert gw.smg_url is None


def test_gateway_router_args_policy():
    """RM policy and capacity use RM configuration independently of rollout."""
    cfg = _make_config()
    cfg.psrl.rollout_coordination = {
        "routing_strategy": {
            "method": "throughput_optimal",
            "cost_model_path": "rollout.json",
            "max_concurrent_seqs_per_instance": 512,
            "delta_throughput_threshold": 0.2,
        }
    }
    cfg.train_actor_rollout_ref = {"model": {"path": "models/Policy"}}
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(cfg, _make_rm_config())
    gw.smg_ip = "127.0.0.1"
    gw.smg_port = 8300

    args = gw._init_router_args()
    assert args.policy == "itl"
    assert args.enable_routing_loop is True
    assert args.worker_selection_strategy == "naive"
    assert args.dp_aware is True
    assert args.max_concurrent_seqs_per_instance == 128
    assert args.cost_model_path == "model.json"
    assert args.max_prompt_length == 40960
    assert args.delta_throughput_threshold == 0.005
    assert args.request_budget == 2048
    assert args.max_num_waiting_reqs_after_preemption == 3
    assert args.model_path == "models/TestRM"
    assert args.tokenizer_path == "models/TestRM"
    assert args.disable_tokenizer_autoload is True


def test_gateway_uses_explicit_reward_tokenizer_path():
    rm_cfg = _make_rm_config()
    rm_cfg.model.tokenizer_path = "tokenizers/TestRM"
    gw = RewardModelGateway(_make_config(), rm_cfg)
    args = gw._init_router_args()
    assert args.model_path == "models/TestRM"
    assert args.tokenizer_path == "tokenizers/TestRM"


def test_gateway_requires_reward_model_path():
    rm_cfg = _make_rm_config()
    del rm_cfg.model.path
    gw = RewardModelGateway(_make_config(), rm_cfg)
    with pytest.raises(ValueError, match="requires model.path"):
        gw._init_router_args()


def test_itl_configuration_requires_cost_model():
    cfg = _make_config()
    rm_cfg = _make_rm_config()
    del rm_cfg.routing_strategy.cost_model_path
    gw = RewardModelGateway(cfg, rm_cfg)
    with pytest.raises(ValueError, match="requires routing_strategy.cost_model_path"):
        gw._init_router_args()


def test_gateway_shutdown_noop_when_not_started():
    """shutdown_router is a no-op when router was never started."""
    cfg = _make_config()
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(cfg, _make_rm_config())
    gw.shutdown_router()  # must not raise


def _mock_tokenizer_api(monkeypatch, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(_gateway_mod.httpx, "Client", lambda **kwargs: client)
    monkeypatch.setattr(_gateway_mod.time, "sleep", lambda seconds: None)


def test_gateway_waits_for_tokenizer_registered_by_model_name(monkeypatch):
    gw = RewardModelGateway(_make_config(), _make_rm_config())
    gw.router_process = MagicMock()
    args = gw._init_router_args()
    events = []
    registration = {}

    def handle(request):
        assert gw.smg_url is None, "Gateway URL must stay unpublished while the tokenizer loads."
        path = request.url.path
        events.append(path)
        if path == "/health":
            return httpx.Response(200)
        if path == "/v1/tokenizers":
            registration.update(json.loads(request.content))
            return httpx.Response(202, json={"id": "tokenizer-1", "status": "pending"})
        if path.endswith("/status"):
            status = "pending" if events.count(path) == 1 else "completed"
            return httpx.Response(200, json={"status": status})
        return httpx.Response(200, json={**registration, "vocab_size": 150000})

    _mock_tokenizer_api(monkeypatch, handle)
    gw._initialize_tokenizer("http://localhost:8300", args)
    assert registration["name"] == "TestRM", "Tokenizer must be registered by request model name."
    assert registration["source"] == "models/TestRM", "Tokenizer must use the reward model path."
    assert events.count("/v1/tokenizers/tokenizer-1/status") == 2, "Initialization must wait for loading."


def test_gateway_surfaces_tokenizer_load_failure(monkeypatch):
    gw = RewardModelGateway(_make_config(), _make_rm_config())
    gw.router_process = MagicMock()

    def handle(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.method == "POST":
            return httpx.Response(202, json={"id": "tokenizer-1"})
        return httpx.Response(200, json={"status": "failed", "message": "Invalid tokenizer JSON."})

    _mock_tokenizer_api(monkeypatch, handle)
    with pytest.raises(RuntimeError, match="TestRM.*models/TestRM.*Invalid tokenizer JSON"):
        gw._initialize_tokenizer("http://localhost:8300", gw._init_router_args())


def test_gateway_rejects_wrong_tokenizer_registration(monkeypatch):
    gw = RewardModelGateway(_make_config(), _make_rm_config())
    gw.router_process = MagicMock()

    def handle(request):
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.method == "POST":
            return httpx.Response(202, json={"id": "tokenizer-1"})
        if request.url.path.endswith("/status"):
            return httpx.Response(200, json={"status": "completed"})
        return httpx.Response(200, json={"name": "Policy", "source": "models/Policy", "vocab_size": 150000})

    _mock_tokenizer_api(monkeypatch, handle)
    with pytest.raises(RuntimeError, match="unexpected tokenizer"):
        gw._initialize_tokenizer("http://localhost:8300", gw._init_router_args())


def test_gateway_detects_router_exit(monkeypatch):
    gw = RewardModelGateway(_make_config(), _make_rm_config())
    gw.router_process = MagicMock()
    _mock_tokenizer_api(monkeypatch, lambda request: httpx.Response(202, json={"id": "tokenizer-1"}))
    gw.router_process.is_alive.return_value = False
    with pytest.raises(RuntimeError, match="exited during initialization"):
        gw._initialize_tokenizer("http://localhost:8300", gw._init_router_args())


def test_gateway_bounds_tokenizer_initialization(monkeypatch):
    gw = RewardModelGateway(_make_config(), _make_rm_config())
    gw.router_process = MagicMock()
    _mock_tokenizer_api(monkeypatch, lambda request: httpx.Response(202, json={"id": "tokenizer-1"}))
    monkeypatch.setattr(_gateway_mod, "_ROUTER_STARTUP_TIMEOUT_S", 0.0)
    with pytest.raises(TimeoutError, match="tokenizer initialization timed out"):
        gw._initialize_tokenizer("http://localhost:8300", gw._init_router_args())


def test_gateway_stops_router_after_initialization_failure(monkeypatch):
    gw = RewardModelGateway(_make_config(), _make_rm_config())
    process = MagicMock()
    monkeypatch.setattr(_gateway_mod.multiprocessing, "Process", MagicMock(return_value=process))
    monkeypatch.setattr(_gateway_mod.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(_gateway_mod, "wait_for_router_ready", lambda *args, **kwargs: None)
    monkeypatch.setattr(gw, "_initialize_tokenizer", MagicMock(side_effect=RuntimeError("Tokenizer failed.")))

    with pytest.raises(RuntimeError, match="Tokenizer failed"):
        gw.launch_router()
    process.terminate.assert_called_once()
    process.join.assert_called_once()
    assert gw.smg_url is None, "Failed initialization must not publish a gateway URL."
