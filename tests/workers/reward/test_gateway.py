"""CPU tests for RewardModelGateway router args construction."""

import importlib

import psrl.workers.reward.reward_model.gateway as _gateway_module
import pytest
import ray
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test

# `RewardModelGateway` is decorated with `@ray.remote`. Reload the module with a pass-through
# decorator so it is a plain class for CPU tests, then restore the real `ray.remote`.
_ray_remote = ray.remote
ray.remote = lambda cls=None, **kwargs: (cls if cls is not None else lambda c: c)
try:
    importlib.reload(_gateway_module)
finally:
    ray.remote = _ray_remote
RewardModelGateway = _gateway_module.RewardModelGateway


def _make_config(tmp_path):
    return OmegaConf.create({"psrl": {"logging_path": str(tmp_path)}})


def test_gateway_init(tmp_path):
    """RewardModelGateway can be instantiated with config + model_name."""
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(_make_config(tmp_path), "TestRM")
    assert gw.model_name == "TestRM"
    assert gw.smg_url is None


def test_gateway_router_args_policy(tmp_path):
    """_init_router_args sets policy to round_robin and disables PSRL routing."""
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(_make_config(tmp_path), "TestRM")
    gw.smg_ip = "127.0.0.1"
    gw.smg_port = 8300

    args = gw._init_router_args()
    assert args.policy == "round_robin"
    assert args.enable_routing_loop is False
    assert args.worker_selection_strategy == "naive"


def test_gateway_shutdown_noop_when_not_started(tmp_path):
    """shutdown_router is a no-op when router was never started."""
    gw = RewardModelGateway.__new__(RewardModelGateway)
    gw.__init__(_make_config(tmp_path), "TestRM")
    gw.shutdown_router()  # must not raise
