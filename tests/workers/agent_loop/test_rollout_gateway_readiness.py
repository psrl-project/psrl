"""
Verify rollout gateways publish URLs only after native control readiness.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import psrl.workers.gen.rollout_gateway as gateway_module
import pytest
from omegaconf import OmegaConf

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def gateway(monkeypatch):
    cls = gateway_module.RolloutGateway.__ray_metadata__.modified_class
    gateway = object.__new__(cls)
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
    gateway.smg_url = None
    gateway.router_process = None
    gateway.session_router_process = None
    gateway._init_router_args = Mock(return_value=SimpleNamespace())
    process = Mock()
    process.is_alive.return_value = True
    monkeypatch.setattr(gateway_module.multiprocessing, "Process", Mock(return_value=process))
    monkeypatch.setattr(gateway_module.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(gateway_module, "find_available_port", lambda **kwargs: 8100)
    monkeypatch.setattr(gateway_module, "wait_for_router_ready", Mock())
    return gateway


def test_rollout_gateway_validates_elastic_runtime_before_publishing_url(gateway):
    def check_ready(url, process, **kwargs):
        assert gateway.smg_url is None, "The gateway must stay unpublished during startup checks."

    gateway_module.wait_for_router_ready.side_effect = check_ready
    assert gateway.launch_router() == "http://127.0.0.1:8100", "Publish the validated gateway URL."
    gateway_module.wait_for_router_ready.assert_called_once_with(
        gateway.smg_url,
        gateway.router_process,
        require_elastic_runtime=True,
    )
    assert gateway.launch_router() == gateway.smg_url, "Launching a ready gateway must be idempotent."
    gateway_module.wait_for_router_ready.assert_called_once()


def test_missing_native_apis_stop_unpublished_rollout_process(gateway):
    gateway_module.wait_for_router_ready.side_effect = RuntimeError("API unavailable.")
    process = gateway_module.multiprocessing.Process.return_value
    with pytest.raises(RuntimeError, match="API unavailable"):
        gateway.launch_router()
    process.terminate.assert_called_once_with()
    process.join.assert_called_once_with()
    assert gateway.smg_url is None, "Startup failure must leave the gateway unpublished."
    assert gateway.router_process is None, "Startup failure must clear the router process."
    gateway.shutdown_router()
    process.terminate.assert_called_once_with()
