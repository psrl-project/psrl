from unittest.mock import MagicMock

import psrl.workers.reward.reward_model.manager as _manager_mod
import pytest
from omegaconf import OmegaConf
from psrl.workers.reward.reward_model.manager import RewardModelManager

pytestmark = pytest.mark.cpu_test


def test_manager_requires_gateway_url():
    """RewardModelManager constructor must accept gateway_url parameter."""
    import inspect

    sig = inspect.signature(RewardModelManager.__init__)
    assert "gateway_url" in sig.parameters, "RewardModelManager.__init__ must have a 'gateway_url' parameter"
    assert "status_queues" not in sig.parameters, "RewardModelManager.__init__ must NOT have 'status_queues' (old API)"


def test_manager_has_get_gateway_url():
    """RewardModelManager must expose get_gateway_url()."""
    assert hasattr(RewardModelManager, "get_gateway_url")


def test_manager_has_get_reward_model_tokenizer():
    """RewardModelManager must expose get_reward_model_tokenizer()."""
    assert hasattr(RewardModelManager, "get_reward_model_tokenizer")


def test_reward_servers_register_with_request_model_name(monkeypatch):
    manager = RewardModelManager.__new__(RewardModelManager)
    manager.gateway_url = "http://localhost:8300"
    manager.reward_model_name = "GLM-Z1-9B-0414"
    manager.reward_model_coordinator = MagicMock()
    replica = MagicMock()
    replica.model_config.path = "models/GLM-Z1-9B-0414"
    monkeypatch.setattr(_manager_mod.ray, "get", MagicMock(return_value=["worker-0"]))

    manager._register_reward_servers([replica])

    replica.servers[0].register_server_to_gateway.remote.assert_called_once_with(
        manager.gateway_url,
        model_id="GLM-Z1-9B-0414",
    )


@pytest.mark.parametrize("tokenizer_path", [None, "tokenizers/TestRM"])
def test_manager_and_gateway_use_same_tokenizer_source(monkeypatch, tokenizer_path):
    rm_config = OmegaConf.create(
        {
            "model": {"path": "models/TestRM", "tokenizer_path": tokenizer_path, "use_shm": False},
            "rollout": {},
        }
    )
    copier = MagicMock(return_value="local_tokenizer")
    tokenizer_loader = MagicMock()
    monkeypatch.setattr(_manager_mod, "copy_to_local", copier)
    monkeypatch.setattr(_manager_mod, "hf_tokenizer", tokenizer_loader)
    monkeypatch.setattr(RewardModelManager, "_initialize_reward_replicas", lambda self: None)
    monkeypatch.setattr(RewardModelManager, "_register_reward_servers", lambda self, replicas: None)

    monkeypatch.setattr(_manager_mod.ray, "remote", MagicMock())
    monkeypatch.setattr(_manager_mod.ray, "get", MagicMock())

    RewardModelManager("TestRM", OmegaConf.create({}), rm_config, [], "http://localhost:8300")

    copier.assert_called_once_with(tokenizer_path or rm_config.model.path, use_shm=False)
    tokenizer_loader.assert_called_once_with("local_tokenizer", trust_remote_code=False)
