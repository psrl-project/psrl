"""
Verify rollout and reward clients share one coordinator owner.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from psrl.trainer.ppo import ray_trainer
from psrl.workers.global_coordination.client import CoordinatorClient
from psrl.workers.global_coordination.coordinator import GlobalCoordinator

pytestmark = pytest.mark.cpu_test


def test_default_config_uses_global_coordination_without_a_switch():
    config_dir = str(Path(__file__).resolve().parents[2] / "psrl/trainer/config")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")
    assert "global_coordinator" not in config.psrl, "Global coordination must not require a configuration switch."


def test_trainer_creates_one_actor_and_passes_scoped_clients(monkeypatch):
    actor_type = Mock()
    actor_type.options.return_value = actor_type
    remote = Mock(return_value=actor_type)
    monkeypatch.setattr(ray_trainer.ray, "remote", remote)
    monkeypatch.setattr(ray_trainer.ray, "get", lambda ref: ref)
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.config = OmegaConf.create(
        {
            "psrl": {"ps_manager_ip": "127.0.0.1"},
            "reward": {
                "active_loops": ["judge"],
                "reward_loops": {
                    "judge": {
                        "reward_loop_type": "gen",
                        "reward_model_name": "judge",
                        "num_replicas": 1,
                        "model": {"path": "models/judge"},
                    }
                },
            },
        }
    )
    trainer.global_coordinator = None
    trainer.rollout_gateway_url = "http://rollout"
    trainer.session_router_url = "http://session"
    trainer.ip_to_node_id = {"127.0.0.1": "node"}
    trainer.ps_manager_handle = Mock()
    trainer.reward_gateway_urls = {"judge": "http://reward"}
    trainer.reward_model_to_manager = {}
    trainer.elastic_mode = False
    manager = Mock()
    monkeypatch.setattr(ray_trainer, "RewardModelManager", manager)

    trainer.init_global_coordinator()
    owner = actor_type.remote.return_value
    remote.assert_called_once_with(GlobalCoordinator)
    actor_type.remote.assert_called_once_with(
        trainer.config, trainer.ps_manager_handle, trainer.rollout_gateway_url, trainer.session_router_url
    )
    assert trainer.global_coordinator is owner, "The trainer must retain the owning actor handle."
    assert trainer.rollout_coordinator == CoordinatorClient(owner), "Policy calls must use the shared actor."

    trainer.init_reward_model_servers({"reward_model_judge_0": Mock()})
    reward_client = manager.call_args.kwargs["coordinator"]
    owner.register_reward_model.remote.assert_called_once_with(
        trainer.config.reward.reward_loops.judge, "http://reward"
    )
    assert reward_client == CoordinatorClient(owner, ("reward", "judge")), (
        "The reward manager must use its own scope on the shared actor."
    )
    remote.assert_called_once_with(GlobalCoordinator)


@pytest.mark.parametrize("timeout", [False, True])
def test_shutdown_targets_the_owning_actor(monkeypatch, timeout):
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    owner = Mock()
    trainer.global_coordinator = owner
    trainer.rollout_coordinator = CoordinatorClient(owner)
    get = Mock(side_effect=ray_trainer.ray.exceptions.GetTimeoutError() if timeout else None)
    kill = Mock()
    monkeypatch.setattr(ray_trainer.ray, "get", get)
    monkeypatch.setattr(ray_trainer.ray, "kill", kill)

    trainer.stop_global_coordinator()

    owner.stop_busy_loop.remote.assert_called_once_with()
    owner.invoke.remote.assert_not_called()
    if timeout:
        kill.assert_called_once_with(owner, no_restart=True)
    else:
        kill.assert_not_called()
    assert trainer.rollout_coordinator is trainer.global_coordinator is None, (
        "Shutdown must clear both coordinator references."
    )
