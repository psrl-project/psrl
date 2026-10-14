# psrl/tests/config/test_reward_config.py
import pytest

pytestmark = pytest.mark.cpu_test


def test_reward_model_config_defaults():
    from psrl.workers.config.reward_model import RewardModelConfig

    cfg = RewardModelConfig()
    assert cfg.enable is False
    assert cfg.reward_manager == "naive"
    assert cfg.n_gpus_per_node == 0
    assert cfg.max_new_tokens == 4096


def test_single_reward_model_config_defaults():
    from psrl.workers.config.reward_model import SingleRewardModelConfig

    cfg = SingleRewardModelConfig()
    assert cfg.reward_model_name is None
    assert cfg.reward_loop_type is None
    assert cfg.reward_fn is None
    assert cfg.num_replicas == 1
    assert cfg.reward_loop_kwargs == {}


def test_reward_loop_worker_config_defaults():
    from psrl.workers.config.reward_model import RewardLoopWorkerConfig

    cfg = RewardLoopWorkerConfig()
    assert cfg.num_workers == 8
    assert cfg.max_concurrency_per_worker == 256


def test_reward_model_configs_are_exported_from_workers_config():
    from psrl.workers.config import MultiRewardModelConfig, RewardModelConfig, SingleRewardModelConfig

    assert MultiRewardModelConfig is not None
    assert RewardModelConfig is not None
    assert SingleRewardModelConfig is not None


def test_multi_reward_model_config_has_managers():
    """Verify that MultiRewardModelConfig uses managers dict + active_managers list."""
    from psrl.workers.config.reward_model import MultiRewardModelConfig

    cfg = MultiRewardModelConfig()
    assert cfg.managers == {}
    assert cfg.active_managers == []
    assert cfg.launch_reward_fn_async is False


def test_resolve_active_managers():
    """Test resolve_active_managers function resolves names to configs."""
    from dataclasses import dataclass

    from psrl.workers.config.reward_model import resolve_active_managers

    @dataclass
    class FakeManagerConfig:
        reward_loop_type: str = "naive"

    @dataclass
    class FakeRewardConfig:
        active_managers: list = None
        managers: dict = None

        def __post_init__(self):
            if self.active_managers is None:
                self.active_managers = []
            if self.managers is None:
                self.managers = {}

    dapo_cfg = FakeManagerConfig(reward_loop_type="dapo")
    naive_cfg = FakeManagerConfig(reward_loop_type="naive")

    config = FakeRewardConfig(
        active_managers=["dapo", "naive"],
        managers={"dapo": dapo_cfg, "naive": naive_cfg},
    )
    resolved = resolve_active_managers(config)
    assert len(resolved) == 2
    assert resolved[0].reward_loop_type == "dapo"
    assert resolved[1].reward_loop_type == "naive"


def test_resolve_active_managers_error():
    """Test resolve_active_managers raises ValueError for unknown name."""
    from dataclasses import dataclass

    from psrl.workers.config.reward_model import resolve_active_managers

    @dataclass
    class FakeRewardConfig:
        active_managers: list = None
        managers: dict = None

        def __post_init__(self):
            if self.active_managers is None:
                self.active_managers = []
            if self.managers is None:
                self.managers = {}

    config = FakeRewardConfig(
        active_managers=["nonexistent"],
        managers={},
    )
    with pytest.raises(ValueError, match="nonexistent"):
        resolve_active_managers(config)
