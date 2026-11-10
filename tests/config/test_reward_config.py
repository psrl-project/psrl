# psrl/tests/config/test_reward_config.py
import pytest

pytestmark = pytest.mark.cpu_test


def test_reward_model_config_defaults():
    from psrl.workers.config.reward_model import RewardModelConfig

    cfg = RewardModelConfig()
    assert cfg.enable is False
    assert cfg.reward_loop == "naive"
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


def test_multi_reward_model_config_has_reward_loops():
    """Verify that MultiRewardModelConfig uses reward_loops dict + active_loops list."""
    from psrl.workers.config.reward_model import MultiRewardModelConfig

    cfg = MultiRewardModelConfig()
    assert cfg.reward_loops == {}
    assert cfg.active_loops == []
    assert cfg.launch_reward_fn_async is False


def test_validation_reward_pool_has_independent_defaults():
    from psrl.workers.config.reward_model import MultiRewardModelConfig, RewardLoopWorkerConfig

    cfg = MultiRewardModelConfig()
    assert cfg.validation_reward_loop_worker.num_workers == 2, "Validation must have its own worker count."
    assert cfg.validation_reward_loop_worker.max_concurrency_per_worker == 256, "Validation must bound concurrency."
    overridden = MultiRewardModelConfig(reward_loop_worker=RewardLoopWorkerConfig(num_workers=16))
    assert overridden.validation_reward_loop_worker.num_workers == 2, "Training overrides must not change validation."


def test_resolve_active_loops():
    """Test resolve_active_loops function resolves names to configs."""
    from dataclasses import dataclass

    from psrl.workers.config.reward_model import resolve_active_loops

    @dataclass
    class FakeLoopConfig:
        reward_loop_type: str = "naive"

    @dataclass
    class FakeRewardConfig:
        active_loops: list = None
        reward_loops: dict = None

        def __post_init__(self):
            if self.active_loops is None:
                self.active_loops = []
            if self.reward_loops is None:
                self.reward_loops = {}

    dapo_cfg = FakeLoopConfig(reward_loop_type="dapo")
    naive_cfg = FakeLoopConfig(reward_loop_type="naive")

    config = FakeRewardConfig(
        active_loops=["dapo", "naive"],
        reward_loops={"dapo": dapo_cfg, "naive": naive_cfg},
    )
    resolved = resolve_active_loops(config)
    assert len(resolved) == 2
    assert resolved[0].reward_loop_type == "dapo"
    assert resolved[1].reward_loop_type == "naive"


def test_resolve_active_loops_error():
    """Test resolve_active_loops raises ValueError for unknown name."""
    from dataclasses import dataclass

    from psrl.workers.config.reward_model import resolve_active_loops

    @dataclass
    class FakeRewardConfig:
        active_loops: list = None
        reward_loops: dict = None

        def __post_init__(self):
            if self.active_loops is None:
                self.active_loops = []
            if self.reward_loops is None:
                self.reward_loops = {}

    config = FakeRewardConfig(
        active_loops=["nonexistent"],
        reward_loops={},
    )
    with pytest.raises(ValueError, match="nonexistent"):
        resolve_active_loops(config)
