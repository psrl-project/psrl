"""CPU-only tests for the current actor configuration load-weight contract."""

import pytest

pytestmark = pytest.mark.cpu_test


def test_fsdp_actor_config_uses_fsdp_strategy():
    from verl.workers.config.actor import FSDPActorConfig

    cfg = FSDPActorConfig(rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    assert cfg.strategy == "fsdp"
    assert cfg.engine is cfg.fsdp_config


def test_mcore_actor_load_weight_defaults_to_true():
    from verl.workers.config.actor import McoreActorConfig

    cfg = McoreActorConfig(rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    assert cfg.engine.load_weight is True


def test_mcore_actor_load_weight_can_be_disabled():
    from verl.workers.config.actor import McoreActorConfig
    from verl.workers.config.engine import McoreEngineConfig

    cfg = McoreActorConfig(
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        megatron=McoreEngineConfig(load_weight=False),
    )
    assert cfg.engine.load_weight is False
