from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.scaling_policy.base import InstanceSignal
from psrl.workers.elastic.scaling_policy.rule_based import RuleBasedScalingPolicy

pytestmark = pytest.mark.cpu_test


def _config():
    return SimpleNamespace(psrl=SimpleNamespace(logging_path="/tmp"))


def _policy(**rule_overrides):
    rule_config = {
        "scale_up_threshold": 0.8,
        "scale_down_threshold": 0.2,
        **rule_overrides,
    }
    return RuleBasedScalingPolicy(
        config=_config(),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": 1,
            "rule_based": rule_config,
        },
    )


def _signal(
    role,
    instance_id,
    *,
    awake,
    kv_cache,
    bundle,
    model="model",
    running=0,
    waiting=0,
    timestamp=None,
):
    return InstanceSignal(
        role_name=role,
        model_name=model,
        instance_id=instance_id,
        is_awaken=awake,
        kv_cache_utilization=kv_cache,
        running_queue_num=running,
        waiting_queue_num=waiting,
        generation_throughput=0.0,
        total_token_num=0,
        snapshot_timestamp=timestamp or datetime.now().isoformat(),
        gpu_keys=frozenset({("pool", bundle)}) if bundle is not None else None,
    )


def test_high_kv_cache_scales_up_only_on_free_device():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.91, bundle=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, kv_cache=0.0, bundle=1),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=2),
    ]

    decision = policy.decide(signals)

    assert decision.reason == "rule_based_kv_cache_high_scale_up"
    assert len(decision.actions) == 1
    action = decision.actions[0]
    assert action.action_type == "scale_up"
    assert action.role_name == PSRL_Role.Rollout
    assert action.preferred_instance_ids == [1]
    assert action.pre_sleep_other_preferred is None
    assert policy.allow_preemptive_scale_up is False
    assert policy.rebalance_after_scale_up is False


def test_high_kv_cache_does_not_preempt_when_target_device_is_occupied():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.91, bundle=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, kv_cache=0.0, bundle=1),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=1),
    ]

    decision = policy.decide(signals)

    assert decision.actions == []
    assert decision.reason == "rule_based_no_action_no_free_device"


def test_high_kv_cache_does_not_use_device_reserved_for_training():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.91, bundle=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, kv_cache=0.0, bundle=1),
        _signal(
            PSRL_Role.RewardModel,
            0,
            awake=True,
            kv_cache=0.0,
            bundle=1,
        ),
    ]

    decision = policy.decide(signals)

    assert decision.actions == []
    assert decision.reason == "rule_based_no_action_no_free_device"


def test_router_backlog_scales_up_when_awake_instance_is_not_saturated():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.5, bundle=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, kv_cache=0.0, bundle=1),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=2),
    ]

    decision = policy.decide(
        signals,
        router_backlog_by_role={PSRL_Role.Rollout: 3},
    )

    assert decision.reason == "rule_based_router_backlog_scale_up"
    assert len(decision.actions) == 1
    action = decision.actions[0]
    assert action.action_type == "scale_up"
    assert action.role_name == PSRL_Role.Rollout
    assert action.preferred_instance_ids == [1]


def test_router_backlog_does_not_duplicate_pending_scale_up():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.5, bundle=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, kv_cache=0.0, bundle=1),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=2),
    ]

    decision = policy.decide(
        signals,
        router_backlog_by_role={PSRL_Role.Rollout: 3},
        pending_scale_up_by_role={PSRL_Role.Rollout: 1},
    )

    assert decision.actions == []
    assert decision.reason == "rule_based_no_action_thresholds_not_crossed"


def test_low_kv_cache_scales_down_even_with_local_requests():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.6, bundle=0),
        _signal(
            PSRL_Role.Rollout,
            1,
            awake=True,
            kv_cache=0.05,
            bundle=1,
            running=3,
            waiting=2,
        ),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=2),
    ]

    decision = policy.decide(signals)

    assert decision.reason == "rule_based_kv_cache_low_scale_down"
    assert len(decision.actions) == 1
    action = decision.actions[0]
    assert action.action_type == "scale_down"
    assert action.role_name == PSRL_Role.Rollout
    assert action.preferred_instance_ids == [1]


def test_low_kv_cache_keeps_minimum_awake_instances():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.05, bundle=0),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=1),
    ]

    decision = policy.decide(signals)

    assert decision.actions == []
    assert decision.reason == "rule_based_no_action_thresholds_not_crossed"


@pytest.mark.parametrize("kv_cache", [0.2, 0.8])
def test_threshold_equality_does_not_trigger(kv_cache):
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.5, bundle=0),
        _signal(PSRL_Role.Rollout, 1, awake=True, kv_cache=kv_cache, bundle=1),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=2),
    ]

    decision = policy.decide(signals)

    assert decision.actions == []


def test_stale_low_signal_does_not_scale_down():
    policy = _policy()
    stale_timestamp = (datetime.now() - timedelta(seconds=10)).isoformat()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, kv_cache=0.5, bundle=0),
        _signal(
            PSRL_Role.Rollout,
            1,
            awake=True,
            kv_cache=0.05,
            bundle=1,
            timestamp=stale_timestamp,
        ),
        _signal(PSRL_Role.RewardModel, 0, awake=True, kv_cache=0.5, bundle=2),
    ]

    decision = policy.decide(signals)

    assert decision.actions == []


@pytest.mark.parametrize(
    ("scale_down_threshold", "scale_up_threshold"),
    [(0.8, 0.8), (0.9, 0.8), (-0.1, 0.8), (0.2, 1.1)],
)
def test_invalid_thresholds_are_rejected(
    scale_down_threshold,
    scale_up_threshold,
):
    with pytest.raises(ValueError, match="must satisfy"):
        _policy(
            scale_down_threshold=scale_down_threshold,
            scale_up_threshold=scale_up_threshold,
        )
