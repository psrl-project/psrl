from datetime import datetime
from types import SimpleNamespace

import pytest
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.scaling_policy.base import InstanceSignal
from psrl.workers.elastic.scaling_policy.thoughput_optimal import ThroughputOptimalScalingPolicy

pytestmark = pytest.mark.cpu_test


def _policy(policy_cls=ThroughputOptimalScalingPolicy, *, max_batch: int = 8):
    return policy_cls(
        config=SimpleNamespace(psrl=SimpleNamespace(logging_path="/tmp")),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": 1,
            "throughput_optimal": {
                "max_scale_instances_per_action": max_batch,
                "enable_heterogeneous_parallelism_candidates": True,
                "model_params": {"default": {"A": 0.001, "B": 10.0, "C": 1.0, "D": 0.0}},
            },
        },
    )


def _signal(role, instance_id, *, awake, running, tokens, bundles):
    return InstanceSignal(
        role_name=role,
        model_name="model",
        instance_id=instance_id,
        is_awaken=awake,
        kv_cache_utilization=0.5,
        running_queue_num=running,
        waiting_queue_num=0,
        generation_throughput=0.0,
        total_token_num=tokens,
        snapshot_timestamp=datetime.now().isoformat(),
        gpu_keys=frozenset(("node-0", bundle) for bundle in bundles),
    )


def test_heterogeneous_count_dp_reuses_one_conflict_group():
    policy = _policy(max_batch=4)
    rollout = [
        _signal(PSRL_Role.Rollout, index, awake=False, running=0, tokens=0, bundles=(index,)) for index in range(8)
    ]
    rm = [
        _signal(
            PSRL_Role.RewardModel,
            0,
            awake=True,
            running=39,
            tokens=167963,
            bundles=(0, 1, 2, 3),
        ),
        _signal(
            PSRL_Role.RewardModel,
            1,
            awake=True,
            running=38,
            tokens=159014,
            bundles=(4, 5, 6, 7),
        ),
    ]

    plans = policy._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=2,
        max_batch=4,
    )

    assert tuple(signal.instance_id for signal in plans[4].wakes) == (0, 1, 2, 3)
    assert tuple(signal.instance_id for signal in plans[4].pre_sleep) == (0,)


def test_heterogeneous_planner_computes_each_victim_phi_once(monkeypatch):
    policy = _policy(max_batch=8)
    rollout = [
        _signal(PSRL_Role.Rollout, index, awake=False, running=0, tokens=0, bundles=(index,)) for index in range(16)
    ]
    rm = [
        _signal(
            PSRL_Role.RewardModel,
            index,
            awake=True,
            running=index + 1,
            tokens=(index + 1) * 100,
            bundles=tuple(range(4 * index, 4 * index + 4)),
        )
        for index in range(4)
    ]
    original_instance_phi = policy._instance_phi
    phi_calls = 0

    def counted_instance_phi(signal, delta_requests):
        nonlocal phi_calls
        phi_calls += 1
        return original_instance_phi(signal, delta_requests)

    monkeypatch.setattr(policy, "_instance_phi", counted_instance_phi)

    plans = policy._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=4,
        max_batch=8,
    )

    assert plans
    assert phi_calls == len(rm)
