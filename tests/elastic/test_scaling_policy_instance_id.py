import pytest
from psrl.workers.elastic.scaling_policy.base import (
    InstanceSignal,
    ScalingAction,
    ScalingPolicy,
)
from psrl.workers.gen.utils import RolloutInstanceId

pytestmark = pytest.mark.cpu_test


# --- Helpers ---


def _make_signal(instance_id: RolloutInstanceId, is_awaken: bool = True, kv: float = 0.5) -> InstanceSignal:
    return InstanceSignal(
        role_name="Rollout",
        model_name="test_model",
        instance_id=instance_id,
        is_awaken=is_awaken,
        kv_cache_utilization=kv,
        running_queue_num=5,
        waiting_queue_num=0,
        generation_throughput=10.0,
        total_token_num=100,
    )


# --- Tests ---


def test_instance_signal_accepts_rollout_instance_id():
    """InstanceSignal.instance_id should accept a RolloutInstanceId tuple."""
    iid: RolloutInstanceId = ("worker_abc", 0)
    sig = _make_signal(iid)
    assert sig.instance_id == ("worker_abc", 0)


def test_instance_signal_rejects_bare_int():
    """A tuple instance ID must retain its components unchanged."""
    iid: RolloutInstanceId = ("worker_xyz", 2)
    sig = _make_signal(iid)
    assert isinstance(sig.instance_id, tuple)
    assert sig.instance_id[0] == "worker_xyz"
    assert sig.instance_id[1] == 2


def test_scaling_action_preferred_instance_ids_accepts_tuple_list():
    """ScalingAction.preferred_instance_ids should accept list[RolloutInstanceId]."""
    iid: RolloutInstanceId = ("worker_abc", 0)
    action = ScalingAction(
        action_type="scale_up",
        role_name="Rollout",
        model_name="test_model",
        preferred_instance_ids=[iid],
    )
    assert action.preferred_instance_ids == [("worker_abc", 0)]


def test_build_mu_maps_uses_tuple_key():
    """_build_mu_maps should produce keys of (str, str, RolloutInstanceId) where the last element is a tuple."""
    policy = ScalingPolicy.__new__(ScalingPolicy)
    policy.cost_config = {}
    policy.config = None

    iid: RolloutInstanceId = ("worker_abc", 0)
    signals = [_make_signal(iid, is_awaken=True, kv=0.5)]
    instance_mu, role_total_mu = policy._build_mu_maps(signals)
    assert ("Rollout", "test_model", ("worker_abc", 0)) in instance_mu


def test_pick_scale_down_candidate_uses_tuple_instance_id():
    """_pick_scale_down_candidate and related helpers should work with RolloutInstanceId keys."""
    policy = ScalingPolicy.__new__(ScalingPolicy)
    policy.theta_low = 0.3
    policy.min_awake_per_role = 0

    iid0: RolloutInstanceId = ("worker_abc", 0)
    iid1: RolloutInstanceId = ("worker_abc", 1)
    signals = [
        _make_signal(iid0, is_awaken=True, kv=0.1),  # low-KV cede candidate
        _make_signal(iid1, is_awaken=True, kv=0.8),
    ]
    instance_mu = {
        ("Rollout", "test_model", iid0): 5.0,
        ("Rollout", "test_model", iid1): 10.0,
    }
    result = policy._pick_scale_down_candidate(signals, instance_mu)
    assert result is not None
    assert result.instance_id == iid0
