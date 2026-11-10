import math
from datetime import datetime
from types import SimpleNamespace

import pytest
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.scaling_policy.base import InstanceSignal
from psrl.workers.elastic.scaling_policy.thoughput_optimal import ThroughputOptimalScalingPolicy

pytestmark = pytest.mark.cpu_test


def _config():
    return SimpleNamespace(psrl=SimpleNamespace(logging_path="/tmp"))


def _policy(
    model_params: dict | None = None,
    *,
    max_scale_instances_per_action: int = 1,
    throughput_objective: str = "balanced_min",
    vllm_current_queue_scope: str = "running_waiting",
    current_state_include_router_waiting: bool = True,
    role_throughput_weight_enable: bool = False,
    role_throughput_weight_basis: str = "request_count",
    role_throughput_weight_mode: str = "share",
    harmonic_denominator_use_max: bool | str = False,
    enable_heterogeneous_parallelism_candidates: bool = False,
    policy_cls: type[ThroughputOptimalScalingPolicy] = ThroughputOptimalScalingPolicy,
) -> ThroughputOptimalScalingPolicy:
    return policy_cls(
        config=_config(),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": 1,
            "throughput_optimal": {
                "decision_window_s": 30.0,
                "migrate_time_s_initial": 30.0,
                "min_gain": 0.0,
                "max_scale_instances_per_action": max_scale_instances_per_action,
                "throughput_objective": throughput_objective,
                "vllm_current_queue_scope": vllm_current_queue_scope,
                "current_state_include_router_waiting": current_state_include_router_waiting,
                "role_throughput_weight_enable": role_throughput_weight_enable,
                "role_throughput_weight_basis": role_throughput_weight_basis,
                "role_throughput_weight_mode": role_throughput_weight_mode,
                "harmonic_denominator_use_max": harmonic_denominator_use_max,
                "enable_heterogeneous_parallelism_candidates": enable_heterogeneous_parallelism_candidates,
                "model_params": model_params or {"default": {"A": 0.0, "B": 10.0, "C": 1.0, "D": 0.0}},
            },
        },
    )


def _signal(
    role: PSRL_Role,
    instance_id: int,
    *,
    awake: bool,
    running: int,
    tokens: int,
    gpus: int | tuple[int, ...],
    waiting: int = 0,
) -> InstanceSignal:
    return InstanceSignal(
        role_name=role,
        model_name="model",
        instance_id=instance_id,
        is_awaken=awake,
        kv_cache_utilization=0.5,
        running_queue_num=running,
        waiting_queue_num=waiting,
        generation_throughput=0.0,
        total_token_num=tokens,
        snapshot_timestamp=datetime.now().isoformat(),
        gpu_keys=frozenset(("node-0", gpu_id) for gpu_id in ((gpus,) if isinstance(gpus, int) else gpus)),
    )


def test_current_state_router_waiting_switch_controls_current_throughput():
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=30, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=True, running=3, tokens=30, gpus=1),
    ]
    grouped = {PSRL_Role.Rollout: rollout}
    router_backlog = {PSRL_Role.Rollout: {"count": 14, "total_tokens": 0}}

    enabled = _policy(throughput_objective="sum", current_state_include_router_waiting=True)
    enabled_tps = enabled._current_role_throughputs(
        grouped,
        rollout_n=2,
        rm_n=0,
        router_backlog_by_role=router_backlog,
    )
    enabled_waiting = enabled._current_router_waiting_load(router_backlog[PSRL_Role.Rollout])
    enabled_snapshot = enabled._current_role_load_snapshot(rollout, 2, enabled_waiting)

    disabled = _policy(throughput_objective="sum", current_state_include_router_waiting=False)
    disabled_tps = disabled._current_role_throughputs(
        grouped,
        rollout_n=2,
        rm_n=0,
        router_backlog_by_role=router_backlog,
    )
    disabled_waiting = disabled._current_router_waiting_load(router_backlog[PSRL_Role.Rollout])
    disabled_snapshot = disabled._current_role_load_snapshot(rollout, 2, disabled_waiting)

    assert enabled_tps[PSRL_Role.Rollout] == pytest.approx(2.0)
    assert [row[1] for row in enabled_snapshot.instance_rows] == pytest.approx([10.0, 10.0])
    assert disabled_tps[PSRL_Role.Rollout] == pytest.approx(0.6)
    assert [row[1] for row in disabled_snapshot.instance_rows] == pytest.approx([3.0, 3.0])


def test_itl_role_throughput_uses_balanced_average_state():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
    ]

    assert policy._role_throughput(signals, 2) == 0.6


def test_itl_role_throughput_can_ignore_vllm_waiting_load():
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=4, waiting=8, tokens=0, gpus=0),
    ]

    legacy_policy = _policy()
    running_only_policy = _policy(vllm_current_queue_scope="running")

    assert legacy_policy._role_throughput(signals, 1) == 1.0
    assert running_only_policy._role_throughput(signals, 1) == 0.4


def test_itl_role_throughput_includes_router_waiting_prefix_load():
    policy = _policy({"default": {"A": 0.1, "B": 10.0, "C": 1.0, "D": 0.0}})
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
    ]
    waiting = policy._normalize_router_waiting_load({"count": 4, "total_tokens": 40})

    assert policy._role_throughput(signals, 2, waiting) == 8.0 / 18.0


def test_itl_role_throughput_waiting_role_without_instances_is_bottleneck():
    policy = _policy()
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=False, running=0, tokens=0, gpus=0),
    ]
    waiting = policy._normalize_router_waiting_load({"count": 1, "total_tokens": 10})

    assert policy._role_throughput(signals, 0, waiting) == 0.0


def test_role_throughput_weights_share_pressure_across_both_roles():
    policy = _policy(role_throughput_weight_enable=True, role_throughput_weight_basis="request_count")
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, gpus=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx((12 + 8) / ((12 + 8) + (3 + 2)))
    assert w_rm == pytest.approx((3 + 2) / ((12 + 8) + (3 + 2)))
    assert w_rollout + w_rm == pytest.approx(1.0)


def test_role_throughput_weights_use_token_basis_when_configured():
    policy = _policy(role_throughput_weight_enable=True, role_throughput_weight_basis="token_count")
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, gpus=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx((120 + 80) / ((120 + 80) + (30 + 20)))
    assert w_rm == pytest.approx((30 + 20) / ((120 + 80) + (30 + 20)))


def test_role_throughput_weights_uniform_when_disabled():
    policy = _policy(role_throughput_weight_enable=False)
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, gpus=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == 0.5
    assert w_rm == 0.5


def test_role_throughput_weights_uniform_when_both_sides_idle():
    policy = _policy(role_throughput_weight_enable=True, role_throughput_weight_basis="request_count")
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=0, tokens=0, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=0, tokens=0, gpus=2)]

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, None, None)

    assert w_rollout == 0.5
    assert w_rm == 0.5


def test_role_throughput_weights_raw_mode_returns_raw_pressure_values():
    # raw mode: divisors are the raw per-role pressure magnitudes, not shares.
    policy = _policy(
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, gpus=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx(20.0)  # 12 + 8
    assert w_rm == pytest.approx(5.0)  # 3 + 2


def test_role_throughput_weights_raw_mode_zero_pressure_falls_back_to_one():
    # A side with zero pressure uses divisor 1.0 so it is not bottlenecked by /0.
    policy = _policy(
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=0, tokens=0, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=7, tokens=70, gpus=2)]

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, None, None)

    assert w_rollout == 1.0
    assert w_rm == pytest.approx(7.0)


def test_role_throughput_weights_raw_mode_token_basis_uses_token_totals():
    policy = _policy(
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="token_count",
        role_throughput_weight_mode="raw",
    )
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=3, tokens=30, gpus=2)]
    rollout_waiting = policy._normalize_router_waiting_load({"count": 8, "total_tokens": 80})
    rm_waiting = policy._normalize_router_waiting_load({"count": 2, "total_tokens": 20})

    w_rollout, w_rm = policy._role_throughput_weights(rollout, rm, rollout_waiting, rm_waiting)

    assert w_rollout == pytest.approx(200.0)  # 120 + 80
    assert w_rm == pytest.approx(50.0)  # 30 + 20


def test_itl_sum_objective_windowed_gain_does_not_multiply_instance_count():
    sum_policy = _policy(throughput_objective="sum")
    balanced_policy = _policy()

    assert sum_policy._windowed_throughput_gain(2.0, 2, 3) == 60.0
    assert balanced_policy._windowed_throughput_gain(2.0, 2, 3) == 300.0


def test_harmonic_system_throughput_unweighted_is_standard_harmonic_mean():
    # ITL = max(10, running). rollout running=12 -> tp=1.0, rm running=5 -> tp=0.5.
    # Unweighted harmonic mean = 2*1.0*0.5 / (1.0 + 0.5) = 0.6667.
    policy = _policy(throughput_objective="sum", policy_cls=ThroughputOptimalScalingPolicy)
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=2)]

    tp = policy._system_throughput({PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}, 1, 1)

    assert tp == pytest.approx(2.0 * 1.0 * 0.5 / (1.0 + 0.5))


def test_harmonic_max_denominator_switch_preserves_default_and_uses_larger_term():
    default_policy = _policy(policy_cls=ThroughputOptimalScalingPolicy)
    max_policy = _policy(
        policy_cls=ThroughputOptimalScalingPolicy,
        harmonic_denominator_use_max="true",
    )

    assert default_policy._weighted_system_throughput(1.0, 0.5, 0.5, 0.5) == pytest.approx(2.0 / 3.0)
    assert max_policy._weighted_system_throughput(1.0, 0.5, 0.5, 0.5) == pytest.approx(1.0)
    assert max_policy.harmonic_denominator_use_max is True


def test_harmonic_idle_instance_contributes_zero_not_infinite():
    # Idle capacity must not inflate the throughput of a loaded role.
    harmonic = _policy(throughput_objective="sum", policy_cls=ThroughputOptimalScalingPolicy)
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm_busy = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=2)]
    rm_idle = [
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=2),
        _signal(PSRL_Role.RewardModel, 1, awake=True, running=0, tokens=0, gpus=3),
    ]

    assert harmonic._role_throughput_sum(rm_idle, 2) == pytest.approx(0.5)

    # System throughput stays finite for harmonic. it does not blow up to inf.
    tp_one = harmonic._system_throughput({PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm_busy}, 1, 1)
    tp_two = harmonic._system_throughput({PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm_idle}, 1, 2)
    assert not math.isinf(tp_two)
    # Adding an idle RM instance (no backlog to redistribute) does not raise the
    # objective above the single-busy-instance baseline.
    assert tp_two == pytest.approx(tp_one)


def test_harmonic_system_throughput_weighted_request_basis():
    # rollout tp=1.0 (pressure 12), rm tp=0.5 (pressure 5).
    # w_rollout=12/17, w_rm=5/17. H = (w_r+w_m) / (w_r/1.0 + w_m/0.5).
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="request_count",
        policy_cls=ThroughputOptimalScalingPolicy,
    )
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=2)]

    tp = policy._system_throughput({PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}, 1, 1)

    w_r, w_m = 12.0 / 17.0, 5.0 / 17.0
    assert tp == pytest.approx((w_r + w_m) / (w_r / 1.0 + w_m / 0.5))


def test_harmonic_system_throughput_weighted_token_basis():
    # rollout tp=1.0 (tokens 120), rm tp=0.5 (tokens 50).
    # w_rollout=120/170, w_rm=50/170.
    policy = _policy(
        throughput_objective="sum",
        role_throughput_weight_enable=True,
        role_throughput_weight_basis="token_count",
        policy_cls=ThroughputOptimalScalingPolicy,
    )
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=12, tokens=120, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=2)]

    tp = policy._system_throughput({PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}, 1, 1)

    w_r, w_m = 120.0 / 170.0, 50.0 / 170.0
    assert tp == pytest.approx((w_r + w_m) / (w_r / 1.0 + w_m / 0.5))


def test_harmonic_system_throughput_zero_side_collapses_to_zero():
    policy = _policy(policy_cls=ThroughputOptimalScalingPolicy)

    assert policy._weighted_system_throughput(1.0, 0.0, 0.5, 0.5) == 0.0
    assert policy._weighted_system_throughput(0.0, 1.0, 0.5, 0.5) == 0.0


def test_harmonic_system_throughput_both_idle_collapses_to_zero():
    policy = _policy(policy_cls=ThroughputOptimalScalingPolicy)
    rollout = [_signal(PSRL_Role.Rollout, 0, awake=True, running=0, tokens=0, gpus=0)]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=0, tokens=0, gpus=2)]

    tp = policy._system_throughput({PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}, 1, 1)

    assert tp == 0.0


def test_itl_transfer_candidate_sets_pre_sleep_before_wake():
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=5, tokens=50, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
    ]
    rm = [
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=1),
        _signal(PSRL_Role.RewardModel, 1, awake=True, running=5, tokens=50, gpus=2),
    ]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    rollout_scale_ups = [
        c for c in candidates if c.action.action_type == "scale_up" and c.action.role_name == PSRL_Role.Rollout
    ]

    assert rollout_scale_ups, "expected at least one rollout scale-up candidate"
    candidate = rollout_scale_ups[0]
    assert candidate.action.preferred_instance_ids == [1]
    assert candidate.action.pre_sleep_other_preferred == [
        {"role_name": PSRL_Role.RewardModel, "model_name": "model", "instance_id": 0}
    ]
    assert candidate.rollout_n == 2
    assert candidate.rm_n == 1


def test_itl_sum_objective_prunes_conflict_scale_up_when_free_bundle_exists():
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=5, tokens=50, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
        _signal(PSRL_Role.Rollout, 2, awake=False, running=0, tokens=0, gpus=2),
    ]
    rm = [
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=12, tokens=120, gpus=1),
        _signal(PSRL_Role.RewardModel, 1, awake=True, running=12, tokens=120, gpus=3),
    ]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    rollout_scale_ups = [
        candidate
        for candidate in policy._enumerate_candidates(grouped, router_backlog_by_role={})
        if candidate.action.action_type == "scale_up" and candidate.action.role_name == PSRL_Role.Rollout
    ]

    assert len(rollout_scale_ups) == 1
    assert rollout_scale_ups[0].action.preferred_instance_ids == [2]
    assert rollout_scale_ups[0].action.pre_sleep_other_preferred is None


def test_itl_balanced_objective_rejects_eviction_below_min_awake():
    policy = _policy(throughput_objective="balanced_min")
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=30, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
    ]
    rm = [
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=1),
        _signal(PSRL_Role.RewardModel, 1, awake=False, running=0, tokens=0, gpus=2),
    ]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    rollout_scale_ups = [
        candidate
        for candidate in policy._enumerate_candidates(grouped, router_backlog_by_role={})
        if candidate.action.action_type == "scale_up" and candidate.action.role_name == PSRL_Role.Rollout
    ]

    assert rollout_scale_ups == []


def test_itl_enumerates_batch_scale_up_candidates_when_enabled():
    policy = _policy(max_scale_instances_per_action=2)
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=30, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
        _signal(PSRL_Role.Rollout, 2, awake=False, running=0, tokens=0, gpus=2),
    ]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=3)]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    batch = [
        c
        for c in candidates
        if c.action.action_type == "scale_up"
        and c.action.role_name == PSRL_Role.Rollout
        and c.action.num_instances == 2
    ]

    assert len(batch) == 1
    assert batch[0].action.preferred_instance_ids == [1, 2]
    assert batch[0].rollout_n == 3


def test_itl_unlimited_max_scale_instances_enumerates_all_wakeable():
    policy = _policy(max_scale_instances_per_action=-1)
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=30, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
        _signal(PSRL_Role.Rollout, 2, awake=False, running=0, tokens=0, gpus=2),
        _signal(PSRL_Role.Rollout, 3, awake=False, running=0, tokens=0, gpus=3),
    ]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=4)]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    assert policy.max_scale_instances_per_action == -1
    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    batch = [
        c
        for c in candidates
        if c.action.action_type == "scale_up"
        and c.action.role_name == PSRL_Role.Rollout
        and c.action.num_instances == 3
    ]

    assert len(batch) == 1
    assert batch[0].action.preferred_instance_ids == [1, 2, 3]
    assert batch[0].rollout_n == 4
    assert {c.action.num_instances for c in candidates if c.action.role_name == PSRL_Role.Rollout} == {1, 2, 3}


def test_itl_keeps_single_step_candidates_by_default():
    policy = _policy()
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=30, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
        _signal(PSRL_Role.Rollout, 2, awake=False, running=0, tokens=0, gpus=2),
    ]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=5, tokens=50, gpus=3)]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})

    assert candidates, "expected at least one candidate"
    assert all(c.action.num_instances == 1 for c in candidates)


def test_heterogeneous_switch_preserves_equal_parallel_log_fixture():
    # Fixture shape and preferred ids come from ScalingPolicy/ElasticExecutor
    # cycle 7816 (decision 354): eight TP1 rollout wakes competing with TP1 RM.
    wake_ids = (6, 12, 20, 22, 16, 1, 2, 11)
    running_loads = (39, 38, 39, 38, 39, 38, 39, 38)
    token_loads = (167963, 163713, 158012, 159014, 145643, 166654, 159014, 163713)
    rollout = [
        _signal(PSRL_Role.Rollout, instance_id, awake=False, running=0, tokens=0, gpus=instance_id)
        for instance_id in wake_ids
    ]
    rm = [
        _signal(
            PSRL_Role.RewardModel,
            instance_id,
            awake=True,
            running=running,
            tokens=tokens,
            gpus=instance_id,
        )
        for instance_id, running, tokens in zip(wake_ids, running_loads, token_loads)
    ]
    rm.append(_signal(PSRL_Role.RewardModel, 99, awake=True, running=38, tokens=159014, gpus=99))

    legacy = _policy(max_scale_instances_per_action=8, throughput_objective="sum")
    enabled = _policy(
        max_scale_instances_per_action=8,
        throughput_objective="sum",
        enable_heterogeneous_parallelism_candidates=True,
    )
    legacy_plans = legacy._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=9,
        max_batch=8,
    )
    enabled_plans = enabled._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=9,
        max_batch=8,
    )

    def summarize(plans):
        return {
            count: (
                tuple(signal.instance_id for signal in plan.wakes),
                tuple(signal.instance_id for signal in plan.pre_sleep),
                tuple(signal.instance_id for signal in plan.pre_wake),
            )
            for count, plan in plans.items()
        }

    assert summarize(enabled_plans) == summarize(legacy_plans)


def test_heterogeneous_large_wake_migrates_small_conflicts_and_preserves_load():
    # RM loads are sampled from ScalingPolicy cycle 7815. A TP4 rollout wake
    # displaces two TP1 RM instances, which can move to free TP1 placements.
    policy = _policy(
        model_params={"default": {"A": 1e-6, "B": 10.0, "C": 1.0, "D": 0.0}},
        throughput_objective="sum",
        enable_heterogeneous_parallelism_candidates=True,
    )
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=30, gpus=(12, 13, 14, 15)),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=(0, 1, 2, 3)),
    ]
    rm = [
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=39, tokens=167963, gpus=0),
        _signal(PSRL_Role.RewardModel, 1, awake=True, running=38, tokens=163713, gpus=1),
        _signal(PSRL_Role.RewardModel, 2, awake=False, running=0, tokens=0, gpus=8),
        _signal(PSRL_Role.RewardModel, 3, awake=False, running=0, tokens=0, gpus=9),
    ]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    candidate = next(
        item
        for item in candidates
        if item.action.action_type == "scale_up" and item.action.role_name == PSRL_Role.Rollout
    )

    assert candidate.action.preferred_instance_ids == [1]
    assert candidate.action.pre_sleep_other_preferred == [
        {"role_name": PSRL_Role.RewardModel, "model_name": "model", "instance_id": 0},
        {"role_name": PSRL_Role.RewardModel, "model_name": "model", "instance_id": 1},
    ]
    assert candidate.action.pre_wake_other_preferred == [
        {"role_name": PSRL_Role.RewardModel, "model_name": "model", "instance_id": 2},
        {"role_name": PSRL_Role.RewardModel, "model_name": "model", "instance_id": 3},
    ]
    assert candidate.rollout_n == 2
    assert candidate.rm_n == 2
    assert candidate.load_overrides_by_role == {PSRL_Role.RewardModel: {2: (39.0, 167963.0), 3: (38.0, 163713.0)}}
    assert candidate.next_throughput > 0.0

    disabled = _policy(throughput_objective="sum")
    disabled_plans = disabled._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=2,
        max_batch=1,
    )
    assert disabled_plans == {}


def test_heterogeneous_wake_cost_treats_conflict_free_as_zero_phi():
    policy = _policy(enable_heterogeneous_parallelism_candidates=True)
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=False, running=0, tokens=0, gpus=(0, 1, 2, 3)),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=(4, 5, 6, 7)),
    ]
    rm = [
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=39, tokens=167963, gpus=0),
        _signal(PSRL_Role.RewardModel, 1, awake=True, running=38, tokens=163713, gpus=8),
    ]

    plans = policy._wake_prefix_plans_for_role(
        target_signals=rollout,
        other_signals=rm,
        current_other_n=2,
        max_batch=1,
    )

    assert [signal.instance_id for signal in plans[1].wakes] == [1]
    assert plans[1].pre_sleep == ()
    assert plans[1].pre_wake == ()


def test_heterogeneous_small_wakes_charge_large_victim_phi_once():
    policy = _policy(
        model_params={"default": {"A": 0.001, "B": 10.0, "C": 1.0, "D": 0.0}},
        max_scale_instances_per_action=4,
        throughput_objective="sum",
        enable_heterogeneous_parallelism_candidates=True,
    )
    rollout = [
        _signal(PSRL_Role.Rollout, 8, awake=True, running=3, tokens=30, gpus=8),
        *[
            _signal(PSRL_Role.Rollout, instance_id, awake=False, running=0, tokens=0, gpus=instance_id)
            for instance_id in range(8)
        ],
    ]
    rm = [
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=39, tokens=167963, gpus=(0, 1, 2, 3)),
        _signal(PSRL_Role.RewardModel, 1, awake=True, running=38, tokens=159014, gpus=(4, 5, 6, 7)),
    ]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(grouped, router_backlog_by_role={})
    candidate = next(
        item
        for item in candidates
        if item.action.action_type == "scale_up"
        and item.action.role_name == PSRL_Role.Rollout
        and item.action.num_instances == 4
    )

    assert candidate.action.preferred_instance_ids == [0, 1, 2, 3]
    assert candidate.action.pre_sleep_other_preferred == [
        {"role_name": PSRL_Role.RewardModel, "model_name": "model", "instance_id": 0}
    ]
    assert candidate.action.pre_wake_other_preferred is None
    assert candidate.rm_n == 1
    assert candidate.phi == pytest.approx(policy._sleep_phi([rm[0]]))
    assert candidate.phi != pytest.approx(4.0 * policy._sleep_phi([rm[0]]))


def test_itl_scale_up_uses_first_free_physical_gpu():
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=5, tokens=50, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
        _signal(PSRL_Role.Rollout, 2, awake=False, running=0, tokens=0, gpus=2),
    ]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=12, tokens=120, gpus=3)]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(
        grouped, router_backlog_by_role={}, trainer_waiting_hint={"trainer_busy": False}
    )
    rollout_scale_ups = [
        c for c in candidates if c.action.action_type == "scale_up" and c.action.role_name == PSRL_Role.Rollout
    ]

    assert rollout_scale_ups
    assert rollout_scale_ups[0].action.preferred_instance_ids == [1]


def test_itl_scale_up_excludes_occupied_physical_gpu():
    policy = _policy(throughput_objective="sum")
    rollout = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=5, tokens=50, gpus=0),
        _signal(PSRL_Role.Rollout, 1, awake=False, running=0, tokens=0, gpus=1),
        _signal(PSRL_Role.Rollout, 2, awake=False, running=0, tokens=0, gpus=2),
    ]
    rm = [_signal(PSRL_Role.RewardModel, 0, awake=True, running=12, tokens=120, gpus=1)]
    grouped = {PSRL_Role.Rollout: rollout, PSRL_Role.RewardModel: rm}

    candidates = policy._enumerate_candidates(
        grouped, router_backlog_by_role={}, trainer_waiting_hint={"trainer_busy": False}
    )
    rollout_scale_ups = [
        c for c in candidates if c.action.action_type == "scale_up" and c.action.role_name == PSRL_Role.Rollout
    ]

    assert rollout_scale_ups
    assert rollout_scale_ups[0].action.preferred_instance_ids == [2]
