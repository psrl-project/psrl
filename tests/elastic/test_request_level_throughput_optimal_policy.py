from datetime import datetime
from types import SimpleNamespace

import psrl.workers.elastic.scaling_policy.thoughput_optimal as itl_policy_module
import pytest
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.policy_runtime import uses_integer_solver
from psrl.workers.elastic.scaling_policy.base import InstanceSignal, ScalingAction
from psrl.workers.elastic.scaling_policy.evaluation.cpp_candidate_evaluator import (
    CppBatchEvaluation,
    CppCandidateEvaluatorError,
)
from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import RoleCandidatePlan
from psrl.workers.elastic.scaling_policy.factory import create_scaling_policy
from psrl.workers.elastic.scaling_policy.thoughput_optimal import (
    ThroughputOptimalScalingPolicy,
    _ITLCandidate,
)

pytestmark = pytest.mark.cpu_test


def test_policy_factory_uses_throughput_optimal():
    policy = create_scaling_policy(_config(), {"scaling_policy_variant": "throughput-optimal"})
    try:
        assert isinstance(policy, ThroughputOptimalScalingPolicy)
        assert uses_integer_solver(policy)
    finally:
        policy.close()


def _config(rollout_method="throughput_optimal", rm_method="itl"):
    return SimpleNamespace(
        psrl=SimpleNamespace(
            logging_path="/tmp",
            rollout_coordination=SimpleNamespace(
                routing_strategy=SimpleNamespace(method=rollout_method),
            ),
        ),
        reward=SimpleNamespace(
            active_managers=["default"],
            managers={
                "default": SimpleNamespace(
                    routing_strategy=SimpleNamespace(method=rm_method),
                )
            },
        ),
    )


def _policy(
    *,
    max_workers=8,
    router_waiting_top_t=-1,
    min_awake_per_role=1,
    config=None,
    policy_cls=ThroughputOptimalScalingPolicy,
    candidate_backend="python",
    cpp_binary=None,
):
    return policy_cls(
        config=config or _config(),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": min_awake_per_role,
            "throughput_optimal": {
                "enable_request_level_candidate_evaluation": True,
                "candidate_evaluation_max_workers": max_workers,
                "candidate_evaluation_backend": candidate_backend,
                "candidate_evaluation_cpp_binary": cpp_binary,
                "router_waiting_top_t": router_waiting_top_t,
                "rm_router_enable": True,
                "throughput_objective": "sum",
                "vllm_current_queue_scope": "running",
                "decision_window_s": 30.0,
                "min_gain": 0.0,
                "model_params": {"default": {"A": 0.1, "B": 1.0, "C": 0.0, "D": 0.0}},
            },
        },
    )


def _aggregate_policy():
    return ThroughputOptimalScalingPolicy(
        config=_config(),
        policy_config={
            "enable_policy": True,
            "cooldown_ms": 0,
            "min_awake_per_role": 1,
            "throughput_optimal": {
                "enable_request_level_candidate_evaluation": False,
                "router_waiting_top_t": -1,
                "rm_router_enable": True,
                "throughput_objective": "sum",
                "vllm_current_queue_scope": "running",
                "decision_window_s": 30.0,
                "min_gain": 0.0,
                "model_params": {"default": {"A": 0.1, "B": 1.0, "C": 0.0, "D": 0.0}},
            },
        },
    )


@pytest.mark.parametrize("first_rollout_awake", [True, False])
def test_pending_queue_wakes_rollout_with_idle_engines_and_released_trainer_gpus(first_rollout_awake):
    policy = _policy(policy_cls=ThroughputOptimalScalingPolicy, min_awake_per_role=0)
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=first_rollout_awake),
        _signal(PSRL_Role.Rollout, 1, awake=False),
        _signal(PSRL_Role.RewardModel, 2, awake=True),
    ]
    signals[1].gpu_keys = frozenset({("train_pool", 0)})
    snapshots = {
        PSRL_Role.Rollout: {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, awake=first_rollout_awake, requests=[]),
                _instance(1, awake=False, requests=[]),
            ],
            "pending_requests": [
                {
                    "request_id": str(i),
                    "seq_len": 10,
                    "route_order": i,
                    "eligible_instance_ids": [1] if first_rollout_awake else [0, 1],
                }
                for i in range(4094)
            ],
            "max_concurrent_requests": 32,
        },
        PSRL_Role.RewardModel: {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [_instance(2, awake=True, requests=[])],
            "pending_requests": [],
        },
    }
    try:
        decision = policy.decide(
            signals,
            router_backlog_by_role={PSRL_Role.Rollout: 0, PSRL_Role.RewardModel: 0},
            trainer_waiting_hint={"trainer_busy": False},
            request_level_snapshots_by_role=snapshots,
        )
        assert decision.actions, "Pending demand must produce a capacity action despite zero observed throughput."
        action = decision.actions[0]
        assert action.action_type == "scale_up", "The pending queue needs added capacity."
        assert action.role_name == PSRL_Role.Rollout, "Rollout owns the outstanding demand."
        assert action.preferred_instance_ids == [1 if first_rollout_awake else 0], (
            "Queued demand must wake an eligible sleeping placement."
        )
    finally:
        policy.close()


def test_itl_policy_rebalance_after_scale_up_defaults_true():
    policy = _aggregate_policy()
    assert policy.rebalance_after_scale_up is True


def test_itl_policy_can_disable_rebalance_after_scale_up():
    policy = ThroughputOptimalScalingPolicy(
        config=_config(),
        policy_config={
            "enable_policy": True,
            "throughput_optimal": {
                "enable_request_level_candidate_evaluation": True,
                "rebalance_after_scale_up": False,
                "router_waiting_top_t": -1,
            },
        },
    )
    assert policy.rebalance_after_scale_up is False
    assert policy.enable_request_level_candidate_evaluation is True


def _signal(role, instance_id, *, awake, running=0, tokens=0):
    return InstanceSignal(
        role_name=role,
        model_name="model",
        instance_id=instance_id,
        is_awaken=awake,
        kv_cache_utilization=0.0,
        running_queue_num=running,
        waiting_queue_num=0,
        generation_throughput=0.0,
        total_token_num=tokens,
        snapshot_timestamp=datetime.now().isoformat(),
        gpu_keys=frozenset({("pool", instance_id)}),
    )


def _instance(instance_id, *, awake, requests):
    return {
        "instance_id": instance_id,
        "is_awake": awake,
        "requests": [
            {
                "request_id": request_id,
                "seq_len": seq_len,
                "source_instance_id": instance_id,
                "is_waiting": False,
            }
            for request_id, seq_len in requests
        ],
        "route_request_count": len(requests),
        "running_count": len(requests),
        "waiting_count": 0,
        "token_count": sum(seq_len for _, seq_len in requests),
        "max_model_len": 10000,
        "route_cost_params": [0.0, 0.0, 0.0, 1.0, 0.01],
    }


def _fixture():
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=3, tokens=13),
        _signal(PSRL_Role.Rollout, 1, awake=False),
        _signal(PSRL_Role.RewardModel, 0, awake=True, running=1, tokens=1),
        _signal(PSRL_Role.RewardModel, 1, awake=False),
    ]
    grouped = {
        PSRL_Role.Rollout: signals[:2],
        PSRL_Role.RewardModel: signals[2:],
    }
    snapshots = {
        PSRL_Role.Rollout: {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, awake=True, requests=[("10", 10), ("2", 2), ("1", 1)]),
                _instance(1, awake=False, requests=[]),
            ],
            "pending_requests": [],
            "max_concurrent_requests": 32,
        },
        PSRL_Role.RewardModel: {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(0, awake=True, requests=[("rm", 1)]),
                _instance(1, awake=False, requests=[]),
            ],
            "pending_requests": [],
            "max_concurrent_requests": 32,
        },
    }
    candidate = _ITLCandidate(
        action=ScalingAction(
            action_type="scale_up",
            role_name=PSRL_Role.Rollout,
            model_name="model",
            num_instances=1,
            preferred_instance_ids=[1],
        ),
        rollout_n=2,
        rm_n=1,
        current_throughput=0.0,
        next_throughput=0.0,
        delta_throughput=0.0,
        phi=0.0,
        gain_l=0.0,
    )
    return grouped, snapshots, candidate


def test_request_level_policy_scores_candidate_and_attaches_rebalance_plan():
    policy = _policy(max_workers=2)
    grouped, snapshots, candidate = _fixture()
    try:
        baseline = policy._evaluate_request_level_candidates(
            candidates=[candidate],
            grouped=grouped,
            router_backlog_by_role={},
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert baseline == candidate.current_throughput
    assert candidate.request_level_results_by_role is not None
    assert [migration["request_id"] for migration in candidate.action.planned_request_migrations] == ["1", "2"]


def test_harmonic_decide_records_planner_phase_breakdown():
    policy = _policy(max_workers=2, policy_cls=ThroughputOptimalScalingPolicy)
    grouped, snapshots, _ = _fixture()
    signals = grouped[PSRL_Role.Rollout] + grouped[PSRL_Role.RewardModel]
    try:
        policy.decide(
            signals,
            router_backlog_by_role={},
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert set(policy.last_planner_breakdown) == {
        "state_analysis_s",
        "candidate_ordering_s",
        "candidate_set_construction_s",
        "simulation_input_preparation_s",
        "candidate_evaluation_wall_s",
        "rebalance_simulation_s",
        "router_simulation_s",
        "simulation_wall_s",
        "rebalance_router_overlap_s",
        "candidate_scoring_s",
        "best_candidate_selection_s",
    }
    assert policy.last_planner_breakdown["state_analysis_s"] > 0.0
    assert policy.last_planner_breakdown["candidate_ordering_s"] > 0.0
    assert policy.last_planner_breakdown["candidate_set_construction_s"] > 0.0
    assert policy.last_planner_breakdown["simulation_input_preparation_s"] > 0.0
    assert policy.last_planner_breakdown["candidate_evaluation_wall_s"] > 0.0
    assert policy.last_planner_breakdown["rebalance_simulation_s"] >= 0.0
    assert policy.last_planner_breakdown["router_simulation_s"] >= 0.0
    assert policy.last_planner_breakdown["simulation_wall_s"] >= 0.0
    assert policy.last_planner_breakdown["rebalance_router_overlap_s"] >= 0.0


def test_harmonic_expands_six_reward_instances_when_ten_rollouts_are_idle():
    policy = _policy(max_workers=2, policy_cls=ThroughputOptimalScalingPolicy, min_awake_per_role=0)
    policy.max_scale_instances_per_action = -1
    rollout = PSRL_Role.Rollout
    reward = PSRL_Role.RewardModel
    signals = [_signal(rollout, index, awake=index < 10) for index in range(16)] + [
        _signal(
            reward, index, awake=index >= 10, running=128 if index >= 10 else 0, tokens=128 * 512 if index >= 10 else 0
        )
        for index in range(16)
    ]
    reward_instances = []
    for index in range(16):
        instance = _instance(
            index,
            awake=index >= 10,
            requests=[(f"rm-{index}-{uid}", 512) for uid in range(128)] if index >= 10 else [],
        )
        for request in instance["requests"]:
            request["movable"] = False
        reward_instances.append(instance)
    snapshots = {
        rollout: {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [_instance(index, awake=index < 10, requests=[]) for index in range(16)],
            "pending_requests": [],
            "max_concurrent_requests": 512,
        },
        reward: {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": reward_instances,
            "pending_requests": [
                {"request_id": f"pending-{index}", "seq_len": 512, "route_order": index} for index in range(1251)
            ],
            "max_concurrent_requests": 128,
        },
    }
    try:
        decision = policy.decide(
            signals,
            router_backlog_by_role={reward: {"request_count": 1251, "token_count": 1251 * 512}},
            request_level_snapshots_by_role=snapshots,
        )
        assert len(decision.actions) == 1
        action = decision.actions[0]
        assert action.action_type == "scale_up"
        assert action.role_name == reward
        assert action.num_instances > 0
        assert action.pre_sleep_other_preferred
        assert action.planned_request_migrations == []
    finally:
        policy.close()
    assert policy.last_planner_breakdown["candidate_scoring_s"] > 0.0
    assert policy.last_planner_breakdown["best_candidate_selection_s"] > 0.0


def test_request_level_policy_selects_scale_up_from_zero_awake_rm(monkeypatch):
    policy = _policy(max_workers=2, min_awake_per_role=0)
    signals = [
        _signal(PSRL_Role.Rollout, 0, awake=True, running=1, tokens=1),
        _signal(PSRL_Role.RewardModel, 0, awake=False),
    ]
    snapshots = {
        PSRL_Role.Rollout: {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, awake=True, requests=[("rollout", 1)]),
            ],
            "pending_requests": [],
            "max_concurrent_requests": 32,
        },
        PSRL_Role.RewardModel: {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(0, awake=False, requests=[]),
            ],
            "pending_requests": [
                {"request_id": "rm-pending", "seq_len": 1, "route_order": 0},
            ],
            "max_concurrent_requests": 32,
        },
    }
    candidate = _ITLCandidate(
        action=ScalingAction(
            action_type="scale_up",
            role_name=PSRL_Role.RewardModel,
            model_name="model",
            num_instances=1,
            preferred_instance_ids=[0],
        ),
        rollout_n=1,
        rm_n=1,
        current_throughput=0.0,
        next_throughput=0.0,
        delta_throughput=0.0,
        phi=0.0,
        gain_l=0.0,
    )
    monkeypatch.setattr(policy, "_enumerate_candidates", lambda *_args, **_kwargs: [candidate])

    try:
        decision = policy.decide(
            signals,
            router_backlog_by_role={PSRL_Role.RewardModel: {"request_count": 1, "token_count": 1}},
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert decision.reason == "itl_best_scale_up_RewardModel"
    assert decision.actions == [candidate.action]
    assert candidate.current_throughput == 0.0
    assert candidate.next_throughput > 0.0
    assert candidate.request_level_results_by_role is not None
    assert candidate.request_level_results_by_role[PSRL_Role.RewardModel].routed_count == 1


def test_request_level_switch_preserves_candidate_construction_and_phi():
    enabled = _policy()
    disabled = _aggregate_policy()
    grouped, _, _ = _fixture()
    try:
        enabled_candidates = enabled._enumerate_candidates(grouped, {})
        disabled_candidates = disabled._enumerate_candidates(grouped, {})
    finally:
        enabled._candidate_evaluation_executor.shutdown(wait=True)

    def planning_fields(candidate):
        action = candidate.action
        return (
            action.action_type,
            action.role_name,
            action.model_name,
            action.num_instances,
            action.preferred_instance_ids,
            action.pre_sleep_other_preferred,
            action.pre_wake_other_preferred,
            candidate.rollout_n,
            candidate.rm_n,
            candidate.phi,
        )

    assert [planning_fields(item) for item in enabled_candidates] == [
        planning_fields(item) for item in disabled_candidates
    ]
    assert all(item.action.planned_request_migrations is None for item in disabled_candidates)


def test_request_level_candidate_results_are_independent_of_worker_count():
    outputs = []
    for max_workers in (1, 8):
        policy = _policy(max_workers=max_workers)
        grouped, snapshots, candidate = _fixture()
        try:
            baseline = policy._evaluate_request_level_candidates(
                candidates=[candidate],
                grouped=grouped,
                router_backlog_by_role={},
                request_level_snapshots_by_role=snapshots,
            )
        finally:
            policy._candidate_evaluation_executor.shutdown(wait=True)
        outputs.append(
            (
                baseline,
                candidate.next_throughput,
                candidate.gain_l,
                candidate.action.planned_request_migrations,
            )
        )

    assert outputs[0] == outputs[1]


def test_request_level_policy_rejects_missing_snapshot_for_whole_cycle():
    policy = _policy()
    grouped, _, _ = _fixture()
    signals = grouped[PSRL_Role.Rollout] + grouped[PSRL_Role.RewardModel]
    try:
        decision = policy.decide(
            signals,
            router_backlog_by_role={},
            request_level_snapshots_by_role={},
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert decision.actions == []
    assert decision.reason == "request_level_candidate_evaluation_failed"


def test_request_level_policy_validates_top_t_and_exact_strategies():
    with pytest.raises(ValueError, match="router_waiting_top_t"):
        _policy(router_waiting_top_t=0)
    with pytest.raises(ValueError, match="throughput_optimal"):
        _policy(config=_config(rollout_method="round_robin"))
    with pytest.raises(ValueError, match="reward-model routing strategy 'itl'"):
        _policy(config=_config(rm_method="round_robin"))
    with pytest.raises(ValueError, match="candidate_evaluation_backend"):
        _policy(candidate_backend="invalid")


def test_request_level_policy_can_use_cpp_backend(monkeypatch):
    closed = []

    class FakeCppCandidateEvaluator:
        def __init__(self, *, binary_path, max_workers, timeout_s):
            assert binary_path == "/fake/elastic_simulator"
            assert max_workers == 2
            assert timeout_s == 30.0

        def evaluate(self, *, snapshots, candidate_plans):
            contexts = {
                role: itl_policy_module.prepare_role_evaluation_context(snapshot)
                for role, snapshot in snapshots.items()
            }
            baseline = {
                role: itl_policy_module.evaluate_role_candidate(
                    context,
                    RoleCandidatePlan(),
                )
                for role, context in contexts.items()
            }
            candidate_results = tuple(
                {
                    role: itl_policy_module.evaluate_role_candidate(
                        contexts[role],
                        plans[role],
                    )
                    for role in contexts
                }
                for plans in candidate_plans
            )
            return CppBatchEvaluation(
                baseline_results=baseline,
                candidate_results=candidate_results,
                logical_task_count=2 * (len(candidate_plans) + 1),
                unique_task_count=3,
                deduplicated_task_count=1,
                bridge_wall_s=0.004,
                evaluation_wall_s=0.003,
                rebalance_simulation_s=0.001,
                router_simulation_s=0.002,
                simulation_wall_s=0.0025,
                rebalance_router_overlap_s=0.0005,
            )

        def close(self):
            closed.append(True)

    monkeypatch.setattr(
        itl_policy_module,
        "CppCandidateEvaluator",
        FakeCppCandidateEvaluator,
    )
    policy = _policy(
        max_workers=2,
        candidate_backend="cpp",
        cpp_binary="/fake/elastic_simulator",
    )
    grouped, snapshots, candidate = _fixture()
    try:
        baseline = policy._evaluate_request_level_candidates(
            candidates=[candidate],
            grouped=grouped,
            router_backlog_by_role={},
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy.close()

    assert policy._candidate_evaluation_executor is None
    assert baseline == candidate.current_throughput
    assert candidate.request_level_results_by_role is not None
    assert [migration["request_id"] for migration in candidate.action.planned_request_migrations] == ["1", "2"]
    assert policy._last_request_level_simulation_timing["rebalance_simulation_s"] == 0.001
    assert policy._last_request_level_simulation_timing["router_simulation_s"] == 0.002
    assert policy._last_request_level_simulation_timing["candidate_evaluation_wall_s"] == 0.003
    assert policy._last_request_level_simulation_timing["simulation_wall_s"] == 0.0025
    assert policy._last_request_level_simulation_timing["rebalance_router_overlap_s"] == 0.0005
    assert closed == [True]


def test_request_level_policy_propagates_cpp_backend_failure(monkeypatch):
    class FailedCppCandidateEvaluator:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self, **_kwargs):
            raise CppCandidateEvaluatorError("C++ candidate evaluator exited with code 1")

        def close(self):
            pass

    monkeypatch.setattr(
        itl_policy_module,
        "CppCandidateEvaluator",
        FailedCppCandidateEvaluator,
    )
    policy = _policy(candidate_backend="cpp", cpp_binary="/fake/elastic_simulator")
    grouped, snapshots, _ = _fixture()
    signals = grouped[PSRL_Role.Rollout] + grouped[PSRL_Role.RewardModel]
    try:
        with pytest.raises(CppCandidateEvaluatorError, match="exited with code 1"):
            policy.decide(
                signals,
                router_backlog_by_role={},
                request_level_snapshots_by_role=snapshots,
            )
    finally:
        policy.close()


def test_request_level_policy_prepares_once_and_deduplicates_unchanged_role(
    monkeypatch,
):
    prepared_roles = []
    evaluated_plans = []
    original_prepare = itl_policy_module.prepare_role_evaluation_context
    original_evaluate = itl_policy_module.evaluate_role_candidate

    def recording_prepare(snapshot):
        context = original_prepare(snapshot)
        prepared_roles.append(context.snapshot.role)
        return context

    def recording_evaluate(context, plan):
        evaluated_plans.append((context.snapshot.role, plan))
        return original_evaluate(context, plan)

    monkeypatch.setattr(
        itl_policy_module,
        "prepare_role_evaluation_context",
        recording_prepare,
    )
    monkeypatch.setattr(
        itl_policy_module,
        "evaluate_role_candidate",
        recording_evaluate,
    )
    policy = _policy(max_workers=2)
    grouped, snapshots, candidate = _fixture()
    try:
        policy._evaluate_request_level_candidates(
            candidates=[candidate],
            grouped=grouped,
            router_backlog_by_role={},
            request_level_snapshots_by_role=snapshots,
        )
    finally:
        policy._candidate_evaluation_executor.shutdown(wait=True)

    assert sorted(prepared_roles) == ["RewardModel", "Rollout"]
    assert len(evaluated_plans) == 3
    assert sum(role == "RewardModel" and plan == RoleCandidatePlan() for role, plan in evaluated_plans) == 1
    timing = policy._last_request_level_simulation_timing
    assert timing["candidate_evaluation_wall_s"] > 0.0
    assert timing["rebalance_simulation_s"] >= 0.0
    assert timing["router_simulation_s"] >= 0.0
    assert timing["simulation_wall_s"] == pytest.approx(
        timing["rebalance_simulation_s"] + timing["router_simulation_s"] - timing["rebalance_router_overlap_s"]
    )
    assert timing["simulation_wall_s"] <= timing["candidate_evaluation_wall_s"]
    assert sum(timing.values()) > 0.0


@pytest.mark.parametrize(
    ("variant", "expected_type", "integer_solver"),
    [("none", "DummyScalingPolicy", True), ("rule-based", "RuleBasedScalingPolicy", False)],
)
def test_policy_factory_constructs_other_variants(variant, expected_type, integer_solver):
    policy = create_scaling_policy(_config(), {"scaling_policy_variant": variant})
    try:
        assert type(policy).__name__ == expected_type
        assert uses_integer_solver(policy) is integer_solver
    finally:
        policy.close()


def test_policy_factory_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown elastic scaling policy"):
        create_scaling_policy(_config(), {"scaling_policy_variant": "unknown"})
