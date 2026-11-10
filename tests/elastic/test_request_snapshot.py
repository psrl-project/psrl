from dataclasses import replace
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.policy_runtime import PolicyIdentityMap
from psrl.workers.elastic.request_snapshot import build_role_snapshot
from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    RoleCandidatePlan,
    evaluate_role_candidate,
)

pytestmark = pytest.mark.cpu_test


def _snapshot(role, pending, *, incomplete=False, sleeping_stats=None, cost_model_path=None):
    identities = PolicyIdentityMap()
    native_ids = [("worker-7", 0), ("worker-21", 0)]
    identities.register(role, "Qwen2.5-1.5B", native_ids)
    signals = [
        SimpleNamespace(instance_id=i, is_awaken=i == 0, running_queue_num=int(i == 0), waiting_queue_num=0)
        for i in range(2)
    ]
    stats = {
        native_ids[0]: {
            "elastic_model_version": 2,
            "scheduler_stats": {
                "num_running_reqs": 1,
                "num_waiting_reqs": 0,
                "req_id_to_prompt_token_num": {} if incomplete else {"cmpl-opaque": 25},
                "req_id_to_response_token_num": {"cmpl-opaque": 5},
            },
        },
        native_ids[1]: {"elastic_model_version": 3},
    }
    if sleeping_stats is not None:
        stats[native_ids[1]] = sleeping_stats
    cfg = OmegaConf.create(
        {
            "method": "throughput_optimal" if role == PSRL_Role.Rollout else "itl",
            "cost_model_path": cost_model_path or "psrl/trainer/config/cost_model/qwen2.5_1.5b.json",
            "candidate_sort_indicator": "reserve_capability",
            "enable_group_sticky": True,
            "max_num_waiting_reqs_after_preemption": 3,
            "max_concurrent_seqs_per_instance": 128,
        }
    )
    return build_role_snapshot(role, "Qwen2.5-1.5B", signals, stats, pending, identities, cfg, 4096, 1, 1, 3)


def test_snapshot_preserves_lengths_queue_order_versions_and_sticky_target():
    result = _snapshot(
        PSRL_Role.Rollout,
        [
            {
                "request_id": "70",
                "seq_len": 50,
                "route_order": 4,
                "version_tag": -1,
                "reservable_versions": [3],
                "reserve_indicators": {2: 0, 3: -1},
                "pinned_instance": ["worker-21", 0],
            }
        ],
    )
    active = result.instances[0].requests[0]
    assert active.seq_len == 30
    assert active.request_id == "cmpl-opaque"
    assert active.eligible_instance_ids == (0, 1)
    assert result.instances[1].model_version == 3
    assert result.instances[0].route_cost_params == (0.0017679, 0.0010714, 2.66e-5, 0.0004751468, 1.88e-8)
    pending = result.pending_requests[0]
    assert pending.route_order == 4
    assert pending.eligible_instance_ids == (1,)
    assert pending.fallback_instance_ids == (1,)
    assert pending.candidate_priorities == ((0, (0, -2)), (1, (-1, -3)))


def test_snapshot_resolves_cost_model_directory():
    result = _snapshot(PSRL_Role.Rollout, [], cost_model_path="psrl/trainer/config/cost_model")
    assert result.instances[0].route_cost_params == (0.0017679, 0.0010714, 2.66e-5, 0.0004751468, 1.88e-8)


def test_nonresumable_reward_requests_remain_on_their_worker():
    result = _snapshot(PSRL_Role.RewardModel, [])
    assert result.instances[0].requests[0].movable is False


def test_reward_snapshot_uses_smg_capacity_with_preemption_queue_config_present():
    snapshot = _snapshot(
        PSRL_Role.RewardModel,
        [
            {"request_id": f"pending-{index}", "seq_len": 32, "route_order": index, "version_tag": 0}
            for index in range(128)
        ],
    )
    result = evaluate_role_candidate(snapshot, RoleCandidatePlan())
    assert result.routed_count == 127
    assert result.unrouted_count == 1


def test_incomplete_scheduler_snapshot_rejects_decision():
    with pytest.raises(ValueError, match="Incomplete scheduler"):
        _snapshot(PSRL_Role.Rollout, [], incomplete=True)


@pytest.mark.parametrize("sleeping_stats", [{}, {"elastic_model_version": -1}])
def test_sleeping_snapshot_uses_committed_wake_version(sleeping_stats):
    snapshot = _snapshot(PSRL_Role.Rollout, [], sleeping_stats=sleeping_stats)
    sleeper = snapshot.instances[1]
    assert sleeper.model_version == 3, "Sleep must use the version loaded by the wake path."
    assert sleeper.requests == (), "Sleeping engines must have no live requests."


def test_pending_requests_dispatch_to_idle_awake_version_eligible_instance():
    pending = [
        {
            "request_id": "71",
            "seq_len": 50,
            "route_order": 1,
            "version_tag": 2,
            "reserve_indicators": {2: 0, 3: -1},
        }
    ]
    snapshot = _snapshot(PSRL_Role.Rollout, pending)
    idle = replace(snapshot.instances[0], requests=(), route_request_count=0, running_count=0, token_count=0)
    snapshot = replace(snapshot, instances=(idle, snapshot.instances[1]))
    result = evaluate_role_candidate(snapshot, RoleCandidatePlan())
    assert result.routed_count == 1, "An idle awake replica must accept eligible pending requests."
    assert result.unrouted_count == 0, "Dispatch must retain and consume the pending request."
