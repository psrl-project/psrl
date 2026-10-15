import psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator as evaluator_module
import pytest
from psrl.workers.elastic.candidate_routing import resolve_candidate_model_versions
from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    RoleCandidatePlan,
    RoleSnapshot,
    evaluate_role_candidate,
    prepare_role_evaluation_context,
)

pytestmark = pytest.mark.cpu_test


def _instance(
    instance_id,
    requests,
    *,
    awake=True,
    throughput_params=(0.1, 1.0, 0.0, 0.0),
    route_cost_params=(0.0, 0.0, 0.0, 1.0, 0.01),
    running_count=None,
    waiting_count=0,
    max_model_len=10_000,
):
    return {
        "instance_id": instance_id,
        "is_awake": awake,
        "requests": [
            {
                "request_id": request_id,
                "seq_len": seq_len,
                "source_instance_id": instance_id,
                "is_waiting": is_waiting,
            }
            for request_id, seq_len, is_waiting in requests
        ],
        "route_request_count": len(requests),
        "running_count": (sum(not item[2] for item in requests) if running_count is None else running_count),
        "waiting_count": waiting_count,
        "token_count": sum(item[1] for item in requests),
        "max_model_len": max_model_len,
        "throughput_params": throughput_params,
        "route_cost_params": route_cost_params,
    }


def test_rebalance_accepts_multiple_strict_gains_and_moves_each_request_once():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, [("10", 10, False), ("2", 2, False), ("1", 1, False)]),
                _instance(1, [], awake=False),
            ],
            "pending_requests": [],
            "max_concurrent_requests": 32,
        }
    )

    result = evaluate_role_candidate(
        snapshot,
        RoleCandidatePlan(
            wake_instance_ids=frozenset({1}),
            primary_scale_up=True,
        ),
    )

    assert [move.request_id for move in result.rebalance_moves] == ["1", "2"]
    assert [move.source_instance_id for move in result.rebalance_moves] == [0, 0]
    assert [move.destination_instance_id for move in result.rebalance_moves] == [1, 1]
    assert len({move.request_id for move in result.rebalance_moves}) == 2


def test_rebalance_stops_on_first_non_increasing_move():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(
                    0,
                    [("0", 1, False), ("1", 100, False)],
                    throughput_params=(1.0, 0.0, 0.0, 0.0),
                ),
                _instance(
                    1,
                    [],
                    awake=False,
                    throughput_params=(1000.0, 0.0, 0.0, 0.0),
                ),
            ],
        }
    )

    result = evaluate_role_candidate(
        snapshot,
        RoleCandidatePlan(
            wake_instance_ids=frozenset({1}),
            primary_scale_up=True,
        ),
    )

    assert result.rebalance_moves == ()


def test_rebalance_ties_use_smallest_instance_and_request_ids():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(0, [("9", 5, False), ("3", 5, False)]),
                _instance(1, [("8", 5, False), ("4", 5, False)]),
                _instance(2, [], awake=False),
                _instance(3, [], awake=False),
            ],
        }
    )

    result = evaluate_role_candidate(
        snapshot,
        RoleCandidatePlan(
            wake_instance_ids=frozenset({2, 3}),
            primary_scale_up=True,
        ),
    )

    first = result.rebalance_moves[0]
    assert first.source_instance_id == 0
    assert first.destination_instance_id == 2
    assert first.request_id == "3"


def test_rollout_selector_respects_priority_capacity_and_pending_order():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, [], max_model_len=5),
                _instance(1, [], max_model_len=100),
            ],
            "pending_requests": [
                {
                    "request_id": "a",
                    "seq_len": 10,
                    "route_order": 0,
                    "eligible_instance_ids": [0, 1],
                    "candidate_priorities": [[0, 0], [1, 1]],
                },
                {
                    "request_id": "b",
                    "seq_len": 1,
                    "route_order": 1,
                    "eligible_instance_ids": [0, 1],
                    "candidate_priorities": [[0, 0], [1, 1]],
                },
            ],
            "max_concurrent_requests": 4,
            "delta_throughput_threshold": 0.0,
        }
    )

    result = evaluate_role_candidate(snapshot, RoleCandidatePlan())

    assert result.routed_count == 2
    assert result.unrouted_count == 0
    assert dict(result.instance_throughputs)[0] > 0
    assert dict(result.instance_throughputs)[1] > 0


def test_scale_down_reroutes_waiting_victim_requests_even_with_running_scope():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "RewardModel",
            "strategy": "itl",
            "queue_scope": "running",
            "instances": [
                _instance(0, [("waiting", 3, True)], waiting_count=1),
                _instance(1, []),
            ],
            "max_concurrent_requests": 4,
        }
    )

    baseline = evaluate_role_candidate(snapshot, RoleCandidatePlan())
    scaled_down = evaluate_role_candidate(
        snapshot,
        RoleCandidatePlan(sleep_instance_ids=frozenset({0})),
    )

    assert baseline.throughput == 0.0
    assert scaled_down.routed_count == 1
    assert scaled_down.unrouted_count == 0
    assert scaled_down.throughput > 0.0


def test_rm_pending_simulation_fills_native_router_concurrency_limit():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(0, [("running", 1, False)], running_count=1),
            ],
            "pending_requests": [
                {"request_id": f"p{index}", "seq_len": 1, "route_order": index} for index in range(128)
            ],
            "max_concurrent_requests": 128,
        }
    )

    result = evaluate_role_candidate(snapshot, RoleCandidatePlan())

    assert result.routed_count == 127
    assert result.unrouted_count == 1


def test_empty_active_role_is_valid_and_leaves_pending_requests_unrouted():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(0, [], awake=False),
                _instance(1, [], awake=False),
            ],
            "pending_requests": [
                {"request_id": "p0", "seq_len": 1, "route_order": 0},
                {"request_id": "p1", "seq_len": 2, "route_order": 1},
            ],
            "max_concurrent_requests": 8,
        }
    )

    baseline = evaluate_role_candidate(snapshot, RoleCandidatePlan())
    scaled_up = evaluate_role_candidate(
        snapshot,
        RoleCandidatePlan(
            wake_instance_ids=frozenset({0}),
            primary_scale_up=True,
        ),
    )

    assert baseline.throughput == 0.0
    assert baseline.routed_count == 0
    assert baseline.unrouted_count == 2
    assert baseline.rebalance_moves == ()
    assert baseline.instance_throughputs == ()
    assert scaled_up.throughput > 0.0
    assert scaled_up.routed_count == 2
    assert scaled_up.unrouted_count == 0


def test_rollout_candidate_uses_current_ps_version_for_sleeping_instances():
    actual_versions = {0: 0, 1: 0, 2: 1}
    assert resolve_candidate_model_versions(
        actual_versions,
        sleeping_instance_ids={1, 2},
        current_ps_model_version=3,
    ) == {0: 0, 1: 3, 2: 3}
    assert actual_versions == {0: 0, 1: 0, 2: 1}

    parsed_instance = RoleSnapshot.from_mapping(
        {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                {
                    **_instance(1, [], awake=False),
                    "model_version": 0,
                    "candidate_model_version": 3,
                }
            ],
        }
    ).instances[0]
    assert parsed_instance.model_version == 3

    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, [], awake=True),
                _instance(1, [], awake=False),
            ],
            "pending_requests": [
                {
                    "request_id": "requires-current-ps-version",
                    "seq_len": 1,
                    "route_order": 0,
                    # The router includes instance 1 after evaluating it at the
                    # current PS version. The awake version-0 instance is not legal.
                    "eligible_instance_ids": [1],
                    "fallback_instance_ids": [1],
                    "candidate_priorities": [[1, [0, -3]]],
                }
            ],
            "max_concurrent_requests": 8,
        }
    )

    baseline = evaluate_role_candidate(snapshot, RoleCandidatePlan())
    scaled_up = evaluate_role_candidate(
        snapshot,
        RoleCandidatePlan(wake_instance_ids=frozenset({1}), primary_scale_up=True),
    )

    assert baseline.routed_count == 0
    assert baseline.unrouted_count == 1
    assert scaled_up.routed_count == 1
    assert scaled_up.unrouted_count == 0
    assert scaled_up.throughput > 0.0


def test_evaluation_is_deterministic():
    raw_snapshot = {
        "role": "RewardModel",
        "strategy": "itl",
        "instances": [
            _instance(0, [("0", 2, False)]),
            _instance(1, [], awake=False),
        ],
        "pending_requests": [
            {"request_id": str(index), "seq_len": index + 1, "route_order": index} for index in range(5)
        ],
        "max_concurrent_requests": 16,
    }
    plan = RoleCandidatePlan(
        wake_instance_ids=frozenset({1}),
        primary_scale_up=True,
    )

    first = evaluate_role_candidate(raw_snapshot, plan)
    for _ in range(10):
        assert evaluate_role_candidate(raw_snapshot, plan) == first


def test_prepared_context_matches_raw_snapshot_evaluation():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "Rollout",
            "strategy": "throughput_optimal",
            "instances": [
                _instance(0, [("9", 9, False), ("1", 1, False)]),
                _instance(1, [], awake=False),
            ],
            "pending_requests": [{"request_id": "pending", "seq_len": 2, "route_order": 0}],
            "max_concurrent_requests": 8,
        }
    )
    plan = RoleCandidatePlan(
        wake_instance_ids=frozenset({1}),
        primary_scale_up=True,
    )

    context = prepare_role_evaluation_context(snapshot)

    assert evaluate_role_candidate(context, plan) == evaluate_role_candidate(snapshot, plan)


def test_request_stream_merge_preserves_global_router_priority():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                {
                    **_instance(0, [], awake=True),
                    "requests": [
                        {
                            "request_id": "victim-2",
                            "seq_len": 1,
                            "routing_priority": [2],
                        },
                        {
                            "request_id": "victim-4",
                            "seq_len": 1,
                            "routing_priority": [4],
                        },
                    ],
                },
                _instance(1, [], awake=True),
            ],
            "pending_requests": [
                {
                    "request_id": "pending-3",
                    "seq_len": 1,
                    "routing_priority": [3],
                },
                {
                    "request_id": "pending-1",
                    "seq_len": 1,
                    "routing_priority": [1],
                },
            ],
        }
    )
    context = prepare_role_evaluation_context(snapshot)

    requests, total = evaluator_module._iter_requests_to_route(
        context,
        frozenset({0}),
    )

    assert total == 4
    assert [item.snapshot.request_id for item in requests] == [
        "pending-1",
        "victim-2",
        "pending-3",
        "victim-4",
    ]


def test_rm_selector_heap_updates_only_chosen_load_and_preserves_ties():
    snapshot = RoleSnapshot.from_mapping(
        {
            "role": "RewardModel",
            "strategy": "itl",
            "instances": [
                _instance(
                    0,
                    [],
                    throughput_params=(0.0, 1.0, 0.0, 0.0),
                ),
                _instance(
                    1,
                    [],
                    throughput_params=(0.0, 1.0, 0.0, 0.0),
                ),
            ],
            "pending_requests": [{"request_id": str(index), "seq_len": 1, "route_order": index} for index in range(5)],
            "max_concurrent_requests": 8,
        }
    )

    result = evaluate_role_candidate(snapshot, RoleCandidatePlan())

    assert result.routed_count == 5
    assert result.instance_throughputs == ((0, 3.0), (1, 2.0))
