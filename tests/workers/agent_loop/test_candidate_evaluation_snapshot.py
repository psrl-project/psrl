"""
Verify native router snapshots and migration settlement for elastic planning.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.policy_runtime import PolicyIdentityMap
from psrl.workers.elastic.request_snapshot import build_role_snapshot
from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    RoleCandidatePlan,
    evaluate_role_candidate,
)
from psrl.workers.global_coordination.commands import CommandService
from psrl.workers.global_coordination.lifecycle import ReplicaLifecycle
from psrl.workers.global_coordination.model_state import ROLLOUT_SCOPE, ModelState
from psrl.workers.global_coordination.telemetry import TelemetryService

pytestmark = pytest.mark.cpu_test


def _snapshot_services(requests, dynamic_weights=True):
    model = ModelState(ROLLOUT_SCOPE, "http://rollout", dynamic_weights, "", None)
    gateway = SimpleNamespace(get=AsyncMock(return_value=requests))
    return TelemetryService(SimpleNamespace(), gateway, None), model


def test_candidate_snapshot_preserves_training_queue_order():
    requests = [
        {"request_id": "3", "is_validate": False},
        {"request_id": "1", "is_validate": True},
        {"request_id": "2", "is_validate": False},
    ]
    telemetry, model = _snapshot_services(requests)
    snapshot = asyncio.run(telemetry.requests(model))
    assert [request["request_id"] for request in snapshot] == ["3", "2"], (
        "Training demand must retain native queue order and exclude dedicated validation."
    )
    telemetry.gateway.get.assert_awaited_once_with(model, "/routing_loop/snapshot")


def test_validation_only_queue_does_not_request_elastic_rollout_capacity():
    telemetry, model = _snapshot_services(
        [{"request_id": "1", "is_validate": True, "seq_len": 50, "route_order": 0, "version_tag": 5}],
    )
    pending = asyncio.run(telemetry.requests(model, [5]))
    identities = PolicyIdentityMap()
    identities.register(PSRL_Role.Rollout, "Qwen2.5-1.5B", [("worker-0", 0)])
    snapshot = build_role_snapshot(
        PSRL_Role.Rollout,
        "Qwen2.5-1.5B",
        [SimpleNamespace(instance_id=0, is_awaken=False, running_queue_num=0, waiting_queue_num=0)],
        {},
        pending,
        identities,
        OmegaConf.create(
            {
                "method": "throughput_optimal",
                "cost_model_path": "psrl/trainer/config/cost_model/qwen2.5_1.5b.json",
                "candidate_sort_indicator": "version",
                "max_concurrent_seqs_per_instance": 128,
            }
        ),
        4096,
        1,
        1,
        5,
    )
    sleeping = evaluate_role_candidate(snapshot, RoleCandidatePlan())
    awake = evaluate_role_candidate(snapshot, RoleCandidatePlan(wake_instance_ids=frozenset({0})))
    assert sleeping.unrouted_count == 0, "Dedicated validation must not enter elastic rollout demand."
    assert awake.routed_count == 0, "Elastic rollout replicas must not serve dedicated validation."
    assert awake.throughput == sleeping.throughput == 0, "Validation must not incentivize elastic rollout wakes."


@pytest.mark.parametrize("invalid_snapshot", [None, {"pending": 0}])
def test_candidate_snapshot_rejects_invalid_native_response(invalid_snapshot):
    telemetry, model = _snapshot_services(invalid_snapshot)
    with pytest.raises(TypeError, match="must be a list"):
        asyncio.run(telemetry.requests(model))


def test_candidate_snapshot_fetch_failure_propagates_to_monitor():
    telemetry, model = _snapshot_services([])
    telemetry.gateway.get.side_effect = RuntimeError("Gateway unavailable.")
    with pytest.raises(RuntimeError, match="Gateway unavailable"):
        asyncio.run(telemetry.requests(model))


def test_static_reward_snapshot_preserves_opaque_request_ids_and_lengths():
    requests = [
        {
            "request_id": "reward-opaque",
            "is_validate": False,
            "prompt_len": 123,
            "version_tag": -1,
        }
    ]
    telemetry, model = _snapshot_services(requests, dynamic_weights=False)
    assert asyncio.run(telemetry.requests(model, [1, 2, 3])) == requests, (
        "Static reward requests must retain opaque IDs and prompt lengths."
    )


def test_candidate_snapshot_queries_only_training_inventory():
    async def scenario():
        requests = [
            {"request_id": "7", "is_validate": False, "version_tag": -1},
            {"request_id": "8", "is_validate": False, "version_tag": 2},
            {"request_id": "9", "is_validate": True, "version_tag": -1},
        ]
        telemetry, model = _snapshot_services(requests)
        model.ps_version = 4
        model.versions = {("worker-1", 0): 1, ("worker-2", 0): 2}
        telemetry.ps_manager = SimpleNamespace(
            can_reserve_request=SimpleNamespace(remote=AsyncMock(return_value=[[True, False, True]])),
            get_reserve_indicator=SimpleNamespace(remote=AsyncMock(side_effect=[[5, 6, 7], [8, 9, 10]])),
        )
        snapshot = await telemetry.requests(model, [1, 2, 3])
        telemetry.ps_manager.can_reserve_request.remote.assert_awaited_once_with([7], [1, 2, 3], False, [False])
        assert snapshot[0]["reservable_versions"] == [1, 3], "Admission must retain the reservable versions."
        assert snapshot[0]["reserve_indicators"] == {1: 5, 2: 6, 3: 7}, "Indicators must retain candidate versions."
        assert snapshot[1]["reserve_indicators"] == {1: 8, 2: 9, 3: 10}, "Bound requests must retain indicators."
        assert "reservable_versions" not in snapshot[1], "Bound requests must not query new admission."
        assert len(snapshot) == 2, "Dedicated validation must not query elastic admission inventory."
        assert [call.args for call in telemetry.ps_manager.get_reserve_indicator.remote.await_args_list] == [
            (7, [1, 2, 3], False),
            (8, [1, 2, 3], False),
        ], "Elastic reservations must use only training inventory."

    asyncio.run(scenario())


def _migration_services():
    model = ModelState(ROLLOUT_SCOPE, "http://rollout", True, "", None)
    config = SimpleNamespace(
        psrl=SimpleNamespace(deployment=SimpleNamespace(elastic={"request_migration_timeout_s": 0.01}))
    )
    source = SimpleNamespace()
    source.get_external_request_id = SimpleNamespace(remote=AsyncMock(return_value="external-1"))
    source.abort_requests = SimpleNamespace(remote=AsyncMock(return_value=True))
    model.add_worker(SimpleNamespace(replica_rank=0), source, "replica-1", 1)
    model.add_worker(SimpleNamespace(replica_rank=1), SimpleNamespace(), "selected-worker", 2)
    gateway = SimpleNamespace(
        get=AsyncMock(return_value={"dispatched": True, "completed": False}),
        post=AsyncMock(),
    )
    commands = CommandService(config, gateway, ReplicaLifecycle(gateway))
    move = {
        "request_id": "opaque-engine-1",
        "source_instance_id": ("replica-1", 0),
        "destination_instance_id": ("selected-worker", 1),
    }
    return commands, model, source, move


def test_migration_preserves_exact_native_destination_and_engine_request_id():
    async def scenario():
        commands, model, source, move = _migration_services()
        await commands.migrate_requests(model, [move])
        source.get_external_request_id.remote.assert_awaited_once_with("opaque-engine-1")
        source.abort_requests.remote.assert_awaited_once_with(["opaque-engine-1"])
        calls = commands.gateway.post.await_args_list
        assert calls[0].args == (
            model,
            "/routing_loop/migration",
            {"request_id": "external-1", "destination": ["selected-worker", 1]},
        ), "Migration must preserve the external ID and selected destination DP rank."
        assert calls[-1].args == (model, "/routing_loop/migration/finish", {"request_id": "external-1"}), (
            "Settled migrations must release their gateway reservation."
        )

    asyncio.run(scenario())


def test_completed_source_request_is_not_migrated_or_aborted():
    async def scenario():
        commands, model, source, move = _migration_services()
        source.get_external_request_id.remote.return_value = None
        await commands.migrate_requests(model, [move])
        source.abort_requests.remote.assert_not_awaited()
        commands.gateway.post.assert_not_awaited()

    asyncio.run(scenario())


def test_failed_abort_clears_native_migration_reservation():
    async def scenario():
        commands, model, source, move = _migration_services()
        source.abort_requests.remote.return_value = False
        await commands.migrate_requests(model, [move])
        commands.gateway.get.assert_not_awaited()
        assert commands.gateway.post.await_args_list[-1].args == (
            model,
            "/routing_loop/migration/finish",
            {"request_id": "external-1"},
        ), "Failed aborts must release their gateway reservation."

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_failed_redispatch_settlement_clears_reservation_and_stops_next_migration(failure):
    async def scenario():
        commands, model, source, move = _migration_services()
        if failure == "http":
            commands.gateway.get.side_effect = RuntimeError("Gateway unavailable.")
            error_type = RuntimeError
        else:
            commands.gateway.get.return_value = {"dispatched": False, "completed": False}
            error_type = TimeoutError
        second = dict(move, request_id="opaque-engine-2")
        with pytest.raises(error_type):
            await commands.migrate_requests(model, [move, second])
        source.abort_requests.remote.assert_awaited_once_with(["opaque-engine-1"])
        assert commands.gateway.post.await_args_list[-1].args == (
            model,
            "/routing_loop/migration/finish",
            {"request_id": "external-1"},
        ), "Failed settlement must release the reservation before propagating the error."

    asyncio.run(scenario())
