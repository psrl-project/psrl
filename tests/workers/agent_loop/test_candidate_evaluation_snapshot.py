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

from tests.elastic.runtime_helpers import _coordinator
from tests.elastic.runtime_helpers import modules as modules

pytestmark = pytest.mark.cpu_test


def _snapshot_coordinator(modules, requests):
    coordinator = object.__new__(modules.status.StatusMixin)
    coordinator._gateway_get_json = AsyncMock(return_value=requests)
    coordinator.ps_manager = None
    return coordinator


def test_candidate_snapshot_preserves_training_queue_order(modules):
    requests = [
        {"request_id": "3", "is_validate": False},
        {"request_id": "1", "is_validate": True},
        {"request_id": "2", "is_validate": False},
    ]
    coordinator = _snapshot_coordinator(modules, requests)
    snapshot = asyncio.run(coordinator.get_router_request_snapshot([1, 2, 3]))
    assert [request["request_id"] for request in snapshot] == ["3", "2"], (
        "Training demand must retain native queue order and exclude dedicated validation."
    )
    coordinator._gateway_get_json.assert_awaited_once_with("/routing_loop/snapshot")


def test_validation_only_queue_does_not_request_elastic_rollout_capacity(modules):
    coordinator = _snapshot_coordinator(
        modules,
        [{"request_id": "1", "is_validate": True, "seq_len": 50, "route_order": 0, "version_tag": 5}],
    )
    pending = asyncio.run(coordinator.get_router_request_snapshot([5]))
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
def test_candidate_snapshot_rejects_invalid_native_response(modules, invalid_snapshot):
    coordinator = _snapshot_coordinator(modules, invalid_snapshot)
    with pytest.raises(TypeError, match="must be a list"):
        asyncio.run(coordinator.get_router_request_snapshot([1, 2, 3]))


def test_candidate_snapshot_fetch_failure_propagates_to_monitor(modules):
    coordinator = _snapshot_coordinator(modules, [])
    coordinator._gateway_get_json.side_effect = RuntimeError("Gateway unavailable.")
    with pytest.raises(RuntimeError, match="Gateway unavailable"):
        asyncio.run(coordinator.get_router_request_snapshot([1, 2, 3]))


def test_static_reward_snapshot_preserves_opaque_request_ids_and_lengths(modules):
    requests = [
        {
            "request_id": "reward-opaque",
            "is_validate": False,
            "prompt_len": 123,
            "version_tag": -1,
        }
    ]
    coordinator = _snapshot_coordinator(modules, requests)
    assert asyncio.run(coordinator.get_router_request_snapshot([1, 2, 3])) == requests


def test_candidate_snapshot_queries_only_training_inventory(modules):
    async def scenario():
        requests = [
            {"request_id": "7", "is_validate": False, "version_tag": -1},
            {"request_id": "8", "is_validate": False, "version_tag": 2},
            {"request_id": "9", "is_validate": True, "version_tag": -1},
        ]
        coordinator = _snapshot_coordinator(modules, requests)
        coordinator.ps_model_version = 4
        coordinator.instance_to_model_version = {("worker-1", 0): 1, ("worker-2", 0): 2}
        coordinator.ps_manager = SimpleNamespace(
            can_reserve_request=SimpleNamespace(remote=AsyncMock(return_value=[[True, False, True]])),
            get_reserve_indicator=SimpleNamespace(remote=AsyncMock(side_effect=[[5, 6, 7], [8, 9, 10]])),
        )
        snapshot = await coordinator.get_router_request_snapshot([1, 2, 3])
        coordinator.ps_manager.can_reserve_request.remote.assert_awaited_once_with([7], [1, 2, 3], False, [False])
        assert snapshot[0]["reservable_versions"] == [1, 3]
        assert snapshot[0]["reserve_indicators"] == {1: 5, 2: 6, 3: 7}
        assert snapshot[1]["reserve_indicators"] == {1: 8, 2: 9, 3: 10}
        assert "reservable_versions" not in snapshot[1]
        assert len(snapshot) == 2, "Dedicated validation must not query elastic admission inventory."
        assert [call.args for call in coordinator.ps_manager.get_reserve_indicator.remote.await_args_list] == [
            (7, [1, 2, 3], False),
            (8, [1, 2, 3], False),
        ], "Elastic reservations must use only training inventory."

    asyncio.run(scenario())


def _migration_coordinator(modules):
    coordinator = _coordinator(modules)
    coordinator.config.psrl.deployment = SimpleNamespace(elastic={"request_migration_timeout_s": 0.01})
    source = coordinator.server_handles["replica-1"]
    source.get_external_request_id = SimpleNamespace(remote=AsyncMock(return_value="external-1"))
    source.abort_requests = SimpleNamespace(remote=AsyncMock(return_value=True))
    coordinator._gateway_get_json = AsyncMock(return_value={"dispatched": True})
    move = {
        "request_id": "opaque-engine-1",
        "source_instance_id": ("replica-1", 0),
        "destination_instance_id": ("selected-worker", 1),
    }
    return coordinator, source, move


def test_migration_preserves_exact_native_destination_and_engine_request_id(modules):
    async def scenario():
        coordinator, source, move = _migration_coordinator(modules)
        await coordinator.execute_request_migrations([move])
        source.get_external_request_id.remote.assert_awaited_once_with("opaque-engine-1")
        source.abort_requests.remote.assert_awaited_once_with(["opaque-engine-1"])
        calls = coordinator._gateway_post_json.await_args_list
        assert calls[0].args == (
            "/routing_loop/migration",
            {"request_id": "external-1", "destination": ["selected-worker", 1]},
        )
        assert calls[-1].args == ("/routing_loop/migration/finish", {"request_id": "external-1"})

    asyncio.run(scenario())


def test_completed_source_request_is_not_migrated_or_aborted(modules):
    async def scenario():
        coordinator, source, move = _migration_coordinator(modules)
        source.get_external_request_id.remote.return_value = None
        await coordinator.execute_request_migrations([move])
        source.abort_requests.remote.assert_not_awaited()
        coordinator._gateway_post_json.assert_not_awaited()

    asyncio.run(scenario())


def test_failed_abort_clears_native_migration_reservation(modules):
    async def scenario():
        coordinator, source, move = _migration_coordinator(modules)
        source.abort_requests.remote.return_value = False
        await coordinator.execute_request_migrations([move])
        coordinator._gateway_get_json.assert_not_awaited()
        assert coordinator._gateway_post_json.await_args_list[-1].args == (
            "/routing_loop/migration/finish",
            {"request_id": "external-1"},
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["timeout", "http"])
def test_failed_redispatch_settlement_clears_reservation_and_stops_next_migration(modules, failure):
    async def scenario():
        coordinator, source, move = _migration_coordinator(modules)
        if failure == "http":
            coordinator._gateway_get_json.side_effect = RuntimeError("Gateway unavailable.")
            error_type = RuntimeError
        else:
            coordinator._gateway_get_json.return_value = {"dispatched": False}
            error_type = TimeoutError
        second = dict(move, request_id="opaque-engine-2")
        with pytest.raises(error_type):
            await coordinator.execute_request_migrations([move, second])
        source.abort_requests.remote.assert_awaited_once_with(["opaque-engine-1"])
        assert coordinator._gateway_post_json.await_args_list[-1].args == (
            "/routing_loop/migration/finish",
            {"request_id": "external-1"},
        )

    asyncio.run(scenario())
