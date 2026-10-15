import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from psrl.workers.gen.rollout_coordination import RolloutCoordinator

pytestmark = pytest.mark.cpu_test


def _make_coordinator():
    coordinator = object.__new__(RolloutCoordinator)
    coordinator.instance_ids = {
        ("rollout-0", 0),
        ("rollout-1", 0),
        ("validate-0", 0),
    }
    coordinator.instance_to_model_version = {instance_id: 0 for instance_id in coordinator.instance_ids}
    coordinator.sleeping_replica_ids = set()
    coordinator.instance_to_version_after_sync = {instance_id: 0 for instance_id in coordinator.instance_ids}
    coordinator.ps_model_version = 0
    coordinator.model_sync_tasks = set()
    coordinator.replica_sync_tasks = {}
    coordinator.replica_operation_locks = {}
    coordinator.routing_operation_lock = asyncio.Lock()
    coordinator.config = SimpleNamespace(
        psrl=SimpleNamespace(
            rollout_coordination=SimpleNamespace(
                partial_rollout=SimpleNamespace(enable=False),
            ),
        ),
    )
    coordinator._set_routing_loop_running = AsyncMock()
    coordinator._publish_weight_version_updates = AsyncMock()
    coordinator.set_rollout_instance_model_version = MagicMock()
    coordinator.server_handles = {}
    for replica_id in ("rollout-0", "rollout-1", "validate-0"):
        handle = MagicMock()
        handle.is_sleeping.remote = AsyncMock(return_value=False)
        handle.pause_for_sync.remote = AsyncMock()
        handle.pull_model_for_sync.remote = AsyncMock(return_value=0)
        handle.resume_after_sync.remote = AsyncMock()
        handle.nixl_wake_up.remote = AsyncMock()
        handle.fail_sync.remote = AsyncMock()
        handle.pull_model.remote = AsyncMock()
        coordinator.server_handles[replica_id] = handle
    return coordinator


def test_validation_sync_targets_registered_validation_replica():
    coordinator = _make_coordinator()

    asyncio.run(
        coordinator.sync_with_ps(
            [("validate-0", 0)],
            wait_model_sync=True,
        )
    )

    validation = coordinator.server_handles["validate-0"]
    validation.pause_for_sync.remote.assert_awaited_once_with()
    validation.pull_model_for_sync.remote.assert_awaited_once_with(0)
    validation.resume_after_sync.remote.assert_awaited_once_with()
    coordinator.server_handles["rollout-0"].pull_model_for_sync.remote.assert_not_awaited()
    coordinator.server_handles["rollout-1"].pull_model_for_sync.remote.assert_not_awaited()


def test_validation_sync_skips_sleeping_replica():
    coordinator = _make_coordinator()
    validation = coordinator.server_handles["validate-0"]
    validation.is_sleeping.remote.return_value = True

    asyncio.run(
        coordinator.sync_with_ps(
            [("validate-0", 0)],
            wait_model_sync=True,
        )
    )

    validation.pause_for_sync.remote.assert_not_awaited()
    validation.pull_model_for_sync.remote.assert_not_awaited()


def test_initial_validation_pull_forces_version_zero_weight_load():
    coordinator = _make_coordinator()
    coordinator.tag_to_replica_ids = {
        "rollout": {"rollout-0", "rollout-1"},
        "validate": {"validate-0"},
    }

    asyncio.run(coordinator.initial_pull_from_ps(tag="validate"))

    coordinator.server_handles["validate-0"].pull_model.remote.assert_awaited_once_with()
    coordinator.server_handles["rollout-0"].pull_model.remote.assert_not_awaited()
    coordinator.server_handles["rollout-1"].pull_model.remote.assert_not_awaited()
    coordinator._publish_weight_version_updates.assert_awaited_once()


@pytest.mark.parametrize("wait_model_sync", [False, True])
def test_sync_updates_only_awake_replicas_without_waiting_for_sleeping_locks(wait_model_sync):
    async def scenario():
        coordinator = _make_coordinator()
        coordinator.ps_model_version = 4
        awake = coordinator.server_handles["rollout-0"]
        asleep = coordinator.server_handles["rollout-1"]
        awake.pull_model_for_sync.remote.return_value = 4
        asleep.is_sleeping.remote.return_value = True
        sleeping_lock = asyncio.Lock()
        await sleeping_lock.acquire()
        coordinator.replica_operation_locks["rollout-1"] = sleeping_lock

        await asyncio.wait_for(
            coordinator.sync_with_ps(
                [("rollout-0", 0), ("rollout-1", 0)],
                wait_model_sync=wait_model_sync,
            ),
            timeout=1.0,
        )
        if coordinator.model_sync_tasks:
            await asyncio.gather(*coordinator.model_sync_tasks)

        awake.pull_model_for_sync.remote.assert_awaited_once_with(4)
        asleep.pause_for_sync.remote.assert_not_awaited()
        asleep.pull_model_for_sync.remote.assert_not_awaited()
        asleep.resume_after_sync.remote.assert_not_awaited()
        assert coordinator.instance_to_version_after_sync[("rollout-1", 0)] == 0
        assert coordinator.instance_to_model_version[("rollout-1", 0)] == 0
        updates = coordinator._publish_weight_version_updates.await_args.args[0]
        assert {update["worker_id"] for update in updates} == {"rollout-0"}
        assert sleeping_lock.locked()
        sleeping_lock.release()

    asyncio.run(scenario())


def test_sync_rechecks_sleep_state_after_acquiring_lifecycle_lock():
    async def scenario():
        coordinator = _make_coordinator()
        server = coordinator.server_handles["rollout-0"]
        acquire = coordinator._acquire_replica_operations

        async def sleep_before_lock(replica_ids):
            server.is_sleeping.remote.return_value = True
            return await acquire(replica_ids)

        coordinator._acquire_replica_operations = sleep_before_lock
        await coordinator.sync_with_ps([("rollout-0", 0)], wait_model_sync=True)
        server.pause_for_sync.remote.assert_not_awaited()
        server.pull_model_for_sync.remote.assert_not_awaited()
        coordinator._publish_weight_version_updates.assert_not_awaited()
        coordinator._set_routing_loop_running.assert_not_awaited()
        assert not coordinator.replica_operation_locks["rollout-0"].locked()

    asyncio.run(scenario())


def test_sleeping_rollout_pulls_latest_weights_when_woken():
    async def scenario():
        coordinator = _make_coordinator()
        server = coordinator.server_handles["rollout-1"]
        server.is_sleeping.remote.return_value = True
        coordinator.ps_model_version = 4
        await coordinator.sync_with_ps([("rollout-1", 0)], wait_model_sync=True)
        server.pull_model_for_sync.remote.assert_not_awaited()
        coordinator._publish_weight_version_updates.assert_not_awaited()

        coordinator.ps_model_version = 7
        server.pull_model_for_sync.remote.return_value = 7
        await coordinator._do_wake_up_instance("rollout-1")
        server.nixl_wake_up.remote.assert_awaited_once()
        server.pull_model_for_sync.remote.assert_awaited_once_with(7)
        coordinator.set_rollout_instance_model_version.assert_called_once_with(("rollout-1", 0), 7)
        assert coordinator.instance_to_version_after_sync[("rollout-1", 0)] == 7

    asyncio.run(scenario())


@pytest.mark.parametrize("strategy", ["greedy", "status_based"])
@pytest.mark.parametrize("partial_rollout", [False, True])
def test_background_sync_ignores_sleeping_replicas_before_version_and_request_checks(strategy, partial_rollout):
    async def scenario():
        coordinator = _make_coordinator()
        coordinator.ps_model_version = 4
        coordinator.tag_to_replica_ids = {"rollout": {"rollout-0", "rollout-1"}, "validate": {"validate-0"}}
        coordinator.stop_sync_and_migrate = False
        coordinator.ready_buffers = set()
        coordinator.config.psrl.rollout_coordination.partial_rollout.enable = partial_rollout
        coordinator.config.psrl.rollout_coordination.sync_and_mig_strategy = SimpleNamespace(
            check_interval_in_ms=0,
            sync=SimpleNamespace(seamless_train_version=0),
            mig=SimpleNamespace(enable=False),
        )
        coordinator.instance_to_engine_status = {
            iid: SimpleNamespace(model_version=0, get_waiting_and_running_queue_size=lambda: 0)
            for iid in coordinator.instance_ids
        }
        asleep = coordinator.server_handles["rollout-1"]
        asleep.is_sleeping.remote.return_value = True
        coordinator._is_selecting = AsyncMock(return_value=False)
        coordinator.check_should_sync = AsyncMock(return_value=True)
        coordinator.check_no_activate_tasks = AsyncMock(return_value=True)
        coordinator.sync_with_ps = AsyncMock()
        probe = coordinator._get_awake_instance_ids

        async def stop_after_probe(instance_ids):
            awake_ids = await probe(instance_ids)
            coordinator.stop_sync_and_migrate = True
            return awake_ids

        coordinator._get_awake_instance_ids = stop_after_probe
        await getattr(coordinator, f"_{strategy}_sync_and_migrate_loop")()
        coordinator.sync_with_ps.assert_awaited_once_with([("rollout-0", 0)])
        for check in (coordinator.check_should_sync, coordinator.check_no_activate_tasks):
            assert all(args.args[0] == ("rollout-0", 0) for args in check.await_args_list)
        coordinator.server_handles["validate-0"].is_sleeping.remote.assert_not_awaited()

    asyncio.run(scenario())


def test_migration_does_not_consider_a_sleeping_replica_with_stale_stats():
    coordinator = _make_coordinator()
    for server in coordinator.server_handles.values():
        server.is_sleeping.remote.return_value = True
    coordinator.instance_to_engine_status = {}
    coordinator.ps_manager = MagicMock()
    assert asyncio.run(coordinator._check_should_migrate()) == []
    coordinator.ps_manager.check_aborted_model_versions.remote.assert_not_called()


@pytest.mark.parametrize("strategy", ["greedy", "status_based"])
@pytest.mark.parametrize("sync_in_progress", [False, True])
def test_sleeping_versions_do_not_count_as_active_background_sync(strategy, sync_in_progress):
    async def scenario():
        coordinator = _make_coordinator()
        coordinator.tag_to_replica_ids = {"rollout": {"rollout-0", "rollout-1"}, "validate": {"validate-0"}}
        coordinator.stop_sync_and_migrate = False
        coordinator.config.psrl.rollout_coordination.sync_and_mig_strategy = SimpleNamespace(
            check_interval_in_ms=0,
            mig=SimpleNamespace(enable=True),
        )
        coordinator.instance_to_engine_status = {
            iid: SimpleNamespace(model_version=0) for iid in coordinator.instance_ids
        }
        for server in coordinator.server_handles.values():
            server.is_sleeping.remote.return_value = True
        if sync_in_progress:
            coordinator.replica_sync_tasks["rollout-0"] = asyncio.Future()
        coordinator.check_and_migrate = AsyncMock()
        coordinator.sync_with_ps = AsyncMock()
        probe = coordinator._get_awake_instance_ids

        async def stop_after_probe(instance_ids):
            awake_ids = await probe(instance_ids)
            coordinator.stop_sync_and_migrate = True
            return awake_ids

        coordinator._get_awake_instance_ids = stop_after_probe
        await getattr(coordinator, f"_{strategy}_sync_and_migrate_loop")()
        coordinator.sync_with_ps.assert_not_awaited()
        if sync_in_progress:
            coordinator.check_and_migrate.assert_not_awaited()
        else:
            coordinator.check_and_migrate.assert_awaited_once()

    asyncio.run(scenario())
