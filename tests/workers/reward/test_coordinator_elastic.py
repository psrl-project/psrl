import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from psrl.workers.gen.rollout_coordination import RolloutCoordinator
from psrl.workers.gen.smg_adapter import ReplicaRoutingStatus
from psrl.workers.reward.reward_model.coordinator import RewardModelCoordinator

pytestmark = pytest.mark.cpu_test


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _make_server_handle():
    """Build a server handle mock where .nixl_sleep.remote and .nixl_wake_up.remote are AsyncMock."""
    handle = MagicMock()
    handle.nixl_sleep.remote = AsyncMock(return_value=None)
    handle.nixl_wake_up.remote = AsyncMock(return_value=None)
    handle.is_sleeping.remote = AsyncMock(return_value=False)
    handle.pull_model_for_sync.remote = AsyncMock(return_value=3)
    return handle


def _make_rollout_coord():
    """Build a minimal RolloutCoordinator stub instance (bypass __init__)."""
    coord = object.__new__(RolloutCoordinator)
    coord.instance_ids = {("wid-0", 0), ("wid-0", 1), ("wid-1", 0)}
    coord.server_handles = {
        "wid-0": _make_server_handle(),
        "wid-1": _make_server_handle(),
    }
    coord._gateway_post_json = AsyncMock(return_value={"updated": 3, "rejected": 0})
    coord.sleeping_replica_ids = set()
    coord.replica_operation_locks = {}
    coord.routing_operation_lock = asyncio.Lock()
    coord._set_routing_loop_running = AsyncMock()
    coord.ps_model_version = 2
    coord._publish_weight_version_updates = AsyncMock()
    coord.set_rollout_instance_model_version = MagicMock()
    coord.instance_to_version_after_sync = {}
    return coord


def _make_reward_coord():
    coord = object.__new__(RewardModelCoordinator)
    coord.instance_ids = {("rm-0", 0), ("rm-1", 0)}

    def _make_rm_handle():
        handle = MagicMock()
        handle.sleep.remote = AsyncMock(return_value=None)
        handle.wake_up.remote = AsyncMock(return_value=None)
        handle.nixl_sleep.remote = AsyncMock(return_value=None)
        handle.nixl_wake_up.remote = AsyncMock(return_value=None)
        handle.is_sleeping.remote = AsyncMock(return_value=False)
        return handle

    coord.server_handles = {
        "rm-0": _make_rm_handle(),
        "rm-1": _make_rm_handle(),
    }
    coord.rollout_gateway_url = ""
    coord.gateway_client = None
    coord.sleeping_replica_ids = set()
    coord.replica_operation_locks = {}
    coord.routing_operation_lock = asyncio.Lock()
    coord._set_routing_loop_running = AsyncMock()
    coord._gateway_post_json = AsyncMock(return_value={"updated": 1, "rejected": 0})
    return coord


# ── Task 1 tests ─────────────────────────────────────────────────────────────


def test_get_all_instance_ids_returns_sorted_tuples():
    coord = _make_rollout_coord()
    ids = coord.get_all_instance_ids()
    assert isinstance(ids, list)
    assert ids == sorted([("wid-0", 0), ("wid-0", 1), ("wid-1", 0)])
    for iid in ids:
        assert isinstance(iid, tuple) and len(iid) == 2


def test_get_sleep_level_rollout_returns_2():
    coord = _make_rollout_coord()
    assert coord._get_sleep_level() == 2


def test_do_sleep_instance_calls_nixl_sleep():
    coord = _make_rollout_coord()
    coord._get_sleep_level = lambda: 2
    asyncio.run(coord._do_sleep_instance("wid-0"))
    coord.server_handles["wid-0"].nixl_sleep.remote.assert_called_once_with(level=2)


def test_do_wake_up_instance_calls_nixl_wake_up():
    coord = _make_rollout_coord()
    asyncio.run(coord._do_wake_up_instance("wid-0"))
    coord.server_handles["wid-0"].nixl_wake_up.remote.assert_called_once_with()
    coord.server_handles["wid-0"].pull_model_for_sync.remote.assert_awaited_once_with(2)
    assert coord.instance_to_version_after_sync == {("wid-0", 0): 3, ("wid-0", 1): 3}
    coord._publish_weight_version_updates.assert_awaited_once()


def test_set_replica_routing_status_targets_selected_replicas():
    coord = _make_rollout_coord()

    asyncio.run(
        coord._set_replica_routing_status(
            ["wid-1", "wid-0", "wid-0"],
            ReplicaRoutingStatus.PAUSED,
        )
    )
    coord._gateway_post_json.assert_awaited_once_with(
        "/workers/pause",
        payload=[
            {"base_worker_id": "wid-0"},
            {"base_worker_id": "wid-1"},
        ],
    )


def test_set_replica_routing_status_rejects_incomplete_acknowledgement():
    coord = _make_rollout_coord()
    coord._gateway_post_json.return_value = {"updated": 1, "rejected": 0}

    with pytest.raises(RuntimeError, match="Incomplete replica routing update"):
        asyncio.run(coord._set_replica_routing_status(["wid-0"], ReplicaRoutingStatus.PAUSED))


def test_sleep_replicas_pauses_only_selected_replica():
    coord = _make_rollout_coord()
    coord._set_replica_routing_status = AsyncMock(return_value=None)

    asyncio.run(coord._sleep_replicas([("wid-0", 0), ("wid-0", 1)]))

    coord._set_replica_routing_status.assert_awaited_once_with(
        ["wid-0"],
        ReplicaRoutingStatus.PAUSED,
    )
    coord.server_handles["wid-0"].nixl_sleep.remote.assert_awaited_once_with(level=2)
    coord.server_handles["wid-1"].nixl_sleep.remote.assert_not_awaited()


def test_sleep_replicas_skips_already_sleeping_replicas():
    coord = _make_rollout_coord()
    coord._set_replica_routing_status = AsyncMock(return_value=None)
    coord.server_handles["wid-0"].is_sleeping.remote = AsyncMock(return_value=True)
    coord.server_handles["wid-1"].is_sleeping.remote = AsyncMock(return_value=False)

    asyncio.run(coord._sleep_replicas([("wid-0", 0), ("wid-1", 0)]))

    coord._set_replica_routing_status.assert_awaited_once_with(
        ["wid-0", "wid-1"],
        ReplicaRoutingStatus.PAUSED,
    )
    coord.server_handles["wid-0"].nixl_sleep.remote.assert_not_awaited()
    coord.server_handles["wid-1"].nixl_sleep.remote.assert_awaited_once_with(level=2)


def test_sleep_replicas_noops_when_all_already_sleeping():
    coord = _make_rollout_coord()
    coord._set_replica_routing_status = AsyncMock(return_value=None)
    for handle in coord.server_handles.values():
        handle.is_sleeping.remote = AsyncMock(return_value=True)

    asyncio.run(coord._sleep_replicas([("wid-0", 0), ("wid-1", 0)]))

    coord._set_replica_routing_status.assert_awaited_once_with(
        ["wid-0", "wid-1"],
        ReplicaRoutingStatus.PAUSED,
    )
    coord.server_handles["wid-0"].nixl_sleep.remote.assert_not_awaited()
    coord.server_handles["wid-1"].nixl_sleep.remote.assert_not_awaited()


def test_sleep_replicas_keeps_routing_paused_when_sleep_fails():
    coord = _make_rollout_coord()
    coord._set_replica_routing_status = AsyncMock(return_value=None)
    coord.server_handles["wid-0"].nixl_sleep.remote.side_effect = RuntimeError("sleep failed")

    with pytest.raises(RuntimeError, match="sleep failed"):
        asyncio.run(coord._sleep_replicas([("wid-0", 0), ("wid-1", 0)]))

    calls = coord._set_replica_routing_status.await_args_list
    assert len(calls) == 1, "Failed sleep must leave uncertain memory state unavailable to routing."
    assert calls[0].args == (["wid-0", "wid-1"], ReplicaRoutingStatus.PAUSED)
    coord.server_handles["wid-1"].nixl_sleep.remote.assert_awaited_once_with(level=2)


def test_wake_up_replicas_resumes_only_after_wake():
    coord = _make_rollout_coord()
    coord.server_handles["wid-1"].is_sleeping.remote = AsyncMock(return_value=True)
    calls = []

    async def _wake(replica_id):
        calls.append(("wake", replica_id))

    async def _set_status(replica_ids, status):
        calls.append(("routing", replica_ids, status))

    coord._do_wake_up_instance = _wake
    coord._set_replica_routing_status = _set_status

    asyncio.run(coord._wake_up_replicas([("wid-1", 0)]))

    assert calls == [
        ("routing", ["wid-1"], ReplicaRoutingStatus.PAUSED),
        ("wake", "wid-1"),
        ("routing", ["wid-1"], ReplicaRoutingStatus.ACTIVE),
    ]


def test_wake_up_replicas_skips_memory_restore_for_already_awake_replica():
    coord = _make_rollout_coord()
    coord._do_wake_up_instance = AsyncMock()
    coord._set_replica_routing_status = AsyncMock()

    asyncio.run(coord._wake_up_replicas([("wid-1", 0)]))

    coord._do_wake_up_instance.assert_not_awaited()
    assert coord._set_replica_routing_status.await_args.args == (["wid-1"], ReplicaRoutingStatus.ACTIVE)


def test_wake_up_replicas_does_not_resume_routing_after_weight_pull_failure():
    coord = _make_rollout_coord()
    coord.server_handles["wid-1"].is_sleeping.remote = AsyncMock(return_value=True)
    coord._do_wake_up_instance = AsyncMock(side_effect=RuntimeError("weight pull failed"))
    coord._set_replica_routing_status = AsyncMock()

    with pytest.raises(RuntimeError, match="weight pull failed"):
        asyncio.run(coord._wake_up_replicas([("wid-1", 0)]))

    coord._set_replica_routing_status.assert_awaited_once_with(["wid-1"], ReplicaRoutingStatus.PAUSED)


# ── Task 2 tests ─────────────────────────────────────────────────────────────


def test_reward_get_sleep_level_returns_2():
    coord = _make_reward_coord()
    assert coord._get_sleep_level() == 2


def test_reward_do_sleep_instance_calls_plain_sleep():
    """RewardModelCoordinator.sleep uses server.sleep (not nixl_sleep)."""
    coord = _make_reward_coord()
    asyncio.run(coord._do_sleep_instance("rm-0"))
    coord.server_handles["rm-0"].sleep.remote.assert_called_once_with(level=2)
    coord.server_handles["rm-0"].nixl_sleep.remote.assert_not_called()


def test_reward_do_wake_up_instance_calls_plain_wake_up():
    """RewardModelCoordinator.wake_up uses server.wake_up (not nixl_wake_up)."""
    coord = _make_reward_coord()
    asyncio.run(coord._do_wake_up_instance("rm-0"))
    coord.server_handles["rm-0"].wake_up.remote.assert_called_once_with()
    coord.server_handles["rm-0"].nixl_wake_up.remote.assert_not_called()


def test_get_router_backlog_size_no_url_returns_0():
    """Without gateway URL, backlog size returns 0 immediately."""
    coord = _make_reward_coord()
    coord.rollout_gateway_url = ""
    result = asyncio.run(coord.get_router_backlog_size())
    assert result == 0


def test_get_router_backlog_size_reads_pending_request_count():
    """get_router_backlog_size reads the SMG routing queue length."""
    coord = _make_reward_coord()
    coord.rollout_gateway_url = "http://127.0.0.1:8300"

    async def _mock_get(path):
        assert path == "/routing_loop/status"
        return {"pending_request_num": 8}

    coord._gateway_get_json = _mock_get
    result = asyncio.run(coord.get_router_backlog_size())
    assert result == 8


def test_get_router_backlog_summary_applies_top_t():
    coord = _make_reward_coord()
    coord.get_router_backlog_size = AsyncMock(return_value=8)

    result = asyncio.run(coord.get_router_backlog_summary(top_t=3))

    assert result == {"pending": 8, "count": 3, "total_tokens": 0}


def test_router_backlog_failure_is_not_reported_as_empty():
    coord = _make_reward_coord()
    coord.rollout_gateway_url = "http://127.0.0.1:8300"
    coord._gateway_get_json = AsyncMock(side_effect=RuntimeError("Gateway unavailable."))
    with pytest.raises(RuntimeError, match="Gateway unavailable"):
        asyncio.run(coord.get_router_backlog_size())


def test_reward_status_sync_starts_with_busy_loop():
    async def check():
        coord = _make_reward_coord()
        coord.command_handler_task = None
        coord.config = MagicMock()
        coord.config.psrl.status_collection.enable = True
        coord.reward_model_name = "TestRM"
        coord._command_handler_loop = AsyncMock()
        coord._process_status_queue = AsyncMock()
        coord._sync_status_to_router = AsyncMock()
        await coord.start_busy_loop()
        await asyncio.gather(
            coord.command_handler_task, coord.process_status_queue_task, coord.sync_status_to_router_task
        )
        coord._sync_status_to_router.assert_awaited_once()

    asyncio.run(check())
