"""
Verify independent global coordination preserves model state and replica lifecycles.
"""

import ast
import asyncio
import itertools
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf
from psrl.utils.server.command import Command, CommandType
from psrl.workers.elastic.cluster_topology import InstanceStatus
from psrl.workers.elastic.sleep_wake_orchestrator import SleepWakeOrchestrator
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.global_coordination import coordinator as global_module
from psrl.workers.global_coordination.client import CoordinatorClient
from psrl.workers.global_coordination.coordinator import GlobalCoordinator
from psrl.workers.global_coordination.lifecycle import GatewayClient
from psrl.workers.global_coordination.model_state import ROLLOUT_SCOPE
from ray import cloudpickle

pytestmark = pytest.mark.cpu_test


class _StatusQueue:
    def __init__(self, endpoint):
        self.queue = asyncio.Queue()
        self.closed = False

    async def get_async(self, block=True, timeout=None):
        if block:
            return await asyncio.wait_for(self.queue.get(), timeout)
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def close(self):
        self.closed = True


@pytest.fixture
def config(tmp_path, monkeypatch):
    ports = itertools.count(28000)
    monkeypatch.setattr(global_module, "find_available_port", lambda **kwargs: next(ports))
    monkeypatch.setattr(global_module.ray.util, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(global_module, "ZMQPullQueue", _StatusQueue)
    return OmegaConf.create(
        {
            "psrl": {
                "logging_path": str(tmp_path),
                "ps_mode": "nixl_cpu",
                "status_collection": {
                    "enable": True,
                    "coordinator_sync_interval_in_ms": 10,
                    "stats_recorder": {"enable": False},
                },
                "rollout_coordination": {
                    "session_strategy": {"thunder_agent": {"enable": False}},
                    "partial_rollout": {"enable": False},
                    "sync_and_mig_strategy": {
                        "method": "greedy",
                        "check_interval_in_ms": 10,
                        "mig": {"enable": False},
                        "sync": {"seamless_train_version": 0},
                    },
                },
            }
        }
    )


async def _make(config):
    coordinator = GlobalCoordinator(config, Mock(), "http://rollout")
    for model in ("judge-a", "judge-b"):
        await coordinator.register_reward_model(OmegaConf.create({"reward_model_name": model}), f"http://{model}")
    coordinator.gateway.pause_replicas = AsyncMock()
    coordinator.gateway.eligibility = AsyncMock()
    coordinator.gateway.publish_versions = AsyncMock()
    coordinator.gateway.routing = AsyncMock()
    return coordinator


async def _call(coordinator, scope, method, *args, **kwargs):
    return await coordinator.invoke(scope, method, args, kwargs)


async def _add(coordinator, scope, worker_id, rank=0, dp_size=1, **kwargs):
    server = SimpleNamespace()
    for method in (
        "sleep",
        "wake_up",
        "nixl_sleep",
        "nixl_wake_up",
        "pause_for_sync",
        "resume_after_sync",
        "fail_sync",
        "pull_model",
        "wait_for_requests_to_drain",
        "init_nixl_client",
        "nixl_protocol",
        "nixl_convert_params",
        "sync_with_ps",
        "abort_all_requests",
        "abort_requests",
    ):
        setattr(server, method, SimpleNamespace(remote=AsyncMock(return_value=0)))
    server.is_sleeping = SimpleNamespace(remote=AsyncMock(return_value=False))
    server.pull_model_for_sync = SimpleNamespace(remote=AsyncMock(return_value=3))
    server.get_active_task_num = SimpleNamespace(remote=AsyncMock(return_value=0))
    replica = SimpleNamespace(replica_rank=rank, world_size=1)
    await _call(coordinator, scope, "add_worker", replica, server, worker_id, dp_size, **kwargs)
    return server


def test_global_implementation_does_not_import_or_inherit_old_coordinators():
    root = Path(global_module.__file__).parent
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert "rollout_coordination" not in (node.module or ""), "Global services must be independent."
                assert "reward_model.coordinator" not in (node.module or ""), "Global services must be independent."
            if isinstance(node, ast.Name):
                assert node.id not in {"RolloutCoordinator", "RewardModelCoordinator"}, (
                    "Global services must not construct or inherit the old coordinators."
                )


def test_validation_pulls_policy_weights_while_rewards_keep_static_weights(config):
    async def scenario():
        coordinator = await _make(config)
        try:
            policy = coordinator.models[ROLLOUT_SCOPE]
            validation = await _add(coordinator, ROLLOUT_SCOPE, "validation", is_validate=True)
            reward = await _add(coordinator, ("reward", "judge-a"), "worker-0")
            await _call(coordinator, ROLLOUT_SCOPE, "set_ps_model_version", 3)
            policy.replicas["validation"].status = InstanceStatus.ASLEEP
            reward_model = coordinator.models[("reward", "judge-a")]
            reward_model.replicas["worker-0"].status = InstanceStatus.ASLEEP
            for scope in (ROLLOUT_SCOPE, ("reward", "judge-a")):
                await _call(coordinator, scope, "start_busy_loop")
            for scope, worker in ((ROLLOUT_SCOPE, "validation"), (("reward", "judge-a"), "worker-0")):
                await _call(
                    coordinator, scope, "exec_command", Command(CommandType.WAKE_UP, instance_ids=[(worker, 0)])
                )
            validation.nixl_wake_up.remote.assert_awaited_once_with()
            validation.pull_model_for_sync.remote.assert_awaited_once_with(3)
            reward.wake_up.remote.assert_awaited_once_with()
            reward.pull_model_for_sync.remote.assert_not_awaited()
            assert policy.versions[("validation", 0)] == 3, "Validation must use policy weights."
            assert reward_model.ps_version == 0, "Policy updates must not alter static rewards."
            assert policy.replica_ids("validate") == ["validation"], "Validation must retain its replica role."
            with pytest.raises(ValueError, match="Unknown coordinator method"):
                await _call(coordinator, ("reward", "judge-a"), "init_nixl_client")
        finally:
            await coordinator.stop_busy_loop()
        assert coordinator.gateway.session.closed, "Global shutdown must close the shared HTTP pool."
        assert all(model.status_queue.closed for model in coordinator.models.values()), (
            "Global shutdown must close every status sink."
        )

    asyncio.run(scenario())


def test_blocked_reward_command_does_not_block_other_models_or_validation(config):
    async def scenario():
        coordinator = await _make(config)
        gate, entered = asyncio.Event(), asyncio.Event()
        blocked = None
        try:
            for scope, model in coordinator.models.items():
                await _add(coordinator, scope, "worker-0")
                model.track("commands", coordinator.commands.run(model))
            validation = await _add(coordinator, ROLLOUT_SCOPE, "validation", rank=1, is_validate=True)
            reward_a = coordinator.models[("reward", "judge-a")]
            reward_b = coordinator.models[("reward", "judge-b")]

            async def release(**kwargs):
                entered.set()
                await gate.wait()

            reward_a.replicas["worker-0"].server.sleep.remote.side_effect = release
            blocked = asyncio.create_task(
                _call(
                    coordinator,
                    ("reward", "judge-a"),
                    "exec_command",
                    Command(CommandType.SLEEP, instance_ids=[("worker-0", 0)]),
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=1)
            await asyncio.wait_for(
                asyncio.gather(
                    _call(
                        coordinator,
                        ("reward", "judge-b"),
                        "exec_command",
                        Command(CommandType.SLEEP, instance_ids=[("worker-0", 0)]),
                    ),
                    _call(
                        coordinator,
                        ROLLOUT_SCOPE,
                        "exec_command",
                        Command(CommandType.SLEEP, instance_ids=[("validation", 0)]),
                    ),
                ),
                timeout=1,
            )
            reward_b.replicas["worker-0"].server.sleep.remote.assert_awaited_once_with(level=2)
            validation.nixl_sleep.remote.assert_awaited_once_with(level=2)
            assert not blocked.done(), "Other models must finish while one model remains blocked."
        finally:
            gate.set()
            if blocked is not None:
                await blocked
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_repeated_replica_ranks_and_gateway_ids_are_model_scoped(config):
    async def scenario():
        coordinator = await _make(config)
        try:
            for version, (scope, model) in enumerate(coordinator.models.items(), start=1):
                await _add(coordinator, scope, "worker-0")
                model.track("telemetry", coordinator.telemetry.collect(model))
                model.status_queue.queue.put_nowait(
                    EngineStats(
                        replica_idx=0,
                        data_parallel_rank=0,
                        model_version=version,
                        snapshot=EngineStats.get_default_snapshot(),
                    )
                )

            async def wait():
                while any(
                    model.engine_status[("worker-0", 0)].model_version != version
                    for version, model in enumerate(coordinator.models.values(), start=1)
                ):
                    await asyncio.sleep(0)

            await asyncio.wait_for(wait(), timeout=1)
            assert len({model.status_endpoint for model in coordinator.models.values()}) == 3, (
                "Every model must retain its own status sink."
            )
            assert {model.gateway_url for model in coordinator.models.values()} == {
                "http://rollout",
                "http://judge-a",
                "http://judge-b",
            }, "Model gateway identities must remain distinct."
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_shutdown_cleans_other_models_after_background_failure(config):
    async def scenario():
        coordinator = await _make(config)

        async def fail():
            raise RuntimeError("Status stream failed.")

        coordinator.models[ROLLOUT_SCOPE].track("status", fail())
        with pytest.raises(RuntimeError, match="Global coordinator shutdown failed"):
            await coordinator.stop_busy_loop()
        assert coordinator.gateway.session.closed, "A failed model must not prevent shared resource cleanup."
        with pytest.raises(RuntimeError, match="shutting down"):
            await _call(coordinator, ROLLOUT_SCOPE, "get_all_instance_ids")

    asyncio.run(scenario())


def test_registration_and_dispatch_reject_unknown_targets(config):
    async def scenario():
        coordinator = await _make(config)
        try:
            with pytest.raises(ValueError, match="already registered"):
                await coordinator.register_reward_model(
                    OmegaConf.create({"reward_model_name": "judge-a"}), "http://new"
                )
            for scope, method, error in (
                (("reward", "unknown"), "get_all_instance_ids", "Unknown coordinator scope"),
                (ROLLOUT_SCOPE, "_sleep_tag", "Private coordinator methods"),
                (ROLLOUT_SCOPE, "config", "Unknown coordinator method"),
            ):
                with pytest.raises(ValueError, match=error):
                    await _call(coordinator, scope, method)
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_scoped_client_survives_serialization_for_executor_and_parameter_server():
    actor = SimpleNamespace(invoke=SimpleNamespace(remote=Mock(return_value="object-ref")))
    client = cloudpickle.loads(cloudpickle.dumps(CoordinatorClient(actor, ("reward", "judge-a"))))
    assert client.exec_command.remote("command", blocking=False) == "object-ref", (
        "Calls must return the Ray reference."
    )
    client.actor.invoke.remote.assert_called_once_with(
        ("reward", "judge-a"), "exec_command", ("command",), {"blocking": False}
    )


@pytest.mark.parametrize("wait_model_sync", [False, True])
def test_sync_corrects_actual_version_and_expands_all_dp_ranks(config, wait_model_sync):
    async def scenario():
        coordinator = await _make(config)
        try:
            policy = coordinator.models[ROLLOUT_SCOPE]
            server = await _add(coordinator, ROLLOUT_SCOPE, "validation", dp_size=2, is_validate=True)
            server.pull_model_for_sync.remote.return_value = 4
            await _call(coordinator, ROLLOUT_SCOPE, "set_ps_model_version", 3)
            await _call(
                coordinator, ROLLOUT_SCOPE, "sync_with_ps", [("validation", 1)], wait_model_sync=wait_model_sync
            )
            if policy.sync_tasks:
                await asyncio.gather(*set(policy.sync_tasks.values()))
            assert policy.versions == {("validation", 0): 4, ("validation", 1): 4}, (
                "Weight pulls apply to complete replicas and must report the actual committed version."
            )
            server.pause_for_sync.remote.assert_awaited_once_with()
            server.resume_after_sync.remote.assert_awaited_once_with()
            assert coordinator.gateway.publish_versions.await_count == 2, (
                "Actual version changes must reach the gateway."
            )
            assert not policy.replicas["validation"].operation_lock.locked(), (
                "Completed sync must release the replica."
            )
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_sync_skips_sleeping_replicas_without_waiting_for_their_locks(config):
    async def scenario():
        coordinator = await _make(config)
        policy = coordinator.models[ROLLOUT_SCOPE]
        try:
            asleep = await _add(coordinator, ROLLOUT_SCOPE, "asleep")
            awake = await _add(coordinator, ROLLOUT_SCOPE, "awake", rank=1)
            asleep.is_sleeping.remote.return_value = True
            lock = policy.replicas["asleep"].operation_lock
            await lock.acquire()
            try:
                await asyncio.wait_for(
                    _call(
                        coordinator, ROLLOUT_SCOPE, "sync_with_ps", [("asleep", 0), ("awake", 0)], wait_model_sync=True
                    ),
                    timeout=1,
                )
            finally:
                lock.release()
            asleep.pause_for_sync.remote.assert_not_awaited()
            awake.pause_for_sync.remote.assert_awaited_once_with()
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_failed_weight_pull_keeps_replica_quarantined(config):
    async def scenario():
        coordinator = await _make(config)
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker-0")
        server.pull_model_for_sync.remote.side_effect = RuntimeError("Transfer failed.")
        with pytest.raises(RuntimeError, match="Weight pull"):
            await _call(coordinator, ROLLOUT_SCOPE, "sync_with_ps", [("worker-0", 0)], wait_model_sync=True)
        server.fail_sync.remote.assert_awaited_once_with()
        server.resume_after_sync.remote.assert_not_awaited()
        model = coordinator.models[ROLLOUT_SCOPE]
        coordinator.gateway.eligibility.assert_awaited_with(model, ["worker-0"], False)
        assert model.replicas["worker-0"].status is InstanceStatus.UNKNOWN, (
            "Failed weight transfers must mark the replica unknown."
        )
        assert await coordinator.lifecycle.awake(model, [("worker-0", 0)]) == [], (
            "Unknown replicas must not participate in later synchronization or routing."
        )
        with pytest.raises(RuntimeError, match="Global coordinator shutdown failed"):
            await coordinator.stop_busy_loop()
        assert coordinator.gateway.session.closed, "Failed sync must still allow global cleanup."

    asyncio.run(scenario())


def test_abort_all_is_deduplicated_per_replica(config):
    async def scenario():
        coordinator = await _make(config)
        try:
            server = await _add(coordinator, ROLLOUT_SCOPE, "worker-0", dp_size=2)
            server.abort_all_requests.remote.return_value = {"aborted_count": 7}
            result = await coordinator.commands.abort(
                coordinator.models[ROLLOUT_SCOPE], [("worker-0", 0), ("worker-0", 1)]
            )
            assert result == 7, "Aborting one replica must not double count requests across DP ranks."
            server.abort_all_requests.remote.assert_awaited_once_with()
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_parameter_server_version_notification_accepts_named_arguments(config):
    async def scenario():
        coordinator = await _make(config)
        try:
            await _add(coordinator, ROLLOUT_SCOPE, "worker-0")
            await _call(
                coordinator,
                ROLLOUT_SCOPE,
                "set_rollout_instance_model_version",
                rollout_instance_id=("worker-0", 0),
                version_tag=7,
            )
            policy = coordinator.models[ROLLOUT_SCOPE]
            assert policy.versions[("worker-0", 0)] == 7, "PS notifications must update the registered replica."
            assert policy.engine_status[("worker-0", 0)].model_version == 7, "Version telemetry must stay consistent."
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_failed_wake_is_retried_before_gateway_eligibility_is_restored(config):
    async def scenario():
        coordinator = await _make(config)
        try:
            model = coordinator.models[("reward", "judge-a")]
            server = await _add(coordinator, model.scope, "worker-0")
            server.is_sleeping.remote.return_value = True
            server.wake_up.remote.side_effect = RuntimeError("Restore failed.")
            with pytest.raises(RuntimeError, match="Wake failed"):
                await coordinator.lifecycle.wake(model, [("worker-0", 0)])
            assert model.replicas["worker-0"].status is InstanceStatus.UNKNOWN, (
                "A failed wake must retain its restoration requirement."
            )
            coordinator.gateway.eligibility.assert_awaited_with(model, [], True)
            server.is_sleeping.remote.return_value = False
            server.wake_up.remote.side_effect = None
            await coordinator.lifecycle.wake(model, [("worker-0", 0)])
            assert server.wake_up.remote.await_count == 2, "A partial restore must retry even when memory is awake."
            coordinator.gateway.eligibility.assert_awaited_with(model, ["worker-0"], True)
            assert model.replicas["worker-0"].status is InstanceStatus.AWAKEN, (
                "Only successful restoration may clear the sleeping state."
            )
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_initial_validation_pull_loads_version_zero_and_excludes_rewards(config):
    async def scenario():
        coordinator = await _make(config)
        try:
            validation = await _add(coordinator, ROLLOUT_SCOPE, "validation", is_validate=True)
            rollout = await _add(coordinator, ROLLOUT_SCOPE, "rollout", rank=1)
            reward = await _add(coordinator, ("reward", "judge-a"), "reward")
            await _call(coordinator, ROLLOUT_SCOPE, "initial_pull_from_ps", tag="validate")
            validation.pull_model.remote.assert_awaited_once_with()
            rollout.pull_model.remote.assert_not_awaited()
            reward.pull_model.remote.assert_not_awaited()
            updates = coordinator.gateway.publish_versions.await_args.args[1]
            assert {item["worker_id"] for item in updates} == {"validation"}, (
                "The initial validation pull must publish only validation instances."
            )
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


def test_backlog_queries_use_their_model_gateway_and_exclude_policy_validation(config):
    async def scenario():
        coordinator = await _make(config)
        try:

            async def get(model, path, params=None):
                if model.dynamic_weights:
                    return [{"is_validate": False}, {"is_validate": True}]
                return {"pending_request_num": 8 if model.scope[1] == "judge-a" else 3}

            coordinator.gateway.get = AsyncMock(side_effect=get)
            results = await asyncio.gather(
                *(_call(coordinator, scope, "get_router_backlog_size") for scope in coordinator.models)
            )
            assert results == [1, 8, 3], "Backlog demand must retain model and validation boundaries."
        finally:
            await coordinator.stop_busy_loop()

    asyncio.run(scenario())


@pytest.mark.parametrize("dp_size", [1, 2])
async def test_validation_cycles_restore_weights_before_routing(config, dp_size):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        server = await _add(coordinator, ROLLOUT_SCOPE, "validation", dp_size=dp_size, is_validate=True)
        rollout = await _add(coordinator, ROLLOUT_SCOPE, "rollout", rank=1)
        state = {"sleeping": False, "version": -1, "routing": True}
        server.is_sleeping.remote.side_effect = lambda: state["sleeping"]

        async def pause(model, replicas):
            assert replicas == ["validation"], "Validation must only pause its own replica."
            state["routing"] = False

        async def sleep(level):
            assert level == 2, "Validation must release all GPU memory."
            assert not state["routing"], "Routing must stop before memory release."
            state.update(sleeping=True, version=-1)

        async def wake():
            assert not state["routing"], "Routing must wait for weight restoration."
            state["sleeping"] = False

        async def pull(version):
            assert not state["sleeping"], "Weight restoration requires allocated memory."
            state["version"] = version
            return version

        async def eligibility(model, replicas, active):
            assert replicas == ["validation"], "Validation must only activate its own replica."
            assert state["version"] == model.ps_version, "Routing requires current policy weights."
            state["routing"] = active

        coordinator.gateway.pause_replicas.side_effect = pause
        coordinator.gateway.eligibility.side_effect = eligibility
        server.nixl_sleep.remote.side_effect = sleep
        server.nixl_wake_up.remote.side_effect = wake
        server.pull_model_for_sync.remote.side_effect = pull
        await _call(coordinator, ROLLOUT_SCOPE, "sleep", "validate")
        for version in (5, 10):
            await _call(coordinator, ROLLOUT_SCOPE, "set_ps_model_version", version)
            await _call(coordinator, ROLLOUT_SCOPE, "wake_up", "validate")
            assert state["routing"], "Restored validation must accept requests."
            assert model.replicas["validation"].status is InstanceStatus.AWAKEN, (
                "Successful wake must clear the recorded sleep state."
            )
            assert all(model.versions[("validation", rank)] == version for rank in range(dp_size)), (
                "Every validation DP rank must advertise restored weights."
            )
            await _call(coordinator, ROLLOUT_SCOPE, "sleep", "validate")
            assert model.replicas["validation"].status is InstanceStatus.ASLEEP, (
                "Every completed sleep must update model state."
            )
        rollout.nixl_sleep.remote.assert_not_awaited()
        rollout.nixl_wake_up.remote.assert_not_awaited()
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("scope", [ROLLOUT_SCOPE, ("reward", "judge-a")])
async def test_sleep_deduplicates_dp_ranks_and_skips_released_memory(config, scope):
    coordinator = await _make(config)
    try:
        model = coordinator.models[scope]
        awake = await _add(coordinator, scope, "awake", dp_size=2)
        asleep = await _add(coordinator, scope, "asleep", rank=1)
        other = await _add(coordinator, scope, "other", rank=2)
        asleep.is_sleeping.remote.return_value = True
        await coordinator.lifecycle.sleep(model, [("awake", 0), ("awake", 1), ("asleep", 0)])
        coordinator.gateway.pause_replicas.assert_awaited_once_with(model, ["asleep", "awake"])
        method = "nixl_sleep" if model.dynamic_weights else "sleep"
        getattr(awake, method).remote.assert_awaited_once_with(level=2)
        getattr(asleep, method).remote.assert_not_awaited()
        getattr(other, method).remote.assert_not_awaited()
        assert all(model.replicas[rid].status is InstanceStatus.ASLEEP for rid in ("awake", "asleep")), (
            "Sleep must record both previously and newly released replicas."
        )
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("operation", ["sleep", "wake_up"])
async def test_failed_lifecycle_keeps_failed_replica_unavailable(config, operation):
    coordinator = await _make(config)
    try:
        scope = ("reward", "judge-a")
        model = coordinator.models[scope]
        failed = await _add(coordinator, scope, "failed")
        sibling = await _add(coordinator, scope, "sibling", rank=1)
        for server in (failed, sibling):
            server.is_sleeping.remote.return_value = operation == "wake_up"
        getattr(failed, operation).remote.side_effect = RuntimeError("Memory transition failed.")
        with pytest.raises(RuntimeError, match="Memory transition failed"):
            await _call(coordinator, scope, operation)
        getattr(sibling, operation).remote.assert_awaited_once()
        if operation == "sleep":
            coordinator.gateway.eligibility.assert_not_awaited()
        else:
            coordinator.gateway.eligibility.assert_awaited_once_with(model, ["sibling"], True)
            assert model.replicas["failed"].status is InstanceStatus.UNKNOWN, (
                "Failed restoration must retain its quarantine."
            )
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("reply", [{"updated": 1}, {"updated": 3}, {"updated": 2, "rejected": 1}])
async def test_gateway_rejects_incomplete_eligibility_acknowledgements(config, reply):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        await _add(coordinator, ROLLOUT_SCOPE, "worker", dp_size=2)
        coordinator.gateway.post = AsyncMock(return_value=reply)
        with pytest.raises(RuntimeError, match="Incomplete gateway eligibility"):
            await GatewayClient.eligibility(coordinator.gateway, model, ["worker"], False)
    finally:
        await coordinator.stop_busy_loop()


async def test_sync_rechecks_logical_sleep_after_acquiring_replica_lock(config):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker")
        lock = model.replicas["worker"].operation_lock
        await lock.acquire()
        transition = asyncio.create_task(_call(coordinator, ROLLOUT_SCOPE, "sync_with_ps", [("worker", 0)]))
        await asyncio.sleep(0.01)
        model.replicas["worker"].status = InstanceStatus.ASLEEP
        lock.release()
        await asyncio.wait_for(transition, timeout=1)
        server.pause_for_sync.remote.assert_not_awaited()
        coordinator.gateway.routing.assert_not_awaited()
        assert not lock.locked(), "Rechecking a sleeping replica must release its operation lock."
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("failure", ["routing", "publish_versions"])
async def test_gateway_failure_settles_weight_pull_before_unlocking(config, failure):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker")
        pulled = asyncio.Future()
        server.pull_model_for_sync.remote = lambda version: pulled
        if failure == "routing":
            coordinator.gateway.routing.side_effect = [None, RuntimeError("Gateway failed.")]
        else:
            coordinator.gateway.publish_versions.side_effect = RuntimeError("Gateway failed.")
        transition = asyncio.create_task(_call(coordinator, ROLLOUT_SCOPE, "sync_with_ps", [("worker", 0)]))
        await asyncio.sleep(0.01)
        lock = model.replicas["worker"].operation_lock
        assert not transition.done() and lock.locked(), "A gateway failure must retain live weight transfers."
        pulled.set_result(3)
        with pytest.raises(RuntimeError, match="Gateway failed"):
            await transition
        server.fail_sync.remote.assert_awaited_once_with()
        assert not lock.locked(), "Settled weight transfers must release the replica lock."
    finally:
        await coordinator.stop_busy_loop()


async def test_background_weight_pull_holds_sleep_reservation(config):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker")
        pulled = asyncio.Future()
        server.pull_model_for_sync.remote = lambda version: pulled
        await _call(coordinator, ROLLOUT_SCOPE, "sync_with_ps", [("worker", 0)])
        sleep = asyncio.create_task(_call(coordinator, ROLLOUT_SCOPE, "sleep"))
        await asyncio.sleep(0.01)
        server.nixl_sleep.remote.assert_not_awaited()
        assert model.replicas["worker"].operation_lock.locked(), "Background pulls must reserve replica memory."
        pulled.set_result(3)
        await asyncio.wait_for(sleep, timeout=1)
        server.nixl_sleep.remote.assert_awaited_once_with(level=2)
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("strategy", ["greedy", "status_based"])
@pytest.mark.parametrize("partial", [False, True])
async def test_background_sync_only_considers_awake_rollout_replicas(config, strategy, partial):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        await _add(coordinator, ROLLOUT_SCOPE, "awake")
        asleep = await _add(coordinator, ROLLOUT_SCOPE, "asleep", rank=1)
        validation = await _add(coordinator, ROLLOUT_SCOPE, "validation", rank=2, is_validate=True)
        model.replicas["asleep"].status = InstanceStatus.ASLEEP
        model.ps_version = 4
        config.psrl.rollout_coordination.sync_and_mig_strategy.method = strategy
        config.psrl.rollout_coordination.partial_rollout.enable = partial
        model.wait_interval = AsyncMock(side_effect=[True, False])
        coordinator.gateway.get = AsyncMock(return_value={"selecting": False})
        coordinator.weights.sync = AsyncMock()
        coordinator.weights.should_sync = AsyncMock(return_value=True)
        await coordinator.weights.run(model)
        coordinator.weights.sync.assert_awaited_once_with(model, [("awake", 0)])
        asleep.is_sleeping.remote.assert_not_awaited()
        validation.is_sleeping.remote.assert_not_awaited()
    finally:
        await coordinator.stop_busy_loop()


async def test_runtime_snapshot_preserves_model_sleep_and_refreshes_idle_stats(config):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        now = datetime.now(timezone.utc).isoformat()
        for rank, rid in enumerate(("awake", "asleep")):
            server = await _add(coordinator, ROLLOUT_SCOPE, rid, rank=rank)
            model.engine_status[(rid, 0)].snapshot["generation_throughput"] = 42.0
            server.get_engine_runtime_snapshots = SimpleNamespace(
                remote=AsyncMock(
                    return_value={
                        0: {
                            "timestamp": now,
                            "is_sleeping": False,
                            "scheduler_stats": {"num_running_reqs": 0, "num_waiting_reqs": 0},
                        }
                    }
                )
            )
        model.ps_version = 7
        model.replicas["asleep"].status = InstanceStatus.ASLEEP
        snapshot = await _call(coordinator, ROLLOUT_SCOPE, "get_instance_runtime_snapshot")
        assert snapshot[("asleep", 0)]["status"] is InstanceStatus.ASLEEP, (
            "Shared engine wake state must not wake a sleeping model."
        )
        assert snapshot[("asleep", 0)]["model_version"] == 7, "Sleeping candidates must use committed PS weights."
        assert snapshot[("awake", 0)]["engine_stats"]["generation_throughput"] == 0, (
            "Idle engine probes must clear stale throughput."
        )
        stats = await _call(coordinator, ROLLOUT_SCOPE, "get_instance_engine_status_snapshot")
        assert stats[("awake", 0)]["timestamp"] == now, "Runtime probes must refresh the router's input snapshot."
    finally:
        await coordinator.stop_busy_loop()


async def test_runtime_probes_run_concurrently_and_preserve_all_dp_ranks(config):
    coordinator = await _make(config)
    try:
        started = set()
        all_started = asyncio.Event()

        async def probe(rid):
            started.add(rid)
            if len(started) == 2:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), timeout=1)
            return {
                rank: {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "is_sleeping": rank == 1,
                    "scheduler_stats": {"num_running_reqs": rank, "num_waiting_reqs": 0},
                }
                for rank in range(2)
            }

        for rank, rid in enumerate(("a", "b")):
            server = await _add(coordinator, ROLLOUT_SCOPE, rid, rank=rank, dp_size=2)
            server.get_engine_runtime_snapshots = SimpleNamespace(remote=lambda rid=rid: probe(rid))
        snapshot = await _call(coordinator, ROLLOUT_SCOPE, "get_instance_runtime_snapshot")
        assert set(snapshot) == {("a", 0), ("a", 1), ("b", 0), ("b", 1)}, (
            "Concurrent probes must preserve every registered DP rank."
        )
        assert snapshot[("a", 1)]["status"] is InstanceStatus.ASLEEP, (
            "Each DP observation must retain its sleep state."
        )
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("engine_sleeping", [False, True])
@pytest.mark.parametrize("scope", [ROLLOUT_SCOPE, ("reward", "judge-a")])
async def test_runtime_snapshot_preserves_unknown_replica_status(config, scope, engine_sleeping):
    coordinator = await _make(config)
    try:
        model = coordinator.models[scope]
        server = await _add(coordinator, scope, "worker")
        model.replicas["worker"].status = InstanceStatus.UNKNOWN
        model.ps_version = 4
        server.get_engine_runtime_snapshots = SimpleNamespace(
            remote=AsyncMock(return_value={0: {"is_sleeping": engine_sleeping}})
        )
        snapshot = await _call(coordinator, scope, "get_instance_runtime_snapshot")
        runtime = cloudpickle.loads(cloudpickle.dumps(snapshot))[("worker", 0)]
        assert runtime["status"] is InstanceStatus.UNKNOWN, (
            "Engine observations and serialization must preserve the authoritative unknown status."
        )
        assert runtime["model_version"] == 0, "Unknown replicas must retain the last confirmed weight version."
        assert runtime["wake_model_version"] == 4, "Recovery must still target the committed PS version."
        assert runtime["engine_stats"] == EngineStats.get_default_snapshot(), (
            "Unknown replicas must not advertise usable inference capacity."
        )
    finally:
        await coordinator.stop_busy_loop()


async def test_failed_reward_command_notifies_waiter_and_keeps_queue_alive(config):
    config.psrl.status_collection.enable = False
    coordinator = await _make(config)
    try:
        scope = ("reward", "judge-a")
        model = coordinator.models[scope]
        server = await _add(coordinator, scope, "worker")
        server.is_sleeping.remote.return_value = True
        server.wake_up.remote.side_effect = [RuntimeError("Restore failed."), None]
        await _call(coordinator, scope, "start_busy_loop")
        for failed in (True, False):
            command = Command(CommandType.WAKE_UP, instance_ids=[("worker", 0)])
            if failed:
                with pytest.raises(RuntimeError, match="Restore failed"):
                    await _call(coordinator, scope, "exec_command", command, timeout=1)
                assert not model.tasks["commands"].done(), "A failed command must leave its consumer running."
            else:
                assert await _call(coordinator, scope, "exec_command", command, timeout=1) is True, (
                    "A later command must run after an earlier failure."
                )
    finally:
        await asyncio.wait_for(coordinator.stop_busy_loop(), timeout=1)


async def test_request_snapshot_checks_versions_without_consuming_validation(config):
    coordinator = await _make(config)
    try:
        queued = [
            {"request_id": "10", "version_tag": -1, "is_validate": False},
            {"request_id": "11", "version_tag": -1, "is_validate": False},
            {"request_id": "12", "version_tag": 2, "is_validate": False},
            {"request_id": "13", "version_tag": -1, "is_validate": True},
        ]
        coordinator.gateway.get = AsyncMock(return_value=queued)
        reserve = AsyncMock(return_value=[[True, False], [False, True]])
        indicator = AsyncMock(return_value=[-1.0, -2.0])
        coordinator.telemetry.ps_manager = SimpleNamespace(
            can_reserve_request=SimpleNamespace(remote=reserve),
            get_reserve_indicator=SimpleNamespace(remote=indicator),
        )
        snapshot = await _call(coordinator, ROLLOUT_SCOPE, "get_router_request_snapshot", [2, 3])
        reserve.assert_awaited_once_with([10, 11], [2, 3], False, [False, False])
        assert [request["request_id"] for request in snapshot] == ["10", "11", "12"], (
            "Demand must preserve training queue order and exclude dedicated validation."
        )
        assert snapshot[0]["reservable_versions"] == [2], "Reservations must retain each candidate version."
        assert snapshot[1]["reservable_versions"] == [3], "Reservations must retain each candidate version."
        assert snapshot[2]["reserve_indicators"] == {2: -1.0, 3: -2.0}, "Inventory indicators must retain versions."
        assert indicator.await_count == 3 and len(queued) == 4, "Snapshot inspection must not consume validation."
    finally:
        await coordinator.stop_busy_loop()


async def test_status_broadcast_retries_after_gateway_failure(config):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        await _add(coordinator, ROLLOUT_SCOPE, "worker")
        model.engine_status[("worker", 0)].snapshot["timestamp"] = datetime.now(timezone.utc).isoformat()
        model.wait_interval = AsyncMock(side_effect=[True, True, False])
        coordinator.gateway.post = AsyncMock(side_effect=[RuntimeError("Gateway failed."), {"updated": 1}])
        await coordinator.telemetry.broadcast(model)
        assert coordinator.gateway.post.await_count == 2, "A transient gateway error must not stop status updates."
    finally:
        await coordinator.stop_busy_loop()


async def test_migrations_wait_for_dispatch_before_aborting_the_next_request(config):
    coordinator = await _make(config)
    try:
        config.psrl.deployment = {"elastic": {"request_migration_timeout_s": 1}}
        scope = ("reward", "judge-a")
        source = await _add(coordinator, scope, "source")
        await _add(coordinator, scope, "destination", rank=1)
        events = []
        source.get_external_request_id = SimpleNamespace(remote=AsyncMock(side_effect=lambda rid: "external-" + rid))
        source.abort_requests.remote.side_effect = lambda ids: events.append(("abort", ids[0])) or 1
        replies = [False, True, True]

        async def get(model, path, params):
            dispatched = replies.pop(0)
            events.append(("dispatch", params["request_id"], dispatched))
            return {"dispatched": dispatched, "completed": False}

        coordinator.gateway.get = get
        coordinator.gateway.post = AsyncMock()
        await _call(
            coordinator,
            scope,
            "execute_request_migrations",
            [
                {"request_id": rid, "source_instance_id": ("source", 0), "destination_instance_id": ("destination", 0)}
                for rid in ("internal-a", "internal-b")
            ],
        )
        assert events.index(("dispatch", "external-internal-a", True)) < events.index(("abort", "internal-b")), (
            "Each migration must settle its dispatch before aborting another request."
        )
    finally:
        await coordinator.stop_busy_loop()


async def test_sync_commands_skip_sleeping_replicas_and_deduplicate_dp_ranks(config):
    config.psrl.status_collection.enable = False
    coordinator = await _make(config)
    try:
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker", dp_size=2)
        await _call(coordinator, ROLLOUT_SCOPE, "start_busy_loop")
        for sleeping in (True, False):
            server.is_sleeping.remote.return_value = sleeping
            command = Command(CommandType.SYNC, instance_ids=[("worker", 0), ("worker", 1)], curr_ps_model_version=3)
            await _call(coordinator, ROLLOUT_SCOPE, "exec_command", command, timeout=1)
            if sleeping:
                server.sync_with_ps.remote.assert_not_awaited()
            else:
                server.sync_with_ps.remote.assert_awaited_once_with(3, pause_generation=True)
    finally:
        await coordinator.stop_busy_loop()


async def test_dispatch_barrier_failure_prevents_memory_release(config):
    coordinator = await _make(config)
    try:
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker")
        coordinator.gateway.pause_replicas.side_effect = RuntimeError("Dispatch did not drain.")
        with pytest.raises(RuntimeError, match="Dispatch did not drain"):
            await _call(coordinator, ROLLOUT_SCOPE, "sleep")
        server.nixl_sleep.remote.assert_not_awaited()
    finally:
        await coordinator.stop_busy_loop()


async def test_colocated_phases_use_scoped_clients_and_preserve_validation(config):
    coordinator = await _make(config)
    try:
        policy = await _add(coordinator, ROLLOUT_SCOPE, "rollout")
        validation = await _add(coordinator, ROLLOUT_SCOPE, "validation", rank=1, is_validate=True)
        reward_scope = ("reward", "judge-a")
        reward = await _add(coordinator, reward_scope, "reward")
        for scope, rid in ((ROLLOUT_SCOPE, "rollout"), (reward_scope, "reward")):
            coordinator.models[scope].replicas[rid].status = InstanceStatus.ASLEEP
        owner = SimpleNamespace(invoke=SimpleNamespace(remote=coordinator.invoke))
        orchestrator = SleepWakeOrchestrator.__ray_metadata__.modified_class(
            "rollout_rm_colocated",
            CoordinatorClient(owner),
            {"judge-a": CoordinatorClient(owner, reward_scope)},
        )
        await orchestrator.start()
        validation.nixl_sleep.remote.reset_mock()
        policy.nixl_sleep.remote.reset_mock()
        reward.sleep.remote.reset_mock()
        await orchestrator.acquire("rollout-a", "rollout")
        policy.nixl_wake_up.remote.assert_awaited_once_with()
        await orchestrator.release("rollout-a")
        await orchestrator.acquire("reward-a", "reward", "judge-a")
        policy.nixl_sleep.remote.assert_awaited_once_with(level=2)
        reward.wake_up.remote.assert_awaited_once_with()
        await orchestrator.release("reward-a")
        await orchestrator.acquire("rollout-b", "rollout")
        reward.sleep.remote.assert_awaited_once_with(level=2)
        assert policy.nixl_wake_up.remote.await_count == 2, "The next rollout phase must restore policy weights."
        validation.nixl_sleep.remote.assert_not_awaited()
        validation.nixl_wake_up.remote.assert_not_awaited()
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("scope", [ROLLOUT_SCOPE, ("reward", "judge-a")])
async def test_failed_wake_cleanup_releases_resident_memory_even_when_logically_unavailable(config, scope):
    coordinator = await _make(config)
    try:
        model = coordinator.models[scope]
        server = await _add(coordinator, scope, "worker")
        model.replicas["worker"].status = InstanceStatus.ASLEEP
        resident = False

        async def restore():
            nonlocal resident
            resident = True
            if not model.dynamic_weights:
                raise RuntimeError("Reward checkpoint restore failed.")

        server.is_sleeping.remote.side_effect = lambda: not resident
        method = server.nixl_wake_up if model.dynamic_weights else server.wake_up
        method.remote.side_effect = restore
        server.pull_model_for_sync.remote.side_effect = RuntimeError("Policy transfer failed.")
        with pytest.raises(RuntimeError, match="Wake failed"):
            await coordinator.lifecycle.wake(model, [("worker", 0)])
        assert resident, "A failed restore may already own GPU memory."
        assert model.replicas["worker"].status is InstanceStatus.UNKNOWN, (
            "Failed restoration must retain an unknown replica status."
        )
        await coordinator.lifecycle.sleep(model, [("worker", 0)])
        release = server.nixl_sleep if model.dynamic_weights else server.sleep
        release.remote.assert_awaited_once_with(level=2, force=True)
        assert model.replicas["worker"].status is InstanceStatus.ASLEEP, (
            "Cleanup may commit sleep only after the server releases its memory."
        )
    finally:
        await coordinator.stop_busy_loop()


async def test_failed_sleep_retries_all_remaining_memory_even_if_executor_reports_sleeping(config):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker")
        server.nixl_sleep.remote.side_effect = [RuntimeError("Partial unmap failed."), None]
        with pytest.raises(RuntimeError, match="Sleep failed"):
            await coordinator.lifecycle.sleep(model, [("worker", 0)])
        server.is_sleeping.remote.return_value = True
        await coordinator.lifecycle.sleep(model, [("worker", 0)])
        server.nixl_sleep.remote.assert_awaited_with(level=2, force=True)
        assert model.replicas["worker"].status is InstanceStatus.ASLEEP, (
            "A partial sleep must be released before another role can own its GPUs."
        )
    finally:
        await coordinator.stop_busy_loop()


async def test_wake_after_partial_sleep_releases_remaining_allocations_before_remapping(config):
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        server = await _add(coordinator, ROLLOUT_SCOPE, "worker")
        server.nixl_sleep.remote.side_effect = RuntimeError("Partial release failed.")
        with pytest.raises(RuntimeError, match="Sleep failed"):
            await coordinator.lifecycle.sleep(model, [("worker", 0)])
        events = []
        server.nixl_sleep.remote.side_effect = lambda **kwargs: events.append(("release", kwargs))
        server.nixl_wake_up.remote.side_effect = lambda: events.append(("wake", {}))
        await coordinator.lifecycle.wake(model, [("worker", 0)])
        assert events == [("release", {"level": 2, "force": True}), ("wake", {})], (
            "Unknown mappings must settle to asleep before wake can restore every memory tag."
        )
        assert model.replicas["worker"].status is InstanceStatus.AWAKEN, (
            "Routing can resume only after recovery and weight restoration complete."
        )
    finally:
        await coordinator.stop_busy_loop()


@pytest.mark.parametrize("ps_mode", ["cpu", "cpu_ref"])
async def test_cpu_policy_lifecycle_restores_weights_without_a_nixl_client(config, ps_mode):
    config.psrl.ps_mode = ps_mode
    coordinator = await _make(config)
    try:
        model = coordinator.models[ROLLOUT_SCOPE]
        server = await _add(coordinator, ROLLOUT_SCOPE, "validation", is_validate=True)
        await coordinator.lifecycle.sleep(model, [("validation", 0)])
        await coordinator.lifecycle.wake(model, [("validation", 0)])
        server.sleep.remote.assert_awaited_once_with(level=2)
        server.wake_up.remote.assert_awaited_once_with()
        server.pull_model_for_sync.remote.assert_awaited_once_with(0)
        server.nixl_sleep.remote.assert_not_awaited()
        server.nixl_wake_up.remote.assert_not_awaited()
    finally:
        await coordinator.stop_busy_loop()
