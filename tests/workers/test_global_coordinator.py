"""
Verify independent global coordination preserves model state and replica lifecycles.
"""

import ast
import asyncio
import itertools
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import OmegaConf
from psrl.utils.server.command import Command, CommandType
from psrl.workers import global_coordinator as global_module
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.global_coordination.state import ROLLOUT_SCOPE
from psrl.workers.global_coordinator import CoordinatorClient, GlobalCoordinator
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
    paths = [Path(global_module.__file__), *root.joinpath("global_coordination").glob("*.py")]
    for path in paths:
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
            policy.replicas["validation"].sleeping = True
            reward_model = coordinator.models[("reward", "judge-a")]
            reward_model.replicas["worker-0"].sleeping = True
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
                model.track("status", coordinator.status.collect(model))
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
        coordinator.gateway.eligibility.assert_awaited_with(coordinator.models[ROLLOUT_SCOPE], ["worker-0"], False)
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
            assert model.replicas["worker-0"].sleeping, "A failed wake must retain its restoration requirement."
            coordinator.gateway.eligibility.assert_awaited_with(model, [], True)
            server.is_sleeping.remote.return_value = False
            server.wake_up.remote.side_effect = None
            await coordinator.lifecycle.wake(model, [("worker-0", 0)])
            assert server.wake_up.remote.await_count == 2, "A partial restore must retry even when memory is awake."
            coordinator.gateway.eligibility.assert_awaited_with(model, ["worker-0"], True)
            assert not model.replicas["worker-0"].sleeping, "Only successful restoration may clear the sleeping state."
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
