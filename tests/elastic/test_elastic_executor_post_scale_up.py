"""
Verify post scale up ordering through native coordinator command interfaces.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.elastic.runtime_helpers import _executor

pytestmark = pytest.mark.cpu_test


def _scenario_executor(modules):
    events = []
    rollout = modules.role.Rollout
    reward = modules.role.RewardModel
    executor = _executor(modules, None)
    executor.roles = [(rollout, "model"), (reward, "rm")]
    executor.scaling_policy.enable_request_level_candidate_evaluation = True
    executor.scaling_policy.rebalance_after_scale_up = True
    executor._interrupt_waiting_after_scale_up = AsyncMock()

    def coordinator(role):
        async def execute(command, **kwargs):
            events.append((role, command.type, command.get_args()["instance_ids"]))
            return True

        async def migrate(moves):
            events.append((role, "migrate", moves))

        return SimpleNamespace(
            exec_command=SimpleNamespace(remote=AsyncMock(side_effect=execute)),
            execute_request_migrations=SimpleNamespace(remote=AsyncMock(side_effect=migrate)),
        )

    executor.coordinators = {rollout: {"model": coordinator(rollout)}, reward: {"rm": coordinator(reward)}}
    placements = [
        (rollout, "model", ("replica-1", 0), 0, modules.topology.InstanceStatus.AWAKEN),
        (rollout, "model", ("rollout-spare", 0), 1, modules.topology.InstanceStatus.ASLEEP),
        (reward, "rm", ("reward-0", 0), 0, modules.topology.InstanceStatus.ASLEEP),
    ]
    for role, model, iid, gpu, status in placements:
        executor.instances_status_flags.setdefault(role, {}).setdefault(model, {})[iid] = status
        executor.instances_engine_stats.setdefault(role, {}).setdefault(model, {})[iid] = {}
        executor.instance_pool_ids[(role, model, iid)] = "shared_rollout_pool"
        executor.topology.register(
            role, model, iid, frozenset([modules.topology.GPUSlot("node-0", gpu)]), status=status
        )
    return executor, events


def _scale_up_task(modules, **overrides):
    task = {
        "decision_id": 1,
        "role_name": modules.role.RewardModel,
        "model_name": "rm",
        "num_instances": 1,
        "preferred_instance_ids": [("reward-0", 0)],
        "pre_sleep_other_preferred": [
            {"role_name": modules.role.Rollout, "model_name": "model", "instance_id": ("replica-1", 0)},
        ],
        "pre_wake_other_preferred": None,
        "planned_request_migrations": [
            {
                "request_id": "opaque-request-1",
                "source_instance_id": ("reward-source", 0),
                "destination_instance_id": ("reward-0", 0),
            },
        ],
    }
    task.update(overrides)
    return task


async def _run_tasks(executor, *tasks):
    executor.scale_up_task_queue = asyncio.Queue()
    for task in tasks:
        executor.scale_up_task_queue.put_nowait(task)
    executor.stop_scale_up = False
    executor._decision_pending_action_counts = {task["decision_id"]: 1 for task in tasks}
    executor._decision_execution_in_progress = True
    executor._decision_settled.clear()
    mark_finished = executor._mark_decision_action_finished

    def finish(decision_id):
        mark_finished(decision_id)
        executor.stop_scale_up = executor.scale_up_task_queue.empty()

    executor._mark_decision_action_finished = finish
    await asyncio.wait_for(executor._scale_up_handler_loop(), timeout=1.0)
    assert executor._decision_pending_action_counts == {}
    assert executor._decision_execution_in_progress is False
    assert executor._decision_settled.is_set()


@pytest.mark.parametrize("other_wakes", [None, []])
def test_scale_up_sleeps_conflict_before_wake_and_planned_migration(modules, other_wakes):
    async def scenario():
        executor, events = _scenario_executor(modules)
        task = _scale_up_task(modules, pre_wake_other_preferred=other_wakes)
        await _run_tasks(executor, task)
        assert events == [
            (modules.role.Rollout, modules.command.CommandType.SLEEP, [("replica-1", 0)]),
            (modules.role.RewardModel, modules.command.CommandType.WAKE_UP, [("reward-0", 0)]),
            (modules.role.RewardModel, "migrate", task["planned_request_migrations"]),
        ]
        assert executor.instances_status_flags[modules.role.Rollout]["model"][("replica-1", 0)] is (
            modules.topology.InstanceStatus.ASLEEP
        )
        assert executor.instances_status_flags[modules.role.RewardModel]["rm"][("reward-0", 0)] is (
            modules.topology.InstanceStatus.AWAKEN
        )
        executor._interrupt_waiting_after_scale_up.assert_not_awaited()

    asyncio.run(scenario())


def test_cross_role_wakes_complete_before_planned_migration(modules):
    async def scenario():
        executor, events = _scenario_executor(modules)
        task = _scale_up_task(
            modules,
            pre_wake_other_preferred=[
                {"role_name": modules.role.Rollout, "model_name": "model", "instance_id": ("rollout-spare", 0)},
            ],
        )
        await _run_tasks(executor, task)
        assert events[-2:] == [
            (modules.role.Rollout, modules.command.CommandType.WAKE_UP, [("rollout-spare", 0)]),
            (modules.role.RewardModel, "migrate", task["planned_request_migrations"]),
        ]

    asyncio.run(scenario())


@pytest.mark.parametrize("failure_stage", ["sleep", "primary_wake", "other_wake"])
def test_failed_transition_prevents_migration_and_releases_decision(modules, failure_stage):
    async def scenario():
        executor, _ = _scenario_executor(modules)
        rollout_coord = executor.coordinators[modules.role.Rollout]["model"]
        reward_coord = executor.coordinators[modules.role.RewardModel]["rm"]
        other_wakes = [
            {"role_name": modules.role.Rollout, "model_name": "model", "instance_id": ("rollout-spare", 0)},
        ]
        if failure_stage == "primary_wake":
            reward_coord.exec_command.remote.side_effect = RuntimeError("Wake failed.")
        elif failure_stage == "sleep":
            rollout_coord.exec_command.remote.side_effect = RuntimeError("Sleep failed.")
        else:
            rollout_coord.exec_command.remote.side_effect = [True, RuntimeError("Other wake failed.")]
        await _run_tasks(executor, _scale_up_task(modules, pre_wake_other_preferred=other_wakes))
        reward_coord.execute_request_migrations.remote.assert_not_awaited()
        executor._interrupt_waiting_after_scale_up.assert_not_awaited()
        if failure_stage == "sleep":
            reward_coord.exec_command.remote.assert_not_awaited()
            assert executor.instances_status_flags[modules.role.Rollout]["model"][("replica-1", 0)] is (
                modules.topology.InstanceStatus.AWAKEN
            )
        elif failure_stage == "primary_wake":
            assert executor.instances_status_flags[modules.role.RewardModel]["rm"][("reward-0", 0)] is (
                modules.topology.InstanceStatus.ASLEEP
            )
        else:
            assert executor.instances_status_flags[modules.role.RewardModel]["rm"][("reward-0", 0)] is (
                modules.topology.InstanceStatus.AWAKEN
            )
            assert executor.instances_status_flags[modules.role.Rollout]["model"][("rollout-spare", 0)] is (
                modules.topology.InstanceStatus.ASLEEP
            )

    asyncio.run(scenario())


def test_disabled_rebalance_skips_all_migrations(modules):
    async def scenario():
        executor, events = _scenario_executor(modules)
        executor.scaling_policy.rebalance_after_scale_up = False
        await _run_tasks(executor, _scale_up_task(modules))
        assert len(events) == 2
        executor.coordinators[modules.role.RewardModel]["rm"].execute_request_migrations.remote.assert_not_awaited()
        executor._interrupt_waiting_after_scale_up.assert_not_awaited()

    asyncio.run(scenario())


def test_empty_migration_plan_does_not_abort_waiting_requests(modules):
    async def scenario():
        executor, events = _scenario_executor(modules)
        await _run_tasks(executor, _scale_up_task(modules, planned_request_migrations=[]))
        assert len(events) == 2
        executor.coordinators[modules.role.RewardModel]["rm"].execute_request_migrations.remote.assert_not_awaited()
        executor._interrupt_waiting_after_scale_up.assert_not_awaited()

    asyncio.run(scenario())


def test_non_request_level_policy_uses_coordinator_rebalance(modules):
    async def scenario():
        executor, events = _scenario_executor(modules)
        executor.scaling_policy.enable_request_level_candidate_evaluation = False
        await _run_tasks(executor, _scale_up_task(modules))
        assert len(events) == 2
        executor._interrupt_waiting_after_scale_up.assert_awaited_once_with(
            role_name=modules.role.RewardModel, model_name="rm"
        )
        executor.coordinators[modules.role.RewardModel]["rm"].execute_request_migrations.remote.assert_not_awaited()

    asyncio.run(scenario())


def test_failed_migration_does_not_strand_next_decision(modules):
    async def scenario():
        executor, events = _scenario_executor(modules)
        coordinator = executor.coordinators[modules.role.RewardModel]["rm"]
        coordinator.execute_request_migrations.remote.side_effect = RuntimeError("Migration failed.")
        second_task = _scale_up_task(
            modules,
            decision_id=2,
            role_name=modules.role.Rollout,
            model_name="model",
            preferred_instance_ids=[("rollout-spare", 0)],
            pre_sleep_other_preferred=[],
            planned_request_migrations=[],
        )
        await _run_tasks(executor, _scale_up_task(modules), second_task)
        assert events[-1] == (modules.role.Rollout, modules.command.CommandType.WAKE_UP, [("rollout-spare", 0)])
        assert executor.instances_status_flags[modules.role.RewardModel]["rm"][("reward-0", 0)] is (
            modules.topology.InstanceStatus.AWAKEN
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("direction", ["sleep", "wake"])
def test_role_batch_runs_replicas_concurrently_through_serial_command_queue(modules, direction):
    from tests.elastic.runtime_helpers import _coordinator

    async def scenario():
        coordinator = _coordinator(modules)
        role = modules.role.Rollout
        executor = _executor(modules, None)
        executor._coordinator_command_timeout_s = 1.0
        method = AsyncMock(side_effect=coordinator.exec_command)
        executor.coordinators[role]["model"] = SimpleNamespace(exec_command=SimpleNamespace(remote=method))
        ids = [("replica-1", 0), ("replica-2", 0)]
        initial = (
            modules.topology.InstanceStatus.ASLEEP if direction == "wake" else modules.topology.InstanceStatus.AWAKEN
        )
        targets = []
        for index, iid in enumerate(ids):
            targets.append({"role_name": role, "model_name": "model", "instance_id": iid})
            coordinator.server_handles[iid[0]] = SimpleNamespace(
                is_sleeping=SimpleNamespace(remote=AsyncMock(return_value=direction == "wake")),
            )
            coordinator.instance_ids.add(iid)
            executor.instances_status_flags[role]["model"][iid] = initial
            executor.instances_engine_stats[role]["model"][iid] = {}
            executor.topology.register(
                role, "model", iid, frozenset([modules.topology.GPUSlot("node-0", index)]), initial
            )
        started = set()
        all_started = asyncio.Event()
        release = asyncio.Event()

        async def operation(replica_id):
            started.add(replica_id)
            if len(started) == len(ids):
                all_started.set()
            await release.wait()

        coordinator._do_sleep_instance = operation
        coordinator._do_wake_up_instance = operation
        handler = asyncio.create_task(coordinator._command_handler_loop())
        transition = asyncio.create_task(
            executor._apply_instance_actions(
                executor._scale_up_instances if direction == "wake" else executor._sleep_planned_instances,
                targets,
            )
        )
        try:
            await asyncio.wait_for(all_started.wait(), timeout=0.5)
            assert not transition.done(), "Every replica must start before any replica completes."
            method.assert_awaited_once()
            assert method.call_args.args[0].get_args()["instance_ids"] == ids
            release.set()
            await asyncio.wait_for(transition, timeout=0.5)
            expected = (
                modules.topology.InstanceStatus.AWAKEN
                if direction == "wake"
                else modules.topology.InstanceStatus.ASLEEP
            )
            assert all(executor.instances_status_flags[role]["model"][iid] is expected for iid in ids)
        finally:
            release.set()
            coordinator.stop_command_handler = True
            await asyncio.wait_for(asyncio.gather(transition, handler, return_exceptions=True), timeout=2.0)

    asyncio.run(scenario())


def test_batch_wake_rejects_reserved_training_gpu_before_any_rpc(modules):
    async def scenario():
        executor, _ = _scenario_executor(modules)
        role = modules.role.Rollout
        executor.instance_pool_ids[(role, "model", ("rollout-spare", 0))] = "train_pool"
        executor._await_elastic_coordinator_command = AsyncMock()
        targets = [
            {"role_name": role, "model_name": "model", "instance_id": iid}
            for iid in [("replica-1", 0), ("rollout-spare", 0)]
        ]
        with pytest.raises(RuntimeError, match="reserved"):
            await executor._scale_up_instances(targets)
        executor._await_elastic_coordinator_command.assert_not_awaited()

    asyncio.run(scenario())


def test_batch_sleep_checks_minimum_before_any_replica_sleeps(modules):
    async def scenario():
        executor, _ = _scenario_executor(modules)
        executor.scaling_policy.min_awake_per_role = 1
        executor._await_elastic_coordinator_command = AsyncMock()
        target = {"role_name": modules.role.Rollout, "model_name": "model", "instance_id": ("replica-1", 0)}
        with pytest.raises(RuntimeError, match="minimum awake"):
            await executor._scale_down_instances([target])
        executor._await_elastic_coordinator_command.assert_not_awaited()

    asyncio.run(scenario())


def test_failed_batch_wake_waits_for_siblings_and_opens_only_successful_replica(modules):
    from tests.elastic.runtime_helpers import _coordinator

    async def scenario():
        coordinator = _coordinator(modules)
        coordinator.server_handles["replica-2"] = SimpleNamespace(
            is_sleeping=SimpleNamespace(remote=AsyncMock(return_value=True)),
        )
        release = asyncio.Event()
        successful_started = asyncio.Event()

        async def wake(replica_id):
            if replica_id == "replica-1":
                raise RuntimeError("Weight restoration failed.")
            successful_started.set()
            await release.wait()

        coordinator._do_wake_up_instance = wake
        transition = asyncio.create_task(coordinator._wake_up_replicas([("replica-1", 0), ("replica-2", 0)]))
        try:
            await asyncio.wait_for(successful_started.wait(), timeout=0.5)
            assert not transition.done(), "A failed wake must not release in flight siblings."
            coordinator._set_replica_routing_status.assert_awaited_once()
            assert (
                coordinator._set_replica_routing_status.call_args.args[1]
                is modules.adapter.ReplicaRoutingStatus.PAUSED
            )
            release.set()
            with pytest.raises(RuntimeError, match="Weight restoration failed"):
                await asyncio.wait_for(transition, timeout=0.5)
            assert coordinator._set_replica_routing_status.call_args.args == (
                ["replica-2"],
                modules.adapter.ReplicaRoutingStatus.ACTIVE,
            )
        finally:
            release.set()
            await asyncio.gather(transition, return_exceptions=True)

    asyncio.run(scenario())
