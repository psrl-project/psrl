"""Exercise elastic commands with the real queue, topology and transition handlers."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import cloudpickle
import pytest

from tests.elastic.runtime_helpers import _coordinator, _executor, _load, _trainer_pool_executor

pytestmark = pytest.mark.cpu_test


def test_trainer_reclaim_waits_for_inflight_decision_and_sleeps_only_train_pool(modules):
    async def scenario():
        coord = _coordinator(modules)
        executor = _executor(modules, coord)
        role = modules.role.Rollout
        train_iid = ("train-replica", 0)
        executor.instances_status_flags[role]["model"][train_iid] = modules.topology.InstanceStatus.AWAKEN
        executor.topology.register(role, "model", train_iid, status=modules.topology.InstanceStatus.AWAKEN)
        executor.instance_pool_ids[(role, "model", train_iid)] = "train_pool"
        executor._trainer_pool_available = True
        executor._decision_settled.clear()
        executor._await_elastic_coordinator_command = AsyncMock()
        reclaim = asyncio.create_task(executor.acquire_trainer_pool())
        await asyncio.sleep(0)
        assert executor._trainer_pool_available is False
        assert executor._trainer_pool_transition is True
        executor._await_elastic_coordinator_command.assert_not_called()
        executor._decision_settled.set()
        await reclaim
        command = executor._await_elastic_coordinator_command.call_args.args[1]
        assert command.get_args()["instance_ids"] == [train_iid]
        assert (
            executor.instances_status_flags[role]["model"][("replica-1", 0)] is modules.topology.InstanceStatus.AWAKEN
        )
        assert executor.instances_status_flags[role]["model"][train_iid] is modules.topology.InstanceStatus.ASLEEP
        assert executor._trainer_pool_transition is False
        await executor.release_trainer_pool()
        assert executor._trainer_pool_available is True

    asyncio.run(scenario())


def test_reclaim_failure_keeps_training_pool_unavailable(modules):
    async def scenario():
        executor = _executor(modules, _coordinator(modules))
        executor.instance_pool_ids[(modules.role.Rollout, "model", ("replica-1", 0))] = "train_pool"
        executor._await_elastic_coordinator_command = AsyncMock(side_effect=RuntimeError("sleep failed"))
        with pytest.raises(RuntimeError, match="sleep failed"):
            await executor.acquire_trainer_pool()
        assert executor._trainer_pool_available is False
        assert (
            executor.instances_status_flags[modules.role.Rollout]["model"][("replica-1", 0)]
            is modules.topology.InstanceStatus.AWAKEN
        )

    asyncio.run(scenario())


def test_reclaim_preserves_snapshot_for_immediate_request_evaluation(modules, monkeypatch):
    async def scenario():
        from omegaconf import OmegaConf
        from psrl.workers.elastic import request_snapshot
        from psrl.workers.elastic.policy_runtime import PolicyIdentityMap
        from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
            RoleCandidatePlan,
            evaluate_role_candidate,
        )

        coord = _coordinator(modules)
        executor = _executor(modules, coord)
        role, iid = modules.role.Rollout, ("replica-1", 0)
        executor.instance_pool_ids[(role, "model", iid)] = "train_pool"
        executor._await_elastic_coordinator_command = AsyncMock(return_value=True)
        await executor.acquire_trainer_pool()
        identities = PolicyIdentityMap()
        identities.register(role, "model", [iid])
        signals = identities.signals(executor._build_instance_signals())
        monkeypatch.setattr(request_snapshot, "_routing_cost_params", lambda *args: (0, 0, 0, 1, 0))
        snapshot = request_snapshot.build_role_snapshot(
            role,
            "model",
            signals,
            executor.instances_engine_stats[role]["model"],
            [],
            identities,
            OmegaConf.create({"method": "throughput_optimal", "cost_model_path": "unused"}),
            4096,
            1,
            1,
            executor.role_model_versions[(role, "model")],
        )
        result = evaluate_role_candidate(snapshot, RoleCandidatePlan(wake_instance_ids=frozenset({0})))
        assert result.unrouted_count == 0, "Reclaim must leave valid inputs for the next decision."
        assert snapshot.instances[0].model_version == 3, "The next wake must advertise committed weights."

    asyncio.run(scenario())


def test_failed_runtime_refresh_publishes_no_partial_cache(modules):
    async def scenario():
        coord = _coordinator(modules)
        executor = _executor(modules, coord)
        role = modules.role.Rollout
        second = ("replica-2", 0)
        executor.instances_status_flags[role]["model"][second] = modules.topology.InstanceStatus.AWAKEN
        coord.get_instance_runtime_snapshot = SimpleNamespace(
            remote=AsyncMock(
                return_value={
                    ("replica-1", 0): {
                        "is_sleeping": True,
                        "engine_stats": {},
                        "model_version": 4,
                        "wake_model_version": 4,
                    },
                    second: {"is_sleeping": True, "engine_stats": None},
                }
            )
        )
        original = executor.instances_engine_stats[role]["model"].copy()
        assert not await executor._sync_engine_status_from_coordinators(), "Invalid observations must reject the tick."
        assert executor.instances_engine_stats[role]["model"] == original, "Refresh must commit all inputs together."
        assert executor.role_model_versions[(role, "model")] == 3, "Failed refresh must retain the committed version."
        assert (
            executor.instances_status_flags[role]["model"][("replica-1", 0)] is modules.topology.InstanceStatus.AWAKEN
        ), "Failed refresh must retain topology state."

    asyncio.run(scenario())


def test_sleeping_role_is_excluded_even_if_shared_engine_reports_awake(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord.sleeping_replica_ids.add("replica-1")
        server = coord.server_handles["replica-1"]
        server.is_sleeping.remote.return_value = False
        assert await coord._get_awake_instance_ids([("replica-1", 0)]) == [], (
            "A sleeping role must not inherit another role's awake state."
        )
        await coord.sync_with_ps([("replica-1", 0)])
        server.pause_for_sync.remote.assert_not_awaited()
        coord._set_routing_loop_running.assert_not_awaited()

    asyncio.run(scenario())


def test_runtime_snapshot_preserves_sleeping_role_despite_shared_engine_state(modules):
    async def scenario():
        coord = modules.status.StatusMixin()
        iid = ("replica-1", 0)
        coord.sleeping_replica_ids = {iid[0]}
        coord.ps_model_version = 4
        coord.instance_to_model_version = {iid: 1}
        coord.instance_to_engine_status = {
            iid: SimpleNamespace(replica_idx=0, model_version=1, snapshot={}),
        }
        coord.server_handles = {
            iid[0]: SimpleNamespace(
                get_engine_runtime_snapshots=SimpleNamespace(
                    remote=AsyncMock(
                        return_value={
                            0: {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "is_sleeping": False,
                                "scheduler_stats": {"num_running_reqs": 2, "num_waiting_reqs": 0},
                            }
                        }
                    ),
                )
            )
        }
        result = await coord.get_instance_runtime_snapshot()
        assert result[iid]["is_sleeping"], "Runtime refresh must preserve the role's acknowledged sleep."
        assert result[iid]["model_version"] == 4, "Sleeping rollout candidates must use committed PS weights."
        assert result[iid]["engine_stats"]["scheduler_stats"]["num_running_reqs"] == 0, (
            "Another role's engine load must not enter the rollout snapshot."
        )

    asyncio.run(scenario())


def test_router_resume_failure_waits_for_live_weight_pull(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord.server_handles["replica-1"].is_sleeping.remote.return_value = False
        pulled = asyncio.Future()
        coord.server_handles["replica-1"].pull_model_for_sync = SimpleNamespace(remote=lambda version: pulled)

        async def routing(running):
            if running:
                raise RuntimeError("Resume failed.")

        coord._set_routing_loop_running = AsyncMock(side_effect=routing)
        task = asyncio.create_task(coord.sync_with_ps([("replica-1", 0)]))
        await asyncio.sleep(0.01)
        assert not task.done(), "Router failures must not release ongoing weight transfers."
        assert coord.replica_operation_locks["replica-1"].locked(), "Weight transfers must retain the lifecycle lock."
        pulled.set_result(3)
        with pytest.raises(RuntimeError, match="Resume failed"):
            await task
        assert not coord.replica_operation_locks["replica-1"].locked(), "Failures must release lifecycle locks."
        coord.server_handles["replica-1"].fail_sync.remote.assert_awaited_once()

    asyncio.run(scenario())


def test_greedy_sync_advances_after_redundant_sync_and_skips_sleeping_role(modules, monkeypatch):
    async def scenario():
        greedy = _load(
            monkeypatch,
            "greedy_sync_runtime_test",
            "psrl/workers/gen/rollout_coordination/sync_and_migrate/greedy.py",
        )
        coord = _coordinator(modules)
        server = coord.server_handles["replica-1"]
        server.is_sleeping.remote.return_value = False
        server.pull_model_for_sync = SimpleNamespace(remote=AsyncMock(return_value=1))
        coord.ps_model_version = 1
        await coord.sync_with_ps([("replica-1", 0)], wait_model_sync=True)
        coord.server_handles["sleeping"] = server
        coord.instance_ids.add(("sleeping", 0))
        coord.sleeping_replica_ids.add("sleeping")
        coord.tag_to_replica_ids = {"rollout": {"replica-1", "sleeping"}}
        coord.ps_model_version = 4
        coord.stop_sync_and_migrate = False
        coord.ready_buffers = set()
        coord.config.psrl.rollout_coordination.sync_and_mig_strategy = SimpleNamespace(
            check_interval_in_ms=1,
            sync=SimpleNamespace(seamless_train_version=0),
            mig=SimpleNamespace(enable=False),
        )
        coord.check_no_activate_tasks = AsyncMock(return_value=True)

        async def sync(instance_ids):
            coord.stop_sync_and_migrate = True
            assert instance_ids == [("replica-1", 0)], "Greedy sync must target only awake rollout roles."

        coord.sync_with_ps = AsyncMock(side_effect=sync)
        await asyncio.wait_for(greedy.GreedySyncMixin._greedy_sync_and_migrate_loop(coord), 1)
        coord.sync_with_ps.assert_awaited_once_with([("replica-1", 0)])

    asyncio.run(scenario())


def test_reserved_training_pool_rejects_wake_before_rpc(modules):
    async def scenario():
        executor = _executor(modules, _coordinator(modules))
        executor.instance_pool_ids[(modules.role.Rollout, "model", ("replica-1", 0))] = "train_pool"
        executor._await_elastic_coordinator_command = AsyncMock()
        with pytest.raises(RuntimeError, match="reserved"):
            await executor._scale_up_instances(
                [{"role_name": modules.role.Rollout, "model_name": "model", "instance_id": ("replica-1", 0)}]
            )
        executor._await_elastic_coordinator_command.assert_not_called()

    asyncio.run(scenario())


def test_trainer_sleep_completes_before_pool_release(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        await executor.release_trainer_pool()
        assert events == ["nixl_sleep"], "Trainer storage must be released before inference can borrow GPUs."
        assert executor._trainer_state == modules.executor.TrainerState.ASLEEP, "Sleep must commit trainer state."
        assert executor._trainer_pool_available is True, "Released trainer storage can host inference."

    asyncio.run(scenario())


def test_sleep_failure_never_releases_trainer_gpus(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor.actor_wg.execute_all_async.side_effect = RuntimeError("NIXL failed.")
        with pytest.raises(RuntimeError, match="NIXL failed"):
            await executor.release_trainer_pool()
        assert events == [], "A failed sleep must not publish resource availability."
        assert executor._trainer_state == modules.executor.TrainerState.AWAKE, "Failed sleep must not commit state."
        assert executor._trainer_pool_available is False, "Failed sleep must reserve trainer storage."

    asyncio.run(scenario())


def test_trainer_wake_restores_weights_after_reclaim(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        await executor.acquire_trainer_pool()
        assert events == ["reclaim", "nixl_wake_up", "nixl_send_local_info_to", "broadcast", "pull_model"], (
            "Reclaim and NIXL reconnection must complete before weight restoration."
        )
        assert executor._trainer_state == modules.executor.TrainerState.AWAKE, "Restored weights permit training."

    asyncio.run(scenario())


def test_trainer_wake_failure_never_restores_memory_before_reclaim(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        executor._reclaim_trainer_pool.side_effect = RuntimeError("Borrower failed.")
        with pytest.raises(RuntimeError, match="Borrower failed"):
            await executor.acquire_trainer_pool()
        assert events == [], "Failed reclaim must prevent trainer memory restoration."

    asyncio.run(scenario())


def test_gpu_registration_tracks_gateway_id_and_each_dp_slice(modules):
    replica = SimpleNamespace(
        gateway_replica_id="worker-10",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=2,
        worker_gpu_locations=(("node", 4), ("node", 5), ("node", 6), ("node", 7)),
    )
    registration = modules.executor.ReplicaRegistration.from_replica(replica, SimpleNamespace(pool_id="train_pool"))
    ids, slots, pools = modules.executor.ElasticExecutor._elastic_instance_registration([registration])
    assert ids == [("worker-10", 0), ("worker-10", 1)], "Each DP instance needs its gateway identity."
    assert [{slot.gpu_id for slot in group} for group in slots] == [{4, 5}, {6, 7}], (
        "DP slices must use disjoint GPUs."
    )
    assert pools == ["train_pool", "train_pool"], "Each instance needs explicit resource ownership."


def test_router_snapshot_checks_all_request_versions_without_consuming_queue(modules):
    async def scenario():
        publisher = modules.status.StatusMixin()
        publisher.ps_model_version = 3
        publisher.instance_to_model_version = {("worker-0", 0): 2}
        queued = [
            {"request_id": "10", "version_tag": -1, "is_validate": False},
            {"request_id": "11", "version_tag": -1, "is_validate": False},
            {"request_id": "12", "version_tag": 2, "is_validate": False},
            {"request_id": "13", "version_tag": -1, "is_validate": True},
        ]
        publisher._gateway_get_json = AsyncMock(return_value=queued)
        reserve = AsyncMock(return_value=[[True, False], [False, True]])
        indicator = AsyncMock(return_value=[-1.0, -2.0])
        publisher.ps_manager = SimpleNamespace(
            can_reserve_request=SimpleNamespace(remote=reserve),
            get_reserve_indicator=SimpleNamespace(remote=indicator),
        )
        result = await publisher.get_router_request_snapshot([2, 3])
        reserve.assert_awaited_once_with([10, 11], [2, 3], False, [False, False])
        assert [request["request_id"] for request in result] == ["10", "11", "12"], (
            "Training rollout demand must remain in native queue order."
        )
        assert result[0]["reservable_versions"] == [2]
        assert result[1]["reservable_versions"] == [3]
        assert result[2]["reserve_indicators"] == {2: -1.0, 3: -2.0}
        assert indicator.await_count == 3, "Dedicated validation must not query elastic inventory."
        assert len(queued) == 4

    asyncio.run(scenario())


def test_policy_identity_translation_preserves_exact_targets(modules):
    from psrl.workers.elastic.policy_runtime import PolicyIdentityMap

    identities = PolicyIdentityMap()
    rollout, rm = modules.role.Rollout, modules.role.RewardModel
    identities.register(rollout, "policy", [("worker-10", 0)])
    identities.register(rm, "reward", [("worker-2", 0)])
    decision = modules.policy.ScalingDecision(
        actions=[
            modules.policy.ScalingAction(
                "scale_up",
                rollout,
                "policy",
                preferred_instance_ids=[0],
                pre_sleep_other_preferred=[{"role_name": rm, "model_name": "reward", "instance_id": 1}],
                planned_request_migrations=[
                    {"request_id": "opaque-id", "source_instance_id": 0, "destination_instance_id": 0}
                ],
            )
        ],
        reason="gain",
        estimated_lambda=1.0,
        role_to_total_mu={},
    )
    translated = identities.decision(decision).actions[0]
    assert translated.preferred_instance_ids == [("worker-10", 0)]
    assert translated.pre_sleep_other_preferred[0]["instance_id"] == ("worker-2", 0)
    assert translated.planned_request_migrations[0]["request_id"] == "opaque-id"


def test_planned_migrations_wait_for_dispatch_before_next_abort(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord.config.psrl.deployment = SimpleNamespace(elastic={})
        events = []
        server = SimpleNamespace(
            get_external_request_id=SimpleNamespace(remote=AsyncMock(side_effect=lambda rid: "external-" + rid)),
            abort_requests=SimpleNamespace(
                remote=AsyncMock(side_effect=lambda ids: events.append(("abort", ids[0])) or 1)
            ),
        )
        coord.server_handles.update({"source": server, "destination": server})
        replies = [{"dispatched": False}, {"dispatched": True}, {"dispatched": True}]

        async def gateway_get(path, params):
            response = replies.pop(0)
            events.append(("dispatch", params["request_id"], response["dispatched"]))
            return response

        coord._gateway_get_json = gateway_get
        moves = [
            {"request_id": rid, "source_instance_id": ("source", 0), "destination_instance_id": ("destination", 0)}
            for rid in ["internal-a", "internal-b"]
        ]
        await coord.execute_request_migrations(moves)
        assert events.index(("dispatch", "external-internal-a", True)) < events.index(("abort", "internal-b"))

    asyncio.run(scenario())


def test_failed_command_notifies_waiter_and_handler_survives(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord._do_wake_up_instance.side_effect = [RuntimeError("weight pull failed"), None]
        loop = asyncio.create_task(coord._command_handler_loop())
        try:
            command = modules.command.Command(
                type=modules.command.CommandType.WAKE_UP,
                instance_ids=[("replica-1", 0)],
            )
            with pytest.raises(RuntimeError, match="weight pull failed"):
                await coord.exec_command(command, timeout=0.5)
            assert not loop.done(), "One failed command must not kill the queue consumer."
            command = modules.command.Command(
                type=modules.command.CommandType.WAKE_UP,
                instance_ids=[("replica-1", 0)],
            )
            assert await coord.exec_command(command, timeout=0.5) is True
            assert coord._set_replica_routing_status.await_args.args[1] is modules.adapter.ReplicaRoutingStatus.ACTIVE
        finally:
            coord.stop_command_handler = True
            await loop

    asyncio.run(scenario())


def test_sleep_completion_updates_flags_and_topology(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord.exec_command = SimpleNamespace(remote=AsyncMock(return_value=True))
        executor = _executor(modules, coord)
        role = modules.role.Rollout
        iid = ("replica-1", 0)
        await executor._scale_down_instances([{"role_name": role, "model_name": "model", "instance_id": iid}])
        assert executor.instances_status_flags[role]["model"][iid] is modules.topology.InstanceStatus.ASLEEP
        assert executor.topology.get_status(role, "model", iid) is modules.topology.InstanceStatus.ASLEEP
        snapshot = executor.instances_engine_stats[role]["model"][iid]
        assert snapshot["elastic_model_version"] == 3, "Sleep must retain an explicit model version."
        assert snapshot["scheduler_stats"]["num_running_reqs"] == 0, "Sleep must clear released engine load."
        assert executor._build_instance_signals()[0].is_awaken is False

    asyncio.run(scenario())


@pytest.mark.parametrize("result", [None, False, {"status": "unexpected"}])
def test_unconfirmed_sleep_does_not_commit_local_state(modules, result):
    async def scenario():
        coord = SimpleNamespace(exec_command=SimpleNamespace(remote=AsyncMock(return_value=result)))
        executor = _executor(modules, coord)
        role = modules.role.Rollout
        iid = ("replica-1", 0)
        with pytest.raises(RuntimeError, match="did not confirm completion"):
            await executor._scale_down_instances([{"role_name": role, "model_name": "model", "instance_id": iid}])
        assert executor.instances_status_flags[role]["model"][iid] is modules.topology.InstanceStatus.AWAKEN

    asyncio.run(scenario())


def test_timeout_tracks_original_command_until_completion(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord.exec_command = SimpleNamespace(remote=AsyncMock(return_value={"status": "RUNNING", "command_id": 7}))
        coord.get_command_status = SimpleNamespace(
            remote=AsyncMock(side_effect=[{"status": "RUNNING"}, {"status": "SUCCEEDED"}]),
        )
        coord.synchronize_command = SimpleNamespace(remote=AsyncMock(return_value=True))
        executor = _executor(modules, coord)
        command = modules.command.Command(type=modules.command.CommandType.SLEEP, instance_ids=[("replica-1", 0)])
        assert await executor._await_elastic_coordinator_command(coord, command, stage="test") is True
        coord.exec_command.remote.assert_awaited_once()
        coord.synchronize_command.remote.assert_awaited_once_with(7)

    asyncio.run(scenario())


def test_timed_out_failure_propagates_from_original_command(modules):
    async def scenario():
        extension = modules.command.CommandExtension()
        command = modules.command.Command(type=modules.command.CommandType.SLEEP)
        pending = await extension.exec_command(command, timeout=0.001)
        extension._start_command(pending["command_id"])
        extension._fail_command(pending["command_id"], ValueError("deregister failed"))
        assert extension.get_command_status(0)["status"] == "FAILED"
        with pytest.raises(RuntimeError, match="deregister failed"):
            await extension.synchronize_command(0)
        assert extension.command_queue.qsize() == 1, "Timeout must not enqueue a replacement command."

    asyncio.run(scenario())


def test_sibling_transition_is_settled_before_action_failure(modules):
    async def scenario():
        executor = _executor(modules, None)
        gate = asyncio.Event()

        async def operation(instances):
            if instances[0]["fail"]:
                raise RuntimeError("sleep failed")
            await gate.wait()

        instances = [
            {"role_name": modules.role.Rollout, "model_name": "model", "fail": True},
            {"role_name": modules.role.RewardModel, "model_name": "reward", "fail": False},
        ]
        task = asyncio.create_task(executor._apply_instance_actions(operation, instances))
        await asyncio.sleep(0.01)
        assert not task.done(), "Failed siblings must not release a decision while another transition runs."
        gate.set()
        with pytest.raises(RuntimeError, match="sleep failed"):
            await task

    asyncio.run(scenario())


@pytest.mark.parametrize("direction", ["up", "down"])
def test_scale_handler_survives_failure_and_releases_decision(modules, direction):
    async def scenario():
        executor = _executor(modules, None)
        task_queue = asyncio.Queue()
        task_queue.put_nowait({"decision_id": 1, "role_name": modules.role.Rollout, "model_name": "model"})
        task_queue.put_nowait({"decision_id": 2, "role_name": modules.role.Rollout, "model_name": "model"})
        executor._decision_pending_action_counts = {1: 1, 2: 1}
        executor._decision_execution_in_progress = True
        setattr(executor, f"scale_{direction}_task_queue", task_queue)
        setattr(executor, f"stop_scale_{direction}", False)
        completed = []

        async def operation(instance):
            completed.append(instance)
            if len(completed) == 1:
                raise RuntimeError("transition failed")
            setattr(executor, f"stop_scale_{direction}", True)

        executor._scale_up_instances = operation
        executor._scale_down_instances = operation
        executor._find_instances_to_scaled_down_for_other_roles = lambda _: []
        target = {"role_name": modules.role.Rollout, "model_name": "model", "instance_id": ("replica-1", 0)}
        executor._find_instances_to_scaled_up = lambda _: [target]
        executor._find_instances_to_scaled_down_in_role = lambda _: [target]
        executor._interrupt_waiting_after_scale_up = AsyncMock()
        await getattr(executor, f"_scale_{direction}_handler_loop")()
        assert len(completed) == 2, "The consumer must process the next action after a failed action."
        assert executor._decision_pending_action_counts == {}
        assert executor._decision_execution_in_progress is False

    asyncio.run(scenario())


def test_failed_pre_sleep_prevents_wake(modules):
    async def scenario():
        executor = _executor(modules, None)
        executor.scale_up_task_queue = asyncio.Queue()
        executor.scale_up_task_queue.put_nowait({"decision_id": 1})
        executor.stop_scale_up = False
        executor._decision_pending_action_counts = {1: 1}
        executor._decision_execution_in_progress = True
        target = {"role_name": modules.role.Rollout, "model_name": "model", "instance_id": ("replica-1", 0)}
        executor._find_instances_to_scaled_down_for_other_roles = lambda _: [target]

        async def fail_sleep(_):
            executor.stop_scale_up = True
            raise RuntimeError("memory release failed")

        executor._scale_down_instances = fail_sleep
        executor._scale_up_instances = AsyncMock()
        executor._find_instances_to_scaled_up = lambda _: [target]
        await executor._scale_up_handler_loop()
        executor._scale_up_instances.assert_not_awaited()
        assert executor._decision_execution_in_progress is False

    asyncio.run(scenario())


@pytest.mark.parametrize("in_progress", [False, True])
def test_runtime_snapshot_reconciles_only_settled_state(modules, in_progress):
    async def scenario():
        iid = ("replica-1", 0)
        coord = SimpleNamespace(
            get_instance_runtime_snapshot=SimpleNamespace(
                remote=AsyncMock(
                    return_value={
                        iid: {
                            "is_sleeping": True,
                            "engine_stats": {},
                            "model_version": 3,
                            "wake_model_version": 3,
                        }
                    }
                ),
            ),
        )
        executor = _executor(modules, coord)
        executor._decision_execution_in_progress = in_progress
        assert await executor._sync_engine_status_from_coordinators() is True
        expected = modules.topology.InstanceStatus.AWAKEN if in_progress else modules.topology.InstanceStatus.ASLEEP
        assert executor.instances_status_flags[modules.role.Rollout]["model"][iid] is expected
        assert executor.topology.get_status(modules.role.Rollout, "model", iid) is expected

    asyncio.run(scenario())


@pytest.mark.parametrize("command_type", ["SLEEP", "WAKE_UP"])
def test_runtime_refresh_cannot_overwrite_a_transition_completed_during_fetch(modules, command_type):
    async def scenario():
        executor = _executor(modules, None)
        role = modules.role.Rollout
        iid = ("replica-1", 0)
        sleeping = command_type == "WAKE_UP"
        status = modules.topology.InstanceStatus.ASLEEP if sleeping else modules.topology.InstanceStatus.AWAKEN
        executor.instances_status_flags[role]["model"][iid] = status
        executor.topology.set_status(role, "model", iid, status)
        executor.instances_engine_stats[role]["model"][iid] = {"sentinel": "committed"}
        sampled = asyncio.Event()
        complete = asyncio.Event()

        async def snapshot():
            observed = {
                iid: {
                    "is_sleeping": sleeping,
                    "engine_stats": {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "scheduler_stats": {},
                    },
                    "model_version": 3,
                    "wake_model_version": 3,
                }
            }
            sampled.set()
            await complete.wait()
            return observed

        coord = SimpleNamespace(
            exec_command=SimpleNamespace(remote=AsyncMock(return_value=True)),
            get_instance_runtime_snapshot=SimpleNamespace(remote=snapshot),
        )
        executor.coordinators[role]["model"] = coord
        refresh = asyncio.create_task(executor._sync_engine_status_from_coordinators())
        await sampled.wait()
        await executor._transition_role_instances(
            getattr(modules.command.CommandType, command_type),
            [{"role_name": role, "model_name": "model", "instance_id": iid}],
        )
        committed_stats = executor.instances_engine_stats[role]["model"][iid]
        complete.set()

        assert await refresh is False, "A snapshot spanning a memory transition must reject the monitor tick."
        expected = modules.topology.InstanceStatus.AWAKEN if sleeping else modules.topology.InstanceStatus.ASLEEP
        assert executor.instances_status_flags[role]["model"][iid] is expected, "The transition must remain committed."
        assert executor.topology.get_status(role, "model", iid) is expected, "GPU ownership must remain committed."
        assert executor.instances_engine_stats[role]["model"][iid] is committed_stats, (
            "Stale metrics must be discarded."
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("fetch_stage", ["_sync_router_backlog_from_coordinators", "_sync_trainer_waiting_hint"])
def test_monitor_rejects_signals_when_transition_settles_after_runtime_fetch(modules, fetch_stage):
    async def scenario():
        executor = _executor(modules, None)
        executor.stop_monitor = False
        executor._decision_execution_in_progress = True
        executor._decision_pending_action_counts = {1: 1}
        executor._sync_engine_status_from_coordinators = AsyncMock(return_value=True)
        executor._sync_router_backlog_from_coordinators = AsyncMock(return_value=True)
        executor._sync_trainer_waiting_hint = AsyncMock()
        executor._fetch_request_level_snapshots = AsyncMock(return_value=None)
        executor.trainer_waiting_hint = {}
        executor.scaling_policy.decide = Mock(return_value=SimpleNamespace(actions=[], reason="idle"))

        async def settle():
            executor._mark_decision_action_finished(1)
            executor.stop_monitor = True
            return True

        setattr(executor, fetch_stage, settle)
        await executor._monitor_loop()
        executor.scaling_policy.decide.assert_not_called()

    asyncio.run(scenario())


def test_failed_router_fetch_is_not_treated_as_empty_queue(modules):
    async def scenario():
        coord = SimpleNamespace(get_router_backlog_size=SimpleNamespace(remote=AsyncMock(side_effect=TimeoutError())))
        executor = _executor(modules, coord)
        executor.router_backlog_by_role = {modules.role.Rollout: 19}
        assert await executor._sync_router_backlog_from_coordinators() is False
        assert executor.router_backlog_by_role == {modules.role.Rollout: 19}

    asyncio.run(scenario())


def test_missing_runtime_instance_invalidates_monitor_input(modules):
    async def scenario():
        coord = SimpleNamespace(get_instance_runtime_snapshot=SimpleNamespace(remote=AsyncMock(return_value={})))
        executor = _executor(modules, coord)
        assert await executor._sync_engine_status_from_coordinators() is False

    asyncio.run(scenario())


def test_monitor_does_not_decide_after_incomplete_fetch(modules):
    async def scenario():
        executor = _executor(modules, None)
        executor.stop_monitor = False

        async def sync_status():
            executor.stop_monitor = True
            return False

        executor._sync_engine_status_from_coordinators = sync_status
        executor._sync_router_backlog_from_coordinators = AsyncMock(return_value=True)
        executor.scaling_policy.decide = AsyncMock()
        await executor._monitor_loop()
        executor.scaling_policy.decide.assert_not_called()

    asyncio.run(scenario())


@pytest.mark.parametrize("has_action", [False, True])
def test_executor_info_logs_only_accepted_decisions_and_completion(modules, monkeypatch, has_action):
    async def scenario():
        executor = _executor(modules, None)
        executor.stop_monitor = False
        executor._next_decision_id = 1
        executor._decision_pending_action_counts = {}
        executor._execution_in_progress_stall_ticks = 0
        executor.scale_up_task_queue = asyncio.Queue()
        executor.scale_down_task_queue = asyncio.Queue()
        executor.trainer_waiting_hint = {}
        logger = Mock()
        monkeypatch.setattr(modules.executor, "psrl_logger", logger)
        monkeypatch.setattr(modules.executor, "monitor_logger", Mock())

        async def sync_status():
            executor.stop_monitor = True
            return True

        action = SimpleNamespace(
            action_type="scale_up",
            role_name=modules.role.Rollout,
            model_name="model",
            num_instances=1,
            preferred_instance_ids=[("replica-1", 0)],
            reason="Scale required.",
            pre_sleep_other_preferred=[],
            pre_wake_other_preferred=[],
            planned_request_migrations=[],
        )
        decision = SimpleNamespace(
            actions=[action] if has_action else [],
            reason="Policy evaluated.",
            estimated_lambda=1.0,
            role_to_total_mu={"Rollout": 1.0},
        )
        executor.scaling_policy.decide = Mock(return_value=decision)
        executor._sync_engine_status_from_coordinators = sync_status
        executor._sync_router_backlog_from_coordinators = AsyncMock(return_value=True)
        executor._sync_trainer_waiting_hint = AsyncMock()
        executor._fetch_request_level_snapshots = AsyncMock(return_value=None)
        await executor._monitor_loop()
        logger.exception.assert_not_called()
        executor.scaling_policy.decide.assert_called_once()
        assert logger.info.call_count == int(has_action), "INFO requires an executable policy decision."
        if has_action:
            assert "Policy decision accepted" in logger.info.call_args.args[0], "Log the accepted decision."
            executor._mark_decision_action_finished(1)
            assert logger.info.call_count == 2, "Decision completion must also be logged."

    asyncio.run(scenario())


@pytest.mark.parametrize("logging_level", [None, "INFO", "WARNING"])
def test_serialized_actor_configures_worker_loggers_and_writes_files(modules, monkeypatch, tmp_path, logging_level):
    if logging_level is None:
        monkeypatch.delenv("PSRL_LOGGING_LEVEL", raising=False)
    else:
        monkeypatch.setenv("PSRL_LOGGING_LEVEL", logging_level)
    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)
    handlers = _load(monkeypatch, "elastic_runtime_logging_test", "psrl/utils/logger/ray_logger.py")
    monkeypatch.setattr(modules.executor, "DualOutputHandler", handlers.DualOutputHandler)
    monkeypatch.setattr(modules.executor, "FileOnlyHandler", handlers.FileOnlyHandler)
    monkeypatch.setattr(
        modules.executor, "create_scaling_policy", lambda **kwargs: SimpleNamespace(monitor_interval_ms=1)
    )

    cloudpickle.register_pickle_by_value(modules.executor)
    try:
        serialized = cloudpickle.dumps(modules.executor.ElasticExecutor)
    finally:
        cloudpickle.unregister_pickle_by_value(modules.executor)
    for logger in (modules.executor.psrl_logger, modules.executor.monitor_logger):
        monkeypatch.delitem(logging.Logger.manager.loggerDict, logger.name)
    actor_cls = cloudpickle.loads(serialized)
    worker_globals = actor_cls.__init__.__globals__
    worker_loggers = [worker_globals[name] for name in ("psrl_logger", "monitor_logger")]
    assert all(logger.getEffectiveLevel() == logging.WARNING for logger in worker_loggers)

    executor = actor_cls(
        config=SimpleNamespace(psrl=SimpleNamespace(logging_path=str(tmp_path))),
        roles=[(modules.role.Rollout, "model")],
        coordinators={},
    )
    try:

        async def scenario():
            signal = modules.policy.InstanceSignal(
                role_name=modules.role.Rollout,
                model_name="model",
                instance_id=("replica-1", 0),
                is_awaken=True,
                kv_cache_utilization=0.1,
                running_queue_num=3,
                waiting_queue_num=2,
                generation_throughput=4.0,
                total_token_num=5,
            )
            executor._maybe_log_instance_signals([signal], engine_ready=False, router_ready=True)
            command = modules.command.Command(type=modules.command.CommandType.SLEEP)
            coordinator = SimpleNamespace(exec_command=SimpleNamespace(remote=AsyncMock(return_value=True)))
            await executor._await_elastic_coordinator_command(coordinator, command, stage="SLEEP")

        asyncio.run(scenario())
        monitor_text = (tmp_path / "ElasticMonitor.log").read_text()
        assert "Elastic monitor initialized" in monitor_text
        assert "engine_ready=False, router_ready=True" in monitor_text
        assert "'running': 3" in monitor_text
        executor_text = (tmp_path / "ElasticExecutor.log").read_text()
        assert not executor_text, "Routine monitoring and command tracing must stay below INFO."
    finally:
        for logger in worker_loggers:
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()


def test_monitor_reports_incomplete_input_before_skipping_policy(modules):
    async def scenario():
        executor = _executor(modules, None)
        executor.stop_monitor = False

        async def sync_status():
            executor.stop_monitor = True
            return False

        executor._sync_engine_status_from_coordinators = sync_status
        executor._sync_router_backlog_from_coordinators = AsyncMock(return_value=True)
        executor.scaling_policy.decide = AsyncMock()
        records = []
        executor._maybe_log_instance_signals = lambda signals, **health: records.append((signals, health))
        await executor._monitor_loop()
        assert len(records) == 1
        assert len(records[0][0]) == 1
        assert records[0][1] == {"engine_ready": False, "router_ready": True}
        executor.scaling_policy.decide.assert_not_called()

    asyncio.run(scenario())


def test_router_status_publisher_retries_after_http_failure(modules):
    async def scenario():
        publisher = modules.status.StatusMixin()
        publisher.stop_sync_status_to_router = False
        publisher.config = SimpleNamespace(
            psrl=SimpleNamespace(status_collection=SimpleNamespace(coordinator_sync_interval_in_ms=1)),
        )
        publisher.instance_to_engine_status = {
            ("replica-1", 0): SimpleNamespace(snapshot={"timestamp": datetime.now(timezone.utc).isoformat()}),
        }
        attempts = []

        async def publish(*args, **kwargs):
            attempts.append(kwargs["payload"])
            if len(attempts) == 1:
                raise RuntimeError("temporary HTTP failure")
            publisher.stop_sync_status_to_router = True
            return {"updated": 1, "rejected": 0, "stale_ignored": 0}

        publisher._gateway_post_json = publish
        await publisher._sync_status_to_router()
        assert len(attempts) == 2, "Transient HTTP errors must not stop scheduler updates."

    asyncio.run(scenario())


def test_sleep_waits_for_existing_replica_weight_operation(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord.server_handles["replica-1"].is_sleeping.remote.return_value = False
        operations = await coord._acquire_replica_operations(["replica-1"])
        task = asyncio.create_task(coord._sleep_replicas([("replica-1", 0)]))
        await asyncio.sleep(0.01)
        coord._do_sleep_instance.assert_not_awaited()
        await operations.aclose()
        await task
        coord._do_sleep_instance.assert_awaited_once_with("replica-1")

    asyncio.run(scenario())


def test_zero_initial_target_selects_no_instances(modules):
    topology = modules.topology.ClusterTopology()
    role = modules.role.Rollout
    topology.register(role, "model", ("replica-1", 0))
    assert topology.select_non_conflicting_awake_ids(role, "model", [("replica-1", 0)], 0, 0) == []


def test_background_weight_pull_holds_sleep_reservation(modules):
    async def scenario():
        coord = _coordinator(modules)
        server = coord.server_handles["replica-1"]
        server.is_sleeping.remote.return_value = False
        pulled = asyncio.Future()
        server.pull_model_for_sync = SimpleNamespace(remote=lambda version: pulled)
        await coord.sync_with_ps([("replica-1", 0)], wait_model_sync=False)
        assert coord.replica_operation_locks["replica-1"].locked()
        sleep = asyncio.create_task(coord._sleep_replicas([("replica-1", 0)]))
        await asyncio.sleep(0.01)
        coord._do_sleep_instance.assert_not_awaited()
        pulled.set_result(3)
        await asyncio.gather(*coord.model_sync_tasks)
        await sleep
        coord._do_sleep_instance.assert_awaited_once_with("replica-1")
        assert not coord.replica_operation_locks["replica-1"].locked()

    asyncio.run(scenario())


def test_publish_failure_settles_started_pull_before_releasing_lock(modules):
    async def scenario():
        coord = _coordinator(modules)
        server = coord.server_handles["replica-1"]
        server.is_sleeping.remote.return_value = False
        pulled = asyncio.Future()
        server.pull_model_for_sync = SimpleNamespace(remote=lambda version: pulled)
        coord._publish_weight_version_updates.side_effect = RuntimeError("publish failed")
        task = asyncio.create_task(coord.sync_with_ps([("replica-1", 0)]))
        await asyncio.sleep(0.01)
        assert not task.done(), "A router error must not release a live weight pull."
        assert coord.replica_operation_locks["replica-1"].locked()
        pulled.set_result(3)
        with pytest.raises(RuntimeError, match="publish failed"):
            await task
        assert not coord.replica_operation_locks["replica-1"].locked()
        server.fail_sync.remote.assert_awaited_once()

    asyncio.run(scenario())


def test_sync_command_skips_sleeping_replica_and_deduplicates_dp_ranks(modules):
    async def scenario():
        coord = _coordinator(modules)
        server = coord.server_handles["replica-1"]
        command = modules.command.Command(
            type=modules.command.CommandType.SYNC,
            instance_ids=[("replica-1", 0), ("replica-1", 1)],
            curr_ps_model_version=3,
        )
        cid = await coord.exec_command(command, blocking=False)
        await coord._execute_queued_command(command)
        await coord.synchronize_command(cid)
        server.sync_with_ps.remote.assert_not_awaited()
        server.is_sleeping.remote.assert_awaited_once()
        server.is_sleeping.remote.reset_mock()
        server.is_sleeping.remote.return_value = False
        command = modules.command.Command(
            type=modules.command.CommandType.SYNC,
            instance_ids=[("replica-1", 0), ("replica-1", 1)],
            curr_ps_model_version=3,
        )
        cid = await coord.exec_command(command, blocking=False)
        await coord._execute_queued_command(command)
        await coord.synchronize_command(cid)
        server.sync_with_ps.remote.assert_awaited_once_with(3, pause_generation=True)
        assert server.is_sleeping.remote.await_count == 2

    asyncio.run(scenario())


def _register_slot(modules, executor, role, iid, gpu_ids, status):
    executor.instances_status_flags.setdefault(role, {}).setdefault("model", {})[iid] = status
    executor.instances_engine_stats.setdefault(role, {}).setdefault("model", {})[iid] = {}
    executor.topology.register(
        role,
        "model",
        iid,
        gpu_slots=frozenset(modules.topology.GPUSlot("node", gpu_id) for gpu_id in gpu_ids),
        status=status,
    )


def test_pre_sleep_targets_every_conflicting_gpu_for_tp_four_wake(modules):
    executor = _executor(modules, None)
    rollout, reward = modules.role.Rollout, modules.role.RewardModel
    executor.roles = [(rollout, "model"), (reward, "model")]
    for index in range(4):
        _register_slot(
            modules,
            executor,
            rollout,
            (f"replica-{index}", 0),
            [index],
            modules.topology.InstanceStatus.AWAKEN,
        )
    _register_slot(modules, executor, reward, ("rm", 0), range(4), modules.topology.InstanceStatus.ASLEEP)
    victims = executor._find_instances_to_scaled_down_for_other_roles(
        {
            "role_name": reward,
            "model_name": "model",
            "preferred_instance_ids": [("rm", 0)],
            "num_instances": 1,
        }
    )
    assert len(victims) == 4, "One TP4 wake must release all four conflicting TP1 instances."


def test_free_gpu_wake_does_not_sleep_unrelated_instance(modules):
    executor = _executor(modules, None)
    rollout, reward = modules.role.Rollout, modules.role.RewardModel
    executor.roles = [(rollout, "model"), (reward, "model")]
    _register_slot(modules, executor, rollout, ("replica-1", 0), [0], modules.topology.InstanceStatus.AWAKEN)
    _register_slot(modules, executor, reward, ("rm", 0), [1], modules.topology.InstanceStatus.ASLEEP)
    assert (
        executor._find_instances_to_scaled_down_for_other_roles(
            {
                "role_name": reward,
                "model_name": "model",
                "preferred_instance_ids": [("rm", 0)],
            }
        )
        == []
    )


def test_unavailable_preferred_wake_is_not_replaced(modules):
    executor = _executor(modules, None)
    _register_slot(modules, executor, modules.role.Rollout, ("other", 0), [2], modules.topology.InstanceStatus.ASLEEP)
    assert (
        executor._find_instances_to_scaled_up(
            {
                "role_name": modules.role.Rollout,
                "model_name": "model",
                "preferred_instance_ids": [("missing", 0)],
            }
        )
        == []
    )


def test_preemption_rejects_minimum_awake_violation_before_sleep(modules):
    executor = _executor(modules, None)
    rollout, reward = modules.role.Rollout, modules.role.RewardModel
    executor.scaling_policy.min_awake_per_role = 1
    executor.roles = [(rollout, "model"), (reward, "model")]
    _register_slot(modules, executor, rollout, ("replica-1", 0), [0], modules.topology.InstanceStatus.AWAKEN)
    _register_slot(modules, executor, reward, ("rm", 0), [0], modules.topology.InstanceStatus.ASLEEP)
    with pytest.raises(RuntimeError, match="minimum awake count"):
        executor._find_instances_to_scaled_down_for_other_roles(
            {
                "role_name": reward,
                "model_name": "model",
                "preferred_instance_ids": [("rm", 0)],
            }
        )


@pytest.mark.parametrize(
    "timestamp",
    [None, "invalid", (datetime.now(timezone.utc) - timedelta(seconds=60)).isoformat()],
)
def test_stale_awake_metrics_invalidate_decision_inputs(modules, timestamp, caplog):
    async def scenario():
        coord = SimpleNamespace(
            get_instance_runtime_snapshot=SimpleNamespace(
                remote=AsyncMock(
                    return_value={
                        ("replica-1", 0): {
                            "is_sleeping": False,
                            "engine_stats": {"timestamp": timestamp},
                            "model_version": 3,
                            "wake_model_version": 3,
                        },
                    }
                )
            )
        )
        executor = _executor(modules, coord)
        assert await executor._sync_engine_status_from_coordinators() is False

    asyncio.run(scenario())
    assert "Skipping elastic decision for stale awake instance" not in caplog.text


def test_timezone_aware_fresh_timestamp_is_accepted(modules):
    snapshot = {"timestamp": datetime.now(timezone.utc).isoformat()}
    assert modules.policy.ScalingPolicy._is_snapshot_staled(snapshot) is False


def test_idle_engine_runtime_probe_refreshes_executor_and_router_inputs(modules, monkeypatch):
    async def scenario():
        iid = ("replica-1", 0)
        now = datetime.now(timezone.utc).isoformat()
        scheduler_stats = {
            "num_running_reqs": 0,
            "num_waiting_reqs": 0,
            "req_id_to_prompt_token_num": {},
            "req_id_to_response_token_num": {},
            "req_id_in_waiting": [],
            "kv_cache_usage": 0.0,
        }
        coord = modules.status.StatusMixin()
        coord.sleeping_replica_ids = set()
        coord.server_handles = {
            iid[0]: SimpleNamespace(
                get_engine_runtime_snapshots=SimpleNamespace(
                    remote=AsyncMock(
                        return_value={
                            0: {
                                "timestamp": now,
                                "scheduler_stats": scheduler_stats,
                                "is_sleeping": False,
                            }
                        }
                    ),
                )
            ),
        }
        coord.instance_to_engine_status = {
            iid: SimpleNamespace(
                replica_idx=0,
                model_version=3,
                snapshot={"timestamp": None, "generation_throughput": 42.0},
            )
        }
        coord.instance_to_model_version = {iid: 3}
        coord.ps_model_version = 3
        executor = _executor(
            modules,
            SimpleNamespace(
                get_instance_runtime_snapshot=SimpleNamespace(remote=coord.get_instance_runtime_snapshot),
            ),
        )
        assert await executor._sync_engine_status_from_coordinators() is True
        snapshot = executor.instances_engine_stats[modules.role.Rollout]["model"][iid]
        assert snapshot["timestamp"] == now
        assert snapshot["scheduler_stats"] == scheduler_stats
        assert snapshot["generation_throughput"] == 0.0
        assert coord.get_instance_engine_status_snapshot()[iid]["timestamp"] == now

    asyncio.run(scenario())


def test_runtime_probes_run_in_parallel_and_preserve_all_dp_ranks(modules, monkeypatch):
    async def scenario():
        started = set()
        all_started = asyncio.Event()

        async def probe(replica_id):
            started.add(replica_id)
            if len(started) == 2:
                all_started.set()
            await asyncio.wait_for(all_started.wait(), 1.0)
            return {
                rank: {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "is_sleeping": rank == 1,
                    "scheduler_stats": {"num_running_reqs": rank, "num_waiting_reqs": 0},
                }
                for rank in range(2)
            }

        coord = modules.status.StatusMixin()
        coord.sleeping_replica_ids = set()
        coord.ps_model_version = 7
        coord.instance_to_model_version = {}
        coord.server_handles = {
            replica_id: SimpleNamespace(
                get_engine_runtime_snapshots=SimpleNamespace(
                    remote=lambda rid=replica_id: probe(rid),
                )
            )
            for replica_id in ("a", "b")
        }
        coord.instance_to_engine_status = {
            (replica_id, rank): SimpleNamespace(replica_idx=0, model_version=0, snapshot={})
            for replica_id in coord.server_handles
            for rank in range(2)
        }
        result = await coord.get_instance_runtime_snapshot()
        assert set(result) == {("a", 0), ("a", 1), ("b", 0), ("b", 1)}
        assert result[("a", 1)]["is_sleeping"] is True
        assert result[("a", 1)]["model_version"] == 7
        assert result[("b", 0)]["engine_stats"]["scheduler_stats"]["num_running_reqs"] == 0

    asyncio.run(scenario())


def test_router_pause_barrier_failure_prevents_memory_release(modules):
    async def scenario():
        coord = _coordinator(modules)
        coord.server_handles["replica-1"].is_sleeping.remote.return_value = False
        coord._set_routing_loop_running.side_effect = RuntimeError("handoff barrier failed")
        with pytest.raises(RuntimeError, match="handoff barrier failed"):
            await coord._sleep_replicas([("replica-1", 0)])
        coord._do_sleep_instance.assert_not_awaited()

    asyncio.run(scenario())


def test_rollout_rebalance_preserves_opaque_engine_request_ids(modules):
    async def scenario():
        coord = SimpleNamespace(exec_command=SimpleNamespace(remote=AsyncMock(return_value=2)))
        executor = _executor(modules, coord)
        role = modules.role.Rollout
        executor.instances_engine_stats[role]["model"][("replica-1", 0)] = {
            "scheduler_stats": {"req_id_in_waiting": ["cmpl-a-0", "cmpl-b-0"]},
        }
        await executor._interrupt_waiting_after_scale_up(role, "model")
        command = coord.exec_command.remote.await_args.args[0]
        assert command.instance_to_uids == {("replica-1", 0): ["cmpl-a-0", "cmpl-b-0"]}

    asyncio.run(scenario())


def test_reward_rebalance_does_not_abort_nonresumable_completions(modules):
    async def scenario():
        executor = _executor(modules, None)
        executor._await_elastic_coordinator_command = AsyncMock()
        await executor._interrupt_waiting_after_scale_up(modules.role.RewardModel, "model")
        executor._await_elastic_coordinator_command.assert_not_awaited()

    asyncio.run(scenario())
