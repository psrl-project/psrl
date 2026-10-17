"""
Verify executor ownership of trainer storage and elastic inference transitions.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

import cloudpickle
import pytest

from tests.elastic.runtime_helpers import _trainer_pool_executor

pytestmark = pytest.mark.cpu_test


@pytest.mark.parametrize("ready", [False, True])
def test_wait_boundary_checks_readiness_before_releasing_storage(modules, ready):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor.elastic_config = {"trainer_ready_grace_s": 0.25}
        executor.agent_loop_manager = SimpleNamespace(
            wait_for_training_readiness=SimpleNamespace(remote=AsyncMock(return_value=ready)),
        )
        await executor.release_trainer_pool_for_wait(4)
        executor.agent_loop_manager.wait_for_training_readiness.remote.assert_awaited_once_with(4, 0.25)
        assert events == ([] if ready else ["nixl_sleep"]), "A ready batch must retain trainer storage."
        assert executor._trainer_pool_available is not ready, "Only released storage can be borrowed."
        assert executor._trainer_waiting_buffer_id == (None if ready else 4), "Publish waits after release."

    asyncio.run(scenario())


def test_initialization_releases_meta_storage_without_admitting_borrowers(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        await executor.initialize_trainer_pool(executor.actor_wg, executor.ps_manager_handle)
        assert events == ["nixl_sleep"], "Initialization must release actor storage."
        executor.actor_wg.execute_all_async.assert_called_once_with("nixl_sleep", "meta")
        assert executor._trainer_pool_available is False, "Inference placements must be registered first."
        assert executor._trainer_state == modules.executor.TrainerState.ASLEEP, "The executor owns sleep state."

    asyncio.run(scenario())


def test_already_sleeping_trainer_publishes_wait_without_readiness_rpc(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        executor.agent_loop_manager = SimpleNamespace(wait_for_training_readiness=SimpleNamespace(remote=AsyncMock()))
        await executor.release_trainer_pool_for_wait(4)
        assert events == [], "A sleeping trainer must retain its released storage."
        executor.agent_loop_manager.wait_for_training_readiness.remote.assert_not_called()
        assert executor._trainer_waiting_buffer_id == 4, "The executor must publish the active wait buffer."

    asyncio.run(scenario())


@pytest.mark.parametrize("failed_stage", ["nixl_wake_up", "broadcast", "pull_model"])
def test_failed_restore_keeps_pool_reserved_until_storage_is_released(modules, failed_stage):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP

        async def execute(method):
            events.append(method)
            if method == failed_stage:
                raise RuntimeError("Restore failed.")

        executor.actor_wg.execute_all_async.side_effect = lambda method, *args: [execute(method)]
        if failed_stage == "broadcast":
            executor._broadcast_trainer_client_infos.side_effect = RuntimeError("Restore failed.")
        with pytest.raises(RuntimeError, match="Restore failed"):
            await executor.acquire_trainer_pool()
        assert executor._trainer_state == modules.executor.TrainerState.RESTORING, "Restore remains incomplete."
        assert executor._trainer_pool_available is False, "Failed restores cannot release GPUs."
        await executor.release_trainer_pool()
        assert events[-1] == "nixl_sleep", "Partially restored storage must be slept before release."
        assert executor._trainer_pool_available is True, "Completed sleep permits inference borrowing."

    asyncio.run(scenario())


def test_worker_failure_settles_other_workers_before_next_transition(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        gate = asyncio.Event()

        async def fail():
            events.append("failure")
            raise RuntimeError("Wake failed.")

        async def finish():
            await gate.wait()
            events.append("settled")

        executor.actor_wg.execute_all_async.side_effect = lambda *args: [fail(), finish()]
        acquire = asyncio.create_task(executor.acquire_trainer_pool())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not acquire.done(), "One worker failure must not abandon other worker operations."
        gate.set()
        with pytest.raises(RuntimeError, match="Wake failed"):
            await acquire
        assert events[-1] == "settled", "Every worker must settle before the transition reports failure."

    asyncio.run(scenario())


def test_release_waits_for_trainer_restore_lock(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        gate = asyncio.Event()

        async def execute(method):
            events.append(method)
            if method == "nixl_wake_up":
                await gate.wait()

        executor.actor_wg.execute_all_async.side_effect = lambda method, *args: [execute(method)]
        acquire = asyncio.create_task(executor.acquire_trainer_pool())
        await asyncio.sleep(0)
        release = asyncio.create_task(executor.release_trainer_pool())
        await asyncio.sleep(0)
        assert "nixl_sleep" not in events, "Release must wait for the complete restore operation."
        gate.set()
        await asyncio.gather(acquire, release)
        assert events.index("pull_model") < events.index("nixl_sleep"), "Transitions must hold one ownership lock."

    asyncio.run(scenario())


def test_deferred_checkpoint_loads_after_weight_restore(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        executor.config.trainer = SimpleNamespace(del_local_ckpt_after_load=True)
        executor.defer_actor_checkpoint("checkpoint/actor")
        await executor.acquire_trainer_pool()
        assert events[-2:] == ["pull_model", "load_checkpoint"], "Checkpoint storage requires restored connections."
        assert call("load_checkpoint", "checkpoint/actor", del_local_after_load=True) in (
            executor.actor_wg.execute_all_async.call_args_list
        ), "Checkpoint deletion policy must reach every trainer worker."
        assert executor._deferred_actor_ckpt_path is None, "A successful checkpoint must be consumed once."

    asyncio.run(scenario())


def test_failed_deferred_checkpoint_remains_pending(modules):
    async def scenario():
        executor, _ = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        executor.config.trainer = SimpleNamespace(del_local_ckpt_after_load=False)

        async def execute(method):
            if method == "load_checkpoint":
                raise RuntimeError("Checkpoint failed.")

        executor.actor_wg.execute_all_async.side_effect = lambda method, *args, **kwargs: [execute(method)]
        executor.defer_actor_checkpoint("checkpoint/actor")
        with pytest.raises(RuntimeError, match="Checkpoint failed"):
            await executor.acquire_trainer_pool()
        assert executor._deferred_actor_ckpt_path == "checkpoint/actor", "Failed checkpoint loads remain pending."
        assert executor._trainer_state == modules.executor.TrainerState.RESTORING, "Training must not start."
        assert executor._trainer_pool_available is False, "Allocated storage must remain reserved."

    asyncio.run(scenario())


@pytest.mark.parametrize("checkpoint_fails", [False, True])
def test_resume_push_releases_actor_storage_before_inference_bootstrap(modules, checkpoint_fails):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        executor.config.trainer = SimpleNamespace(del_local_ckpt_after_load=False)

        async def execute(method):
            events.append(method)
            if checkpoint_fails and method == "load_checkpoint":
                raise RuntimeError("Checkpoint failed.")

        executor.actor_wg.execute_all_async.side_effect = lambda method, *args, **kwargs: [execute(method)]
        version_rpc = AsyncMock()
        executor.ps_manager_handle.init_model_version_for_resume = SimpleNamespace(remote=version_rpc)
        if checkpoint_fails:
            with pytest.raises(RuntimeError, match="Checkpoint failed"):
                await executor.load_and_push_actor_checkpoint("checkpoint/actor", 4)
            version_rpc.assert_not_called()
            assert "push_model" not in events, "Failed checkpoints must not publish resume weights."
        else:
            await executor.load_and_push_actor_checkpoint("checkpoint/actor", 4)
            version_rpc.assert_awaited_once_with(3)
            assert events[-3:] == ["load_checkpoint", "push_model", "nixl_sleep"], "Resume push must release storage."
        assert events[-1] == "nixl_sleep", "Actor storage must be released even if checkpoint loading fails."
        assert executor._trainer_pool_available is False, "Bootstrap must register inference placements first."

    asyncio.run(scenario())


@pytest.mark.parametrize("wake_fails", [False, True])
def test_validation_reclaims_then_wakes_and_always_sleeps_before_release(modules, wake_fails):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        executor._validation_instance_ids = [("validator", 0)]

        async def command(coordinator, cmd, *, stage):
            events.append(stage)
            assert cmd.get_args()["instance_ids"] == [("validator", 0)], "Validation commands must target validators."
            if wake_fails and stage == "VALIDATION_WAKE":
                raise RuntimeError("Validation wake failed.")

        executor._await_elastic_coordinator_command = AsyncMock(side_effect=command)
        if wake_fails:
            with pytest.raises(RuntimeError, match="Validation wake failed"):
                await executor.begin_validation()
        else:
            await executor.begin_validation()
        assert executor._validation_active is True, "Partial validation wakes require cleanup."
        assert events == ["reclaim", "nixl_sleep", "VALIDATION_WAKE"], "Validation must reserve trainer GPUs first."
        await executor.release_trainer_pool()
        assert events[-1] == "VALIDATION_SLEEP", "Validation storage must be slept before borrowing resumes."
        assert executor._validation_active is False, "Cleanup must clear validation ownership."
        assert executor._trainer_pool_available is True, "Released validation storage can host inference."

    asyncio.run(scenario())


def test_initial_validation_sleep_uses_role_coordinator(modules):
    async def scenario():
        executor, _ = _trainer_pool_executor(modules)
        executor._rollout_coordinator().sleep = SimpleNamespace(remote=AsyncMock())
        await executor.initialize_validation_workers([("validator", 0)])
        executor._rollout_coordinator().sleep.remote.assert_awaited_once_with("validate")
        assert executor._validation_instance_ids == [("validator", 0)], "The executor owns validation membership."

    asyncio.run(scenario())


def test_validation_waits_for_scaling_then_reserves_pool_until_release(modules):
    async def scenario():
        executor, events = _trainer_pool_executor(modules)
        del executor._reclaim_trainer_pool
        role = modules.role.Rollout
        borrower = ("replica-1", 0)
        executor.instance_pool_ids[(role, "model", borrower)] = "train_pool"
        executor._trainer_pool_available = True
        executor._decision_settled.clear()
        executor._validation_instance_ids = [("validator", 0)]

        async def command(coordinator, cmd, *, stage):
            events.append(stage)
            assert not executor._trainer_pool_available, "Validation must reserve GPUs before any storage transition."

        executor._await_elastic_coordinator_command = AsyncMock(side_effect=command)
        validation = asyncio.create_task(executor.begin_validation())
        await asyncio.sleep(0)
        assert not executor._trainer_pool_available, "In flight scaling must immediately lose borrowing permission."
        assert events == [], "Validation must wait for the previous scaling decision to settle."
        assert not validation.done(), "Validation cannot wake while scaling is in flight."
        executor._decision_settled.set()
        await validation
        assert events == ["TRAIN_POOL_RECLAIM role=Rollout model=model", "nixl_sleep", "VALIDATION_WAKE"], (
            "Borrowers must release memory before validation wakes."
        )
        assert executor.instances_status_flags[role]["model"][borrower] == modules.topology.InstanceStatus.ASLEEP, (
            "Reclaimed inference borrowers must remain asleep."
        )
        with pytest.raises(RuntimeError, match="Trainer pool is reserved"):
            await executor._scale_up_instances([{"role_name": role, "model_name": "model", "instance_id": borrower}])
        assert executor._validation_active, "Validation ownership must persist throughout inference."
        await executor.release_trainer_pool()
        assert events[-1] == "VALIDATION_SLEEP", "Validation must release memory before borrowing resumes."
        assert executor._trainer_pool_available, "Only finished validation releases the reservation."

    asyncio.run(scenario())


def test_trainer_metadata_broadcast_reaches_every_nixl_client(modules):
    async def scenario():
        from psrl.utils.common.worker_naming import train_client_name

        executor, _ = _trainer_pool_executor(modules)
        del executor._broadcast_trainer_client_infos
        executor.actor_wg.world_size = 2
        executor.ps_wg = SimpleNamespace(world_size=2, execute_all_async=Mock(return_value=[]))
        servers = [SimpleNamespace(nixl_update_client_infos=SimpleNamespace(remote=AsyncMock())) for _ in range(2)]
        executor.configure_trainer_pool_clients(executor.ps_wg, {"rollout": servers[:1], "validate": servers[1:]})
        updates = {train_client_name(rank): {"info": bytes([rank]), "temp_mapping": {}} for rank in range(2)}

        async def execute(method, *args):
            if method == "nixl_get_local_client_info":
                return {train_client_name(0): updates[train_client_name(0)]}

        async def second_rank():
            return {train_client_name(1): updates[train_client_name(1)]}

        executor.actor_wg.execute_all_async.side_effect = lambda method, *args: (
            [execute(method), second_rank()] if method == "nixl_get_local_client_info" else [execute(method, *args)]
        )
        await executor._broadcast_trainer_client_infos()
        for server in servers:
            server.nixl_update_client_infos.remote.assert_awaited_once_with(updates)
        executor.ps_wg.execute_all_async.assert_called_once_with("nixl_update_client_infos", updates)
        executor.actor_wg.execute_all_async.assert_has_calls(
            [call("nixl_get_local_client_info"), call("nixl_update_client_infos", updates)]
        )

    asyncio.run(scenario())


@pytest.mark.parametrize("receiver_fails", [False, True])
def test_restore_waits_for_metadata_installation_before_pulling_weights(modules, receiver_fails):
    async def scenario():
        from psrl.utils.common.worker_naming import train_client_name

        executor, events = _trainer_pool_executor(modules)
        del executor._broadcast_trainer_client_infos
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        started = asyncio.Event()
        installed = asyncio.Event()
        updates = {train_client_name(0): {"info": b"descriptor", "temp_mapping": {}}}

        async def execute(method, *args):
            events.append(method)
            if method == "nixl_get_local_client_info":
                return updates

        async def install(payload):
            assert payload == updates, "The receiver must install the current trainer descriptors."
            started.set()
            await installed.wait()
            if receiver_fails:
                raise RuntimeError("Metadata installation failed.")

        executor.actor_wg.execute_all_async.side_effect = lambda method, *args: [execute(method, *args)]
        executor.ps_wg = SimpleNamespace(execute_all_async=Mock(return_value=[]))
        server = SimpleNamespace(nixl_update_client_infos=SimpleNamespace(remote=AsyncMock(side_effect=install)))
        executor.configure_trainer_pool_clients(executor.ps_wg, {"rollout": [server]})
        restore = asyncio.create_task(executor.acquire_trainer_pool())
        await asyncio.wait_for(started.wait(), timeout=1)
        assert "pull_model" not in events, "Weights must wait for every receiver to install descriptors."
        assert not restore.done(), "An unacknowledged metadata update must hold the restore operation."
        assert executor._trainer_state == modules.executor.TrainerState.RESTORING, "Storage is not ready yet."
        installed.set()
        if receiver_fails:
            with pytest.raises(RuntimeError, match="Metadata installation failed"):
                await restore
            assert "pull_model" not in events, "Failed metadata installation must prevent weight transfers."
            assert executor._trainer_state == modules.executor.TrainerState.RESTORING, "Failed storage stays reserved."
        else:
            await restore
            assert events[-1] == "pull_model", "Acknowledged descriptors permit weight restoration."
            assert executor._trainer_state == modules.executor.TrainerState.AWAKE, "Restored weights permit training."
        assert executor._trainer_pool_available is False, "Storage must stay reserved throughout trainer restore."

    asyncio.run(scenario())


def test_missing_rank_metadata_prevents_broadcast_and_weight_pull(modules):
    async def scenario():
        from psrl.utils.common.worker_naming import train_client_name

        executor, events = _trainer_pool_executor(modules)
        del executor._broadcast_trainer_client_infos
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        executor.actor_wg.world_size = 2

        async def execute(method):
            events.append(method)
            if method == "nixl_get_local_client_info":
                return {train_client_name(0): {"info": b"descriptor", "temp_mapping": {}}}

        executor.actor_wg.execute_all_async.side_effect = lambda method, *args: [execute(method)]
        executor.ps_wg = SimpleNamespace(execute_all_async=Mock(return_value=[]))
        executor.configure_trainer_pool_clients(executor.ps_wg, {})
        with pytest.raises(ValueError, match="missing=.*NIXLTrainClient_1"):
            await executor.acquire_trainer_pool()
        executor.ps_wg.execute_all_async.assert_not_called()
        assert "pull_model" not in events, "Missing ranks must prevent weight transfers."
        assert executor._trainer_state == modules.executor.TrainerState.RESTORING, "Incomplete restore stays reserved."

    asyncio.run(scenario())


def test_replica_registration_serializes_only_placement_metadata(modules):
    registration = modules.executor.ReplicaRegistration("rollout", 2, 1, 1, (("node", 0), ("node", 1)), "train_pool")
    restored = cloudpickle.loads(cloudpickle.dumps(registration))
    ids, slots, pools = modules.executor.ElasticExecutor._elastic_instance_registration([restored])
    assert ids == [("rollout", 0)], "Gateway identities must survive actor transfer."
    assert len(slots[0]) == 2, "All physical workers belong to the DP instance."
    assert pools == ["train_pool"], "Resource pool identity must survive actor transfer."


def test_replica_registration_rejects_invalid_parallelism(modules):
    registration = modules.executor.ReplicaRegistration("rollout", 2, 1, 2, (("node", 0),), "train_pool")
    with pytest.raises(ValueError, match="GPU locations"):
        modules.executor.ElasticExecutor._elastic_instance_registration([registration])


@pytest.mark.parametrize("trainer_pool_enabled", [False, True])
def test_inference_bootstrap_registers_roles_and_respects_pool_ownership(modules, trainer_pool_enabled):
    async def scenario():
        from psrl.workers.elastic.policy_runtime import PolicyIdentityMap

        executor, _ = _trainer_pool_executor(modules)
        executor.roles = []
        executor.topology = modules.topology.ClusterTopology()
        executor.instances_status_flags = {}
        executor.instances_engine_stats = {}
        executor.policy_identity_map = PolicyIdentityMap()
        executor._trainer_state = modules.executor.TrainerState.ASLEEP
        executor.elastic_config = {"enable_trainer_pool": trainer_pool_enabled, "min_awake_per_role": 1}
        executor.start_busy_loop = AsyncMock()
        rollout, reward = modules.role.Rollout, modules.role.RewardModel
        rollout_coord = SimpleNamespace(
            get_all_instance_ids=SimpleNamespace(remote=AsyncMock(return_value=[("r0", 0), ("r1", 0), ("v0", 0)])),
        )
        reward_coord = SimpleNamespace(
            get_all_instance_ids=SimpleNamespace(remote=AsyncMock(return_value=[("rm0", 0)])),
        )
        coordinators = {rollout: {"rollout": rollout_coord}, reward: {"reward": reward_coord}}
        register = modules.executor.ReplicaRegistration
        registrations = {
            (rollout, "rollout"): [
                register("r0", 1, 1, 1, (("node", 0),), "shared_rollout_pool"),
                register(
                    "r1", 1, 1, 1, (("node", 1),), "train_pool" if trainer_pool_enabled else "shared_rollout_pool"
                ),
            ],
            (reward, "reward"): [register("rm0", 1, 1, 1, (("node", 0),), "shared_rollout_pool")],
        }
        commands = []

        async def execute(coordinator, command, *, stage):
            commands.append((stage, command.get_args()["instance_ids"]))

        executor._await_elastic_coordinator_command = AsyncMock(side_effect=execute)
        await executor.initialize_inference_runtime(coordinators, registrations, [("v0", 0)])
        assert executor.instances_status_flags[reward]["reward"][("rm0", 0)] == modules.topology.InstanceStatus.AWAKEN
        assert executor.instances_status_flags[rollout]["rollout"][("r0", 0)] == modules.topology.InstanceStatus.ASLEEP
        assert executor.instances_status_flags[rollout]["rollout"][("r1", 0)] == modules.topology.InstanceStatus.AWAKEN
        wakes = [ids for stage, ids in commands if stage.startswith("INITIAL_WAKE")]
        assert wakes == [[("rm0", 0)], [("r1", 0)]], (
            "Bootstrap must reserve reward GPUs before selecting rollout GPUs."
        )
        assert (rollout, "rollout", ("v0", 0)) not in executor.instance_pool_ids, (
            "Validators are not elastic borrowers."
        )
        assert executor._trainer_pool_available is trainer_pool_enabled, "Only released trainer pools can be borrowed."
        executor.start_busy_loop.assert_awaited_once_with()

    asyncio.run(scenario())
