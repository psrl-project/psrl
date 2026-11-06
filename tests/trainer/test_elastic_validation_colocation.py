import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from psrl.trainer.ppo import ray_trainer

pytestmark = pytest.mark.cpu_test


def test_trainer_inherits_only_the_ppo_trainer():
    assert ray_trainer.PSRL_RayPPOTrainer.__bases__ == (ray_trainer.RayPPOTrainer,), (
        "Elastic resource lifecycle must belong to the executor."
    )


def _remote_handle():
    handle = MagicMock()
    handle.remote.return_value = object()
    return handle


def _trainer(elastic=True):
    trainer = ray_trainer.PSRL_RayPPOTrainer.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.config = SimpleNamespace(
        psrl=SimpleNamespace(
            colocate_validate_and_train=True,
            ps_mode="nixl_cpu",
            deployment=SimpleNamespace(mode="elastic_rl"),
        ),
        train_actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(data_parallel_size=1)),
    )
    trainer.is_rollout_mode_in_actor = True
    trainer.elastic_mode = elastic
    trainer._trainer_lease_id = None
    trainer.n_rollout_instances = 2
    trainer.n_validate_instances = 1
    trainer.tag_to_base_worker_ids = {"validate": ["worker-validate"]}
    server = SimpleNamespace(
        **{
            name: _remote_handle()
            for name in (
                "nixl_wake_up",
                "nixl_sleep",
                "nixl_get_local_client_info",
                "pause_generation",
            )
        }
    )
    trainer.tag_to_server_handles = {"validate": [server]}
    trainer.rollout_coordinator = SimpleNamespace(
        sleep=_remote_handle(),
        wake_up=_remote_handle(),
        sync_with_ps=_remote_handle(),
        exec_command=_remote_handle(),
    )
    trainer.elastic_executor = SimpleNamespace(
        acquire_trainer_pool=_remote_handle(), release_trainer_pool=_remote_handle(), begin_validation=_remote_handle()
    )
    trainer.actor_wg = MagicMock()
    trainer.actor_wg.world_size = 2
    trainer.actor_wg.execute_all_async.return_value = [object()]
    trainer.ps_manager_handle = SimpleNamespace()
    trainer._post_gateway_worker_routing_control = MagicMock()
    trainer._broadcast_updated_client_infos = MagicMock()
    trainer.worker_to_ps_idx = {ray_trainer.WorkerKey("validate", 0, 0): 0}
    return trainer


@pytest.mark.parametrize("start_succeeds", [True, False])
def test_elastic_monitor_binds_manager_after_pipeline_start(start_succeeds):
    trainer = _trainer()
    trainer.agent_loop_manager = SimpleNamespace(start_busy_loop=_remote_handle())
    trainer.elastic_executor.set_agent_loop_manager = _remote_handle()
    start_ref = trainer.agent_loop_manager.start_busy_loop.remote.return_value
    events = []

    def get(ref):
        if ref is start_ref:
            if not start_succeeds:
                raise RuntimeError("Pipeline startup failed.")
            events.append("started")
        else:
            assert events == ["started"], "The monitor must bind after the pipeline is ready."
            events.append("bound")

    with patch.object(ray_trainer.ray, "get", side_effect=get):
        if start_succeeds:
            trainer.start_agent_loop_manager()
        else:
            with pytest.raises(RuntimeError, match="Pipeline startup failed"):
                trainer.start_agent_loop_manager()

    if start_succeeds:
        assert events == ["started", "bound"], "Startup must precede elastic observation."
        trainer.elastic_executor.set_agent_loop_manager.remote.assert_called_once_with(trainer.agent_loop_manager)
    else:
        trainer.elastic_executor.set_agent_loop_manager.remote.assert_not_called()


@pytest.mark.parametrize("enable_trainer_pool", [False, True])
def test_switch_to_validation_delegates_to_executor(enable_trainer_pool):
    trainer = _trainer()
    trainer.config.psrl.deployment = SimpleNamespace(
        mode="elastic_rl", elastic=SimpleNamespace(enable=True, enable_trainer_pool=enable_trainer_pool)
    )
    with patch.object(ray_trainer.ray, "get", return_value=True):
        trainer.switch_to_rollout_mode()
    trainer.elastic_executor.begin_validation.remote.assert_called_once_with()
    trainer.actor_wg.execute_all_async.assert_not_called()
    trainer._post_gateway_worker_routing_control.assert_not_called()
    trainer.rollout_coordinator.exec_command.remote.assert_not_called()


def test_elastic_validation_returns_pool_without_waking_trainer():
    trainer = _trainer()
    with patch.object(ray_trainer.ray, "get", return_value=True):
        trainer.switch_to_trainer_mode(restore_trainer=False)
    trainer.elastic_executor.release_trainer_pool.remote.assert_called_once_with()
    trainer.actor_wg.execute_all_async.assert_not_called()
    trainer.rollout_coordinator.exec_command.remote.assert_not_called()


def test_elastic_training_delegates_restoration_to_executor():
    trainer = _trainer()
    with patch.object(ray_trainer.ray, "get", return_value=True):
        trainer.switch_to_trainer_mode()
    trainer.elastic_executor.acquire_trainer_pool.remote.assert_called_once_with()
    trainer.actor_wg.execute_all_async.assert_not_called()


def test_elastic_resume_delegates_the_complete_actor_storage_cycle():
    trainer = _trainer()
    trainer.config.psrl.ps_mode = "nixl_cpu"
    trainer._resolve_resume_checkpoint_paths = MagicMock(return_value=("checkpoint/global_step_4/actor", None))
    trainer.elastic_executor.load_and_push_actor_checkpoint = _remote_handle()
    with patch.object(ray_trainer.ray, "get", return_value=True):
        trainer._resume_load_and_push_to_ps()
    trainer.elastic_executor.load_and_push_actor_checkpoint.remote.assert_called_once_with(
        "checkpoint/global_step_4/actor",
        4,
    )
    trainer.actor_wg.execute_all_async.assert_not_called()
    assert trainer._actor_resume_loaded is True, "Initialization must record completed checkpoint loading."


def test_elastic_runtime_hands_off_registration_without_sleeping_workers():
    trainer = _trainer()
    trainer.elastic_mode = True
    trainer.config.gen_actor_rollout_ref = SimpleNamespace(model=SimpleNamespace(path="models/rollout"))
    trainer.start_rollout_coordinator = MagicMock()
    replica = SimpleNamespace(
        tag="rollout",
        gateway_replica_id="worker-rollout",
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        worker_gpu_locations=(("node", 0), ("node", 1)),
    )
    trainer.rollout_replicas = [replica]
    trainer.elastic_replica_placements = {"rollout_0": SimpleNamespace(pool_id="train_pool")}
    trainer.reward_model_to_manager = {}
    trainer.elastic_executor.initialize_inference_runtime = _remote_handle()
    with patch.object(ray_trainer.ray, "get", return_value=True):
        trainer.init_elastic_runtime()
    args = trainer.elastic_executor.initialize_inference_runtime.remote.call_args.args
    coordinators, registrations, validators = args
    assert coordinators[ray_trainer.PSRL_Role.Rollout]["rollout"] is trainer.rollout_coordinator, (
        "The executor must own the initialized rollout coordinator."
    )
    assert registrations[(ray_trainer.PSRL_Role.Rollout, "rollout")] == [
        ray_trainer.ReplicaRegistration.from_replica(replica, trainer.elastic_replica_placements["rollout_0"])
    ], "The executor must receive physical replica metadata."
    assert validators == [], "Only initialized validators participate in resource ownership."
    trainer.rollout_coordinator.exec_command.remote.assert_not_called()
    trainer.actor_wg.execute_all_async.assert_not_called()


def test_non_elastic_validation_wakes_through_coordinator_before_returning():
    trainer = _trainer(elastic=False)
    trainer.is_rollout_mode_in_actor = False
    events = []
    client_name = ray_trainer.WorkerKey("validate", 0, 0).to_nixl_client_name(trainer.n_rollout_instances)
    groups = [[{client_name: {"info": b"validation", "temp_mapping": {}}}]]

    def get(refs):
        if refs is trainer.rollout_coordinator.wake_up.remote.return_value:
            assert events == ["actor_sleep"], "The actor must release memory before validation wakes."
            events.append("validation_ready")
        elif refs is trainer.actor_wg.execute_all_async.return_value:
            events.append("actor_sleep")
        return groups

    with patch.object(ray_trainer.ray, "get", side_effect=get):
        trainer.switch_to_rollout_mode()

    assert events == ["actor_sleep", "validation_ready"], "Validation must wait for coordinated weight restoration."
    trainer.rollout_coordinator.wake_up.remote.assert_called_once_with("validate")
    trainer.rollout_coordinator.sync_with_ps.remote.assert_not_called()
    trainer.actor_wg.execute_all_async.assert_called_once_with("nixl_sleep", "full")
    trainer.tag_to_server_handles["validate"][0].nixl_wake_up.remote.assert_not_called()
    trainer._post_gateway_worker_routing_control.assert_not_called()
    trainer._broadcast_updated_client_infos.assert_not_called()
    assert trainer.is_rollout_mode_in_actor is True, "Validation must own the colocated GPUs after wake."


def test_non_elastic_validation_restores_trainer_after_validation_sleep():
    trainer = _trainer(elastic=False)
    groups = [{ray_trainer.train_client_name(rank): {"info": bytes([rank]), "temp_mapping": {}}} for rank in range(2)]
    events = []

    def get(refs):
        if refs is trainer.rollout_coordinator.sleep.remote.return_value:
            events.append("validation_asleep")
        elif refs is trainer.actor_wg.execute_all_async.return_value:
            assert events, "Validation must release memory before the actor wakes."
        return groups

    with patch.object(ray_trainer.ray, "get", side_effect=get):
        trainer.switch_to_trainer_mode()
    trainer.rollout_coordinator.sleep.remote.assert_called_once_with("validate")
    trainer.tag_to_server_handles["validate"][0].nixl_sleep.remote.assert_not_called()
    trainer.actor_wg.execute_all_async.assert_has_calls(
        [
            call("nixl_wake_up"),
            call("nixl_get_local_client_info"),
            call("pull_model"),
        ]
    )
    trainer._post_gateway_worker_routing_control.assert_not_called()
    trainer.elastic_executor.release_trainer_pool.remote.assert_not_called()
    assert trainer.is_rollout_mode_in_actor is False, "The actor must own the colocated GPUs after restoration."


def test_non_elastic_validation_wake_failure_runs_coordinated_cleanup():
    trainer = _trainer(elastic=False)
    trainer.is_rollout_mode_in_actor = False
    trainer._run_validation = MagicMock()
    groups = [{ray_trainer.train_client_name(rank): {"info": bytes([rank]), "temp_mapping": {}}} for rank in range(2)]

    def get(refs):
        if refs is trainer.rollout_coordinator.wake_up.remote.return_value:
            raise RuntimeError("Weight restoration failed.")
        return groups

    with patch.object(ray_trainer.ray, "get", side_effect=get):
        with pytest.raises(RuntimeError, match="Weight restoration failed"):
            trainer._validate()

    trainer._run_validation.assert_not_called()
    trainer.rollout_coordinator.sleep.remote.assert_called_once_with("validate")
    assert trainer.is_rollout_mode_in_actor is False, "Wake failures must return the GPUs to training."


@pytest.mark.parametrize("merged", [False, True])
def test_validation_failure_returns_training_pool(merged):
    trainer = _trainer()
    trainer.switch_to_rollout_mode = MagicMock()
    trainer._run_validation = MagicMock(side_effect=RuntimeError("Validation failed."))
    trainer.switch_to_trainer_mode = MagicMock()
    with pytest.raises(RuntimeError, match="Validation failed"):
        trainer._validate(merged=merged)
    trainer.switch_to_trainer_mode.assert_called_once_with(restore_trainer=False)


def test_partial_validation_wake_failure_runs_cleanup():
    trainer = _trainer()
    trainer.is_rollout_mode_in_actor = False
    trainer.elastic_executor.begin_validation.remote.side_effect = RuntimeError("Wake failed.")
    trainer._run_validation = MagicMock()
    trainer.switch_to_trainer_mode = MagicMock()
    with patch.object(ray_trainer.ray, "get", return_value=True), pytest.raises(RuntimeError, match="Wake failed"):
        trainer._validate()
    trainer._run_validation.assert_not_called()
    trainer.switch_to_trainer_mode.assert_called_once_with(restore_trainer=False)


def test_colocated_validation_failure_releases_admission_only_after_actor_storage_cleanup():
    trainer = _trainer()
    trainer.config.psrl.deployment.mode = "colocated"
    events = []
    trainer.inference_orchestrator = SimpleNamespace(
        acquire=SimpleNamespace(remote=MagicMock(side_effect=lambda *args: events.append("reserved"))),
        release=SimpleNamespace(remote=MagicMock(side_effect=lambda *args: events.append("admission_released"))),
    )
    trainer.elastic_executor.begin_validation.remote.side_effect = RuntimeError("Validation wake failed.")
    trainer.elastic_executor.release_trainer_pool.remote.side_effect = lambda: events.append("storage_released")
    with patch.object(ray_trainer.ray, "get", side_effect=lambda ref: ref):
        with pytest.raises(RuntimeError, match="Validation wake failed"):
            trainer._validate()
    assert events == ["reserved", "storage_released", "admission_released"], (
        "A failed initial validation cannot strand a training admission or admit inference before cleanup."
    )
    assert trainer._trainer_lease_id is None, "Successful storage cleanup must relinquish the driver's lease."


def test_trainer_sleep_clears_gradients_and_preserves_optimizer_before_memory_release():
    path = Path(ray_trainer.__file__).parents[2] / "workers/train/engine_train_worker.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PSRL_EngineTrainWorker")
    methods = [
        node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in ("nixl_sleep", "nixl_wake_up")
    ]
    namespace = {}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)
    events = []
    engine = SimpleNamespace(
        optimizer=SimpleNamespace(zero_grad=lambda **kw: events.append("clear_grads")),
        is_optimizer_offload_enabled=False,
    )
    worker = SimpleNamespace(
        config=SimpleNamespace(actor=SimpleNamespace(strategy="fsdp2")),
        actor=SimpleNamespace(engine=engine, to=lambda **kw: events.append(f"optimizer_{kw['device']}")),
        nixl_storage_client=SimpleNamespace(
            deregister_local_tensors=lambda: events.append("deregister"),
            register_local_tensors=lambda *args: events.append("register"),
        ),
        sleep_model=lambda: events.append("sleep"),
        wake_up_model=lambda: events.append("wake"),
        unified_state_dict={},
        unified_sharding_dict={},
    )
    namespace["nixl_sleep"](worker)
    namespace["nixl_wake_up"](worker)
    assert events == ["clear_grads", "optimizer_cpu", "deregister", "sleep", "wake", "optimizer_device", "register"]
