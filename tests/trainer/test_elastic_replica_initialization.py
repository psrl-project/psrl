"""
Verify that disjoint replicas of one elastic role initialize concurrently.
"""

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from psrl.trainer.ppo import ray_trainer
from psrl.workers.reward.reward_model import manager as reward_manager

pytestmark = pytest.mark.cpu_test


def _replica_class(events, count):
    class Replica:
        def __init__(self, *, replica_rank, **kwargs):
            self.replica_rank = replica_rank
            self.tag = kwargs.get("tag", "reward")
            self._server_address = f"replica-{replica_rank}"
            self.gateway_replica_id = f"worker-{replica_rank}"
            server = SimpleNamespace()
            for name in ("init_nixl_client", "nixl_convert_params", "sleep"):

                def execute(*args, operation=name, **kwargs):
                    events.append((operation, self.replica_rank))

                setattr(server, name, SimpleNamespace(remote=Mock(side_effect=execute)))
            self._server_handle = server
            self.servers = [server]

        async def init_model(self, worker_group):
            if not hasattr(type(self), "gate"):
                type(self).gate = asyncio.Event()
            events.append(("start", self.replica_rank))
            if sum(event[0] == "start" for event in events) == count:
                type(self).gate.set()
            await asyncio.wait_for(type(self).gate.wait(), timeout=0.5)
            events.append(("ready", self.replica_rank))

    return Replica


def _rollout_config():
    return SimpleNamespace(
        tensor_model_parallel_size=1,
        data_parallel_size=1,
        pipeline_model_parallel_size=1,
        prometheus=SimpleNamespace(enable=False),
    )


def test_rollout_role_initializes_every_replica_before_sleeping(monkeypatch):
    events = []
    count = 4
    monkeypatch.setattr(ray_trainer, "PSRL_vLLMReplica", _replica_class(events, count))
    monkeypatch.setattr(ray_trainer, "GenInterface", lambda **kwargs: None)
    monkeypatch.setattr(ray_trainer.ray, "get", lambda refs: refs)
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.rollout_coordinator = SimpleNamespace(
        get_status_sink_endpoint=SimpleNamespace(remote=Mock(return_value="status-endpoint")),
    )
    trainer.ps_manager_handle = None
    trainer.config = SimpleNamespace(
        psrl=SimpleNamespace(deployment=SimpleNamespace(rollout_ngpus_per_node_per_instance=1)),
        gen_actor_rollout_ref=SimpleNamespace(rollout=_rollout_config()),
        train_actor_rollout_ref=SimpleNamespace(model=SimpleNamespace(path="rollout-model")),
    )
    trainer._rollout_replica_alloc_lock = threading.Lock()
    trainer.rollout_replicas = []
    trainer.tag_to_server_handles = {}
    trainer.server_addresses = []
    trainer.tag_to_base_worker_ids = {}
    trainer._post_gateway_worker_routing_control = Mock(side_effect=lambda *args: events.append(("pause", None)))
    trainer.register_rollout_servers = Mock(side_effect=lambda *args: events.append(("register", None)))
    groups = [SimpleNamespace(world_size=1) for _ in range(count)]

    trainer.init_rollout_servers(groups, sleep_after_init=True)

    assert events[:count] == [("start", index) for index in range(count)]
    operations = [event[0] for event in events]
    assert operations.count("ready") == count
    assert operations.index("register") > max(index for index, name in enumerate(operations) if name == "ready")
    assert operations.index("pause") < operations.index("init_nixl_client")
    assert max(index for index, name in enumerate(operations) if name == "init_nixl_client") < (
        operations.index("nixl_convert_params")
    )
    assert max(index for index, name in enumerate(operations) if name == "nixl_convert_params") < operations.index(
        "sleep"
    )
    assert operations.count("sleep") == count
    assert len(trainer.rollout_replicas) == count
    assert len(trainer.tag_to_server_handles["rollout"]) == count
    trainer._post_gateway_worker_routing_control.assert_called_once_with(
        "pause", [f"worker-{i}" for i in range(count)]
    )


@pytest.mark.parametrize("elastic", [True, False])
def test_reward_role_initializes_every_replica_concurrently(monkeypatch, elastic):
    events = []
    count = 4
    monkeypatch.setattr(reward_manager, "RewardModelReplica", _replica_class(events, count))
    monkeypatch.setattr(reward_manager, "GenInterface", lambda **kwargs: None)
    monkeypatch.setattr(reward_manager.ray, "get", lambda refs: refs)
    manager = object.__new__(reward_manager.RewardModelManager)
    manager.reward_model_coordinator = SimpleNamespace(
        get_status_sink_endpoint=SimpleNamespace(remote=Mock(return_value="status-endpoint")),
    )
    manager.rollout_config = _rollout_config()
    manager.config = SimpleNamespace(
        psrl=SimpleNamespace(
            deployment=SimpleNamespace(
                elastic=SimpleNamespace(enable=elastic),
            )
        )
    )
    manager.reward_model_config = SimpleNamespace(rollout_ngpus_per_instance_per_node=1)
    manager.reward_model_name = "reward-model"
    manager.hf_model_config = SimpleNamespace(path="reward-model")
    manager.reward_model_wg_list = [SimpleNamespace(world_size=1) for _ in range(count)]
    manager.replicas = []

    manager._initialize_reward_replicas()

    assert events[:count] == [("start", index) for index in range(count)]
    operations = [event[0] for event in events]
    assert operations.count("ready") == count
    assert operations.count("sleep") == (count if elastic else 0)
    if elastic:
        assert max(index for index, name in enumerate(operations) if name == "ready") < operations.index("sleep")
    assert len(manager.replicas) == count


def test_reward_managers_reserve_separate_request_and_callback_capacity(monkeypatch):
    actor = Mock()
    actor.options.return_value = actor
    monkeypatch.setattr(ray_trainer.ray, "remote", Mock(return_value=actor))
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.config = SimpleNamespace(
        psrl=SimpleNamespace(reward_service_ip="reward-service"),
        reward=SimpleNamespace(
            reward_loop_worker=SimpleNamespace(num_workers=8, max_concurrency_per_worker=256),
            validation_reward_loop_worker=SimpleNamespace(num_workers=2, max_concurrency_per_worker=256),
        ),
    )
    trainer.max_concurrency = 6144
    trainer.ip_to_node_id = {"reward-service": "node"}
    trainer.data_processor = object()
    trainer.rollout_coordinator = object()
    trainer.ps_manager_handle = object()
    trainer.reward_loop_workers = [object() for _ in range(8)]
    trainer.validation_reward_loop_workers = [object() for _ in range(2)]
    trainer.init_reward_loop_workers = Mock()
    handles = [object(), object()]
    actor.remote.side_effect = handles

    trainer.init_reward_manager()

    capacities = [call.kwargs["max_concurrency"] for call in actor.options.call_args_list]
    assert capacities == [4128, 1056], "Each reward manager must reserve its own request and callback capacity."
    assert trainer.reward_manager is handles[0], "Training must use its own manager actor."
    assert trainer.validation_reward_manager is handles[1], "Validation must use its own manager actor."
    trainer.init_reward_loop_workers.assert_called_once_with()
    actor.remote.assert_any_call(
        config=trainer.config,
        ps_manager_handle=trainer.ps_manager_handle,
        reward_loop_workers=trainer.reward_loop_workers,
        is_validate=False,
    )
    actor.remote.assert_any_call(
        config=trainer.config,
        ps_manager_handle=trainer.ps_manager_handle,
        reward_loop_workers=trainer.validation_reward_loop_workers,
        is_validate=True,
    )


def test_reward_worker_pools_use_separate_roles_and_configuration(monkeypatch):
    actor = Mock()
    actor.options.return_value = actor
    actor.remote.side_effect = [object() for _ in range(5)]
    monkeypatch.setattr(ray_trainer, "RewardLoopWorker", actor)
    monkeypatch.setattr(
        ray_trainer.ray, "nodes", lambda: [{"NodeID": "cpu-node", "Alive": True, "Resources": {"CPU": 1}}]
    )
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.config = SimpleNamespace(
        psrl=SimpleNamespace(reward_service_ip="reward-service", ps_manager_ip="ps-service"),
        reward=SimpleNamespace(
            reward_loop_worker=SimpleNamespace(
                num_workers=2, max_concurrency_per_worker=7, placement="reward_service_ip"
            ),
            validation_reward_loop_worker=SimpleNamespace(
                num_workers=3, max_concurrency_per_worker=11, placement="all_cpu_nodes"
            ),
            active_managers=[],
            managers={},
        ),
    )
    trainer.ip_to_node_id = {"reward-service": "reward-node", "ps-service": "ps-node"}
    trainer.tokenizer = None
    trainer.processor = None
    trainer._build_reward_model_runtime_infos = Mock(return_value={})

    trainer.init_reward_loop_workers()

    assert len(trainer.reward_loop_workers) == 2, "Training pool must use its configured worker count."
    assert len(trainer.validation_reward_loop_workers) == 3, "Validation pool must use its configured worker count."
    options = [call.kwargs for call in actor.options.call_args_list]
    assert [option["max_concurrency"] for option in options] == [7, 7, 11, 11, 11], (
        "Each pool must use its configured actor concurrency."
    )
    assert len({option["name"] for option in options}) == 5, "Reward worker actor names must be unique across pools."
    assert [option["scheduling_strategy"].node_id for option in options] == ["reward-node"] * 2 + ["cpu-node"] * 3, (
        "Each pool must honor its placement configuration."
    )
    roles = [call.kwargs["is_validate"] for call in actor.remote.call_args_list]
    assert roles == [False, False, True, True, True], "Each worker must receive its pool role."


def test_reward_pools_start_together(monkeypatch):
    get = Mock()
    monkeypatch.setattr(ray_trainer.ray, "get", get)
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.reward_manager = Mock()
    trainer.validation_reward_manager = Mock()

    trainer.start_reward_manager()

    trainer.reward_manager.start_busy_loop.remote.assert_called_once_with()
    trainer.validation_reward_manager.start_busy_loop.remote.assert_called_once_with()
    get.assert_called_once_with(
        [
            trainer.reward_manager.start_busy_loop.remote.return_value,
            trainer.validation_reward_manager.start_busy_loop.remote.return_value,
        ]
    )


@pytest.mark.parametrize("shutdown_error", [None, ray_trainer.ray.exceptions.GetTimeoutError("Shutdown stalled.")])
def test_reward_pools_release_all_actors_on_shutdown(monkeypatch, shutdown_error):
    get = Mock(side_effect=shutdown_error)
    kill = Mock()
    monkeypatch.setattr(ray_trainer.ray, "get", get)
    monkeypatch.setattr(ray_trainer.ray, "kill", kill)
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    managers = [Mock(), Mock()]
    workers = [Mock(), Mock(), Mock()]
    trainer.reward_manager, trainer.validation_reward_manager = managers
    trainer.reward_loop_workers = workers[:2]
    trainer.validation_reward_loop_workers = workers[2:]

    trainer.stop_reward_manager()

    get.assert_called_once_with([manager.stop_busy_loop.remote.return_value for manager in managers], timeout=60)
    assert [call.args[0] for call in kill.call_args_list] == managers + workers, (
        "Both pools must release their actors."
    )
    assert trainer.reward_manager is None, "Training manager handle must be cleared."
    assert trainer.validation_reward_manager is None, "Validation manager handle must be cleared."
    assert not trainer.reward_loop_workers, "Training worker handles must be cleared."
    assert not trainer.validation_reward_loop_workers, "Validation worker handles must be cleared."
