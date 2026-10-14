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


def test_reward_manager_reserves_callback_capacity_independently_of_rollout_buffer(monkeypatch):
    actor = Mock()
    actor.options.return_value = actor
    monkeypatch.setattr(ray_trainer.ray, "remote", Mock(return_value=actor))
    trainer = object.__new__(ray_trainer.PSRL_RayPPOTrainer)
    trainer.config = SimpleNamespace(
        psrl=SimpleNamespace(reward_service_ip="reward-service"),
        reward=SimpleNamespace(
            reward_loop_worker=SimpleNamespace(num_workers=8, max_concurrency_per_worker=256),
        ),
    )
    trainer.max_concurrency = 6144
    trainer.ip_to_node_id = {"reward-service": "node"}
    trainer.data_processor = object()
    trainer.rollout_coordinator = object()
    trainer.ps_manager_handle = object()
    trainer.reward_loop_workers = [object() for _ in range(8)]
    trainer.init_reward_loop_workers = Mock()

    trainer.init_reward_manager()

    assert actor.options.call_args.kwargs["max_concurrency"] == 4128
    trainer.init_reward_loop_workers.assert_called_once_with()
    actor.remote.assert_called_once_with(
        config=trainer.config,
        ps_manager_handle=trainer.ps_manager_handle,
        reward_loop_workers=trainer.reward_loop_workers,
    )
