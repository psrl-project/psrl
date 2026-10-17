"""
Verify scoped clients cross Ray actor boundaries and retain one coordinator owner.
"""

from types import SimpleNamespace

import pytest
import ray
from omegaconf import OmegaConf
from psrl.utils.server.command import Command, CommandType
from psrl.workers.global_coordinator import CoordinatorClient, GlobalCoordinator

pytestmark = pytest.mark.integration


@ray.remote
class _CoordinatorConsumer:
    async def inspect_instances(self, client):
        return await client.get_all_instance_ids.remote()

    async def abort(self, client):
        return await client.exec_command.remote(Command(CommandType.ABORT, instance_ids=[]))


def test_scoped_clients_share_one_owner_across_actor_boundaries(ray_cluster, tmp_path):
    config = OmegaConf.create(
        {
            "psrl": {
                "staleness": 0,
                "logging_path": str(tmp_path),
                "status_collection": {"enable": False, "stats_recorder": {"enable": False}},
                "rollout_coordination": {"session_strategy": {"thunder_agent": {"enable": False}}},
            },
        }
    )
    owner = ray.remote(GlobalCoordinator).remote(config, None, "http://rollout")
    consumer = _CoordinatorConsumer.remote()
    rollout = CoordinatorClient(owner)
    reward = CoordinatorClient(owner, ("reward", "judge"))
    try:
        ray.get(owner.register_reward_model.remote(OmegaConf.create({"reward_model_name": "judge"}), "http://reward"))
        for client in (rollout, reward):
            ray.get(client.add_worker.remote(SimpleNamespace(replica_rank=0, world_size=1), None, "worker-0", 1))
        ray.get(
            rollout.add_worker.remote(
                SimpleNamespace(replica_rank=1, world_size=1), None, "validation-0", 1, is_validate=True
            )
        )
        results = ray.get(
            [
                consumer.inspect_instances.remote(rollout),
                consumer.inspect_instances.remote(reward),
            ],
            timeout=60,
        )
        assert results == [[("validation-0", 0), ("worker-0", 0)], [("worker-0", 0)]], (
            "Clients serialized into another actor must retain their own model instances."
        )
        endpoints = ray.get([client.get_status_sink_endpoint.remote() for client in (rollout, reward)])
        assert len(set(endpoints)) == 2, "The shared owner must bind a separate status sink per controller."
        ray.get(reward.start_busy_loop.remote())
        assert ray.get(consumer.abort.remote(reward), timeout=60) == 0, (
            "Forwarded commands must run on the shared actor event loop."
        )
        ray.get(owner.stop_busy_loop.remote(), timeout=60)
    finally:
        ray.kill(consumer, no_restart=True)
        ray.kill(owner, no_restart=True)
