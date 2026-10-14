"""
Test elastic role registration with gateway identities and resource pools.
"""

import pytest
from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.cluster_topology import ClusterTopology, GPUSlot, InstanceStatus
from psrl.workers.elastic.executor import ElasticExecutor
from psrl.workers.elastic.policy_runtime import PolicyIdentityMap

pytestmark = pytest.mark.cpu_test


def _executor():
    cls = ElasticExecutor.__ray_metadata__.modified_class
    executor = object.__new__(cls)
    executor.instances_status_flags = {}
    executor.instances_engine_stats = {}
    executor.instance_pool_ids = {}
    executor.role_model_versions = {}
    executor.topology = ClusterTopology()
    executor.policy_identity_map = PolicyIdentityMap()
    return executor


def test_register_role_tracks_identity_placement_and_pool():
    executor = _executor()
    role = PSRL_Role.Rollout
    iid = ("worker", 0)
    slots = frozenset({GPUSlot("node", 0)})
    executor.register_role(role, "model", [iid], [slots], ["rollout_pool"])
    assert executor.instances_status_flags[role]["model"][iid] is InstanceStatus.ASLEEP, "Instances start asleep."
    assert executor.topology.get_gpu_slots(role, "model", iid) == slots, "The topology must retain placement."
    assert executor.instance_pool_ids[(role, "model", iid)] == "rollout_pool", "The instance must retain its pool."
    assert executor.role_model_versions[(role, "model")] == 0, "The initial role model version must be zero."


@pytest.mark.parametrize("placements,pools", [([], ["pool"]), ([frozenset()], [])])
def test_register_role_rejects_incomplete_registration(placements, pools):
    with pytest.raises(ValueError):
        _executor().register_role(PSRL_Role.Rollout, "model", [("worker", 0)], placements, pools)
