"""CPU tests: ElasticExecutor uses RolloutInstanceId keys throughout."""

import importlib
from unittest.mock import MagicMock

import psrl.utils.elastic_rm.elastic_executor as _executor_module
import pytest
import ray
from psrl.utils.elastic_rm.cluster_topology import GPUSlot, InstanceStatus

pytestmark = pytest.mark.cpu_test

# `ElasticExecutor` is decorated with `@ray.remote`. Reload the module with a pass-through
# decorator so it is a plain class for CPU tests, then restore the real `ray.remote`.
_ray_remote = ray.remote
ray.remote = lambda cls=None, **kwargs: (cls if cls is not None else lambda c: c)
try:
    importlib.reload(_executor_module)
finally:
    ray.remote = _ray_remote
ClusterTopology = _executor_module.ClusterTopology
ElasticExecutor = _executor_module.ElasticExecutor


class _FakePolicy:
    min_awake_per_role = 0

    def decide(self, *args, **kwargs):
        return MagicMock(actions=[])


def _make_executor():
    role = MagicMock()
    model = "test_model"
    coord = MagicMock()
    ex = ElasticExecutor.__new__(ElasticExecutor)
    ex.coordinators = {role: {model: coord}}
    ex.roles = [(role, model)]
    ex.instances_status_flags = {}
    ex.instances_engine_stats = {}
    ex.topology = ClusterTopology()
    ex.scaling_policy = _FakePolicy()
    ex.elastic_rm_config = {}
    ex._post_scale_up_abort_waiting_ratio = 0.0
    ex._coordinator_command_timeout_s = None
    ex._coordinator_sync_timeout_s = 60.0
    ex._decision_execution_in_progress = False
    ex._next_decision_id = 1
    ex._decision_pending_action_counts = {}
    ex._execution_in_progress_stall_ticks = 0
    ex._decision_abandon_stall_ticks = 0
    ex.router_backlog_by_role = {}
    ex.trainer_waiting_hint = {}
    ex._last_monitor_instance_log_ms = 0.0
    ex._monitor_instance_log_interval_ms = 5000
    ex._enable_monitor_instance_log = False
    return ex, role, model


def test_register_role_accepts_rollout_instance_id_list():
    """register_role must key instances by RolloutInstanceId tuples, not an int count."""
    ex, role, model = _make_executor()
    ids = [("wid-0", 0), ("wid-0", 1), ("wid-1", 0)]
    ex.register_role(role, model, ids, [frozenset() for _ in ids])
    assert ("wid-0", 0) in ex.instances_status_flags[role][model]
    assert ("wid-0", 1) in ex.instances_status_flags[role][model]
    assert ("wid-1", 0) in ex.instances_status_flags[role][model]


def test_register_role_rejects_int():
    """register_role must NOT accept a plain integer (old API)."""
    ex, role, model = _make_executor()
    with pytest.raises(TypeError):
        ex.register_role(role, model, 3, [])


def test_register_role_records_gpu_slots_under_tuple_key():
    ex, role, model = _make_executor()
    ids = [("wid-0", 0)]
    gpu_slots = [frozenset({GPUSlot(node_id="node1", gpu_id=0)})]
    ex.register_role(role, model, ids, gpu_slots)
    assert ex.topology.get_gpu_slots(role, model, ("wid-0", 0)) == gpu_slots[0]


def test_select_initial_awake_ids_uses_tuple_ids():
    ex, role, model = _make_executor()
    ids = [("wid-0", 0), ("wid-1", 0)]
    ex.register_role(role, model, ids, [frozenset() for _ in ids])
    awake = ex.select_initial_awake_ids(role, model, target_awake_num=1)
    assert awake == [("wid-0", 0)]
    assert ex.instances_status_flags[role][model][("wid-0", 0)] == InstanceStatus.AWAKEN
    assert ex.instances_status_flags[role][model][("wid-1", 0)] == InstanceStatus.ASLEEP


def test_register_role_stores_engine_stats_under_tuple_key():
    """Engine stats are registered under the RolloutInstanceId tuple key."""
    ex, role, model = _make_executor()
    ids = [("wid-0", 0)]
    ex.register_role(role, model, ids, [frozenset()])
    assert ("wid-0", 0) in ex.instances_engine_stats[role][model]
