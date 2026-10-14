"""Verify pool counts through the actual driver and independently tiled placements."""

import ast
import logging
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from psrl.utils.deployment_mode import expand_ngpus_per_node
from psrl.workers.elastic.pool_layout import ReplicaPlacement, elastic_pool_specs, plan_replicas

pytestmark = pytest.mark.cpu_test

ROOT = Path(__file__).resolve().parents[2]


def _config(enabled=True, rm_tp=1):
    return OmegaConf.create(
        {
            "psrl": {
                "colocate_validate_and_train": False,
                "deployment": {
                    "train_nnodes": 1,
                    "train_ngpus_per_node": 8,
                    "n_rollout_instances": 8,
                    "elastic": {
                        "enable": True,
                        "enable_trainer_pool": enabled,
                        "shared_nnodes": 1,
                        "shared_ngpus_per_node": 8,
                    },
                    "heterogeneous_rollout": {"enable": False},
                },
            },
            "gen_actor_rollout_ref": {
                "rollout": {
                    "tensor_model_parallel_size": 1,
                    "pipeline_model_parallel_size": 1,
                    "data_parallel_size": 1,
                }
            },
            "reward": {
                "manager": {
                    "reward_loop_type": "gen",
                    "reward_model_name": "reward",
                    "num_replicas": 8,
                    "rollout": {
                        "tensor_model_parallel_size": rm_tp,
                        "pipeline_model_parallel_size": 1,
                        "data_parallel_size": 1,
                    },
                }
            },
        }
    )


@pytest.mark.parametrize(
    "enabled,rm_tp,rollout_count,rm_count", [(True, 1, 16, 16), (False, 1, 8, 8), (True, 4, 16, 4)]
)
def test_driver_counts_both_pools_without_allocating_extra_gpus(monkeypatch, enabled, rm_tp, rollout_count, rm_count):
    class Role(Enum):
        Rollout = 1
        RewardModel = 2
        Validate = 3

    def resource_manager(**kwargs):
        return SimpleNamespace(**kwargs)

    stub = ModuleType("psrl.trainer.ppo.utils")
    stub.ResourcePoolManager = resource_manager
    monkeypatch.setitem(sys.modules, stub.__name__, stub)
    tree = ast.parse((ROOT / "psrl/trainer/main_ppo.py").read_text())
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "TaskRunner")
    method = next(
        node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "init_resource_pool_mgr"
    )
    namespace = {
        "PSRL_Role": Role,
        "elastic_pool_specs": elastic_pool_specs,
        "plan_replicas": plan_replicas,
        "expand_ngpus_per_node": expand_ngpus_per_node,
        "resolve_active_managers": lambda cfg: [cfg.manager],
        "psrl_logger": logging.getLogger(__name__),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "main_ppo.py", "exec"), namespace)
    cfg = _config(enabled, rm_tp)
    manager = namespace[method.name](SimpleNamespace(mapping={}), cfg)
    assert cfg.psrl.deployment.n_rollout_instances == rollout_count
    assert cfg.reward.manager.num_replicas == rm_count
    assert manager.mapping[Role.Rollout] == [f"rollout_{i}" for i in range(rollout_count)]
    assert manager.mapping[Role.RewardModel] == [f"reward_model_reward_{i}" for i in range(rm_count)]
    assert manager.resource_pool_spec == {"train_pool": [8], "shared_rollout_pool": [8]}
    rollout_pool_ids = [manager.elastic_replica_placements[tag].pool_id for tag in manager.mapping[Role.Rollout]]
    rm_pool_ids = [manager.elastic_replica_placements[tag].pool_id for tag in manager.mapping[Role.RewardModel]]
    assert rollout_pool_ids.count("train_pool") == (8 if enabled else 0)
    assert rm_pool_ids.count("train_pool") == (8 // rm_tp if enabled else 0)
    expected_rollout = plan_replicas(elastic_pool_specs(cfg.psrl.deployment), 1)
    assert [manager.elastic_replica_placements[tag] for tag in manager.mapping[Role.Rollout]] == expected_rollout


def test_replica_placements_are_disjoint_within_each_pool():
    placements = plan_replicas({"shared_rollout_pool": [8], "train_pool": [8]}, 4)
    occupied = set()
    for placement in placements:
        slots = {
            (placement.pool_id, i)
            for i in range(
                placement.start_bundle_index,
                placement.start_bundle_index + placement.world_size,
            )
        }
        assert occupied.isdisjoint(slots)
        occupied.update(slots)
    assert len(placements) == 4
    assert len(occupied) == 16


def test_remainder_capacity_does_not_cross_pool_boundaries():
    assert plan_replicas({"shared_rollout_pool": [3], "train_pool": [3]}, 4) == []


def test_nonuniform_node_counts_are_preserved():
    cfg = _config()
    cfg.psrl.deployment.elastic.shared_nnodes = 2
    cfg.psrl.deployment.elastic.shared_ngpus_per_node = [4, 8]
    assert elastic_pool_specs(cfg.psrl.deployment)["shared_rollout_pool"] == [4, 8]


def test_zero_replica_world_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        plan_replicas({"train_pool": [8]}, 0)


def test_elastic_replica_pools_are_sliced_from_parent():
    from psrl.trainer.ppo.utils import PSRL_Role, ResourcePoolManager

    parent = MagicMock()
    parent.store = [8]
    parent.use_gpu = True
    parent.name_prefix = "shared_rollout_pool"
    parent.max_colocate_count = 4
    parent.detached = False
    parent.accelerator_type = None
    parent.resource_num_per_bundle = 1.0
    parent.pgs = object()
    placement = ReplicaPlacement(pool_id="shared_rollout_pool", start_bundle_index=2, world_size=2)
    manager = ResourcePoolManager(
        resource_pool_spec={"shared_rollout_pool": [8]},
        mapping={PSRL_Role.Rollout: ["rollout_0"]},
        elastic_replica_placements={"rollout_0": placement},
    )
    manager.resource_pool_dict["shared_rollout_pool"] = parent
    manager._create_elastic_replica_pools(device_name="cuda")
    parent.get_placement_groups.assert_called_once_with(strategy="STRICT_PACK", device_name="cuda")
    replica = manager.get_resource_pool(PSRL_Role.Rollout, 0)
    assert replica.start_bundle_index == 2
    assert replica.world_size == 2
    assert replica.pgs is parent.pgs
