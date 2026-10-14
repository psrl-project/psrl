from unittest.mock import MagicMock

import pytest
from psrl.trainer.ppo.node_placement import (
    get_node_ids_from_resource_pool,
    get_resource_pool_for_wg_name,
)
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import SubRayResourcePool

pytestmark = pytest.mark.cpu_test


class _FakePlacementGroup:
    def __init__(self, pg_id: str, bundle_count: int):
        self.id = pg_id
        self.bundle_count = bundle_count


def _patch_pg_table(monkeypatch, tables: dict[str, dict[int, str]]) -> None:
    import ray

    def _placement_group_table(pg_id):
        return {"bundles_to_node_id": tables[pg_id]}

    monkeypatch.setattr(ray._private.state.state, "placement_group_table", _placement_group_table)


def test_get_resource_pool_for_wg_name_returns_owning_pool():
    pool_a = MagicMock(name="pool_a")
    pool_b = MagicMock(name="pool_b")
    mapping = {
        pool_a: {"rollout_0": object()},
        pool_b: {"actor": object(), "dummy": object()},
    }

    assert get_resource_pool_for_wg_name(mapping, "rollout_0") is pool_a
    assert get_resource_pool_for_wg_name(mapping, "actor") is pool_b


def test_get_resource_pool_for_wg_name_missing_raises_key_error():
    with pytest.raises(KeyError, match="rollout_9"):
        get_resource_pool_for_wg_name({MagicMock(): {"actor": object()}}, "rollout_9")


def test_subresource_pool_maps_only_owned_bundle_slice(monkeypatch):
    pg = _FakePlacementGroup("pg-shared", 8)
    resource_pool = SubRayResourcePool(
        placement_groups=[pg],
        start_bundle_index=3,
        subgroup_world_size=2,
        process_on_nodes=[8],
        name_prefix="elastic_rollout",
    )
    _patch_pg_table(
        monkeypatch,
        {"pg-shared": {i: f"node-{i}" for i in range(8)}},
    )

    node_ids = get_node_ids_from_resource_pool(resource_pool)

    assert node_ids == ["node-3", "node-4"]


def test_subresource_pool_spans_placement_groups(monkeypatch):
    pg0 = _FakePlacementGroup("pg0", 4)
    pg1 = _FakePlacementGroup("pg1", 4)
    resource_pool = SubRayResourcePool(
        placement_groups=[pg0, pg1],
        start_bundle_index=5,
        subgroup_world_size=2,
        process_on_nodes=[4, 4],
        name_prefix="elastic_rollout",
    )
    _patch_pg_table(
        monkeypatch,
        {
            "pg0": {i: "node-a" for i in range(4)},
            "pg1": {i: f"node-b-{i}" for i in range(4)},
        },
    )

    node_ids = get_node_ids_from_resource_pool(resource_pool)

    assert node_ids == ["node-b-1", "node-b-2"]


def test_full_resource_pool_maps_every_local_rank(monkeypatch):
    import ray

    pg = _FakePlacementGroup("pg-full", 4)
    resource_pool = RayResourcePool(process_on_nodes=[4], name_prefix="train")
    resource_pool.pgs = [pg]
    monkeypatch.setattr(
        ray,
        "nodes",
        lambda: [{"NodeID": "nid", "NodeManagerAddress": "10.0.0.1"}],
    )
    _patch_pg_table(monkeypatch, {"pg-full": {i: "nid" for i in range(4)}})

    node_ids = get_node_ids_from_resource_pool(resource_pool)

    assert node_ids == ["nid", "nid", "nid", "nid"]
