"""Map Ray resource pools to per-worker node IDs via placement-group metadata."""

import ray
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import (
    SubRayResourcePool,
    _global_bundle_index_to_pg_local_rank,
    sort_placement_group_by_node_ip,
)


def get_resource_pool_for_wg_name(
    resource_pool_to_cls: dict[RayResourcePool, dict],
    wg_name: str,
) -> RayResourcePool:
    """
    Look up the resource pool that owns a worker-group name.

    Args:
        resource_pool_to_cls (dict[RayResourcePool, dict]): Trainer mapping from pool to role classes.
        wg_name (str): Worker-group key such as `rollout_0` or `actor`.

    Returns:
        RayResourcePool: Pool used to create that worker group.

    Raises:
        KeyError: If wg_name is not registered on any pool.
    """
    for pool, class_dict in resource_pool_to_cls.items():
        if wg_name in class_dict:
            return pool
    raise KeyError(f"No resource pool registered for worker group {wg_name!r}.")


def get_node_ids_from_resource_pool(resource_pool: RayResourcePool) -> list[str]:
    """
    Read per-worker Ray node IDs from placement-group GCS metadata.

    Elastic `SubRayResourcePool` workers occupy a contiguous bundle slice of a
    shared placement group. Mapping every bundle would invent ranks that do not exist.

    Args:
        resource_pool (RayResourcePool): Pool whose placement groups have already been created.

    Returns:
        list[str]: Node ID for each worker rank, in worker-creation order.
    """
    pgs = resource_pool.pgs
    assert pgs is not None, "Placement groups must be created before querying node IDs."

    if isinstance(resource_pool, SubRayResourcePool):
        bundle_slots = []
        start = resource_pool.start_bundle_index
        end = start + resource_pool.world_size
        for bundle_idx in range(start, end):
            pg_idx, local_rank = _global_bundle_index_to_pg_local_rank(pgs, bundle_idx)
            bundle_slots.append((pgs[pg_idx], local_rank))
    else:
        local_world_size = resource_pool.store[0]
        bundle_slots = [
            (pg, local_rank) for pg in sort_placement_group_by_node_ip(pgs) for local_rank in range(local_world_size)
        ]

    node_ids = []
    pg_table_cache: dict = {}
    for pg, local_rank in bundle_slots:
        pg_id = pg.id
        if pg_id not in pg_table_cache:
            pg_table_cache[pg_id] = ray._private.state.state.placement_group_table(pg_id)
        bundles_to_node = pg_table_cache[pg_id]["bundles_to_node_id"]
        node_ids.append(bundles_to_node[local_rank])
    return node_ids
