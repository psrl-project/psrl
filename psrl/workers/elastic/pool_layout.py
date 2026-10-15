"""
Plan elastic replicas within the physical boundaries of each resource pool.
"""

from dataclasses import dataclass

from psrl.utils.deployment_mode import expand_ngpus_per_node


@dataclass(frozen=True)
class ReplicaPlacement:
    pool_id: str
    start_bundle_index: int
    world_size: int


def elastic_pool_specs(deployment) -> dict[str, list[int]]:
    """
    Return GPU counts for pools that may host elastic replicas.
    """
    cfg = deployment.elastic
    pools = {
        "shared_rollout_pool": expand_ngpus_per_node(cfg.shared_ngpus_per_node, cfg.shared_nnodes),
    }
    if cfg.get("enable_trainer_pool", False):
        pools["train_pool"] = expand_ngpus_per_node(deployment.train_ngpus_per_node, deployment.train_nnodes)
    return pools


def plan_replicas(pools: dict[str, list[int]], world_size: int) -> list[ReplicaPlacement]:
    """
    Tile each pool independently without reusing a bundle within one role.
    """
    if world_size <= 0:
        raise ValueError(f"Replica world size must be positive, got {world_size}.")
    return [
        ReplicaPlacement(pool_id, start, world_size)
        for pool_id, node_gpus in pools.items()
        for start in range(0, sum(node_gpus) - world_size + 1, world_size)
    ]
