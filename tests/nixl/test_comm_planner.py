#!/usr/bin/env python
"""
Tests for NIXL communication planner and network topology.

test_network_topology_integration: CPU-only (pytestmark = cpu_test).
test_communication_planner: CPU-only. NIXLTensorInfo is constructable without the
  nixl C library when desc_bytes_list is None (no real agent descriptors needed for
  testing the planner's routing/assignment logic).
"""

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.comm_plan import (
    NIXLClientInfo,
    NIXLClientType,
    global_comm_planner,
)
from psrl.utils.nixl.network_topology import LinkType, NetworkTopology
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test


def _make_sharding(shard_indices):
    """Helper: build a NIXLSharding with 4-shard mesh and given local indices."""
    return NIXLSharding(
        shard_mesh=OrderedDict([(0, 4)]),
        shard_indices=[(i,) for i in shard_indices],
    )


def _make_tensor_info(shard_indices):
    """
    Build a NIXLTensorInfo for testing comm plan logic.

    Uses None for desc_bytes_list entries, so the nixl C library is not needed
    for testing the planner's routing/assignment logic.
    NIXLShardMetaInfo requires torch.dtype, torch.device, torch.Size.
    """
    sharding = _make_sharding(shard_indices)
    meta = NIXLShardMetaInfo(
        dtype=torch.float32,
        device=torch.device("cuda:0"),
        shape=torch.Size([1000, 1000]),
        stride=(1000, 1),
        is_contiguous=True,
    )
    n = len(shard_indices)
    return NIXLTensorInfo(
        desc_bytes_list=[None] * n,
        temp_desc_bytes_list=[None] * n,
        sharding=sharding,
        shard_meta_infos=[meta] * n,
    )


def test_communication_planner():
    """Test the communication planner with custom sorting.

    Verifies that make_comm_plan() produces a plan with the expected structure
    when given PUSH_SIDE, PS_FOR_PUSH, and PULL_SIDE clients with different
    network topologies (same node/GPU, same node/diff GPU, CPU, different node).
    """
    clients = {}

    # PUSH_SIDE clients
    # Client A: Same node as PS, same GPU (LOCAL)
    clients["push_A"] = NIXLClientInfo(
        name="push_A",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": _make_tensor_info([0, 1])},
        meta=b"push_meta_A",
    )

    # Client B: Same node as PS, different GPU (NVLINK)
    clients["push_B"] = NIXLClientInfo(
        name="push_B",
        node_ip="192.168.1.1",
        node_gpu_id=1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": _make_tensor_info([1, 2])},
        meta=b"push_meta_B",
    )

    # Client C: Same node as PS, CPU (PCIE)
    clients["push_C"] = NIXLClientInfo(
        name="push_C",
        node_ip="192.168.1.1",
        node_gpu_id=-1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": _make_tensor_info([0, 1])},
        meta=b"push_meta_C",
    )

    # Client D: Different node (IB)
    clients["push_D"] = NIXLClientInfo(
        name="push_D",
        node_ip="192.168.1.2",
        node_gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": _make_tensor_info([0, 1, 2])},
        meta=b"push_meta_D",
    )

    # PS_FOR_PUSH client (replaces old NIXLClientType.PS)
    clients["ps_1"] = NIXLClientInfo(
        name="ps_1",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PS_FOR_PUSH,
        tensor_infos={"weight": _make_tensor_info([0, 1, 2])},
        meta=b"ps_meta_1",
    )

    # PULL_SIDE clients
    clients["pull_1"] = NIXLClientInfo(
        name="pull_1",
        node_ip="192.168.1.3",
        node_gpu_id=0,
        type=NIXLClientType.PULL_SIDE,
        tensor_infos={"weight": _make_tensor_info([0, 1])},
        meta=b"pull_meta_1",
    )

    # PS_FOR_PULL client
    clients["ps_for_pull_1"] = NIXLClientInfo(
        name="ps_for_pull_1",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PS_FOR_PULL,
        tensor_infos={"weight": _make_tensor_info([0, 1, 2])},
        meta=b"ps_for_pull_meta_1",
    )

    # Generate communication plan
    comm_plan = global_comm_planner.make_comm_plan(clients)

    # Verify plan structure
    assert hasattr(comm_plan, "push_to_ps_plan"), "comm_plan must have push_to_ps_plan"
    assert hasattr(comm_plan, "rollout_pull_from_ps_plan"), "comm_plan must have rollout_pull_from_ps_plan"

    # Verify PUSH_SIDE -> PS_FOR_PUSH plan has entries and link types are valid
    for push_client, key_plans in comm_plan.push_to_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(push_client, ps_client)
                assert link_type is not None, (
                    f"Link type is missing. push_client={push_client!r}, ps_client={ps_client!r}."
                )
                assert isinstance(link_type, LinkType), f"Expected LinkType instance, got {type(link_type)}"

    # Verify PULL_SIDE <- PS_FOR_PULL plan has entries and link types are valid
    for pull_client, key_plans in comm_plan.rollout_pull_from_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(ps_client, pull_client)
                assert link_type is not None, (
                    f"Link type is missing. ps_client={ps_client!r}, pull_client={pull_client!r}."
                )
                assert isinstance(link_type, LinkType), f"Expected LinkType instance, got {type(link_type)}"


def test_network_topology_integration():
    """Test network topology link-type and bandwidth determination.

    CPU-only and does not require the nixl C library or a GPU.
    """
    topology = NetworkTopology()

    # Register four clients with different topological relationships
    topology.register_client("client_A", "192.168.1.1", 0)  # GPU 0 on node 1
    topology.register_client("client_B", "192.168.1.1", 1)  # GPU 1 on node 1 (same node, diff GPU)
    topology.register_client("client_C", "192.168.1.1", -1)  # CPU on node 1
    topology.register_client("client_D", "192.168.1.2", 0)  # GPU 0 on node 2 (different node)

    link_aa = topology.get_link_type("client_A", "client_A")
    link_ab = topology.get_link_type("client_A", "client_B")
    link_ac = topology.get_link_type("client_A", "client_C")
    link_ad = topology.get_link_type("client_A", "client_D")

    # Verify specific link types based on PSRL topology heuristics
    assert link_aa == LinkType.LOCAL  # same GPU → local memory
    assert link_ab == LinkType.NVLINK  # same node, different GPU → NVLink
    assert link_ac == LinkType.PCIE  # same node, CPU (gpu_id=-1) → PCIe
    assert link_ad in (LinkType.IB, LinkType.IB_PCIE)  # different node → InfiniBand

    # All bandwidths must be positive
    bw_local = topology.get_bandwidth_gbps("client_A", "client_A")
    bw_nvlink = topology.get_bandwidth_gbps("client_A", "client_B")
    bw_pcie = topology.get_bandwidth_gbps("client_A", "client_C")
    bw_ib = topology.get_bandwidth_gbps("client_A", "client_D")
    assert bw_local > 0
    assert bw_nvlink > 0
    assert bw_pcie > 0
    assert bw_ib > 0
    assert bw_local >= bw_nvlink, f"Expected LOCAL ({bw_local}) >= NVLINK ({bw_nvlink})"

    # Print bandwidth summary when run as a benchmark (not during pytest CI)
    if __name__ == "__main__":
        print("\nBandwidth results:")
        print(f"  LOCAL  (same GPU):       {bw_local:.1f} Gbps")
        print(f"  NVLINK (same node, GPU): {bw_nvlink:.1f} Gbps")
        print(f"  PCIE   (same node, CPU): {bw_pcie:.1f} Gbps")
        print(f"  IB     (cross-node):     {bw_ib:.1f} Gbps")


if __name__ == "__main__":
    test_network_topology_integration()
    test_communication_planner()
