#!/usr/bin/env python
"""
Test communication planning with custom sorting and shard coverage.
"""

import os
import sys
from collections import OrderedDict

import pytest
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from psrl.utils.nixl.comm_plan import global_comm_planner
from psrl.utils.nixl.network_topology import NetworkTopology
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
    NIXLShardMetaInfo,
    NIXLTensorInfo,
)
from psrl.utils.nixl.server import NIXLMetaServer

pytestmark = pytest.mark.cpu_test


def _make_weight_tensor_info(global_shard_indices: list[int], total_global_shards: int = 4) -> NIXLTensorInfo:
    """Build a valid NIXLTensorInfo: 1D shard along dim 0, equal-sized shards."""
    shard_mesh = OrderedDict([(0, total_global_shards)])
    shard_indices = [(i,) for i in global_shard_indices]
    rows = 1000 // total_global_shards
    shape = torch.Size((rows, 1000))
    stride = (1000, 1)
    desc_bytes_list = [f"shard_{i}".encode() for i in global_shard_indices]
    temp_desc_bytes_list = [b""] * len(global_shard_indices)
    shard_meta_infos = [
        NIXLShardMetaInfo(
            dtype=torch.float32,
            device=torch.device("cuda", 0),
            shape=shape,
            stride=stride,
            is_contiguous=True,
        )
        for _ in global_shard_indices
    ]
    sharding = NIXLSharding(shard_mesh=shard_mesh, shard_indices=shard_indices)
    return NIXLTensorInfo(
        desc_bytes_list=desc_bytes_list,
        temp_desc_bytes_list=temp_desc_bytes_list,
        sharding=sharding,
        shard_meta_infos=shard_meta_infos,
    )


def create_test_tensor_desc_info_1():
    """This client holds global shards 0 and 1."""
    return _make_weight_tensor_info([0, 1])


def create_test_tensor_desc_info_2():
    """This client holds global shards 1 and 2."""
    return _make_weight_tensor_info([1, 2])


def create_test_tensor_desc_info_3():
    """This client holds global shards 0, 1 and 2."""
    return _make_weight_tensor_info([0, 1, 2])


def create_test_tensor_desc_info_ps():
    """PS holds global shards 0..2 (union of all PUSH_SIDE shards in this scenario)."""
    return _make_weight_tensor_info([0, 1, 2])


def test_communication_planner():
    """Test the communication planner with custom sorting"""
    print("Testing communication planner with custom sorting...")

    clients = {}

    clients["push_A"] = NIXLClientInfo(
        name="push_A",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"push_meta_A",
    )

    clients["push_B"] = NIXLClientInfo(
        name="push_B",
        node_ip="192.168.1.1",
        node_gpu_id=1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_2()},
        meta=b"push_meta_B",
    )

    clients["push_C"] = NIXLClientInfo(
        name="push_C",
        node_ip="192.168.1.1",
        node_gpu_id=-1,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"push_meta_C",
    )

    clients["push_D"] = NIXLClientInfo(
        name="push_D",
        node_ip="192.168.1.2",
        node_gpu_id=0,
        type=NIXLClientType.PUSH_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_3()},
        meta=b"push_meta_D",
    )

    # Split PS: push path vs rollout pull path (matches production NIXL comm_plan).
    clients["ps_push"] = NIXLClientInfo(
        name="ps_push",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PS_FOR_PUSH,
        tensor_infos={"weight": create_test_tensor_desc_info_ps()},
        meta=b"ps_meta_push",
    )

    clients["ps_pull"] = NIXLClientInfo(
        name="ps_pull",
        node_ip="192.168.1.1",
        node_gpu_id=0,
        type=NIXLClientType.PS_FOR_PULL,
        tensor_infos={"weight": create_test_tensor_desc_info_ps()},
        meta=b"ps_meta_pull",
    )

    clients["pull_1"] = NIXLClientInfo(
        name="pull_1",
        node_ip="192.168.1.3",
        node_gpu_id=0,
        type=NIXLClientType.PULL_SIDE,
        tensor_infos={"weight": create_test_tensor_desc_info_1()},
        meta=b"pull_meta_1",
    )

    comm_plan = global_comm_planner.make_comm_plan(clients)

    print("\nGenerated communication plan:")
    print("=" * 50)

    print("PUSH_SIDE -> PS_FOR_PUSH Plan:")
    for push_client, key_plans in comm_plan.push_to_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(push_client, ps_client)
                print(f"  {push_client} -> {ps_client}: {key} shards {shards} ({link_type.name})")

    print("\nPULL_SIDE <- PS_FOR_PULL Plan (rollout):")
    for pull_client, key_plans in comm_plan.rollout_pull_from_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(ps_client, pull_client)
                print(f"  {ps_client} -> {pull_client}: {key} shards {shards} ({link_type.name})")

    print("\nPUSH_SIDE <- PS_FOR_PUSH Plan (train pull):")
    for push_client, key_plans in comm_plan.train_pull_from_ps_plan.items():
        for key, ps_plans in key_plans.items():
            for ps_client, shards in ps_plans.items():
                link_type = global_comm_planner._get_link_type_for_test(ps_client, push_client)
                print(f"  {ps_client} -> {push_client}: {key} shards {shards} ({link_type.name})")

    print("\nCommunication planner test completed!")


@pytest.mark.parametrize("tp_size", [4, 8])
def test_pull_planner_supports_replicated_kv_shard_indices(tp_size):
    num_kv_heads = 2
    replicas = tp_size // num_kv_heads
    ps_client = f"replicated_kv_ps_tp{tp_size}"
    clients = {
        ps_client: NIXLClientInfo(
            name=ps_client,
            node_ip="192.168.20.1",
            node_gpu_id=0,
            type=NIXLClientType.PS_FOR_PULL,
            tensor_infos={"k_proj.weight": _make_weight_tensor_info([0, 1], total_global_shards=2)},
            meta=b"ps_meta",
        )
    }

    for rank in range(tp_size):
        shard_index = rank // replicas
        client_name = f"replicated_kv_pull_tp{tp_size}_rank{rank}"
        clients[client_name] = NIXLClientInfo(
            name=client_name,
            node_ip=f"192.168.21.{rank + 1}",
            node_gpu_id=rank,
            type=NIXLClientType.PULL_SIDE,
            tensor_infos={"k_proj.weight": _make_weight_tensor_info([shard_index], total_global_shards=2)},
            meta=f"pull_meta_{rank}".encode(),
        )

    comm_plan = global_comm_planner.make_comm_plan(clients)

    for rank in range(tp_size):
        client_name = f"replicated_kv_pull_tp{tp_size}_rank{rank}"
        expected_shard = (rank // replicas,)
        assert comm_plan.rollout_pull_from_ps_plan[client_name]["k_proj.weight"] == {ps_client: [expected_shard]}


def test_push_planner_supports_sparse_per_expert_source_ownership():
    num_experts = 8
    world_size = 4
    ps_client = "expert_ps_push"
    expert_keys = {
        expert_id: [
            f"model.layers.0.mlp.experts.{expert_id}.{projection}_proj.weight" for projection in ("gate", "up", "down")
        ]
        for expert_id in range(num_experts)
    }
    clients = {
        ps_client: NIXLClientInfo(
            name=ps_client,
            node_ip="192.168.30.1",
            node_gpu_id=0,
            type=NIXLClientType.PS_FOR_PUSH,
            tensor_infos={
                key: _make_weight_tensor_info([0], total_global_shards=1)
                for keys in expert_keys.values()
                for key in keys
            },
            meta=b"ps_meta",
        )
    }

    experts_per_rank = num_experts // world_size
    for rank in range(world_size):
        client_name = f"expert_push_rank{rank}"
        first_expert = rank * experts_per_rank
        local_keys = [
            key for expert_id in range(first_expert, first_expert + experts_per_rank) for key in expert_keys[expert_id]
        ]
        clients[client_name] = NIXLClientInfo(
            name=client_name,
            node_ip=f"192.168.31.{rank + 1}",
            node_gpu_id=rank,
            type=NIXLClientType.PUSH_SIDE,
            tensor_infos={key: _make_weight_tensor_info([0], total_global_shards=1) for key in local_keys},
            meta=f"push_meta_{rank}".encode(),
        )

    comm_plan = global_comm_planner.make_comm_plan(clients)

    for expert_id, keys in expert_keys.items():
        owner = f"expert_push_rank{expert_id // experts_per_rank}"
        for key in keys:
            assert comm_plan.push_to_ps_plan[owner][key] == {ps_client: [(0,)]}


def test_network_topology_integration():
    """Test network topology integration with communication planner"""
    print("\nTesting network topology integration...")

    topology = NetworkTopology()

    topology.register_client("client_A", "192.168.1.1", 0)
    topology.register_client("client_B", "192.168.1.1", 1)
    topology.register_client("client_C", "192.168.1.1", -1)
    topology.register_client("client_D", "192.168.1.2", 0)

    print("Link type determination:")
    print(f"  client_A -> client_A: {topology.get_link_type('client_A', 'client_A').name}")
    print(f"  client_A -> client_B: {topology.get_link_type('client_A', 'client_B').name}")
    print(f"  client_A -> client_C: {topology.get_link_type('client_A', 'client_C').name}")
    print(f"  client_A -> client_D: {topology.get_link_type('client_A', 'client_D').name}")

    print("\nBandwidth values:")
    print(f"  LOCAL: {topology.get_bandwidth_gbps('client_A', 'client_A')} Gbps")
    print(f"  NVLINK: {topology.get_bandwidth_gbps('client_A', 'client_B')} Gbps")
    print(f"  PCIE: {topology.get_bandwidth_gbps('client_A', 'client_C')} Gbps")
    print(f"  IB: {topology.get_bandwidth_gbps('client_A', 'client_D')} Gbps")

    print("\nNetwork topology integration test completed!")


def test_missing_source_key_reports_planner_context():
    clients = {
        "push": NIXLClientInfo(
            name="push",
            node_ip="192.168.1.1",
            node_gpu_id=0,
            type=NIXLClientType.PUSH_SIDE,
            tensor_infos={"source.weight": _make_weight_tensor_info([0])},
            meta=b"push_meta",
        ),
        "ps_push": NIXLClientInfo(
            name="ps_push",
            node_ip="192.168.1.2",
            node_gpu_id=0,
            type=NIXLClientType.PS_FOR_PUSH,
            tensor_infos={"target.weight": _make_weight_tensor_info([0])},
            meta=b"ps_meta",
        ),
    }

    with pytest.raises(RuntimeError, match="phase=push_to_ps.*target=ps_push.*target.weight"):
        global_comm_planner.make_comm_plan(clients)


def test_incomplete_source_shards_report_missing_coverage():
    clients = {
        name: NIXLClientInfo(
            name=name,
            node_ip="127.0.0.1",
            node_gpu_id=0,
            type=client_type,
            tensor_infos={"weight": _make_weight_tensor_info(indices)},
            meta=b"meta",
        )
        for name, client_type, indices in [
            ("push", NIXLClientType.PUSH_SIDE, [0]),
            ("ps_push", NIXLClientType.PS_FOR_PUSH, [0, 1]),
        ]
    }
    with pytest.raises(AssertionError, match="Not all shards assigned.*weight.*ps_push"):
        global_comm_planner.make_comm_plan(clients)


def test_unified_sharding_refinement_does_not_alias_split_keys():
    shared_ps_sharding = NIXLSharding.default()
    server = object.__new__(NIXLMetaServer)
    server._is_all_client_shardings_recved = True
    server.client_unified_sharding_dicts = {}
    server.client_sharding_dicts = {
        "ps": {
            "weight_q": shared_ps_sharding,
            "weight_v": shared_ps_sharding,
        },
        "train": {
            "weight_q": NIXLSharding(
                shard_mesh=OrderedDict([(0, 2)]),
                shard_indices=[(0,)],
            ),
            "weight_v": NIXLSharding(
                shard_mesh=OrderedDict([(0, 4)]),
                shard_indices=[(0,)],
            ),
        },
    }

    server.make_unified_sharding()

    ps_q = server.client_unified_sharding_dicts["ps"]["weight_q"]
    ps_v = server.client_unified_sharding_dicts["ps"]["weight_v"]
    assert ps_q.shard_mesh == OrderedDict([(0, 2)])
    assert ps_v.shard_mesh == OrderedDict([(0, 4)])
    assert ps_q is not ps_v
    assert shared_ps_sharding.shard_mesh == OrderedDict([(0, 1)])


if __name__ == "__main__":
    test_network_topology_integration()
    test_communication_planner()
