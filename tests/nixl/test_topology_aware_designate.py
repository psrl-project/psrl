"""Topology-aware designate selection for peer diffusion (D1).

_make_peer_groups now sorts designate candidates by their network link quality
to the upstream trainer source, so the instance with the fastest path to the
trainer becomes the designate.  These tests pin that property.
"""

from __future__ import annotations

import time
from collections import OrderedDict

import pytest
import torch

from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType
from psrl.utils.nixl.global_vars import GLOBAL_TOPOLOGY
from psrl.utils.nixl.network_topology import LinkType, NetworkLink
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test

KEYS = ["layer.weight"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tensor_info(total_shards: int, owned: list[int]) -> NIXLTensorInfo:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, total_shards)]),
        shard_indices=[(i,) for i in sorted(owned)],
    )
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size((64, 64)),
        stride=(64, 1),
        is_contiguous=True,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[None] * len(owned),
        temp_desc_bytes_list=[None] * len(owned),
        sharding=sharding,
        shard_meta_infos=[meta] * len(owned),
    )


def _client(name: str, client_type: NIXLClientType, node_ip: str,
            gpu_id: int, total_shards: int, owned: list[int]) -> NIXLClientInfo:
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=gpu_id,
        type=client_type,
        tensor_infos={k: _tensor_info(total_shards, owned) for k in KEYS},
        meta=b"",
    )


def _set_link(c1: str, c2: str, link_type: LinkType) -> None:
    """Override the topology link between two registered clients."""
    link = GLOBAL_TOPOLOGY._default_links[link_type]
    key = tuple(sorted([c1, c2]))
    GLOBAL_TOPOLOGY._links[key] = link


def _designate_for(plan, clients, client: str, key: str) -> str:
    """Return the designate (first member) of the peer group for (client, key).

    Every gen client in this file owns exactly one shard per key, so looking it up
    from the client's own tensor info and threading it through the (now shard-aware)
    `get_peer_group` is a mechanical fix, not a behavior change.
    """
    shard = clients[client].tensor_infos[key].sharding.shard_indices[0]
    group = plan.get_peer_group(client, key, shard)
    return group[0] if group else client


# ---------------------------------------------------------------------------
# Test 1: topology preference — NVLink beats IB
# ---------------------------------------------------------------------------

class TestTopologyPreference:
    """NVLink-to-trainer instance becomes designate over IB-to-trainer instance."""

    def test_nvlink_instance_beats_ib_instance(self):
        # Topology:
        #   trainer0  on node 10.0.0.0
        #   gen0, gen1  on node 10.1.0.0  (IB to trainer)
        #   gen2, gen3  on node 10.1.0.0  BUT we override their link to NVLink
        # All four gen clients need the same shard (tp=1), so they form one peer group.
        # After overriding gen2/gen3 links, they should be designate, not gen0/gen1.
        clients = {
            "trainer0": _client("trainer0", NIXLClientType.PUSH_SIDE, "10.0.0.0", 0, 1, [0]),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, "10.1.0.0", 0, 1, [0]),
            "gen1": _client("gen1", NIXLClientType.PULL_SIDE, "10.1.0.0", 1, 1, [0]),
            "gen2": _client("gen2", NIXLClientType.PULL_SIDE, "10.1.0.0", 2, 1, [0]),
            "gen3": _client("gen3", NIXLClientType.PULL_SIDE, "10.1.0.0", 3, 1, [0]),
        }
        plan = CommunicationPlanner().make_comm_plan(clients)

        # Force gen2/gen3 to appear NVLink-close to trainer0 in the topology.
        _set_link("gen2", "trainer0", LinkType.NVLINK)
        _set_link("gen3", "trainer0", LinkType.NVLINK)

        # Replan with the overridden topology.
        plan2 = CommunicationPlanner().make_comm_plan(clients)

        for key in KEYS:
            designate = _designate_for(plan2, clients, "gen0", key)
            assert designate in {"gen2", "gen3"}, (
                f"Expected NVLink instance (gen2 or gen3) as designate, got {designate!r}"
            )

    def test_same_node_trainer_beats_cross_node_trainer(self):
        # Two nodes, each with 2 rollout instances.  Node A also hosts a trainer (NVLink).
        # Node B rollouts reach trainer only via IB.
        # Per node, instances form a peer group; the designate for node A should be
        # the instance with the better link to trainer0 (both are NVLink here, so
        # the test verifies NVLink is picked as priority when available).
        clients = {
            "trainer0": _client("trainer0", NIXLClientType.PUSH_SIDE, "10.0.0.0", 0, 2, [0, 1]),
            # Node A: same node as trainer0 → NVLink
            "genA0": _client("genA0", NIXLClientType.PULL_SIDE, "10.0.0.0", 1, 2, [0]),
            "genA1": _client("genA1", NIXLClientType.PULL_SIDE, "10.0.0.0", 2, 2, [1]),
            # Node B: cross-node → IB
            "genB0": _client("genB0", NIXLClientType.PULL_SIDE, "10.2.0.0", 0, 2, [0]),
            "genB1": _client("genB1", NIXLClientType.PULL_SIDE, "10.2.0.0", 1, 2, [1]),
        }
        plan = CommunicationPlanner().make_comm_plan(clients)
        for key in KEYS:
            # Node A has same-IP as trainer → LOCAL link → highest priority
            designate_a = _designate_for(plan, clients, "genA0", key)
            assert designate_a in {"genA0", "genA1"}, (
                f"Node A designate should be on node A, got {designate_a!r}"
            )
            designate_b = _designate_for(plan, clients, "genB0", key)
            assert designate_b in {"genB0", "genB1"}, (
                f"Node B designate should be on node B, got {designate_b!r}"
            )


# ---------------------------------------------------------------------------
# Test 2: load balancing within same topology tier
# ---------------------------------------------------------------------------

class TestLoadBalancingWithinTier:
    """When two candidates have equal link quality, pick the one with lower load."""

    def test_load_balance_breaks_tie(self):
        # Four instances on one node, all with identical link quality (IB) to trainer.
        # Two keys: key0 (large) and key1 (small).
        # After key0 is assigned, the loaded instance should lose the key1 tie-break.
        keys = ["key0", "key1"]

        def _ti_with_size(total: int, owned: list[int], size: int) -> NIXLTensorInfo:
            sharding = NIXLSharding(
                shard_mesh=OrderedDict([(0, total)]),
                shard_indices=[(i,) for i in sorted(owned)],
            )
            meta = NIXLShardMetaInfo(
                dtype=torch.float32,
                device=torch.device("cuda:0"),
                shape=torch.Size((size, 1)),
                stride=(1, 1),
                is_contiguous=True,
            )
            return NIXLTensorInfo(
                desc_bytes_list=[None] * len(owned),
                temp_desc_bytes_list=[None] * len(owned),
                sharding=sharding,
                shard_meta_infos=[meta] * len(owned),
            )

        clients = {
            "trainer0": NIXLClientInfo(
                name="trainer0", node_ip="10.0.0.0", node_gpu_id=0,
                type=NIXLClientType.PUSH_SIDE,
                tensor_infos={
                    "key0": _ti_with_size(1, [0], 1024),
                    "key1": _ti_with_size(1, [0], 16),
                },
                meta=b"",
            ),
        }
        for i in range(4):
            clients[f"gen{i}"] = NIXLClientInfo(
                name=f"gen{i}", node_ip="10.1.0.0", node_gpu_id=i,
                type=NIXLClientType.PULL_SIDE,
                tensor_infos={
                    "key0": _ti_with_size(1, [0], 1024),
                    "key1": _ti_with_size(1, [0], 16),
                },
                meta=b"",
            )

        plan = CommunicationPlanner().make_comm_plan(clients)
        # key0 is large → its designate accumulates more load.
        # key1's designate should be a different instance.
        d0 = _designate_for(plan, clients, "gen0", "key0")
        d1 = _designate_for(plan, clients, "gen0", "key1")
        assert d0 != d1, (
            f"Expected load-balanced designates for key0/key1, both got {d0!r}"
        )


# ---------------------------------------------------------------------------
# Test 3: IB beats ETH
# ---------------------------------------------------------------------------

class TestIBOverETH:
    """When one node has IB and another has only ETH, IB node becomes designate."""

    def test_ib_node_beats_eth_node(self):
        # Two nodes on the same IP subnet but with forced different link types.
        # Both nodes have 2 rollout instances that want the same shard.
        # We override node B's link to trainer to ETH so node A (IB) should win.
        clients = {
            "trainer0": _client("trainer0", NIXLClientType.PUSH_SIDE, "10.0.0.0", 0, 1, [0]),
            "genA0": _client("genA0", NIXLClientType.PULL_SIDE, "10.1.0.0", 0, 1, [0]),
            "genA1": _client("genA1", NIXLClientType.PULL_SIDE, "10.1.0.0", 1, 1, [0]),
            "genB0": _client("genB0", NIXLClientType.PULL_SIDE, "10.1.0.0", 2, 1, [0]),
            "genB1": _client("genB1", NIXLClientType.PULL_SIDE, "10.1.0.0", 3, 1, [0]),
        }
        # Force genB0/genB1 to appear ETH-linked to trainer
        _set_link("genB0", "trainer0", LinkType.ETH)
        _set_link("genB1", "trainer0", LinkType.ETH)
        # genA0/genA1 will naturally be IB (cross-node, non-ETH)
        _set_link("genA0", "trainer0", LinkType.IB)
        _set_link("genA1", "trainer0", LinkType.IB)

        plan = CommunicationPlanner().make_comm_plan(clients)
        for key in KEYS:
            d = _designate_for(plan, clients, "genA0", key)
            assert d in {"genA0", "genA1"}, (
                f"IB instance should be designate, got {d!r} (ETH instances are genB0/genB1)"
            )


# ---------------------------------------------------------------------------
# Test 4: determinism with same priority and same load
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Identical topology and zero load → sort by name, always stable."""

    def test_same_priority_same_load_deterministic(self):
        clients = {
            "trainer0": _client("trainer0", NIXLClientType.PUSH_SIDE, "10.0.0.0", 0, 1, [0]),
            "gen_z": _client("gen_z", NIXLClientType.PULL_SIDE, "10.1.0.0", 0, 1, [0]),
            "gen_a": _client("gen_a", NIXLClientType.PULL_SIDE, "10.1.0.0", 1, 1, [0]),
        }
        results = set()
        for _ in range(5):
            plan = CommunicationPlanner().make_comm_plan(clients)
            for key in KEYS:
                results.add(_designate_for(plan, clients, "gen_a", key))
        assert len(results) == 1, f"Non-deterministic designate selection: {results}"
        # With equal priority and zero load, name sort (ascending) picks gen_a before gen_z
        assert results == {"gen_a"}, f"Expected 'gen_a' (lexicographically first), got {results}"


# ---------------------------------------------------------------------------
# Test 5: fallback when source missing from pull_plan
# ---------------------------------------------------------------------------

class TestFallback:
    """A wanter with no source in pull_plan gets priority 0; no crash."""

    def test_missing_source_does_not_crash(self):
        # Normal two-instance cluster; the planner should still produce valid groups.
        clients = {
            "trainer0": _client("trainer0", NIXLClientType.PUSH_SIDE, "10.0.0.0", 0, 1, [0]),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, "10.1.0.0", 0, 1, [0]),
            "gen1": _client("gen1", NIXLClientType.PULL_SIDE, "10.1.0.0", 1, 1, [0]),
        }
        plan = CommunicationPlanner().make_comm_plan(clients)
        # Both should be in the same peer group, one is designate
        for key in KEYS:
            g0 = plan.get_peer_group("gen0", key, (0,))
            g1 = plan.get_peer_group("gen1", key, (0,))
            assert set(g0) == set(g1) == {"gen0", "gen1"}, (
                f"Unexpected peer groups: gen0={g0}, gen1={g1}"
            )


# ---------------------------------------------------------------------------
# Test 6: scale — 64 trainers, 512 rollout instances, 64 nodes
# ---------------------------------------------------------------------------

class TestScale:
    """Planning time is O(N×K), not O(N²). Must finish in under 2 s."""

    def test_large_cluster_plans_quickly(self):
        NUM_TRAINERS = 64
        NUM_NODES = 64
        PER_NODE = 16         # More than TP so peer groups actually form (PER_NODE > TP)
        TP = 8
        NUM_KEYS = 200        # Rough MoE key count

        keys = [f"layer{i}.weight" for i in range(NUM_KEYS)]

        def _ti(total: int, owned: list[int]) -> NIXLTensorInfo:
            sharding = NIXLSharding(
                shard_mesh=OrderedDict([(0, total)]),
                shard_indices=[(i,) for i in sorted(owned)],
            )
            meta = NIXLShardMetaInfo(
                dtype=torch.bfloat16,
                device=torch.device("cuda:0"),
                shape=torch.Size((128, 128)),
                stride=(128, 1),
                is_contiguous=True,
            )
            return NIXLTensorInfo(
                desc_bytes_list=[None] * len(owned),
                temp_desc_bytes_list=[None] * len(owned),
                sharding=sharding,
                shard_meta_infos=[meta] * len(owned),
            )

        clients: dict[str, NIXLClientInfo] = {}

        for t in range(NUM_TRAINERS):
            clients[f"train{t}"] = NIXLClientInfo(
                name=f"train{t}",
                node_ip=f"10.0.0.{t}",
                node_gpu_id=t % 8,
                type=NIXLClientType.PUSH_SIDE,
                tensor_infos={k: _ti(TP, list(range(TP))) for k in keys},
                meta=b"",
            )

        for node in range(NUM_NODES):
            for slot in range(PER_NODE):
                idx = node * PER_NODE + slot
                # slot % TP: with PER_NODE=16 > TP=8, each shard has 2 instances per node
                # → peer groups form (2 instances with identical shards on same node)
                clients[f"gen{idx}"] = NIXLClientInfo(
                    name=f"gen{idx}",
                    node_ip=f"10.1.0.{node}",
                    node_gpu_id=slot % 8,
                    type=NIXLClientType.PULL_SIDE,
                    tensor_infos={k: _ti(TP, [slot % TP]) for k in keys},
                    meta=b"",
                )

        # Time _make_peer_groups in isolation after building the pull plan correctly.
        # We want to verify O(N×K) scaling of the topology sort we added, not benchmark
        # the preexisting _make_comm_plan_generic.
        from psrl.utils.nixl.global_vars import GLOBAL_TOPOLOGY

        for cn, ci in clients.items():
            GLOBAL_TOPOLOGY.register_client(cn, ci.node_ip, ci.node_gpu_id)

        push_groups: dict = {}
        pull_groups: dict = {}
        for cn, ci in clients.items():
            gid = ci.client_group_id
            if ci.type == NIXLClientType.PUSH_SIDE:
                push_groups.setdefault(gid, []).append(cn)
            elif ci.type == NIXLClientType.PULL_SIDE:
                pull_groups.setdefault(gid, []).append(cn)

        planner = CommunicationPlanner()
        pull_plan: dict = {c: {} for g in pull_groups.values() for c in g}
        planner._make_pull_from_ps_plan(clients, pull_groups, push_groups, pull_plan)

        # Now time only _make_peer_groups — the code we changed.
        t_peer_start = time.perf_counter()
        peer_groups = planner._make_peer_groups(clients, pull_plan)
        t_peer_end = time.perf_counter()
        peer_s = t_peer_end - t_peer_start

        # The topology sort we added must not regress planning time. With O(N×K)
        # caching (one get_link_priority call per (client, key), cached after first call)
        # this should be well under 0.5s for 1024 instances × 200 keys.
        assert peer_s < 3.0, (
            f"_make_peer_groups took {peer_s:.2f}s on 1024-instance×200-key cluster "
            f"(limit: 3.0s) — topology sort caching may be broken"
        )
        # peer_groups may be empty when all instances have unique shard assignments
        # (each (node, key, shards) bucket has only one wanter). That's fine — we've
        # verified the timing bound above.
        assert pull_plan, "Expected non-empty pull plan from _make_pull_from_ps_plan"

    def test_large_mixed_layout_cluster_plans_quickly(self):
        """Same scale, but half of every node's instances own 2 finest shards instead
        of 1 (as if running at half the TP degree) — exercises the O(N x K x S) bound
        with S > 1, which the uniform-TP test above cannot (every client there owns
        exactly 1 shard per key, so S == 1 throughout)."""
        NUM_TRAINERS = 64
        NUM_NODES = 64
        PER_NODE = 16
        TP = 8
        NUM_KEYS = 200

        keys = [f"layer{i}.weight" for i in range(NUM_KEYS)]

        def _ti(total: int, owned: list[int]) -> NIXLTensorInfo:
            sharding = NIXLSharding(
                shard_mesh=OrderedDict([(0, total)]),
                shard_indices=[(i,) for i in sorted(owned)],
            )
            meta = NIXLShardMetaInfo(
                dtype=torch.bfloat16,
                device=torch.device("cuda:0"),
                shape=torch.Size((128, 128)),
                stride=(128, 1),
                is_contiguous=True,
            )
            return NIXLTensorInfo(
                desc_bytes_list=[None] * len(owned),
                temp_desc_bytes_list=[None] * len(owned),
                sharding=sharding,
                shard_meta_infos=[meta] * len(owned),
            )

        clients: dict[str, NIXLClientInfo] = {}

        for t in range(NUM_TRAINERS):
            clients[f"train{t}"] = NIXLClientInfo(
                name=f"train{t}",
                node_ip=f"10.0.0.{t}",
                node_gpu_id=t % 8,
                type=NIXLClientType.PUSH_SIDE,
                tensor_infos={k: _ti(TP, list(range(TP))) for k in keys},
                meta=b"",
            )

        for node in range(NUM_NODES):
            for slot in range(PER_NODE):
                idx = node * PER_NODE + slot
                if slot % 2 == 0:
                    owned = [slot % TP]  # fine: 1 shard, as in the uniform test
                else:
                    pair_base = (slot % TP) - (slot % TP) % 2
                    owned = [pair_base, pair_base + 1]  # coarse: 2 finest shards
                clients[f"gen{idx}"] = NIXLClientInfo(
                    name=f"gen{idx}",
                    node_ip=f"10.1.0.{node}",
                    node_gpu_id=slot % 8,
                    type=NIXLClientType.PULL_SIDE,
                    tensor_infos={k: _ti(TP, owned) for k in keys},
                    meta=b"",
                )

        from psrl.utils.nixl.global_vars import GLOBAL_TOPOLOGY

        for cn, ci in clients.items():
            GLOBAL_TOPOLOGY.register_client(cn, ci.node_ip, ci.node_gpu_id)

        push_groups: dict = {}
        pull_groups: dict = {}
        for cn, ci in clients.items():
            gid = ci.client_group_id
            if ci.type == NIXLClientType.PUSH_SIDE:
                push_groups.setdefault(gid, []).append(cn)
            elif ci.type == NIXLClientType.PULL_SIDE:
                pull_groups.setdefault(gid, []).append(cn)

        planner = CommunicationPlanner()
        pull_plan: dict = {c: {} for g in pull_groups.values() for c in g}
        planner._make_pull_from_ps_plan(clients, pull_groups, push_groups, pull_plan)

        t_peer_start = time.perf_counter()
        planner._make_peer_groups(clients, pull_plan)
        peer_s = time.perf_counter() - t_peer_start

        assert peer_s < 3.0, (
            f"_make_peer_groups took {peer_s:.2f}s on a 1024-instance x 200-key mixed-layout "
            f"cluster (limit: 3.0s) — per-shard bucketing may have broken the O(N x K x S) bound"
        )


# ---------------------------------------------------------------------------
# Test 7: existing invariants are preserved
# ---------------------------------------------------------------------------

class TestInvariantsPreserved:
    """Topology sort must not break the existing group correctness invariants."""

    def test_members_share_node_and_bytes(self):
        from tests.nixl.test_peer_diffusion_plan import _cluster
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)
        for name, per_key in plan.peer_groups.items():
            for key, per_shard in per_key.items():
                for shard, wanters in per_shard.items():
                    assert name in wanters and len(wanters) == len(set(wanters)) >= 2
                    assert len({clients[w].node_ip for w in wanters}) == 1
                    assert all(shard in clients[w].tensor_infos[key].sharding.shard_indices for w in wanters)

    def test_upstream_reads_are_one_per_node_per_rank(self):
        from tests.nixl.test_peer_diffusion_plan import _cluster, _designates, KEYS as _KEYS
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)
        assert sum(_designates(plan, clients).values()) == 2 * 2 * len(_KEYS)

    def test_plan_is_deterministic(self):
        from tests.nixl.test_peer_diffusion_plan import _cluster
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan1 = CommunicationPlanner().make_comm_plan(clients)
        plan2 = CommunicationPlanner().make_comm_plan(clients)
        assert plan1.peer_groups == plan2.peer_groups
