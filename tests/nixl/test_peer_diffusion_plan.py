"""Same node diffusion groups (implementation item D1).

Rollout clients on one node with the same parallel rank hold byte identical shards. The
planner groups them per key so that one member reads upstream and the rest read it over
NVLink. These tests pin the group invariants the pull engine relies on.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo

pytestmark = pytest.mark.cpu_test

KEYS = ["w0", "w1"]


def _tensor_info(shard_count: int, owned: list[int]) -> NIXLTensorInfo:
    sharding = NIXLSharding(
        shard_mesh=OrderedDict([(0, shard_count)]),
        shard_indices=[(i,) for i in sorted(owned)],
    )
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size((16, 16)),
        stride=(16, 1),
        is_contiguous=True,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[None] * len(owned),
        temp_desc_bytes_list=[None] * len(owned),
        sharding=sharding,
        shard_meta_infos=[meta] * len(owned),
    )


def _client(name, client_type, node_ip, shard_count, owned):
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=0,
        type=client_type,
        tensor_infos={key: _tensor_info(shard_count, owned) for key in KEYS},
        meta=b"",
    )


def _cluster(nodes: int, per_node: int, tp: int):
    """Rollouts whose tp_rank repeats within a node, plus trainers holding every shard."""
    clients = {
        f"train{i}": _client(f"train{i}", NIXLClientType.PUSH_SIDE, f"10.0.0.{i}", tp, list(range(tp)))
        for i in range(2)
    }
    for node in range(nodes):
        for slot in range(per_node):
            rank = node * per_node + slot
            clients[f"gen{rank}"] = _client(
                f"gen{rank}", NIXLClientType.PULL_SIDE, f"10.1.0.{node}", tp, [rank % tp]
            )
    return clients


def _designates(plan, clients, participants=None) -> dict[str, int]:
    """Upstream (key, shard) count per client when everyone in `participants` pulls."""
    counts: dict[str, int] = {}
    for name, info in clients.items():
        if info.type is not NIXLClientType.PULL_SIDE:
            continue
        for key in KEYS:
            for shard in info.tensor_infos[key].sharding.shard_indices:
                group = [
                    w for w in plan.get_peer_group(name, key, shard) if participants is None or w in participants
                ]
                if not group or group[0] == name:
                    counts[name] = counts.get(name, 0) + 1
    return counts


class TestGroups:
    def test_members_share_node_and_bytes(self):
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)
        for name, per_key in plan.peer_groups.items():
            for key, per_shard in per_key.items():
                for shard, wanters in per_shard.items():
                    assert name in wanters and len(wanters) == len(set(wanters)) >= 2
                    assert len({clients[w].node_ip for w in wanters}) == 1
                    assert all(shard in clients[w].tensor_infos[key].sharding.shard_indices for w in wanters)

    def test_upstream_reads_are_one_per_node_per_rank(self):
        clients = _cluster(nodes=2, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)
        assert sum(_designates(plan, clients).values()) == 2 * 2 * len(KEYS)

    @pytest.mark.parametrize("per_node", [2, 4, 8])
    def test_upstream_reads_stay_flat_as_clients_per_node_grow(self, per_node):
        clients = _cluster(nodes=1, per_node=per_node, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)
        assert sum(_designates(plan, clients).values()) == 2 * len(KEYS)

    def test_designates_are_spread_over_members(self):
        clients = _cluster(nodes=1, per_node=4, tp=1)
        plan = CommunicationPlanner().make_comm_plan(clients)
        assert len(_designates(plan, clients)) == len(KEYS)

    def test_a_subset_still_has_exactly_one_designate_per_group(self):
        clients = _cluster(nodes=1, per_node=8, tp=2)
        plan = CommunicationPlanner().make_comm_plan(clients)
        subset = {"gen1", "gen2", "gen5"}
        counts = _designates(plan, clients, subset)
        assert sum(counts[n] for n in subset) == 2 * len(KEYS)

    def test_nothing_is_grouped_without_duplication(self):
        clients = _cluster(nodes=2, per_node=2, tp=2)
        assert CommunicationPlanner().make_comm_plan(clients).peer_groups == {}

    def test_disabled_means_no_groups(self):
        clients = _cluster(nodes=1, per_node=8, tp=2)
        assert CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients).peer_groups == {}

    def test_the_plan_is_deterministic_and_survives_serialisation(self):
        clients = _cluster(nodes=2, per_node=8, tp=2)
        first = CommunicationPlanner().make_comm_plan(clients)
        second = CommunicationPlanner().make_comm_plan(clients)
        assert first.peer_groups == second.peer_groups
        assert type(first).deserialize(first.serialize()).peer_groups == first.peer_groups

    def test_a_ps_cluster_also_groups(self):
        clients = _cluster(nodes=1, per_node=8, tp=2)
        for name in [n for n in clients if n.startswith("train")]:
            clients[name].type = NIXLClientType.PS_FOR_PULL
        assert CommunicationPlanner().make_comm_plan(clients).peer_groups

    def test_different_target_layouts_share_an_individual_shard(self):
        """Phase 2: a finer-TP reader's one shard and a coarser-TP reader's matching half
        can share a designate at the shard level, even though their whole shard sets
        (one shard vs two) never match -- the gap exact-shard-set grouping left open."""
        trainer = _client("train0", NIXLClientType.PUSH_SIDE, "10.0.0.0", 4, [0, 1, 2, 3])
        fine = _client("gen_fine", NIXLClientType.PULL_SIDE, "10.1.0.0", 4, [0])
        coarse = _client("gen_coarse", NIXLClientType.PULL_SIDE, "10.1.0.0", 4, [0, 1])
        clients = {"train0": trainer, "gen_fine": fine, "gen_coarse": coarse}
        plan = CommunicationPlanner().make_comm_plan(clients)
        for key in KEYS:
            # Their whole shard sets never matched -- this grouping only exists because
            # _make_peer_groups now looks at individual shards, not whole shard sets.
            assert tuple(fine.tensor_infos[key].sharding.shard_indices) != tuple(
                coarse.tensor_infos[key].sharding.shard_indices
            )
            group = plan.get_peer_group("gen_fine", key, (0,))
            assert set(group) == {"gen_fine", "gen_coarse"}
            assert group == plan.get_peer_group("gen_coarse", key, (0,))
            # gen_coarse's other shard, (1,), is wanted by nobody else -- no peer group for it.
            assert plan.get_peer_group("gen_coarse", key, (1,)) == ()
