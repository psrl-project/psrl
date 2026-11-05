"""A peer-diffusion follower must hold its designate's descriptors (E4 two-stage join).

The two-stage join loads only ~401 B of connection info for a target unless the client is a
*reader* of it, and `_is_reader_of` decided that from the three plan directions. A same-node
follower reads its designate through `peer_groups`, which is not one of them, so it held
connection info only and `prep_xfer_dlist` raised NIXL_ERR_NOT_FOUND. The first pull never
exercises the peer path, so the failure appeared on the second pull of a real run, after
weights had been pulled once.
"""

from __future__ import annotations

import pytest
from psrl.utils.nixl.comm_plan import CommunicationPlanner
from psrl.utils.nixl.registration_policy import (
    descriptor_readers_by_source,
    needs_rdma_registration,
    readers_by_source,
)

from tests.nixl.test_peer_diffusion_plan import _cluster

pytestmark = pytest.mark.cpu_test

LAYOUTS = [(1, 2, 1), (1, 4, 1), (1, 8, 2), (2, 8, 2), (4, 8, 2)]


def _follower_pairs(plan):
    """Yield (follower, designate, key), deduplicated -- a key's shards can now have
    more than one distinct designate, but the registration question this file checks
    ("is the follower a descriptor reader of this designate for this key") only cares
    about the pair, not which specific shard drove it."""
    seen = set()
    for follower, per_key in plan.peer_groups.items():
        for key, per_shard in per_key.items():
            for group in per_shard.values():
                designate = group[0]
                if follower != designate and (follower, designate, key) not in seen:
                    seen.add((follower, designate, key))
                    yield follower, designate, key


@pytest.mark.parametrize("nodes,per_node,tp", LAYOUTS)
def test_every_follower_is_a_descriptor_reader_of_its_designate(nodes, per_node, tp):
    plan = CommunicationPlanner().make_comm_plan(_cluster(nodes=nodes, per_node=per_node, tp=tp))
    readers = descriptor_readers_by_source(plan)

    pairs = list(_follower_pairs(plan))
    assert pairs, "no peer groups: the test would pass vacuously"
    for follower, designate, key in pairs:
        assert follower in readers.get(designate, {}).get(key, set()), (
            f"{follower} reads {designate}/{key} in a peer wave but is not a descriptor reader of it"
        )


@pytest.mark.parametrize("nodes,per_node,tp", LAYOUTS)
def test_the_plain_inversion_still_misses_them(nodes, per_node, tp):
    """Documents why a separate function exists: the registration verdict must not change."""
    plan = CommunicationPlanner().make_comm_plan(_cluster(nodes=nodes, per_node=per_node, tp=tp))
    plain = readers_by_source(plan)

    assert any(f not in plain.get(d, {}).get(k, set()) for f, d, k in _follower_pairs(plan))


@pytest.mark.parametrize("nodes,per_node,tp", LAYOUTS)
def test_registration_verdict_is_unchanged_by_peer_readers(nodes, per_node, tp):
    clients = _cluster(nodes=nodes, per_node=per_node, tp=tp)
    on = CommunicationPlanner(peer_diffusion=True).make_comm_plan(clients)
    off = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

    for name in clients:
        assert needs_rdma_registration(on, clients, name) == needs_rdma_registration(off, clients, name)


def test_no_peer_groups_means_no_extra_readers():
    clients = _cluster(nodes=2, per_node=4, tp=1)
    plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

    assert descriptor_readers_by_source(plan) == readers_by_source(plan)
