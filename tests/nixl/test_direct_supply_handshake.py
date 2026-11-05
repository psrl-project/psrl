"""Descriptor fan-out under trainer direct supply (implementation item D3).

The planner routing rollouts to train clients is not enough on its own: the meta server
decides which remote descriptors each agent is handed, and it only ever gave a
PULL_SIDE client the PS_FOR_PULL ones. With no PS registered that set is empty, and the
rollout's first `client_read` fails with "target client not found in client infos" --
which reads as a handshake bug rather than as a filter nobody updated.

The fan-out decides by what is registered, matching how the planner dispatches. These
tests check both directions, because a filter that always included train clients would
also pass a one-sided test while widening the PS path's fan-out for no reason.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

pytest.importorskip("nixl._api", reason="NIXL not installed")

from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
    NIXLShardMetaInfo,
    NIXLTensorInfo,
)
from psrl.utils.nixl.server import NIXLMetaServer

pytestmark = pytest.mark.cpu_test


def _tensor_info() -> NIXLTensorInfo:
    sharding = NIXLSharding(shard_mesh=OrderedDict([(0, 1)]), shard_indices=[(0,)])
    meta = NIXLShardMetaInfo(
        dtype=torch.bfloat16,
        device=torch.device("cuda:0"),
        shape=torch.Size((4, 4)),
        stride=(4, 1),
        is_contiguous=True,
    )
    return NIXLTensorInfo(
        desc_bytes_list=[None],
        temp_desc_bytes_list=[None],
        sharding=sharding,
        shard_meta_infos=[meta],
    )


def _info(name: str, client_type: NIXLClientType) -> NIXLClientInfo:
    return NIXLClientInfo(
        name=name,
        node_ip="10.0.0.1",
        node_gpu_id=0,
        type=client_type,
        tensor_infos={"w": _tensor_info()},
        meta=b"",
    )


class _Server:
    """`NIXLMetaServer` without constructing a NIXL agent.

    The method under test touches only `connected_clients` and `client_infos`, so
    binding it to a bare object keeps this a CPU test; building a real agent costs
    minutes on this host and would prove nothing extra about a filter.
    """

    def __init__(self, clients: dict[str, NIXLClientType], broadcast_init=False, comm_plan=None):
        self.client_infos = {name: _info(name, t) for name, t in clients.items()}
        self.connected_clients = {f"agent_{name}": [name] for name in clients}
        self._broadcast_init_enabled = broadcast_init
        self.comm_plan = comm_plan

    relevant_for = NIXLMetaServer._get_relevant_client_names_for_agent


DIRECT = {
    "train0": NIXLClientType.PUSH_SIDE,
    "train1": NIXLClientType.PUSH_SIDE,
    "gen0": NIXLClientType.PULL_SIDE,
}

WITH_PS = {
    **DIRECT,
    "ps_push": NIXLClientType.PS_FOR_PUSH,
    "ps_pull": NIXLClientType.PS_FOR_PULL,
}


class TestDirectSupplyFanOut:
    def test_a_rollout_is_given_the_train_descriptors(self):
        server = _Server(DIRECT)
        relevant = server.relevant_for("agent_gen0")

        assert {"train0", "train1"} <= relevant, (
            "a direct-supply rollout received no train descriptors, so its first read "
            "would fail with 'target client not found in client infos'"
        )

    def test_a_rollout_keeps_its_own_client(self):
        server = _Server(DIRECT)
        assert "gen0" in server.relevant_for("agent_gen0")

    def test_trainers_are_not_given_rollout_descriptors(self):
        """The trainer never initiates under direct supply; it is read from."""
        server = _Server(DIRECT)
        relevant = server.relevant_for("agent_train0")

        assert "gen0" not in relevant


class TestPSPathIsUnchanged:
    """A filter that always added train clients would pass the test above too."""

    def test_a_rollout_with_a_ps_is_given_the_ps_descriptors(self):
        server = _Server(WITH_PS)
        assert "ps_pull" in server.relevant_for("agent_gen0")

    def test_a_rollout_with_a_ps_is_not_given_train_descriptors(self):
        server = _Server(WITH_PS)
        relevant = server.relevant_for("agent_gen0")

        assert "train0" not in relevant, (
            "the PS path's fan-out widened; every rollout would now carry every "
            "trainer's descriptors for no reason"
        )

    def test_a_trainer_with_a_ps_is_given_the_push_target(self):
        server = _Server(WITH_PS)
        assert "ps_push" in server.relevant_for("agent_train0")


class TestDispatchFollowsRegistration:
    def test_the_decision_is_made_on_ps_for_pull_presence(self):
        """Matching how the planner dispatches keeps the two from drifting apart."""
        with_pull_only = {
            "train0": NIXLClientType.PUSH_SIDE,
            "gen0": NIXLClientType.PULL_SIDE,
            "ps_pull": NIXLClientType.PS_FOR_PULL,
        }
        server = _Server(with_pull_only)

        relevant = server.relevant_for("agent_gen0")
        assert "ps_pull" in relevant
        assert "train0" not in relevant

    def test_a_push_only_ps_does_not_count_as_a_pull_source(self):
        """PS_FOR_PUSH receives the trainer's writes; it never serves rollouts."""
        push_only = {
            "train0": NIXLClientType.PUSH_SIDE,
            "gen0": NIXLClientType.PULL_SIDE,
            "ps_push": NIXLClientType.PS_FOR_PUSH,
        }
        server = _Server(push_only)

        relevant = server.relevant_for("agent_gen0")
        assert "train0" in relevant, (
            "a PS that only accepts pushes cannot serve the rollout, so the rollout "
            "still needs the train descriptors"
        )


class TestPeerDescriptorsAreIncluded:
    """D1: a group member reads its designate and notifies its followers, so it needs every peer of its groups."""

    class _Plan:
        def __init__(self, groups):
            self.peer_groups = {}
            for key, wanters in groups.items():
                for client in wanters:
                    self.peer_groups.setdefault(client, {})[key] = wanters

        def peer_group_members(self, pull_client):
            return {w for wanters in self.peer_groups.get(pull_client, {}).values() for w in wanters}

    def test_every_group_member_is_included_in_both_directions(self):
        clients = {**DIRECT, "gen1": NIXLClientType.PULL_SIDE}
        server = _Server(clients, comm_plan=self._Plan({"w": ("gen0", "gen1")}))
        assert "gen0" in server.relevant_for("agent_gen1")
        assert "gen1" in server.relevant_for("agent_gen0")

    def test_no_plan_means_no_extra_fan_out(self):
        clients = {**DIRECT, "gen1": NIXLClientType.PULL_SIDE}
        server = _Server(clients, comm_plan=None)
        assert "gen0" not in server.relevant_for("agent_gen1")

    def test_the_fan_out_stays_within_groups(self):
        clients = {**DIRECT, **{f"gen{i}": NIXLClientType.PULL_SIDE for i in range(1, 4)}}
        server = _Server(clients, comm_plan=self._Plan({"w": ("gen0", "gen1"), "v": ("gen2", "gen3")}))
        relevant = server.relevant_for("agent_gen0")
        assert "gen1" in relevant and "gen2" not in relevant and "gen3" not in relevant
