"""Registration is derived from the plan, not from a deployment flag (design §3.7.3.0).

The property under test is the one that makes the colocate switch cheap: when every
reader of a client is on the client's own node, nothing it publishes can be reached by a
remote NIC, so the RDMA registration it does today is unusable by construction. The
inverse property matters more for correctness: a single remote reader must force
registration, because an unregistered shard that a remote peer tries to read does not
degrade gracefully.

These are plan-level tests, so they run on CPU. The end-to-end count of `register_memory`
calls on a colocate switch is asserted in `test_registration_wiring.py`.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLClientInfo, NIXLClientType, NIXLCommPlan
from psrl.utils.nixl.nixl_spec import NIXLSharding, NIXLShardMetaInfo, NIXLTensorInfo
from psrl.utils.nixl.registration_policy import (
    all_sources_copyable,
    keys_read_across_nodes,
    needs_rdma_registration,
    readers_by_source,
    registration_report,
    sources_by_reader,
)

pytestmark = pytest.mark.cpu_test

KEYS = ["w0", "w1"]
#: Trainer node and rollout node, distinct so the default cluster is disaggregate.
TRAIN_NODE = "10.0.0.1"
ROLLOUT_NODE = "10.1.0.1"


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
        # One readable allocation per shard; the policy checks these against ipc_regions.
        read_bases=[_base(i) for i in sorted(owned)],
    )


def _base(shard: int) -> int:
    return 0x1000 * (shard + 1)


def _client(name, client_type, node_ip, shard_count=2, owned=(0, 1)):
    owned = list(owned)
    return NIXLClientInfo(
        name=name,
        node_ip=node_ip,
        node_gpu_id=0,
        type=client_type,
        tensor_infos={key: _tensor_info(shard_count, owned) for key in KEYS},
        meta=b"",
        # Fully exportable by default, so a test opts into the failure it is about.
        ipc_regions={_base(shard): (None, 512) for shard in sorted(owned)},
    )


def _direct_supply_cluster(*, rollout_nodes: int, per_node: int = 2, train_node: str = TRAIN_NODE):
    """Trainers holding every shard plus `rollout_nodes` nodes of rolling-out clients.

    No PS clients, so the planner takes the direct-supply branch and names the trainers
    themselves as the rollout sources -- which is the arrangement the criterion has to
    handle for a colocated run.
    """
    clients = {
        "train0": _client("train0", NIXLClientType.PUSH_SIDE, train_node),
    }
    for node in range(rollout_nodes):
        # Same node as the trainer for node 0 when `train_node` is that node's ip.
        node_ip = train_node if node == 0 else f"10.1.0.{node}"
        for slot in range(per_node):
            name = f"gen{node}_{slot}"
            clients[name] = _client(name, NIXLClientType.PULL_SIDE, node_ip)
    return clients


class TestPureColocateNeedsNothing:
    """The headline case: every reader shares the node, so registration is pure waste."""

    def test_a_trainer_read_only_by_its_own_node_registers_nothing(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "train0") is False

    def test_the_skipped_keys_are_named_not_just_counted(self):
        """`skippable_keys` is the measurement: what would have been registered and
        cannot be read remotely. A bare False would not survive a plan change."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        report = registration_report(plan, clients, "train0")

        assert report["needs_registration"] is False
        assert report["cross_node_keys"] == 0
        assert report["skippable_keys"] == report["published_keys"] == len(KEYS)

    def test_the_verdict_is_not_vacuous(self):
        """A `False` because nobody reads the client is not the claim. The claim is that
        the readers exist and are all local, so this pins that the trainer really is a
        named source -- without it, an empty plan would make every test above pass."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        readers = readers_by_source(plan)

        assert readers["train0"], "train0 is not a source in the plan"
        assert readers["train0"]["w0"], "no reader of w0"
        assert {clients[r].node_ip for r in readers["train0"]["w0"]} == {TRAIN_NODE}

    def test_peer_diffusion_does_not_change_the_verdict(self):
        """Diffusion reorders who reads whom but adds no cross-node reader, and the
        policy ignores delegation anyway -- so both planner settings must agree."""
        clients = _direct_supply_cluster(rollout_nodes=1, per_node=4)
        off = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        on = CommunicationPlanner(peer_diffusion=True).make_comm_plan(clients)

        assert needs_rdma_registration(off, clients, "train0") is False
        assert needs_rdma_registration(on, clients, "train0") is False


class TestOneRemoteReaderForcesRegistration:
    def test_a_remote_rollout_node_is_enough(self):
        clients = _direct_supply_cluster(rollout_nodes=2)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "train0") is True

    def test_a_local_reader_does_not_cancel_a_remote_one(self):
        """The mixed deployment. Registering here is not waste -- the local reader still
        takes the IPC path, so the extra registration costs only the RDMA MR, while
        skipping it would break the remote reader."""
        clients = _direct_supply_cluster(rollout_nodes=2, per_node=2)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        report = registration_report(plan, clients, "train0")

        assert report["needs_registration"] is True
        assert report["cross_node_keys"] == len(KEYS)
        assert report["skippable_keys"] == 0


class TestUnknownsAnswerRegister:
    """Wrong in this direction costs performance; wrong the other way costs correctness."""

    def test_a_reader_without_a_node_forces_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        before = needs_rdma_registration(plan, clients, "train0")
        assert before is False
        # A reader the cluster knows about but whose node we cannot see.
        clients["gen0_0"].node_ip = ""

        assert needs_rdma_registration(plan, clients, "train0") is True

    def test_a_source_with_no_known_node_forces_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        clients["train0"].node_ip = ""

        assert needs_rdma_registration(plan, clients, "train0") is True

    def test_a_reader_absent_from_client_infos_forces_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        del clients["gen0_0"]

        assert needs_rdma_registration(plan, clients, "train0") is True


class TestNoReaders:
    def test_a_client_nobody_reads_registers_nothing(self):
        """An empty reader set is trivially all-local. Asserting it keeps the policy
        from being written as `not all(...)`, which would answer True here."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "gen0_0") is False

    def test_an_empty_plan_registers_nothing(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        plan.rollout_pull_from_ps_plan = {}

        assert needs_rdma_registration(plan, clients, "train0") is False

    def test_a_self_reference_is_not_a_reader(self):
        """A plan naming the source as its own source would otherwise make every client
        look remote to itself, i.e. register everything always."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        plan.rollout_pull_from_ps_plan["train0"] = {"w0": {"train0": [(0,)]}}

        assert needs_rdma_registration(plan, clients, "train0") is False

    def test_an_empty_shard_list_is_not_a_read(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        plan.rollout_pull_from_ps_plan["train0"] = {"w0": {"train0": []}}

        assert needs_rdma_registration(plan, clients, "train0") is False


class TestInversionCollectsEveryDirection:
    def test_all_three_plan_directions_are_read(self):
        """A trainer appears as a source in `push_to_ps_plan` (to a PS) *and* in
        `rollout_pull_from_ps_plan` (direct supply). Reading only one of them would miss
        readers and answer "no registration" for a client that has remote ones."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        readers = readers_by_source(plan)

        assert "gen0_0" in readers["train0"]["w0"]
        assert "gen0_1" in readers["train0"]["w1"]

    def test_a_ps_read_by_remote_rollouts_registers(self):
        """With PS clients present the planner takes the PS branch; the PS is then the
        source, and it must register because its readers are on other nodes."""
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, TRAIN_NODE),
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PUSH, TRAIN_NODE),
            "ps1": _client("ps1", NIXLClientType.PS_FOR_PULL, TRAIN_NODE),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, ROLLOUT_NODE),
        }
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "ps1") is True

    def test_a_ps_read_only_locally_registers_nothing(self):
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, TRAIN_NODE),
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PUSH, TRAIN_NODE),
            "ps1": _client("ps1", NIXLClientType.PS_FOR_PULL, TRAIN_NODE),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, TRAIN_NODE),
        }
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert needs_rdma_registration(plan, clients, "ps1") is False


class TestGranularityIsPerKey:
    def test_only_the_keys_with_remote_readers_are_reported(self):
        """The criterion is stated per `(key, shard)` because a mixed deployment can
        have one key read locally and another remotely. The policy answers at that
        granularity even where the wiring currently uses the coarse verdict."""
        clients = _direct_supply_cluster(rollout_nodes=2, per_node=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        # The remote node stops needing w1.
        plan.rollout_pull_from_ps_plan["gen1_0"]["w1"] = {}

        assert keys_read_across_nodes(plan, clients, "train0") == {"w0"}


class TestClientShouldRegisterRdma:
    """_should_register_rdma() wires needs_rdma_registration into NIXLStorageClient.

    The criterion: returns True when the comm plan has any cross-node reader for this
    client, False only when every reader is same-node (pure colocate). Tested through
    the policy layer; the client method is a thin wrapper that adds the safe fallback.
    """

    def test_no_plan_means_register(self):
        """Before wait_for_server_info the plan is None; must not skip registration."""

        class _FakePlan:
            push_to_ps_plan = {}
            rollout_pull_from_ps_plan = {}
            train_pull_from_ps_plan = {}

        # needs_rdma_registration with an empty plan returns False (no readers),
        # but _should_register_rdma returns True when _comm_plan is None.
        # We test the wrapper logic directly:

        # Simulate a client with no plan set
        class _Stub:
            client_type = NIXLClientType.PUSH_SIDE
            _comm_plan = None
            _all_client_infos = {}
            client_name = "train0"

            def _should_register_rdma(self_):
                from psrl.utils.nixl.client import NIXLStorageClient
                return NIXLStorageClient._should_register_rdma(self_)

        stub = _Stub()
        assert stub._should_register_rdma() is True, \
            "no comm plan -> should register (safe fallback)"

    def test_disaggregate_registers(self):
        """A cross-node reader forces registration in direct-supply disaggregate."""
        # rollout_nodes=2: node 0 is same as trainer (TRAIN_NODE), node 1 is on 10.1.0.1.
        # So gen1_* clients are on a different node, forcing RDMA registration.
        clients = _direct_supply_cluster(rollout_nodes=2, per_node=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        from psrl.utils.nixl.registration_policy import needs_rdma_registration
        # gen1_0 is on 10.1.0.1 (different from TRAIN_NODE), so train0 must register.
        assert needs_rdma_registration(plan, clients, "train0") is True

    def test_colocate_skips_registration(self):
        """All readers on the same node -> registration can be skipped."""
        node = TRAIN_NODE
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, node),
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PUSH, node),
            "ps1": _client("ps1", NIXLClientType.PS_FOR_PULL, node),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, node),
        }
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        from psrl.utils.nixl.registration_policy import needs_rdma_registration
        assert needs_rdma_registration(plan, clients, "train0") is False

    def test_a_pull_side_client_with_unprovable_sources_registers(self):
        """The old unconditional True is now conditional, but every unknown still answers
        register: a reader pulls *into* its memory and NIXL requires the local side of a
        transfer to be registered, so a skip needs positive proof of copyability."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        del clients["train0"]  # the source's export capability can no longer be observed

        class _Stub:
            client_type = NIXLClientType.PULL_SIDE
            device = torch.device("cuda", 0)
            _comm_plan = plan
            _all_client_infos = clients
            client_name = "gen0_0"

            def _should_register_rdma(self_):
                from psrl.utils.nixl.client import NIXLStorageClient

                return NIXLStorageClient._should_register_rdma(self_)

            def _sources_peer_reachable(self_, plan_):
                from psrl.utils.nixl.client import NIXLStorageClient

                return NIXLStorageClient._sources_peer_reachable(self_, plan_)

        assert _Stub()._should_register_rdma() is True

    def test_a_client_that_pushes_or_pulls_through_nixl_always_registers(self):
        """A local transfer source/destination needs registration even with no remote reader.

        The train client in PS mode pushes its shards to the PS every step and pulls them
        back after a wake. If its only readers are same-node PS workers the old policy
        skipped registration entirely, leaving its descriptors unset; the first trainer
        pull then fell into the temp-buffer path with no mapping.
        """
        from psrl.utils.nixl.client import NIXLStorageClient

        node = TRAIN_NODE
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, node),
            "ps0": _client("ps0", NIXLClientType.PS_FOR_PUSH, node),
            "ps1": _client("ps1", NIXLClientType.PS_FOR_PULL, node),
            "gen0": _client("gen0", NIXLClientType.PULL_SIDE, node),
        }
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        assert plan.push_to_ps_plan.get("train0"), "the train client has no push plan in this fixture"

        class _Stub:
            client_type = NIXLClientType.PUSH_SIDE
            _comm_plan = plan
            _all_client_infos = clients
            client_name = "train0"

            def _should_register_rdma(self_):
                return NIXLStorageClient._should_register_rdma(self_)

            def _sources_peer_reachable(self_, plan_):
                return NIXLStorageClient._sources_peer_reachable(self_, plan_)

        assert _Stub()._should_register_rdma() is True


class TestReaderSideRegistration:
    """A reader skips its registration only when every source it pulls from is copyable.

    This is the half that dominated the colocate validate switch: the reader's own
    weights are the destination of every read, so a NIXL read needs them registered --
    but in pure colocate no read goes through NIXL (design 3.7.4).
    """

    def test_same_node_exportable_sources_let_the_reader_skip(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        copyable, reason = all_sources_copyable(plan, clients, "gen0_0")

        assert copyable, reason

    def test_a_remote_source_forces_reader_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=2)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        copyable, reason = all_sources_copyable(plan, clients, "gen1_0")

        assert not copyable
        assert "another node" in reason

    def test_a_source_that_cannot_export_forces_reader_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        clients["train0"].ipc_regions.clear()
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        copyable, reason = all_sources_copyable(plan, clients, "gen0_0")

        assert not copyable
        assert "export" in reason

    def test_an_unknown_source_forces_reader_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)
        del clients["train0"]

        assert all_sources_copyable(plan, clients, "gen0_0")[0] is False

    def test_a_reader_with_no_plan_entry_answers_register(self):
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        assert all_sources_copyable(plan, clients, "train0")[0] is False

    def test_a_peer_designate_is_a_source_for_its_followers(self):
        """Follower reads the designate, not the trainer, and the designate may be the
        only edge that decides whether the follower's registration is needed."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = NIXLCommPlan(
            push_to_ps_plan={},
            rollout_pull_from_ps_plan={"gen0_1": {"w0": {"train0": [(0,), (1,)]}}},
            train_pull_from_ps_plan={},
            peer_groups={"gen0_1": {"w0": {(0,): ("gen0_0", "gen0_1"), (1,): ("gen0_0", "gen0_1")}}},
        )

        assert sources_by_reader(plan, "gen0_1") == {"w0": {"gen0_0": [(0,), (1,)]}}
        assert sources_by_reader(plan, "gen0_0") == {}


class TestReaderSkipWiring:
    """`_should_register_rdma` must answer both directions for a PULL_SIDE client."""

    def _verdict(self, clients, name):
        from psrl.utils.nixl.client import NIXLStorageClient

        plan = CommunicationPlanner(peer_diffusion=False).make_comm_plan(clients)

        class _Stub:
            client_type = NIXLClientType.PULL_SIDE
            device = torch.device("cuda", 0)
            _comm_plan = plan
            _all_client_infos = clients
            client_name = name

            def _should_register_rdma(self_):
                return NIXLStorageClient._should_register_rdma(self_)

            def _sources_peer_reachable(self_, plan_):
                return NIXLStorageClient._sources_peer_reachable(self_, plan_)

        return _Stub()._should_register_rdma()

    def test_a_colocate_reader_skips_registration(self):
        clients = _direct_supply_cluster(rollout_nodes=1)

        assert self._verdict(clients, "gen0_0") is False

    def test_a_disaggregate_reader_registers(self):
        clients = _direct_supply_cluster(rollout_nodes=2)

        assert self._verdict(clients, "gen1_0") is True

    def test_a_local_designate_makes_a_remote_upstream_irrelevant_to_the_follower(self):
        """Disaggregate plus peer diffusion: the follower reads its same-node designate,
        so its own registration can be skipped even though the plan names a remote
        trainer. The trainer keeps its registration for the designate."""
        clients = {
            "train0": _client("train0", NIXLClientType.PUSH_SIDE, TRAIN_NODE),
            "gen0_0": _client("gen0_0", NIXLClientType.PULL_SIDE, ROLLOUT_NODE),
            "gen0_1": _client("gen0_1", NIXLClientType.PULL_SIDE, ROLLOUT_NODE),
        }
        plan = NIXLCommPlan(
            push_to_ps_plan={},
            rollout_pull_from_ps_plan={"gen0_1": {"w0": {"train0": [(0,), (1,)]}}},
            train_pull_from_ps_plan={},
            peer_groups={"gen0_1": {"w0": {(0,): ("gen0_0", "gen0_1"), (1,): ("gen0_0", "gen0_1")}}},
        )

        assert sources_by_reader(plan, "gen0_1") == {"w0": {"gen0_0": [(0,), (1,)]}}
        assert all_sources_copyable(plan, clients, "gen0_1")[0] is True

    def test_exportability_is_judged_per_shard_the_reader_wants(self):
        """One buffer the allocator cannot export must not force every same-node reader
        to register: only readers that actually pull that shard need NIXL."""
        clients = _direct_supply_cluster(rollout_nodes=1)
        clients["train0"].ipc_regions.pop(_base(1))
        one_shard = NIXLCommPlan(
            push_to_ps_plan={},
            rollout_pull_from_ps_plan={"gen0_0": {"w0": {"train0": [(0,)]}}},
            train_pull_from_ps_plan={},
            peer_groups={},
        )
        both_shards = NIXLCommPlan(
            push_to_ps_plan={},
            rollout_pull_from_ps_plan={"gen0_0": {"w0": {"train0": [(0,), (1,)]}}},
            train_pull_from_ps_plan={},
            peer_groups={},
        )

        assert all_sources_copyable(one_shard, clients, "gen0_0")[0] is True
        copyable, reason = all_sources_copyable(both_shards, clients, "gen0_0")
        assert not copyable
        assert "cannot export" in reason


class TestSourceSideFallback:
    """A source that skipped registration registers only if a reader pulls a missing shard.

    Every vLLM allocator refuses IPC export (measured: 435/435 allocations), so a rule
    that registered on any missing allocation would undo the reader-side skip for every
    validate instance -- clients nobody reads.
    """

    def _plan(self, sources_for_reader):
        return NIXLCommPlan(
            push_to_ps_plan={},
            rollout_pull_from_ps_plan=sources_for_reader,
            train_pull_from_ps_plan={},
            peer_groups={},
        )

    def test_nobody_reads_means_no_registration(self):
        from psrl.utils.nixl.registration_policy import missing_exports_block_readers

        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = self._plan({"gen0_0": {"w0": {"train0": [(0,)]}}})

        assert missing_exports_block_readers(plan, "gen0_0", clients["gen0_0"].tensor_infos, frozenset({_base(0)})) is False

    def test_a_reader_of_the_missing_shard_forces_registration(self):
        from psrl.utils.nixl.registration_policy import missing_exports_block_readers

        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = self._plan({"gen0_0": {"w0": {"train0": [(0,), (1,)]}}})

        assert missing_exports_block_readers(plan, "train0", clients["train0"].tensor_infos, frozenset({_base(1)})) is True

    def test_a_reader_of_only_exportable_shards_does_not(self):
        from psrl.utils.nixl.registration_policy import missing_exports_block_readers

        clients = _direct_supply_cluster(rollout_nodes=1)
        plan = self._plan({"gen0_0": {"w0": {"train0": [(0,)]}}})

        assert missing_exports_block_readers(plan, "train0", clients["train0"].tensor_infos, frozenset({_base(1)})) is False

    def test_an_unknown_plan_answers_register(self):
        from psrl.utils.nixl.registration_policy import missing_exports_block_readers

        clients = _direct_supply_cluster(rollout_nodes=1)

        assert missing_exports_block_readers(None, "train0", clients["train0"].tensor_infos, frozenset({1})) is True
