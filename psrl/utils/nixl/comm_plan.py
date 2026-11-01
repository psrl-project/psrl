"""
NIXL Communication Planning Module

This module provides intelligent load-balanced communication planning for NIXL,
analyzing network topology and tensor distribution to generate optimal
PUSH_SIDE to PS and PULL_SIDE from PS communication plans.
"""

import logging
import os
import pickle
from dataclasses import dataclass, field

from psrl.utils.nixl.global_vars import GLOBAL_TOPOLOGY
from psrl.utils.nixl.nixl_spec import NIXLClientInfo, NIXLClientType

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


@dataclass
class NIXLCommPlan:
    """Communication plan defining communication scheme for each client's keys"""

    # PUSH_SIDE -> PS write plan
    # {push_client: {key: {ps_client: [shard_indices]}}}
    push_to_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]]

    # PULL_SIDE <- PS read plan
    # {pull_client: {key: {ps_client: [shard_indices]}}}
    rollout_pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]]

    # PUSH_SIDE <- PS read plan
    # {pull_client: {key: {ps_client: [shard_indices]}}
    train_pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]]

    # Same node diffusion groups (D1, refined for phase 2's cross-layout sharing).
    # {pull_client: {key: {shard: wanters}}} where the wanters of one shard hold byte
    # identical bytes of that shard (not necessarily the same whole shard set for the
    # key) and are ordered by designate preference. For a set of participants the
    # first participating wanter reads upstream and the rest read it. Grouping at
    # individual-shard granularity, rather than a client's whole shard set for the
    # key, is what lets two different target layouts (e.g. TP4 and TP2 rollouts on
    # the same node) share a designate for the shards they happen to overlap on, even
    # though their whole shard sets never match.
    peer_groups: dict[str, dict[str, dict[tuple[int, ...], tuple[str, ...]]]] = field(default_factory=dict)

    def serialize(self):
        """Serialize communication plan"""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        """Deserialize communication plan"""
        return pickle.loads(data)

    def get_push_plan(self, push_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get write plan for specific PUSH_SIDE client and key"""
        return self.push_to_ps_plan.get(push_client, {}).get(key, {})

    def get_rollout_pull_plan(self, pull_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PULL_SIDE client and key"""
        return self.rollout_pull_from_ps_plan.get(pull_client, {}).get(key, {})

    def get_peer_group(self, pull_client: str, key: str, shard: tuple[int, ...]) -> tuple[str, ...]:
        """Wanters of this one shard of `key` on `pull_client`'s node, designate first.

        Empty if `shard` has no peer group (the shard's source is already local, or
        only one instance wants it).
        """
        return self.peer_groups.get(pull_client, {}).get(key, {}).get(shard, ())

    def get_peer_shards(self, pull_client: str, key: str) -> dict[tuple[int, ...], tuple[str, ...]]:
        """Every shard of `key` that has a peer group for `pull_client`, designate first."""
        return self.peer_groups.get(pull_client, {}).get(key, {})

    def peer_group_members(self, pull_client: str) -> set[str]:
        """Every client that shares any peer group with `pull_client`, across every key/shard.

        For callers that only need "who might this client read from or notify" (handshake
        fan-out, participant-set resolution) and should not have to understand the
        per-shard structure at all.
        """
        return {
            w
            for per_key in self.peer_groups.get(pull_client, {}).values()
            for wanters in per_key.values()
            for w in wanters
        }

    def get_train_pull_plan(self, pull_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PUSH_SIDE client and key"""
        return self.train_pull_from_ps_plan.get(pull_client, {}).get(key, {})

    def get_ps_write_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get write plan for specific PS client and key (receiving from PUSH_SIDE)"""
        result = {}
        for push_client, key_plans in self.push_to_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[push_client] = key_plans[key][ps_client]
        return result

    def get_rollout_ps_read_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PS client and key (sending to PULL_SIDE)"""
        result = {}
        for pull_client, key_plans in self.rollout_pull_from_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[pull_client] = key_plans[key][ps_client]
        return result

    def get_train_ps_read_plan(self, ps_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PS client and key (sending to PUSH_SIDE)"""
        result = {}
        for pull_client, key_plans in self.train_pull_from_ps_plan.items():
            if key in key_plans and ps_client in key_plans[key]:
                result[pull_client] = key_plans[key][ps_client]
        return result


def merge_comm_plans(plans: list[NIXLCommPlan]) -> NIXLCommPlan:
    """
    Merge the per model plans into one.

    Every client belongs to exactly one model, so the per client dicts have disjoint keys and
    the merge cannot silently drop or overwrite an entry. The assertion states that.
    """
    merged = NIXLCommPlan(push_to_ps_plan={}, rollout_pull_from_ps_plan={}, train_pull_from_ps_plan={}, peer_groups={})
    for plan in plans:
        for name in ("push_to_ps_plan", "rollout_pull_from_ps_plan", "train_pull_from_ps_plan", "peer_groups"):
            target, source = getattr(merged, name), getattr(plan, name)
            clash = set(target).intersection(source)
            assert not clash, f"{name}: {sorted(clash)} appear in two models' plans."
            target.update(source)
    return merged


class CommunicationPlanner:
    """Intelligent communication planning with load balancing"""

    def __init__(
        self,
        restrict_client_group_comm: bool = False,
        peer_diffusion: bool = True,
    ):
        """Initialize the communication planner"""
        # NOTE(lhy): Restrict communication to matching client groups when enabled.
        self.restrict_client_group_comm = restrict_client_group_comm
        self.peer_diffusion = peer_diffusion

    def make_comm_plan(self, clients: dict[str, NIXLClientInfo]) -> NIXLCommPlan:
        """
        Generate communication plan for all clients

        Args:
            clients: Dictionary mapping client names to client info

        Returns:
            NIXLCommPlan: Generated communication plan
        """
        # Register all clients to network topology
        for client_name, client_info in clients.items():
            GLOBAL_TOPOLOGY.register_client(client_name, client_info.node_ip, client_info.node_gpu_id)

        # Classify clients
        push_client_groups = {}
        ps_for_push_client_groups = {}
        rollout_pull_client_groups = {}
        ps_for_rollout_pull_client_groups = {}
        train_pull_client_groups = {}
        ps_for_train_pull_client_groups = {}

        for client_name, client_info in clients.items():
            if client_info.type == NIXLClientType.PUSH_SIDE:
                if client_info.client_group_id not in push_client_groups:
                    push_client_groups[client_info.client_group_id] = []
                push_client_groups[client_info.client_group_id].append(client_name)
                if client_info.client_group_id not in train_pull_client_groups:
                    train_pull_client_groups[client_info.client_group_id] = []
                train_pull_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PUSH:
                if client_info.client_group_id not in ps_for_push_client_groups:
                    ps_for_push_client_groups[client_info.client_group_id] = []
                ps_for_push_client_groups[client_info.client_group_id].append(client_name)
                if client_info.client_group_id not in ps_for_train_pull_client_groups:
                    ps_for_train_pull_client_groups[client_info.client_group_id] = []
                ps_for_train_pull_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PULL_SIDE:
                if client_info.client_group_id not in rollout_pull_client_groups:
                    rollout_pull_client_groups[client_info.client_group_id] = []
                rollout_pull_client_groups[client_info.client_group_id].append(client_name)
            elif client_info.type == NIXLClientType.PS_FOR_PULL:
                if client_info.client_group_id not in ps_for_rollout_pull_client_groups:
                    ps_for_rollout_pull_client_groups[client_info.client_group_id] = []
                ps_for_rollout_pull_client_groups[client_info.client_group_id].append(client_name)
            else:
                raise ValueError(f"Unknown client type: {client_info.type}")

        # Initialize communication plans
        push_to_ps_plan = {client: {} for client_group in push_client_groups.values() for client in client_group}
        rollout_pull_from_ps_plan = {
            client: {} for client_group in rollout_pull_client_groups.values() for client in client_group
        }
        train_pull_from_ps_plan = {
            client: {} for client_group in train_pull_client_groups.values() for client in client_group
        }

        # Generate PUSH_SIDE -> PS_FOR_PUSH write plan
        if push_client_groups and ps_for_push_client_groups:
            self._make_push_to_ps_plan(clients, push_client_groups, ps_for_push_client_groups, push_to_ps_plan)

        # Generate PUSH_SIDE <- PS_FOR_PUSH read plan
        if train_pull_client_groups and ps_for_train_pull_client_groups:
            self._make_pull_from_ps_plan(
                clients, train_pull_client_groups, ps_for_train_pull_client_groups, train_pull_from_ps_plan
            )

        # Generate PULL_SIDE <- PS_FOR_PULL read plan
        if rollout_pull_client_groups and ps_for_rollout_pull_client_groups:
            self._make_pull_from_ps_plan(
                clients, rollout_pull_client_groups, ps_for_rollout_pull_client_groups, rollout_pull_from_ps_plan
            )
        elif rollout_pull_client_groups and push_client_groups:
            # D3: trainer direct supply. With no PS in the cluster the rollout reads
            # from the train clients themselves, which is the same planner with a
            # different source set rather than a second code path -- the thing being
            # removed is the intermediate copy, not the planning.
            #
            # Dispatching on "is there a PS" rather than on a mode flag keeps the two
            # arrangements from drifting: a cluster either has PS clients registered or
            # it does not, and that fact is already in `clients`.
            self._make_pull_from_ps_plan(
                clients, rollout_pull_client_groups, push_client_groups, rollout_pull_from_ps_plan
            )

        # D1: collapse duplicate reads of the same bytes into one upstream read plus
        # same-node peer reads. Done after the plan is built, because it rewrites which
        # source serves whom rather than which shards anyone needs.
        peer_groups = self._make_peer_groups(clients, rollout_pull_from_ps_plan)

        return NIXLCommPlan(
            push_to_ps_plan=push_to_ps_plan,
            rollout_pull_from_ps_plan=rollout_pull_from_ps_plan,
            train_pull_from_ps_plan=train_pull_from_ps_plan,
            peer_groups=peer_groups,
        )

    def _make_peer_groups(
        self,
        clients: dict[str, NIXLClientInfo],
        pull_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
    ) -> dict[str, dict[str, dict[tuple[int, ...], tuple[str, ...]]]]:
        """
        Group same node pull clients that need byte identical bytes of one shard (D1, phase 2).

        Peers read each other's local tensors, so a group is keyed by one individual finest-mesh
        shard, not by a reader's whole shard set for the key. Every client already expresses its
        needed shards in the same unified finest mesh (the server's `_unify_role` runs every
        client's sharding through `find_finest_shard_mesh` / `refactor_based_on_finer_shard_mesh`
        before planning), so grouping at this granularity is what lets two different target
        layouts on the same node (say vLLM TP4 and TP2 for the same key) share a designate for the
        shards they overlap on, even though their whole shard sets never match. The first member
        of every group is the designate: it reads the shard upstream and the rest read it from the
        designate over NVLink.

        Designate ordering uses a three-level sort key:

        1. **Trainer link quality** (primary, descending): the instance with the best
           network path to this shard's upstream trainer source becomes the designate.
           LOCAL > NVLINK > IB > PCIE > IB_PCIE > ETH.  This puts the fastest link on
           the critical path (trainer → designate) and preserves scarce IB bandwidth for
           cross-node hops that have no NVLink alternative.

        2. **Accumulated upstream load** (secondary, ascending): within instances that
           share the same link tier, pick the one that has been assigned the least data
           so far.  This spreads NIC ingress across the node instead of concentrating it
           on one instance's NICs.

        3. **Client name** (tertiary, ascending): deterministic tie-break so plans are
           reproducible regardless of dict-iteration order.
        """
        if not self.peer_diffusion:
            return {}
        node_of = {name: info.node_ip for name, info in clients.items()}
        wanted: dict[tuple[str, str, tuple[int, ...]], list[str]] = {}
        key_bytes: dict[tuple[str, str, tuple[int, ...]], int] = {}
        shard_source: dict[tuple[str, str, tuple[int, ...]], str] = {}
        for pull_client, key_plans in pull_plan.items():
            info = clients.get(pull_client)
            if info is None:
                continue
            for key, per_source in key_plans.items():
                tensor_info = info.tensor_infos[key]
                positions = {s: i for i, s in enumerate(tensor_info.sharding.shard_indices)}
                for source, shards in per_source.items():
                    if node_of.get(source) == node_of[pull_client]:
                        # Diffusion exists to collapse an expensive upstream read. When this
                        # shard's source is already on the reader's node, the read is a
                        # device copy and the extra designate hop only adds a reader's memory
                        # to the path -- and a reader-side buffer is often not IPC-exportable,
                        # which would force the followers to register after all (3.7.4).
                        continue
                    for shard in shards:
                        wanted.setdefault((node_of[pull_client], key, shard), []).append(pull_client)
                        key_bytes[(pull_client, key, shard)] = tensor_info.get_shard_size_bytes(positions[shard])
                        shard_source[(pull_client, key, shard)] = source

        psrl_logger.debug(
            f"Peer-diffusion candidate groups: {len(wanted)} total, "
            f"{sum(1 for w in wanted.values() if len(w) >= 2)} with >= 2 wanters, "
            f"sizes {sorted({len(w) for w in wanted.values()})}."
        )

        # Re-coalesce shards of the same key that share the exact same wanter set into
        # one election unit before load-balancing. Two shards of one key wanted by the
        # identical set of readers (the common case: every shard of a key, same readers
        # throughout -- exactly what phase 1's whole-key grouping always was) must be
        # decided atomically, or the greedy "least accumulated load" tie-break can split
        # them across two different designates purely from processing order, fragmenting
        # one peer read into two for zero cross-layout benefit. A shard whose wanter set
        # genuinely differs (the phase 2 case -- different target layouts overlapping on
        # part of a key) still lands in its own, separate unit. Different *keys* are
        # deliberately kept as separate units even when their wanter sets happen to
        # match, same as phase 1, so designates still spread across keys for load
        # balancing instead of one instance taking every key a group shares.
        units: dict[tuple[str, str, tuple[str, ...]], list[tuple[int, ...]]] = {}
        for (node, key, shard), wanters in wanted.items():
            if len(wanters) < 2:
                continue
            units.setdefault((node, key, tuple(sorted(wanters))), []).append(shard)

        upstream_load: dict[str, int] = {}
        groups: dict[str, dict[str, dict[tuple[int, ...], tuple[str, ...]]]] = {}

        # Pre-compute per-(client, key, shard) trainer link priority once, outside the
        # sort. get_link_priority is already O(1) (cached after first call), but calling
        # it inside a sort key closure re-invokes it O(G log G) times per group. With 512
        # rollout instances and 200 keys that adds up. Computing once per (client, key,
        # shard) tuple keeps the overall complexity O(N x K x S).
        _trainer_priority_cache: dict[tuple[str, str, tuple[int, ...]], int] = {}

        def _trainer_link_priority(client: str, key: str, shard: tuple[int, ...]) -> int:
            cache_key = (client, key, shard)
            if cache_key in _trainer_priority_cache:
                return _trainer_priority_cache[cache_key]
            source = shard_source.get(cache_key)
            priority = GLOBAL_TOPOLOGY.get_link_priority(client, source) if source is not None else 0
            _trainer_priority_cache[cache_key] = priority
            return priority

        def _unit_priority(client: str, key: str, shards: list[tuple[int, ...]]) -> int:
            # Best link across every shard this unit covers -- the same "max over
            # sources" phase 1 used per key, generalized to a unit that can be a subset
            # of a key's shards (the phase 2 case).
            return max(_trainer_link_priority(client, key, shard) for shard in shards)

        # Largest units first so the greedy balance has small units left to even out the tail.
        for (_node, key, wanters), shards in sorted(
            units.items(), key=lambda item: -sum(key_bytes[(item[0][2][0], item[0][1], shard)] for shard in item[1])
        ):
            # Pre-fetch priorities for this unit's wanters so the sort key is a pure dict
            # lookup — no function call overhead inside Python's sort comparisons.
            priorities = {c: _unit_priority(c, key, shards) for c in wanters}

            # Sort by: best trainer link first (NVLink beats IB), then least accumulated
            # upstream load (spread NIC ingress within the same tier), then name for
            # determinism. This ensures the designate is always the instance closest to
            # its trainer source, leaving scarce IB slots for cross-node hops only.
            ordered = tuple(sorted(
                wanters,
                key=lambda client: (-priorities[client], upstream_load.get(client, 0), client),
            ))
            total_bytes = sum(key_bytes[(ordered[0], key, shard)] for shard in shards)
            upstream_load[ordered[0]] = upstream_load.get(ordered[0], 0) + total_bytes
            for client in ordered:
                for shard in shards:
                    groups.setdefault(client, {}).setdefault(key, {})[shard] = ordered
        return groups

    def _make_push_to_ps_plan(
        self,
        clients: dict[str, NIXLClientInfo],
        push_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        ps_for_push_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        push_to_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
    ):
        """Generate PUSH_SIDE to PS write plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_client_groups=push_client_groups,
            target_client_groups=ps_for_push_client_groups,
            comm_plan=push_to_ps_plan,
            is_push_to_ps=True,
        )

    def _make_pull_from_ps_plan(
        self,
        clients: dict[str, NIXLClientInfo],
        pull_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        ps_for_pull_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
    ):
        """Generate PULL_SIDE from PS read plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_client_groups=ps_for_pull_client_groups,
            target_client_groups=pull_client_groups,
            comm_plan=pull_from_ps_plan,
            is_push_to_ps=False,
        )

    def _make_comm_plan_generic(
        self,
        clients: dict[str, NIXLClientInfo],
        source_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        target_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        comm_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
        is_push_to_ps: bool,
    ):
        """
        Generic communication plan generation with intelligent load balancing

        This method implements a custom sorting algorithm that prioritizes:
        1. Network connection quality (LOCAL > NVLINK > PCIE > IB > ETH)
        2. Current data volume (lower volume gets priority)

        Args:
            clients: Dictionary mapping client names to client info
            source_client_groups: Dict of source client groups (PUSH_SIDE or PS_FOR_PULL)
            target_client_groups: Dict of target client groups (PS_FOR_PUSH or PULL_SIDE)
            comm_plan: Communication plan to update
            is_push_to_ps: True if PUSH_SIDE to PS_FOR_PUSH, False if PS_FOR_PULL to PULL_SIDE
        """
        # Track data volume for each source client
        source_client_volumes = {
            client: 0.0 for client_group in source_client_groups.values() for client in client_group
        }

        restrict_target_to_source_client_group_mapping = {}
        if self.restrict_client_group_comm:
            # NOTE(lhy): Push traffic uses round robin group pairing, while pull
            # traffic remains unrestricted.
            if is_push_to_ps:
                for target_client_group_id in target_client_groups.keys():
                    source_client_group_ids = [
                        client_group_id
                        for client_group_id in source_client_groups.keys()
                        if client_group_id % len(target_client_groups)
                        == target_client_group_id % len(source_client_groups)
                    ]
                    assert len(source_client_group_ids) > 0, (
                        f"No source client group ids found for target client group "
                        f"(id: {target_client_group_id}, "
                        f"client_names: {target_client_groups[target_client_group_id]})"
                    )
                    restrict_target_to_source_client_group_mapping[target_client_group_id] = source_client_group_ids
            else:
                for target_client_group_id in target_client_groups.keys():
                    restrict_target_to_source_client_group_mapping[target_client_group_id] = list(
                        source_client_groups.keys()
                    )
        else:
            for target_client_group_id in target_client_groups.keys():
                restrict_target_to_source_client_group_mapping[target_client_group_id] = list(
                    source_client_groups.keys()
                )

        # Cache source clients by restriction set to avoid rescanning every target key.
        key_to_sources_by_restriction: dict[tuple, dict[str, list[str]]] = {}
        for target_client_group_id in target_client_groups.keys():
            restrict_ids = restrict_target_to_source_client_group_mapping[target_client_group_id]
            # Use a frozen tuple as cache key (group restriction sets are small and reused)
            restriction_key = tuple(sorted(restrict_ids))
            if restriction_key not in key_to_sources_by_restriction:
                key_to_srcs: dict[str, list[str]] = {}
                for src_group_id in restrict_ids:
                    for src_client in source_client_groups[src_group_id]:
                        for key in clients[src_client].tensor_infos:
                            key_to_srcs.setdefault(key, []).append(src_client)
                key_to_sources_by_restriction[restriction_key] = key_to_srcs

        # Lazily build {shard: local_pos} per (source_client, key) on first use:
        # O(1) lookup vs O(S) list.index() per call.
        shard_pos_maps: dict[tuple[str, str], dict] = {}  # (source_client, key) -> {shard: local_pos}

        def get_shard_pos_map(src: str, key: str) -> dict:
            cache_key = (src, key)
            if cache_key not in shard_pos_maps:
                shard_pos_maps[cache_key] = {
                    s: i for i, s in enumerate(clients[src].tensor_infos[key].sharding.shard_indices)
                }
            return shard_pos_maps[cache_key]

        # For each target client, process all its keys
        for target_client_group_id, target_client_group in target_client_groups.items():
            restrict_ids = restrict_target_to_source_client_group_mapping[target_client_group_id]
            restriction_key = tuple(sorted(restrict_ids))
            key_to_srcs = key_to_sources_by_restriction[restriction_key]

            for target_client in target_client_group:
                target_info = clients[target_client]

                for key, target_tensor_info in target_info.tensor_infos.items():
                    if getattr(target_tensor_info.sharding, "local_only", False):
                        continue
                    # Use the precomputed key->sources mapping and a set for O(1) discard,
                    # avoiding O(G × C_s) per-key scan and O(C_s) list.remove().
                    if key not in key_to_srcs:
                        client_keys_info = {}
                        for source_client_group_id in restrict_ids:
                            for source_client in source_client_groups[source_client_group_id]:
                                client_keys_info[source_client] = list(clients[source_client].tensor_infos.keys())
                        error_msg = (
                            f"No available key {key} in source client groups {source_client_groups}, "
                            f"which is required by target client {target_client}, "
                            f"keys of them are {client_keys_info}"
                        )
                        raise AssertionError(error_msg)

                    # Build a working set for this (target_client, key) assignment round:
                    # set.discard() is O(1) vs list.remove() O(C_s).
                    available_source_clients: set[str] = set(key_to_srcs[key])

                    # Get all shards needed by target client
                    needed_shards = set(target_tensor_info.sharding.shard_indices)
                    assigned_shards = set()

                    # Greedy assignment: prioritize source clients with
                    # optimal network connection and least data volume
                    while assigned_shards != needed_shards:
                        # Custom sorting: first by link type (LOCAL > NVLINK > PCIE > IB > ETH), then by data volume
                        def sort_key(source_client, target_client=target_client):
                            # Get link priority (higher is better)
                            link_priority = GLOBAL_TOPOLOGY.get_link_priority(source_client, target_client)
                            # We want best first, so we negate the value
                            link_priority = -link_priority
                            # Secondary sort by data volume (lower is better)
                            volume = source_client_volumes[source_client]
                            return (link_priority, volume)

                        # Find source client with optimal network connection and minimum data volume
                        min_volume_client = min(available_source_clients, key=sort_key)

                        source_tensor_info = clients[min_volume_client].tensor_infos[key]

                        # Find shards this client can provide (and needed by the target client)
                        available_shards = (
                            set(source_tensor_info.sharding.shard_indices) - assigned_shards
                        ) & needed_shards
                        if not available_shards:
                            # O(1) set discard vs O(C_s) list.remove()
                            available_source_clients.discard(min_volume_client)
                            if not available_source_clients:
                                break
                            continue

                        # Assign shards
                        shards_to_assign = sorted(list(available_shards))
                        assigned_shards.update(shards_to_assign)

                        # Update plan
                        if is_push_to_ps:
                            # PUSH_SIDE -> PS_FOR_PUSH: push_client -> {key -> {ps_client -> shards}}
                            if key not in comm_plan[min_volume_client]:
                                comm_plan[min_volume_client][key] = {}
                            comm_plan[min_volume_client][key][target_client] = shards_to_assign
                        else:
                            # PS_FOR_PULL -> PULL_SIDE: pull_client -> {key -> {ps_client -> shards}}
                            if key not in comm_plan[target_client]:
                                comm_plan[target_client][key] = {}
                            comm_plan[target_client][key][min_volume_client] = shards_to_assign

                        # O(1) lookup via precomputed shard_pos_map vs O(S) list.index()
                        shard_pos_map = get_shard_pos_map(min_volume_client, key)
                        local_indices = [shard_pos_map[shard] for shard in shards_to_assign]
                        shard_size_bytes = sum(
                            source_tensor_info.get_shard_size_bytes(local_idx) for local_idx in local_indices
                        )
                        bandwidth_gbps = GLOBAL_TOPOLOGY.get_bandwidth_gbps(min_volume_client, target_client)
                        volume_increase = shard_size_bytes / (bandwidth_gbps * 1e9)  # Convert to time
                        source_client_volumes[min_volume_client] += volume_increase

                        # O(1) set discard vs O(C_s) list.remove()
                        available_source_clients.discard(min_volume_client)
                        if not available_source_clients:
                            break

                    # Verify all shards are assigned
                    assert assigned_shards == needed_shards, (
                        f"Not all shards assigned for key {key} on target client {target_client}. "
                        f"Needed shards: {needed_shards}. Assigned shards: {assigned_shards}."
                    )

    def _get_link_type_for_test(self, client1: str, client2: str):
        """Helper method for testing to get link type between clients"""
        return GLOBAL_TOPOLOGY.get_link_type(client1, client2)


# Global communication planner instance
global_comm_planner = CommunicationPlanner()
