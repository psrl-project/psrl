"""
NIXL Communication Planning Module

This module provides intelligent load-balanced communication planning for NIXL,
analyzing network topology and tensor distribution to generate optimal
PUSH_SIDE to PS and PULL_SIDE from PS communication plans.
"""

import logging
import pickle
import time
from dataclasses import dataclass

from psrl.utils.nixl.global_vars import GLOBAL_TOPOLOGY
from psrl.utils.nixl.nixl_spec import NIXLClientInfo, NIXLClientType

psrl_logger = logging.getLogger(__file__)


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

    def serialize(self):
        """Serialize communication plan"""
        return pickle.dumps(self)

    @staticmethod
    def deserialize(data):
        """Deserialize communication plan"""
        return pickle.loads(data)

    def for_clients(self, client_names: list[str] | set[str] | tuple[str, ...]) -> "NIXLCommPlan":
        """Return the initiating-side plan entries used by ``client_names``."""
        selected = set(client_names)
        return NIXLCommPlan(
            push_to_ps_plan={
                client_name: self.push_to_ps_plan[client_name]
                for client_name in selected
                if client_name in self.push_to_ps_plan
            },
            rollout_pull_from_ps_plan={
                client_name: self.rollout_pull_from_ps_plan[client_name]
                for client_name in selected
                if client_name in self.rollout_pull_from_ps_plan
            },
            train_pull_from_ps_plan={
                client_name: self.train_pull_from_ps_plan[client_name]
                for client_name in selected
                if client_name in self.train_pull_from_ps_plan
            },
        )

    def target_clients_for(self, client_names: list[str] | set[str] | tuple[str, ...]) -> set[str]:
        """Return remote clients referenced by the selected initiating-side plans."""
        targets: set[str] = set()
        selected = set(client_names)
        for plan in (
            self.push_to_ps_plan,
            self.rollout_pull_from_ps_plan,
            self.train_pull_from_ps_plan,
        ):
            for client_name in selected:
                for target_plans in plan.get(client_name, {}).values():
                    targets.update(target_plans)
        return targets

    def get_push_plan(self, push_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get write plan for specific PUSH_SIDE client and key"""
        return self.push_to_ps_plan.get(push_client, {}).get(key, {})

    def get_rollout_pull_plan(self, pull_client: str, key: str) -> dict[str, list[tuple[int, ...]]]:
        """Get read plan for specific PULL_SIDE client and key"""
        return self.rollout_pull_from_ps_plan.get(pull_client, {}).get(key, {})

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


class CommunicationPlanner:
    """Intelligent communication planning with load balancing"""

    def __init__(self, restrict_client_group_comm: bool = False):
        """Initialize the communication planner"""
        # NOTE(lhy): Restrict communication to matching client groups when enabled.
        self.restrict_client_group_comm = restrict_client_group_comm

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

        client_type_counts: dict[str, int] = {}
        for client_info in clients.values():
            client_type = client_info.type.value
            client_type_counts[client_type] = client_type_counts.get(client_type, 0) + 1
        psrl_logger.info(
            "[comm-plan] start clients=%d key_refs=%d shard_refs=%d client_types=%s",
            len(clients),
            sum(len(client.tensor_infos) for client in clients.values()),
            sum(
                len(tensor_info.sharding.shard_indices)
                for client in clients.values()
                for tensor_info in client.tensor_infos.values()
            ),
            client_type_counts,
        )

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
            self._make_push_to_ps_plan(
                clients,
                push_client_groups,
                ps_for_push_client_groups,
                push_to_ps_plan,
                phase_name="push_to_ps",
            )

        # Generate PUSH_SIDE <- PS_FOR_PUSH read plan
        if train_pull_client_groups and ps_for_train_pull_client_groups:
            self._make_pull_from_ps_plan(
                clients,
                train_pull_client_groups,
                ps_for_train_pull_client_groups,
                train_pull_from_ps_plan,
                phase_name="train_pull_from_ps",
            )

        # Generate PULL_SIDE <- PS_FOR_PULL read plan
        if rollout_pull_client_groups and ps_for_rollout_pull_client_groups:
            self._make_pull_from_ps_plan(
                clients,
                rollout_pull_client_groups,
                ps_for_rollout_pull_client_groups,
                rollout_pull_from_ps_plan,
                phase_name="rollout_pull_from_ps",
            )

        return NIXLCommPlan(
            push_to_ps_plan=push_to_ps_plan,
            rollout_pull_from_ps_plan=rollout_pull_from_ps_plan,
            train_pull_from_ps_plan=train_pull_from_ps_plan,
        )

    def _make_push_to_ps_plan(
        self,
        clients: dict[str, NIXLClientInfo],
        push_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        ps_for_push_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        push_to_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
        phase_name: str,
    ):
        """Generate PUSH_SIDE to PS write plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_client_groups=push_client_groups,
            target_client_groups=ps_for_push_client_groups,
            comm_plan=push_to_ps_plan,
            is_push_to_ps=True,
            phase_name=phase_name,
        )

    def _make_pull_from_ps_plan(
        self,
        clients: dict[str, NIXLClientInfo],
        pull_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        ps_for_pull_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        pull_from_ps_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
        phase_name: str,
    ):
        """Generate PULL_SIDE from PS read plan with load balancing"""
        self._make_comm_plan_generic(
            clients=clients,
            source_client_groups=ps_for_pull_client_groups,
            target_client_groups=pull_client_groups,
            comm_plan=pull_from_ps_plan,
            is_push_to_ps=False,
            phase_name=phase_name,
        )

    @staticmethod
    def _summarize_shards(shards, limit: int = 24) -> dict[str, object]:
        shard_list = sorted(shards)
        return {
            "count": len(shard_list),
            "sample": shard_list[:limit],
            "truncated": len(shard_list) > limit,
        }

    def _make_comm_plan_generic(
        self,
        clients: dict[str, NIXLClientInfo],
        source_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        target_client_groups: dict[int, list[str]],  # {client_group_id: [client_name_1, client_name_2, ...]}
        comm_plan: dict[str, dict[str, dict[str, list[tuple[int, ...]]]]],
        is_push_to_ps: bool,
        phase_name: str,
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
        phase_start = time.monotonic()
        last_progress_time = phase_start
        processed_target_keys = 0
        processed_target_shards = 0
        source_selection_steps = 0
        source_clients = [client for group in source_client_groups.values() for client in group]
        target_clients = [client for group in target_client_groups.values() for client in group]
        psrl_logger.info(
            "[comm-plan] phase=%s start source_clients=%d target_clients=%d "
            "source_key_refs=%d target_key_refs=%d source_shard_refs=%d target_shard_refs=%d",
            phase_name,
            len(source_clients),
            len(target_clients),
            sum(len(clients[client].tensor_infos) for client in source_clients),
            sum(len(clients[client].tensor_infos) for client in target_clients),
            sum(
                len(tensor_info.sharding.shard_indices)
                for client in source_clients
                for tensor_info in clients[client].tensor_infos.values()
            ),
            sum(
                len(tensor_info.sharding.shard_indices)
                for client in target_clients
                for tensor_info in clients[client].tensor_infos.values()
            ),
        )

        # Track data volume for each source client
        source_client_volumes = {client: 0.0 for client in source_clients}
        source_key_counts = {client: len(clients[client].tensor_infos) for client in source_clients}

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
                    # Use the precomputed key->sources mapping and a set for O(1) discard,
                    # avoiding O(G × C_s) per-key scan and O(C_s) list.remove().
                    if key not in key_to_srcs:
                        eligible_source_clients = [
                            source_client
                            for source_client_group_id in restrict_ids
                            for source_client in source_client_groups[source_client_group_id]
                        ]
                        sources_with_key = [
                            source_client
                            for source_client in source_clients
                            if key in clients[source_client].tensor_infos
                        ]
                        raise RuntimeError(
                            f"Communication plan has no source for target key: phase={phase_name}, "
                            f"target={target_client}, key={key!r}, "
                            f"target_shards={self._summarize_shards(target_tensor_info.sharding.shard_indices)}, "
                            f"restricted_source_group_ids={restrict_ids}, "
                            f"eligible_source_clients={eligible_source_clients}, "
                            f"sources_with_key={sources_with_key}, "
                            f"source_key_counts={source_key_counts}"
                        )

                    # Build a working set for this (target_client, key) assignment round:
                    # set.discard() is O(1) vs list.remove() O(C_s).
                    available_source_clients: set[str] = set(key_to_srcs[key])

                    # Get all shards needed by target client
                    needed_shards = set(target_tensor_info.sharding.shard_indices)
                    assigned_shards = set()
                    processed_target_keys += 1
                    processed_target_shards += len(needed_shards)

                    # Greedy assignment: prioritize source clients with
                    # optimal network connection and least data volume
                    while assigned_shards != needed_shards:
                        source_selection_steps += 1

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

                    now = time.monotonic()
                    if now - last_progress_time >= 5.0:
                        psrl_logger.info(
                            "[comm-plan] phase=%s progress elapsed_s=%.3f processed_target_keys=%d "
                            "processed_target_shards=%d source_selection_steps=%d current_target=%s current_key=%s",
                            phase_name,
                            now - phase_start,
                            processed_target_keys,
                            processed_target_shards,
                            source_selection_steps,
                            target_client,
                            key,
                        )
                        last_progress_time = now

        psrl_logger.info(
            "[comm-plan] phase=%s done elapsed_s=%.3f processed_target_keys=%d "
            "processed_target_shards=%d source_selection_steps=%d",
            phase_name,
            time.monotonic() - phase_start,
            processed_target_keys,
            processed_target_shards,
            source_selection_steps,
        )

    def _get_link_type_for_test(self, client1: str, client2: str):
        """Helper method for testing to get link type between clients"""
        return GLOBAL_TOPOLOGY.get_link_type(client1, client2)


# Global communication planner instance
global_comm_planner = CommunicationPlanner()
