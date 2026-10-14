import copy
import logging
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from nixl._api import nixl_agent, nixl_agent_config
from omegaconf import DictConfig

from psrl.utils.nixl.comm_plan import CommunicationPlanner, NIXLCommPlan
from psrl.utils.nixl.nixl_spec import (
    NIXLClientInfo,
    NIXLClientType,
    NIXLSharding,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))


class NIXLMetaServer:
    def __init__(self, server_name: str, nixl_config: DictConfig, broadcast_init_enabled: bool = False):
        self.server_name = server_name
        self.server_ip = nixl_config.server_ip
        self.server_port = nixl_config.server_port
        self.metadata_broadcast_max_workers = int(nixl_config.get("metadata_broadcast_max_workers", 8))
        if self.metadata_broadcast_max_workers < 1:
            raise ValueError("nixl.metadata_broadcast_max_workers must be at least 1")
        psrl_logger.info(
            f"[nixl-handshake] Initializing server agent={self.server_name!r} "
            f"UCX_TLS={os.environ.get('UCX_TLS')!r} "
            f"UCX_NET_DEVICES={os.environ.get('UCX_NET_DEVICES')!r}."
        )
        self.agent = nixl_agent(
            self.server_name,
            nixl_agent_config(
                True,
                True,
                self.server_port,
                num_workers=self.metadata_broadcast_max_workers,
            ),
        )
        self.connected_clients: dict[
            str, list[str]
        ] = {}  # agent_name -> [client_name1, client_name2, ...], one agent can bind to multiple clients
        self.client_sharding_dicts: dict[str, dict[str, NIXLSharding]] = {}
        self.client_infos: dict[str, NIXLClientInfo] = {}
        self.client_info_bytes: dict[str, bytes] = {}

        self.client_unified_sharding_dicts: dict[str, dict[str, NIXLSharding]] = {}
        self.comm_plan: NIXLCommPlan | None = None
        self._client_temp_mappings: dict[str, dict] = {}

        self._is_all_client_shardings_recved = False
        self._is_all_client_infos_recved = False
        self._is_all_temp_mappings_recved = False

        # NOTE(claude): Broadcast initialization makes PS workers direct peers, so
        # each receives every PS GPU descriptor.
        self._broadcast_init_enabled = broadcast_init_enabled

    def _add_client(self, agent_name: str, client_name: str):
        if agent_name not in self.connected_clients:
            self.connected_clients[agent_name] = []
        if client_name not in self.connected_clients[agent_name]:
            self.connected_clients[agent_name].append(client_name)

    def _remote_metadata_ready(self, agent_name: str) -> bool:
        """Return whether this server can address `agent_name` as a remote NIXL peer."""
        try:
            return bool(self.agent.check_remote_metadata(agent_name))
        except Exception as e:
            psrl_logger.warning(
                f"[nixl-handshake] check_remote_metadata({agent_name}) raised {type(e).__name__}: {e}."
            )
            return False

    def _log_remote_metadata_probe(self, stage: str) -> tuple[list[str], list[str]]:
        """Probe remote metadata for every connected agent and log ready vs missing."""
        ready_agents: list[str] = []
        missing_agents: list[str] = []
        for agent_name in self.connected_clients:
            if self._remote_metadata_ready(agent_name):
                ready_agents.append(agent_name)
            else:
                missing_agents.append(agent_name)
        psrl_logger.info(
            f"[nixl-handshake] {stage}: remote_md_ready={ready_agents} "
            f"remote_md_missing={missing_agents} "
            f"connected_clients={dict(self.connected_clients)}."
        )
        return ready_agents, missing_agents

    def wait_for_client_shardings(self, expected_agents: int = 1, timeout: float = 600.0):
        """
        Wait for all agents to connect and send sharding.
        """
        psrl_logger.info(f"Waiting for {expected_agents} agents to connect and send sharding...")
        if self._is_all_client_shardings_recved:
            # TODO(lhy): Support elastic clients after initial registration.
            return True
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_shardings = pickle.loads(notif)
                        assert isinstance(multi_shardings, dict), (
                            f"Expected a dict of multi_shardings, but got {multi_shardings}"
                        )
                        is_new_agent = agent_name not in already_recved_agents
                        for client_name, sharding_dict in multi_shardings.items():
                            self.client_sharding_dicts[client_name] = sharding_dict
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                        if is_new_agent:
                            # NOTE(yfzhao): A received sharding notif does not prove the
                            # server loaded this agent's metadata for reverse send_notif
                            psrl_logger.info(
                                f"[nixl-handshake] received sharding from agent={agent_name} "
                                f"clients={list(multi_shardings)} "
                                f"recv={len(already_recved_agents)}/{expected_agents} "
                                f"elapsed={time.time() - start:.3f}s "
                                f"remote_md_ready={self._remote_metadata_ready(agent_name)}."
                            )
                    except Exception:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.01)
        self._is_all_client_shardings_recved = True
        psrl_logger.info(
            f"All {len(self.client_sharding_dicts)} clients of {expected_agents} agents "
            f"sent sharding after {time.time() - start} seconds."
        )
        self._log_remote_metadata_probe("after wait_for_client_shardings")

    def wait_for_client_infos(self, expected_agents: int = 1, timeout: float = 600.0):
        """
        Wait for all agents to connect and send client infos.
        """
        psrl_logger.info(f"Waiting for {expected_agents} agents to send client infos...")
        if self._is_all_client_infos_recved:
            # TODO(lhy): Support elastic clients after initial registration.
            return True
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_infos = pickle.loads(notif)
                        assert isinstance(multi_infos, dict), f"Expected a dict of multi_infos, but got {multi_infos}"
                        for client_name, info in multi_infos.items():
                            self.client_infos[client_name] = NIXLClientInfo.deserialize(info)
                            self.client_info_bytes[client_name] = info
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                    except Exception:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.01)
        self._is_all_client_infos_recved = True
        psrl_logger.info(
            f"All {len(self.client_infos)} clients of {expected_agents} agents "
            f"sent client infos after {time.time() - start} seconds."
        )

    def wait_for_client_temp_mappings(self, expected_agents: int = 1, timeout: float = 600.0):
        """
        Wait for all agents to send temporary mappings.
        """
        psrl_logger.info(f"Waiting for {expected_agents} agents to send temp mappings...")
        if self._is_all_temp_mappings_recved:
            return True
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_temp_mappings = pickle.loads(notif)
                        assert isinstance(multi_temp_mappings, dict), (
                            f"Expected a dict of multi_temp_mappings, but got {multi_temp_mappings}"
                        )
                        for client_name, temp_mapping in multi_temp_mappings.items():
                            self._client_temp_mappings[client_name] = temp_mapping
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                    except Exception:
                        continue
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents temp mappings.")
            time.sleep(0.01)
        self._is_all_temp_mappings_recved = True
        psrl_logger.info(
            f"All {len(self._client_temp_mappings)} clients of {expected_agents} agents "
            f"sent temp mappings after {time.time() - start} seconds."
        )

    def make_unified_sharding(self):
        """
        Make unified sharding for all clients.
        """
        assert self._is_all_client_shardings_recved, "Not all clients sent sharding yet."
        assert not self.client_unified_sharding_dicts, "Unified sharding already made."
        _t_start = time.time()
        # We first need to guarantee that all client shardings have the same keys
        all_keys = set()
        for client_name, sharding_dict in self.client_sharding_dicts.items():
            all_keys.update(sharding_dict.keys())
        _t_keys = time.time()
        # Build each key's finest common sharding representation.
        for key in all_keys:
            shard_mesh_list = []
            for client_name, sharding_dict in self.client_sharding_dicts.items():
                if key not in sharding_dict:
                    # Pipeline stages may omit keys but still need an empty shard descriptor.
                    sharding_dict[key] = NIXLSharding.empty()
                shard_mesh_list.append(sharding_dict[key].shard_mesh)
            finest_shard_mesh = NIXLSharding.find_finest_shard_mesh(shard_mesh_list)
            for client_name, sharding_dict in self.client_sharding_dicts.items():
                if client_name not in self.client_unified_sharding_dicts:
                    self.client_unified_sharding_dicts[client_name] = {}
                # Split parameters may share their input descriptor, so each key needs its own refinement.
                sharding = copy.deepcopy(sharding_dict[key])
                sharding.refactor_based_on_finer_shard_mesh(finest_shard_mesh)
                self.client_unified_sharding_dicts[client_name][key] = sharding
        psrl_logger.info(
            f"[timing] make_unified_sharding done: total={time.time() - _t_start:.3f}s, "
            f"collect_keys={_t_keys - _t_start:.3f}s, "
            f"refactor={time.time() - _t_keys:.3f}s, "
            f"n_clients={len(self.client_sharding_dicts)}, n_keys={len(all_keys)}"
        )

    def make_comm_plan(self):
        """
        Make communication plan for all clients.
        """
        assert self._is_all_client_infos_recved, "Not all clients sent client infos yet."
        assert not self.comm_plan, "Communication plan already made."

        psrl_logger.info("Making communication plan...")
        start = time.time()
        try:
            self.comm_plan = CommunicationPlanner().make_comm_plan(self.client_infos)
        except Exception:
            client_type_counts: dict[str, int] = {}
            for client_info in self.client_infos.values():
                client_type = client_info.type.value
                client_type_counts[client_type] = client_type_counts.get(client_type, 0) + 1
            psrl_logger.exception(
                "Failed to make communication plan after %.3f seconds: clients=%d, "
                "key_refs=%d, shard_refs=%d, client_types=%s",
                time.time() - start,
                len(self.client_infos),
                sum(len(client.tensor_infos) for client in self.client_infos.values()),
                sum(
                    len(tensor_info.sharding.shard_indices)
                    for client in self.client_infos.values()
                    for tensor_info in client.tensor_infos.values()
                ),
                client_type_counts,
            )
            raise
        psrl_logger.info(f"Communication plan made after {time.time() - start} seconds.")

    def notify_all_client_shardings(self):
        """
        Notify all connected clients with their sharding.
        """
        assert self._is_all_client_shardings_recved, "Not all clients sent sharding yet."
        assert self.client_unified_sharding_dicts, "Unified sharding not made yet."
        agent_items = list(self.connected_clients.items())
        self._log_remote_metadata_probe("before notify_all_client_shardings")
        already_sent: list[str] = []
        for idx, (agent_name, client_names) in enumerate(agent_items, start=1):
            client_sharding_dicts = {}
            for client_name in client_names:
                assert client_name in self.client_unified_sharding_dicts, (
                    f"Client {client_name} not found in unified sharding dicts."
                )
                sharding_dict = self.client_unified_sharding_dicts[client_name]
                client_sharding_dicts[client_name] = sharding_dict
            payload = pickle.dumps(client_sharding_dicts)
            md_ready = self._remote_metadata_ready(agent_name)
            remaining = [name for name, _ in agent_items[idx:]]
            psrl_logger.info(
                f"[nixl-handshake] notify sharding {idx}/{len(agent_items)}: "
                f"agent={agent_name} clients={client_names} "
                f"payload_bytes={len(payload)} remote_md_ready={md_ready}."
            )
            try:
                send_start = time.monotonic()
                self.agent.send_notif(agent_name, payload)
                psrl_logger.info(
                    f"[nixl-handshake] notify sharding ok: agent={agent_name} "
                    f"send={time.monotonic() - send_start:.3f}s."
                )
            except Exception:
                psrl_logger.exception(
                    f"[nixl-handshake] notify sharding FAILED: agent={agent_name} "
                    f"clients={client_names} payload_bytes={len(payload)} "
                    f"remote_md_ready={md_ready} already_sent={already_sent} "
                    f"remaining={remaining}."
                )
                raise
            already_sent.append(agent_name)

    def _get_relevant_client_names_for_agent(self, agent_name: str) -> set[str]:
        """
        Return this agent's clients plus the exact remote clients referenced by its plan.

        Only the initiating side needs the remote descriptors, so the rules are:
          - PUSH_SIDE needs: all PS_FOR_PUSH (train pushes to PS)
          - PULL_SIDE needs: all PS_FOR_PULL (gen pulls from PS)
          - PS clients do not initiate transfers and need no remote infos beyond their own,
            unless broadcast_init is enabled. In that case PS workers write to each other
            and therefore need every other PS worker's descriptors.

        PS clients are passive transfer targets, so they only retain their own infos.
        PUSH_SIDE and PULL_SIDE clients receive descriptors only for PS clients that
        occur in their per-key communication plans.
        """

        # TODO(zyf): check the correctness
        # my_clients: set[str] = set(self.connected_clients[agent_name])
        # return my_clients | self.comm_plan.target_clients_for(my_clients)

        my_clients: set[str] = set(self.connected_clients[agent_name])
        relevant: set[str] = set(my_clients)  # always include own clients

        # Determine which types this agent's clients have.
        my_types: set[NIXLClientType] = {self.client_infos[c].type for c in my_clients if c in self.client_infos}

        # Initiating side → counterpart type mapping.
        type_needs: dict[NIXLClientType, NIXLClientType] = {
            NIXLClientType.PUSH_SIDE: NIXLClientType.PS_FOR_PUSH,
            NIXLClientType.PULL_SIDE: NIXLClientType.PS_FOR_PULL,
        }
        needed_types: set[NIXLClientType] = {dst for src, dst in type_needs.items() if src in my_types}

        for client_name, client_info in self.client_infos.items():
            if client_info.type in needed_types:
                relevant.add(client_name)

        # Broadcast initialization uses direct PS writes, so include every PS
        # descriptor in the existing info exchange.
        if self._broadcast_init_enabled:
            ps_types = {NIXLClientType.PS_FOR_PUSH, NIXLClientType.PS_FOR_PULL}
            if my_types & ps_types:
                for client_name, client_info in self.client_infos.items():
                    if client_info.type in ps_types:
                        relevant.add(client_name)

        return relevant

    def notify_all_client_infos_and_comm_plan(self):
        """
        Notify all connected agents with relevant client infos and the comm plan.

        Each agent receives only the client infos and communication-plan entries it uses.
        """
        assert self._is_all_client_infos_recved, "Not all clients sent client infos yet."
        assert self.comm_plan, "Communication plan not made yet."

        # Reuse the original bytes received in step 4. The fallback keeps tests and
        # programmatically constructed servers compatible.
        cached_info_bytes = getattr(self, "client_info_bytes", {})
        all_serialized: dict[str, bytes] = {
            name: cached_info_bytes.get(name) or info.serialize() for name, info in self.client_infos.items()
        }

        agent_names = list(self.connected_clients)
        if not agent_names:
            return
        max_workers = min(self.metadata_broadcast_max_workers, len(agent_names))

        def send_to_agent(agent_name: str) -> tuple[int, int, int, float, float]:
            serialize_start = time.monotonic()
            relevant = self._get_relevant_client_names_for_agent(agent_name)
            missing_infos = relevant.difference(all_serialized)
            if missing_infos:
                raise RuntimeError(f"Communication plan references missing client infos: {sorted(missing_infos)}")
            agent_comm_plan = self.comm_plan.for_clients(self.connected_clients[agent_name])
            comm_plan_bytes = agent_comm_plan.serialize()
            payload = pickle.dumps(
                {
                    "client_infos": {n: all_serialized[n] for n in relevant},
                    "comm_plan": comm_plan_bytes,
                }
            )
            serialize_elapsed = time.monotonic() - serialize_start
            send_start = time.monotonic()
            self.agent.send_notif(agent_name, payload)
            send_elapsed = time.monotonic() - send_start
            return len(payload), len(relevant), len(comm_plan_bytes), serialize_elapsed, send_elapsed

        start = time.monotonic()
        completed = 0
        total_payload_bytes = 0
        total_comm_plan_bytes = 0
        min_relevant_infos: int | None = None
        max_relevant_infos = 0
        last_progress = start
        psrl_logger.info(
            "[timing] notify client infos start: agents=%d workers=%d cached_client_infos=%d",
            len(agent_names),
            max_workers,
            len(cached_info_bytes),
        )
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nixl-meta-notify") as executor:
            futures = {executor.submit(send_to_agent, agent_name): agent_name for agent_name in agent_names}
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    payload_bytes, relevant_infos, comm_plan_bytes, serialize_elapsed, send_elapsed = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    psrl_logger.exception("Failed to notify NIXL agent %s", agent_name)
                    raise

                completed += 1
                total_payload_bytes += payload_bytes
                total_comm_plan_bytes += comm_plan_bytes
                min_relevant_infos = (
                    relevant_infos if min_relevant_infos is None else min(min_relevant_infos, relevant_infos)
                )
                max_relevant_infos = max(max_relevant_infos, relevant_infos)
                now = time.monotonic()
                psrl_logger.debug(
                    "[timing] notified agent=%s payload_bytes=%d relevant_infos=%d "
                    "comm_plan_bytes=%d serialize=%.3fs send=%.3fs",
                    agent_name,
                    payload_bytes,
                    relevant_infos,
                    comm_plan_bytes,
                    serialize_elapsed,
                    send_elapsed,
                )
                if completed == len(agent_names) or now - last_progress >= 5.0:
                    psrl_logger.info(
                        "[timing] notify client infos progress: completed=%d/%d elapsed=%.3fs "
                        "submitted_payload_bytes=%d avg_payload_bytes=%d relevant_infos_range=%d-%d "
                        "avg_comm_plan_bytes=%d",
                        completed,
                        len(agent_names),
                        now - start,
                        total_payload_bytes,
                        total_payload_bytes // completed,
                        min_relevant_infos,
                        max_relevant_infos,
                        total_comm_plan_bytes // completed,
                    )
                    last_progress = now

    def notify_all_client_temp_mappings(self):
        """
        Notify each connected agent with only its local clients' temp mappings.

        Temporary descriptors are used only for the initiating client's local
        non-contiguous tensors, so remote-client mappings must not be broadcast.
        """
        assert self._is_all_temp_mappings_recved, "Not all clients sent temp mappings yet."
        agent_names = list(self.connected_clients)
        if not agent_names:
            return
        max_workers = min(self.metadata_broadcast_max_workers, len(agent_names))

        def send_to_agent(agent_name: str) -> int:
            payload = pickle.dumps(
                {
                    client_name: self._client_temp_mappings[client_name]
                    for client_name in self.connected_clients[agent_name]
                }
            )
            self.agent.send_notif(agent_name, payload)
            return len(payload)

        start = time.monotonic()
        completed = 0
        total_payload_bytes = 0
        last_progress = start
        psrl_logger.info(
            "[timing] notify temp mappings start: agents=%d workers=%d",
            len(agent_names),
            max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nixl-meta-temp-notify") as executor:
            futures = {executor.submit(send_to_agent, agent_name): agent_name for agent_name in agent_names}
            for future in as_completed(futures):
                agent_name = futures[future]
                try:
                    payload_bytes = future.result()
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    psrl_logger.exception("Failed to notify NIXL temp mappings to agent %s", agent_name)
                    raise

                completed += 1
                total_payload_bytes += payload_bytes
                now = time.monotonic()
                if completed == len(agent_names) or now - last_progress >= 5.0:
                    psrl_logger.info(
                        "[timing] notify temp mappings progress: completed=%d/%d elapsed=%.3fs "
                        "submitted_payload_bytes=%d",
                        completed,
                        len(agent_names),
                        now - start,
                        total_payload_bytes,
                    )
                    last_progress = now

    def wait_for_update_infos(self, expected_agents: int, timeout: float = 600.0):
        """
        Wait for expected number of agents to send updated client infos.
        """
        psrl_logger.debug(f"Waiting for {expected_agents} agents to send updated client infos...")
        start = time.time()
        already_recved_agents = set()
        while len(already_recved_agents) < expected_agents:
            notifs = self.agent.get_new_notifs()
            for agent_name, notif_list in notifs.items():
                for notif in notif_list:
                    try:
                        multi_infos = pickle.loads(notif)
                        assert isinstance(multi_infos, dict), f"Expected a dict of multi_infos, but got {multi_infos}"
                        for client_name, info_and_temp_mapping in multi_infos.items():
                            info = info_and_temp_mapping["info"]
                            client_temp_mapping = info_and_temp_mapping["temp_mapping"]
                            client_info = NIXLClientInfo.deserialize(info)
                            self.client_infos[client_name] = client_info
                            self.client_info_bytes[client_name] = info
                            self._client_temp_mappings[client_name] = client_temp_mapping
                            self._add_client(agent_name, client_name)
                            already_recved_agents.add(agent_name)
                    except Exception as e:
                        psrl_logger.error(f"Failed to parse updated client infos from agent {agent_name}: {e}")
                        raise
            if time.time() - start > timeout:
                raise TimeoutError("Timeout waiting for agents.")
            time.sleep(0.01)

        psrl_logger.info(
            f"{self.server_name}: Successfully received all {len(already_recved_agents)}/{expected_agents} "
            f"agents in {time.time() - start:.2f} seconds"
        )

    def broadcast_update_client_infos(self, dst_agent_names: list[str], update_client_names: list[str]):
        """
        Broadcast updated client infos to specified agents.

        Args:
            dst_agent_names (List[str]): List of destination agent names
            update_client_names (List[str]): List of updated client names to broadcast
        """
        # Prepare notification data with updated client infos
        payload_dict = {}
        for client_name in update_client_names:
            payload_dict[client_name] = {
                "info": self.client_infos[client_name].serialize(),
                "temp_mapping": self._client_temp_mappings[client_name],
            }
        payload = pickle.dumps(payload_dict)

        for agent_name in dst_agent_names:
            # Send notification with updated client infos
            try:
                self.agent.send_notif(agent_name, payload)
            except Exception as e:
                raise RuntimeError(
                    f"{self.server_name}: Failed to send update client infos to agent {agent_name}: {e}, "
                    f"connected clients: {self.connected_clients}, "
                    f"dst agent names: {dst_agent_names}, "
                    f"update client names: {update_client_names}"
                ) from e

        psrl_logger.debug(
            f"Broadcast update client infos to agents: {dst_agent_names}, include clients: {update_client_names}"
        )

    def shutdown(self):
        """
        Shutdown the meta server.
        """
        for agent_name in self.connected_clients:
            self.agent.remove_remote_agent(agent_name)
        self.agent.invalidate_local_metadata(self.server_ip, self.server_port)
