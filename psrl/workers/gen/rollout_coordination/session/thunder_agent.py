"""Capacity scheduler for TITO session hang and continue decisions.

The scheduler evicts sessions from over-capacity instances and readmits hung
sessions when capacity permits.
"""

from __future__ import annotations

import asyncio
import logging
import os

from psrl.workers.gen.session_scheduler import InstanceCapacity, SessionInfo, ThunderAgentScheduler

from .base import (
    SessionSchedulingBase,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class ThunderAgentSessionMixin(SessionSchedulingBase):
    """Mixin providing the periodic thunder_agent session hang/continue loop.

    Inherits the generic session-router HTTP helpers from ``SessionSchedulingBase``.
    Expects ``self`` to have the following attributes (initialized by RolloutCoordinator):
      self._thunder_agent_cfg
      self.stop_thunder_agent
      self.instance_ids
      self.instance_to_total_kv_tokens
      self.tag_to_replica_ids
      self.server_handles
      self.instance_to_engine_status
      self.session_router_url
      self._session_client
    """

    async def _refresh_kv_token_capacities(self) -> None:
        """Fetch absolute KV-token capacity (get_total_kv_cache_tokens) for rollout
        instances that don't have it cached yet."""
        missing_replicas = {
            instance_id[0]
            for instance_id in self.instance_ids
            if instance_id not in self.instance_to_total_kv_tokens
            and instance_id[0] in self.tag_to_replica_ids["rollout"]
        }
        for replica_id in missing_replicas:
            total_tokens = int(await self.server_handles[replica_id].get_total_kv_cache_tokens.remote())
            for instance_id in self.instance_ids:
                if instance_id[0] == replica_id:
                    self.instance_to_total_kv_tokens[instance_id] = total_tokens

    def _build_instance_capacities(self):
        """Snapshot per-instance KV capacity for the scheduler (rollout instances only)."""
        capacities = []
        for instance_id, engine_status in self.instance_to_engine_status.items():
            if instance_id[0] not in self.tag_to_replica_ids["rollout"]:
                continue
            total = self.instance_to_total_kv_tokens.get(instance_id)
            if not total:
                # Skip until `get_total_kv_cache_tokens` resolves capacity.
                continue
            used = int(round(engine_status.get_kv_cache_utilization() * total))
            capacities.append(InstanceCapacity(instance_id=instance_id, total_kv_tokens=total, used_tokens=used))
        return capacities

    @staticmethod
    def _parse_session_infos(payload):
        """Convert /control/sessions JSON into SessionInfo objects."""
        sessions = []
        for item in payload.get("sessions", []):
            # A session that is closing should not be scheduled.
            if item.get("closing"):
                continue
            base_worker_id = item.get("base_worker_id")
            target_dp_rank = item.get("target_dp_rank")
            instance_id = None
            if base_worker_id is not None and target_dp_rank is not None:
                instance_id = (str(base_worker_id), int(target_dp_rank))
            sessions.append(
                SessionInfo(
                    session_id=str(item["session_id"]),
                    instance_id=instance_id,
                    status=str(item.get("status", "env")),
                    hang_state=str(item.get("hang_state", "running")),
                    total_tokens=int(item.get("total_tokens", 0)),
                )
            )
        return sessions

    async def _thunder_agent_loop(self):
        """Periodically hang/continue sessions to keep instances within KV capacity."""
        cfg = self._thunder_agent_cfg
        # Scope controls relocation, while force pinning controls the next turn.
        global_scope = str(cfg.get("continue_scope", "bucketed")) == "global"
        force_pin = bool(cfg.get("continue_force_pin", False))
        self._thunder_scheduler = ThunderAgentScheduler(
            env_token_weight=float(cfg.get("env_token_weight", 1.0)),
            buffer_per_session=int(cfg.get("buffer_per_session", 100)),
            global_scope=global_scope,
        )
        interval = int(cfg.get("check_interval_in_ms", 1000)) / 1000.0
        psrl_logger.info(
            f"Starting thunder_agent (session hang/continue) loop, interval={interval:.3f}s, "
            f"continue_scope={'global' if global_scope else 'bucketed'} "
            f"continue_force_pin={force_pin}."
        )

        while not self.stop_thunder_agent:
            await asyncio.sleep(interval)
            if self.stop_thunder_agent:
                break
            await self._refresh_kv_token_capacities()
            payload = await self._session_get_json("/control/sessions")
            sessions = self._parse_session_infos(payload)
            if not sessions:
                continue
            instances = self._build_instance_capacities()
            to_hang, to_continue = self._thunder_scheduler.decide(instances, sessions)
            if to_hang:
                resp = await self._session_post_json("/control/hang", [{"session_id": sid} for sid in to_hang])
                # In-flight sessions defer hanging until their next turn boundary.
                psrl_logger.info(
                    f"Requested session hangs: count={len(to_hang)}, "
                    f"hung={resp.get('hung', [])}, deferred={resp.get('deferred', [])}, "
                    f"missing={resp.get('missing', [])}."
                )
            if to_continue:
                # Force pinning applies the selected instance only to the next turn.
                if force_pin:
                    continue_payload = [
                        {
                            "session_id": sid,
                            "base_worker_id": str(instance_id[0]),
                            "target_dp_rank": str(instance_id[1]),
                        }
                        for sid, instance_id in to_continue
                    ]
                else:
                    continue_payload = [{"session_id": sid} for sid, _ in to_continue]
                resp = await self._session_post_json("/control/continue", continue_payload)
                psrl_logger.info(
                    f"Requested session continues: count={len(to_continue)}, "
                    f"continued={resp.get('continued', [])}, missing={resp.get('missing', [])}."
                )
        psrl_logger.info("Stopped thunder_agent loop.")
