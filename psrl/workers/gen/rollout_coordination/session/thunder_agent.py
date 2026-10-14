"""Capacity scheduler for TITO session hang and continue decisions.

The scheduler evicts sessions from over-capacity instances and readmits hung
sessions when capacity permits.
"""

from __future__ import annotations

import asyncio
import logging
import os

from .base import (
    SESSION_HUNG,
    STATUS_ENV,
    InstanceCapacity,
    RolloutInstanceId,
    SessionInfo,
    SessionScheduler,
    SessionSchedulingBase,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


class ThunderAgentScheduler(SessionScheduler):
    """
    Decide session hang and continue actions from KV capacity.

    Args:
        env_token_weight: Reservation coefficient for environment-session tokens.
        buffer_per_session: Decode headroom (tokens) reserved per running session.
        global_scope: Whether readmission may relocate a session.
    """

    def __init__(
        self,
        env_token_weight: float = 1.0,
        buffer_per_session: int = 100,
        global_scope: bool = False,
    ):
        self.env_token_weight = float(env_token_weight)
        self.buffer_per_session = int(buffer_per_session)
        self.global_scope = bool(global_scope)

    def decide(
        self,
        instances: list[InstanceCapacity],
        sessions: list[SessionInfo],
    ) -> tuple[list[str], list[tuple[str, RolloutInstanceId]]]:
        """Return ``(session_ids_to_hang, [(session_id, continue_instance), ...])``.

        Hang is always decided per instance (a session is hung on whichever
        instance it currently occupies). Continue depends on ``global_scope``:
        per-instance readmission when False, else a global BFD that may relocate
        the session and reports the chosen target instance.
        """
        instance_by_id: dict = {inst.instance_id: inst for inst in instances}

        # Group sessions by their current instance. Sessions with no pinning yet
        # (instance_id is None) or on an unknown instance are ignored.
        running_by_instance: dict = {}
        hung_by_instance: dict = {}
        for sess in sessions:
            if sess.instance_id is None or sess.instance_id not in instance_by_id:
                continue
            if sess.hang_state == SESSION_HUNG:
                hung_by_instance.setdefault(sess.instance_id, []).append(sess)
            else:
                running_by_instance.setdefault(sess.instance_id, []).append(sess)

        to_hang: list[str] = []
        # Remaining KV capacity per instance AFTER this tick's hang decisions.
        remaining_by_instance: dict[RolloutInstanceId, int] = {}

        # Evict environment sessions before generation sessions, then prefer
        # smaller footprints within each class.
        for inst in instances:
            running = running_by_instance.get(inst.instance_id, [])
            hung_now: set[str] = set()
            remaining = self._remaining_capacity(inst, running, hung_now)
            if remaining < 0:
                psrl_logger.debug(
                    f"Instance over capacity: instance_id={inst.instance_id!r}, "
                    f"total_kv={inst.total_kv_tokens}, used={inst.used_tokens}, "
                    f"remaining={remaining}, running={len(running)}, "
                    f"hung={len(hung_by_instance.get(inst.instance_id, []))}. Shedding sessions."
                )
                for sess in self._hang_order(running):
                    if remaining >= 0:
                        break
                    to_hang.append(sess.session_id)
                    hung_now.add(sess.session_id)
                    remaining = self._remaining_capacity(inst, running, hung_now)
            remaining_by_instance[inst.instance_id] = remaining

        # --- Continue phase: readmit hung sessions that now fit. ---
        if self.global_scope:
            to_continue = self._continue_global_bfd(hung_by_instance, remaining_by_instance)
        else:
            to_continue = self._continue_pinned(hung_by_instance, remaining_by_instance)

        if to_hang or to_continue:
            psrl_logger.info(f"ThunderAgentScheduler decide: hang={to_hang} continue={to_continue}.")
        return to_hang, to_continue

    def _continue_pinned(
        self,
        hung_by_instance: dict,
        remaining_by_instance: dict,
    ) -> list[tuple[str, RolloutInstanceId]]:
        """Sticky mode: readmit each hung session on its own instance.

        Smallest-tokens first within each instance so we readmit as many as
        possible. The continue instance is the session's current instance.
        """
        to_continue: list[tuple[str, RolloutInstanceId]] = []
        for instance_id, hung in hung_by_instance.items():
            remaining = remaining_by_instance.get(instance_id, 0)
            if remaining <= self.buffer_per_session:
                continue
            for sess in sorted(hung, key=lambda s: s.total_tokens):
                need = sess.total_tokens + self.buffer_per_session
                if need <= remaining:
                    to_continue.append((sess.session_id, instance_id))
                    remaining -= need
        return to_continue

    def _continue_global_bfd(
        self,
        hung_by_instance: dict,
        remaining_by_instance: dict,
    ) -> list[tuple[str, RolloutInstanceId]]:
        """Global scope: global Best-Fit-Decreasing readmission.

        Faithful port of ThunderAgent's ``_greedy_resume``:
        1. Collect per-instance remaining capacity (only instances with room).
        2. Select the max set of hung sessions whose cumulative required tokens
           fit the total capacity (smallest first, to readmit as many as possible).
        3. BFD placement: largest session first onto the instance with the most
           remaining capacity, re-sorting instances after each placement.
        """
        buffer = self.buffer_per_session
        # Mutable [instance_id, remaining] rows for instances with room.
        caps = [
            [instance_id, remaining] for instance_id, remaining in remaining_by_instance.items() if remaining > buffer
        ]
        total_capacity = sum(row[1] for row in caps)
        if not caps or total_capacity <= 0:
            return []

        all_hung = [sess for hung in hung_by_instance.values() for sess in hung]
        if not all_hung:
            return []
        # Hung sessions have no in-flight turn, so token footprint alone
        # determines admission order.
        candidates = sorted(all_hung, key=lambda s: s.total_tokens)

        # Step 1: select the max prefix (smallest first) fitting total capacity.
        resumable: list[SessionInfo] = []
        cumulative = 0
        for sess in candidates:
            need = sess.total_tokens + buffer
            if cumulative + need <= total_capacity:
                resumable.append(sess)
                cumulative += need
        if not resumable:
            return []

        # Place the largest session on the emptiest instance first.
        resumable.sort(key=lambda s: -s.total_tokens)
        caps.sort(key=lambda row: -row[1])
        to_continue: list[tuple[str, RolloutInstanceId]] = []
        min_need = resumable[-1].total_tokens + buffer
        for sess in resumable:
            if not caps:
                break
            need = sess.total_tokens + buffer
            max_cap = caps[0][1]
            # Even the smallest remaining session can't fit the emptiest
            # instance → nothing more can be placed.
            if min_need > max_cap:
                break
            # A smaller following session may still fit.
            if need > max_cap:
                continue
            to_continue.append((sess.session_id, caps[0][0]))
            caps[0][1] -= need
            if caps[0][1] > buffer:
                caps.sort(key=lambda row: -row[1])
            else:
                caps.pop(0)
        return to_continue

    def _remaining_capacity(
        self,
        inst: InstanceCapacity,
        running: list[SessionInfo],
        hung_now: set[str],
    ) -> int:
        """
        Return remaining KV-token capacity after planned hangs.

        Prefix-cache savings use the full running set and remain fixed throughout
        one hang pass.
        """
        generate_tokens_full = sum(s.total_tokens for s in running if s.status != STATUS_ENV)
        shared = max(0, generate_tokens_full - inst.used_tokens)

        generate_tokens = 0
        env_tokens = 0
        count = 0
        for sess in running:
            if sess.session_id in hung_now:
                continue
            count += 1
            if sess.status == STATUS_ENV:
                env_tokens += sess.total_tokens
            else:
                generate_tokens += sess.total_tokens

        active = int(generate_tokens + self.env_token_weight * env_tokens)
        buffer = count * self.buffer_per_session
        used = active - shared + buffer
        return inst.total_kv_tokens - used

    @staticmethod
    def _hang_order(running: list[SessionInfo]) -> list[SessionInfo]:
        """Eviction order: env-status before generate-status, smallest tokens first."""
        env = sorted(
            (s for s in running if s.status == STATUS_ENV),
            key=lambda s: s.total_tokens,
        )
        generate = sorted(
            (s for s in running if s.status != STATUS_ENV),
            key=lambda s: s.total_tokens,
        )
        return env + generate


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
