"""
Expand demand on free GPUs and shrink replicas with empty queues.
"""

from __future__ import annotations

import time
from typing import Any

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.scaling_policy.base import (
    InstanceSignal,
    ScalingAction,
    ScalingDecision,
    ScalingPolicy,
)


class DummyScalingPolicy(ScalingPolicy):
    """
    Expand a loaded role when its peer is idle, and shrink empty replicas.

    Every action preserves the configured minimum awake count.
    """

    uses_integer_instance_ids = True

    def _pick_scale_down_idle_queue(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
    ) -> InstanceSignal | None:
        """
        Scale-down candidate: empty local queues. KV tie-break matches spontaneous shrink.
        """
        awaken = [s for s in role_signals if s.is_awaken]
        if len(awaken) <= self.min_awake_per_role:
            return None
        idle = [s for s in awaken if s.running_queue_num == 0 and s.waiting_queue_num == 0]
        if not idle:
            return None
        idle.sort(
            key=lambda s: (
                s.kv_cache_utilization,
                instance_mu.get((s.role_name, s.model_name, s.instance_id), 0.0),
            )
        )
        return idle[0]

    def _try_expand(
        self,
        *,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
        rm_hot: bool,
        rollout_hot: bool,
        backlog_rollout: int,
        backlog_rm: int,
    ) -> tuple[list[ScalingAction], str] | None:
        """
        Demand = router backlog or KV full. Single-sided hot → expand that role. both hot → backlog order.
        """
        if rm_hot and not rollout_hot:
            rm_free_up = self._pick_scale_up_candidate_on_free_gpu(rm_signals, rollout_signals, instance_mu)
            if rm_free_up is not None:
                return (
                    [
                        ScalingAction(
                            action_type="scale_up",
                            role_name=rm_free_up.role_name,
                            model_name=rm_free_up.model_name,
                            preferred_instance_ids=[rm_free_up.instance_id],
                            reason="rm_demand_free_gpu_scale_up",
                        )
                    ],
                    "rm_demand_free_gpu_scale_up",
                )
            return None

        if rollout_hot and not rm_hot:
            rollout_free_up = self._pick_scale_up_candidate_on_free_gpu(rollout_signals, rm_signals, instance_mu)
            if rollout_free_up is not None:
                return (
                    [
                        ScalingAction(
                            action_type="scale_up",
                            role_name=rollout_free_up.role_name,
                            model_name=rollout_free_up.model_name,
                            preferred_instance_ids=[rollout_free_up.instance_id],
                            reason="rollout_demand_free_gpu_scale_up",
                        )
                    ],
                    "rollout_demand_free_gpu_scale_up",
                )
            return None

        if rm_hot and rollout_hot:
            if backlog_rollout > 0:
                up = self._pick_scale_up_candidate_on_free_gpu(rollout_signals, rm_signals, instance_mu)
                if up is not None:
                    return (
                        [
                            ScalingAction(
                                action_type="scale_up",
                                role_name=up.role_name,
                                model_name=up.model_name,
                                preferred_instance_ids=[up.instance_id],
                                reason="router_backlog_rollout_scale_up",
                            )
                        ],
                        "router_backlog_rollout_scale_up",
                    )
            if backlog_rm > 0:
                up = self._pick_scale_up_candidate_on_free_gpu(rm_signals, rollout_signals, instance_mu)
                if up is not None:
                    return (
                        [
                            ScalingAction(
                                action_type="scale_up",
                                role_name=up.role_name,
                                model_name=up.model_name,
                                preferred_instance_ids=[up.instance_id],
                                reason="router_backlog_reward_scale_up",
                            )
                        ],
                        "router_backlog_reward_scale_up",
                    )
            return None

        return None

    def _make_stepwise_decision(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        instance_mu: dict[tuple[PSRL_Role, str, int], float],
        router_backlog_by_role: dict[PSRL_Role, int],
    ) -> tuple[list[ScalingAction], str]:
        rollout_role = PSRL_Role.Rollout
        rm_role = PSRL_Role.RewardModel
        rollout_signals = grouped.get(rollout_role, [])
        rm_signals = grouped.get(rm_role, [])
        if not rollout_signals or not rm_signals:
            self._policy_log(
                "decision",
                outcome="no_action",
                reason="skip_decision_missing_rollout_or_rm_signals",
                has_rm_signals=bool(rm_signals),
                has_rollout_signals=bool(rollout_signals),
            )
            return [], "skip_decision_missing_rollout_or_rm_signals"

        rollout_full = self._role_full_load(rollout_signals, self.theta_max, mode=self.full_load_mode)
        rm_full = self._role_full_load(rm_signals, self.theta_max, mode=self.full_load_mode)

        backlog_rollout = int(router_backlog_by_role.get(rollout_role, 0))
        backlog_rm = int(router_backlog_by_role.get(rm_role, 0))
        # Demand / "saturated": router backlog or KV full-load for that role.
        rm_hot = backlog_rm > 0 or rm_full
        rollout_hot = backlog_rollout > 0 or rollout_full

        rollout_down = self._pick_scale_down_idle_queue(rollout_signals, instance_mu)
        rm_down = self._pick_scale_down_idle_queue(rm_signals, instance_mu)
        rollout_down_waiting = rollout_down.waiting_queue_num if rollout_down is not None else -1
        rm_down_waiting = rm_down.waiting_queue_num if rm_down is not None else -1

        # --- 1. Expand: demand on one side only, or both-hot backlog tie-break (free GPU only) ---
        expanded = self._try_expand(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            instance_mu=instance_mu,
            rm_hot=rm_hot,
            rollout_hot=rollout_hot,
            backlog_rollout=backlog_rollout,
            backlog_rm=backlog_rm,
        )
        if expanded is not None:
            return expanded

        # --- 2. Shrink (idle queues only. no peer *_hot gate) ---
        if rollout_down is not None:
            return (
                [
                    ScalingAction(
                        action_type="scale_down",
                        role_name=rollout_down.role_name,
                        model_name=rollout_down.model_name,
                        preferred_instance_ids=[rollout_down.instance_id],
                        reason="rollout_idle_queue_scale_down",
                    )
                ],
                "rollout_idle_queue_scale_down",
            )

        if rm_down is not None:
            return (
                [
                    ScalingAction(
                        action_type="scale_down",
                        role_name=rm_down.role_name,
                        model_name=rm_down.model_name,
                        preferred_instance_ids=[rm_down.instance_id],
                        reason="rm_idle_queue_scale_down",
                    )
                ],
                "rm_idle_queue_scale_down",
            )

        rollout_low = self._role_has_low_load(rollout_signals, self.theta_low)
        rm_low = self._role_has_low_load(rm_signals, self.theta_low)
        detail = self._no_action_detail_strings(
            trainer_busy=True,
            pending_total=0,
            waiting_on="none",
            rollout_full=rollout_full,
            rm_full=rm_full,
            rollout_low=rollout_low,
            rm_low=rm_low,
            rollout_down_waiting=rollout_down_waiting,
            rm_down_waiting=rm_down_waiting,
            rollout_up=None,
            rm_up=None,
            rollout_down=rollout_down,
            rm_down=rm_down,
            rollout_free_up=None,
            rm_free_up=None,
            rollout_down_xfer=None,
            rm_down_xfer=None,
            p2_gain_rejected=None,
            p2_branch_notes=[
                "dummy_expand_first_then_shrink_idle_queue",
                f"rollout_hot={rollout_hot}",
                f"rm_hot={rm_hot}",
                f"backlog_rollout={backlog_rollout}",
                f"backlog_rm={backlog_rm}",
            ],
        )
        final_reason = f"no_action_{' ;; '.join(detail)}"[:4096]
        self._policy_log_no_action(final_reason, [], detail)
        return [], final_reason

    def decide(
        self,
        signals: list[InstanceSignal],
        execution_in_progress: bool = False,
        router_backlog_by_role: dict[PSRL_Role, int] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> ScalingDecision:
        _ = trainer_waiting_hint
        backlog_map = dict(router_backlog_by_role or {})
        if not self.enable:
            self._policy_log("decision", outcome="skipped", reason="policy_disabled")
            return ScalingDecision(actions=[], reason="policy_disabled", estimated_lambda=0.0, role_to_total_mu={})
        if not signals:
            self._policy_log("decision", outcome="skipped", reason="empty_signals")
            return ScalingDecision(actions=[], reason="empty_signals", estimated_lambda=0.0, role_to_total_mu={})
        if execution_in_progress:
            self._policy_log("decision", outcome="skipped", reason="decision_execution_in_progress")
            return ScalingDecision(
                actions=[],
                reason="decision_execution_in_progress",
                estimated_lambda=0.0,
                role_to_total_mu={},
            )

        now_ms = time.time() * 1000
        if now_ms - self.last_action_time_ms < self.cooldown_ms:
            remain = self.cooldown_ms - (now_ms - self.last_action_time_ms)
            self._policy_log(
                "decision",
                cooldown_remaining_ms=remain,
                outcome="skipped",
                reason="cooldown",
            )
            return ScalingDecision(actions=[], reason="cooldown", estimated_lambda=0.0, role_to_total_mu={})

        grouped = self._group_by_role(signals)
        instance_mu, role_total_mu, _ = self._build_mu_maps(signals)
        estimated_lambda = self._estimate_lambda(signals, role_total_mu)

        stale_count = 0
        for signal in signals:
            if self._is_signal_staled(signal):
                stale_count += 1
        if stale_count == len(signals):
            self._policy_log(
                "decision",
                outcome="skipped",
                reason="all_signals_stale",
                stale_count=stale_count,
            )
            return ScalingDecision(actions=[], reason="all_signals_stale", estimated_lambda=0.0, role_to_total_mu={})

        actions, reason = self._make_stepwise_decision(grouped, instance_mu, backlog_map)

        if actions:
            self.last_action_time_ms = now_ms
        self._policy_log(
            "decision_summary",
            actions_count=len(actions),
            estimated_lambda=estimated_lambda,
            reason=reason,
            role_to_total_mu=dict(role_total_mu),
        )
        return ScalingDecision(
            actions=actions,
            reason=reason,
            estimated_lambda=estimated_lambda,
            role_to_total_mu=role_total_mu,
        )
