"""
Non-preemptive KV-cache threshold scaling policy.
"""

from __future__ import annotations

import math
import time
from typing import Any

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.workers.elastic.scaling_policy.base import (
    InstanceSignal,
    ScalingAction,
    ScalingDecision,
    ScalingPolicy,
)


class RuleBasedScalingPolicy(ScalingPolicy):
    """
    Scale on KV-cache thresholds without evicting another instance.

    A role/model group expands when any fresh awake instance is above the high
    threshold and an asleep instance has a placement that is currently free.
    A role with router backlog also expands onto a free placement.
    It shrinks the lowest-utilization fresh instance below the low threshold,
    while preserving ``min_awake_per_role``. Expansion takes precedence over
    shrinking, and one decision changes at most one instance.
    """

    # Scale-up may only consume an already idle placement. Waking that capacity
    # is the complete operation. existing requests stay where they are.
    allow_preemptive_scale_up = False
    rebalance_after_scale_up = False

    def __init__(self, config: dict, policy_config: dict | None = None):
        super().__init__(config=config, policy_config=policy_config)
        cfg = policy_config or {}
        rule_cfg = cfg.get("rule_based", {}) or {}
        if not isinstance(rule_cfg, dict):
            rule_cfg = {}
        self.scale_up_threshold = float(rule_cfg.get("scale_up_threshold", 0.85))
        self.scale_down_threshold = float(rule_cfg.get("scale_down_threshold", 0.3))
        self._validate_thresholds()

    def _validate_thresholds(self) -> None:
        values = (self.scale_down_threshold, self.scale_up_threshold)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rule-based KV-cache thresholds must be finite")
        if not 0.0 <= self.scale_down_threshold < self.scale_up_threshold <= 1.0:
            raise ValueError(
                "rule-based KV-cache thresholds must satisfy 0 <= scale_down_threshold < scale_up_threshold <= 1"
            )

    @staticmethod
    def _group_by_role_and_model(
        signals: list[InstanceSignal],
    ) -> dict[tuple[PSRL_Role, str], list[InstanceSignal]]:
        grouped: dict[tuple[PSRL_Role, str], list[InstanceSignal]] = {}
        for signal in signals:
            grouped.setdefault((signal.role_name, signal.model_name), []).append(signal)
        return grouped

    @staticmethod
    def _group_sort_key(group_key: tuple[PSRL_Role, str]) -> tuple[str, str]:
        role_name, model_name = group_key
        return (getattr(role_name, "name", str(role_name)), model_name)

    def _fresh_awake(self, signals: list[InstanceSignal]) -> list[InstanceSignal]:
        return [signal for signal in signals if signal.is_awaken and not self._is_signal_staled(signal)]

    def _free_scale_up_candidates(
        self,
        group_signals: list[InstanceSignal],
        all_signals: list[InstanceSignal],
    ) -> list[InstanceSignal]:
        occupied_gpu_keys: set[tuple[str, int]] = set()
        for signal in all_signals:
            if signal.is_awaken and signal.gpu_keys:
                occupied_gpu_keys.update(signal.gpu_keys)
        candidates = [
            signal
            for signal in group_signals
            if (not signal.is_awaken and signal.gpu_keys and not signal.gpu_keys.intersection(occupied_gpu_keys))
        ]
        return sorted(candidates, key=lambda signal: signal.instance_id)

    @staticmethod
    def _decision(
        actions: list[ScalingAction],
        reason: str,
    ) -> ScalingDecision:
        return ScalingDecision(
            actions=actions,
            reason=reason,
            estimated_lambda=0.0,
            role_to_total_mu={},
        )

    def decide(
        self,
        signals: list[InstanceSignal],
        execution_in_progress: bool = False,
        router_backlog_by_role: dict[PSRL_Role, int] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
        pending_scale_up_by_role: dict[PSRL_Role, int] | None = None,
    ) -> ScalingDecision:
        # This policy deliberately uses no throughput or trainer hints.
        _ = trainer_waiting_hint

        if not self.enable:
            return self._decision([], "policy_disabled")
        if not signals:
            return self._decision([], "empty_signals")
        if execution_in_progress:
            return self._decision([], "decision_execution_in_progress")

        now_ms = time.time() * 1000
        if now_ms - self.last_action_time_ms < self.cooldown_ms:
            return self._decision([], "cooldown")

        grouped = self._group_by_role_and_model(signals)
        backlog_map = router_backlog_by_role or {}
        pending_scale_up = pending_scale_up_by_role or {}
        backlog_expansion_candidates: list[tuple[int, str, InstanceSignal]] = []
        for role_name in {role_name for role_name, _ in grouped}:
            role_signals = [signal for signal in signals if signal.role_name == role_name]
            backlog = int(backlog_map.get(role_name, 0))
            pending_count = int(pending_scale_up.get(role_name, 0))
            if backlog <= 0 or pending_count > 0:
                continue

            free_candidates = self._free_scale_up_candidates(role_signals, signals)
            if free_candidates:
                role_key = getattr(role_name, "name", str(role_name))
                backlog_expansion_candidates.append((backlog, role_key, free_candidates[0]))

        if backlog_expansion_candidates:
            backlog_expansion_candidates.sort(key=lambda item: (-item[0], item[1], item[2].instance_id))
            backlog, _, target = backlog_expansion_candidates[0]
            action = ScalingAction(
                action_type="scale_up",
                role_name=target.role_name,
                model_name=target.model_name,
                preferred_instance_ids=[target.instance_id],
                reason=(f"rule_based_router_backlog_scale_up backlog={backlog}"),
            )
            self.last_action_time_ms = now_ms
            self._policy_log(
                "rule_based_decision",
                action="scale_up",
                backlog=backlog,
                instance_id=target.instance_id,
                model_name=target.model_name,
                role_name=getattr(target.role_name, "name", target.role_name),
            )
            return self._decision(
                [action],
                "rule_based_router_backlog_scale_up",
            )

        awake_signals = [signal for signal in signals if signal.is_awaken]
        fresh_awake_signals = self._fresh_awake(signals)
        if awake_signals and not fresh_awake_signals:
            return self._decision([], "all_awake_signals_stale")

        overloaded_groups: dict[tuple[PSRL_Role, str], float] = {}
        for group_key, group_signals in grouped.items():
            fresh_awake = self._fresh_awake(group_signals)
            high_values = [
                signal.kv_cache_utilization
                for signal in fresh_awake
                if signal.kv_cache_utilization > self.scale_up_threshold
            ]
            if high_values:
                overloaded_groups[group_key] = max(high_values)

        expansion_candidates: list[tuple[float, tuple[PSRL_Role, str], InstanceSignal]] = []
        for group_key, high_watermark in overloaded_groups.items():
            free_candidates = self._free_scale_up_candidates(grouped[group_key], signals)
            if free_candidates:
                expansion_candidates.append((high_watermark, group_key, free_candidates[0]))

        if expansion_candidates:
            expansion_candidates.sort(
                key=lambda item: (
                    -item[0],
                    self._group_sort_key(item[1]),
                    item[2].instance_id,
                )
            )
            high_watermark, _, target = expansion_candidates[0]
            action = ScalingAction(
                action_type="scale_up",
                role_name=target.role_name,
                model_name=target.model_name,
                preferred_instance_ids=[target.instance_id],
                reason=(
                    "rule_based_kv_cache_high_free_device_scale_up "
                    f"kv_cache={high_watermark:.6f} "
                    f"threshold={self.scale_up_threshold:.6f}"
                ),
            )
            self.last_action_time_ms = now_ms
            self._policy_log(
                "rule_based_decision",
                action="scale_up",
                instance_id=target.instance_id,
                kv_cache=high_watermark,
                model_name=target.model_name,
                role_name=getattr(target.role_name, "name", target.role_name),
                threshold=self.scale_up_threshold,
            )
            return self._decision([action], "rule_based_kv_cache_high_scale_up")

        shrink_candidates: list[InstanceSignal] = []
        for group_key, group_signals in grouped.items():
            # Do not issue contradictory scale-up and scale-down rules for one group.
            if group_key in overloaded_groups:
                continue
            awake_count = sum(1 for signal in group_signals if signal.is_awaken)
            if awake_count <= self.min_awake_per_role:
                continue
            fresh_awake = self._fresh_awake(group_signals)
            shrink_candidates.extend(
                signal for signal in fresh_awake if signal.kv_cache_utilization < self.scale_down_threshold
            )

        if shrink_candidates:
            shrink_candidates.sort(
                key=lambda signal: (
                    signal.kv_cache_utilization,
                    self._group_sort_key((signal.role_name, signal.model_name)),
                    signal.instance_id,
                )
            )
            target = shrink_candidates[0]
            action = ScalingAction(
                action_type="scale_down",
                role_name=target.role_name,
                model_name=target.model_name,
                preferred_instance_ids=[target.instance_id],
                reason=(
                    "rule_based_kv_cache_low_scale_down "
                    f"kv_cache={target.kv_cache_utilization:.6f} "
                    f"threshold={self.scale_down_threshold:.6f}"
                ),
            )
            self.last_action_time_ms = now_ms
            self._policy_log(
                "rule_based_decision",
                action="scale_down",
                instance_id=target.instance_id,
                kv_cache=target.kv_cache_utilization,
                model_name=target.model_name,
                role_name=getattr(target.role_name, "name", target.role_name),
                threshold=self.scale_down_threshold,
            )
            return self._decision([action], "rule_based_kv_cache_low_scale_down")

        reason = (
            "rule_based_no_action_no_free_device"
            if overloaded_groups
            else "rule_based_no_action_thresholds_not_crossed"
        )
        self._policy_log(
            "rule_based_decision",
            action="none",
            overloaded_groups=len(overloaded_groups),
            reason=reason,
        )
        return self._decision([], reason)
