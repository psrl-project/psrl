import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psrl.utils.elastic.cost_model import estimate_instance_throughput
from psrl.utils.elastic.logging import log_policy_event, log_policy_no_action
from psrl.utils.logger import FileOnlyHandler
from psrl.workers.gen.utils import RolloutInstanceId

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def _role_name_to_str(role_name: Any) -> str:
    if hasattr(role_name, "name"):
        return str(role_name.name)
    return str(role_name)


def _config_section(cfg: dict | None, key: str) -> dict:
    """
    Return a nested mapping from `cfg[key]`, or empty if it is missing or not a dict.
    """
    if not isinstance(cfg, dict):
        return {}
    value = cfg.get(key, {}) or {}
    return value if isinstance(value, dict) else {}


@dataclass
class InstanceSignal:
    role_name: Any
    model_name: str
    instance_id: RolloutInstanceId
    is_awaken: bool
    kv_cache_utilization: float
    running_queue_num: int
    waiting_queue_num: int
    generation_throughput: float
    total_token_num: int
    snapshot_timestamp: str | None = None
    # Set of (node_id, gpu_id) pairs occupied by this instance.
    # Populated by `ElasticExecutor`. None means the mapping is unavailable.
    gpu_keys: frozenset | None = None


@dataclass
class ScalingAction:
    action_type: str  # "scale_up" or "scale_down"
    role_name: Any
    model_name: str
    num_instances: int = 1
    preferred_instance_ids: list[RolloutInstanceId] | None = None
    reason: str = ""
    # Candidate scoring includes exact cross-role transitions and migration targets.
    pre_sleep_other_preferred: list[dict[str, Any]] | None = None
    pre_wake_other_preferred: list[dict[str, Any]] | None = None
    planned_request_migrations: list[dict[str, Any]] | None = None


@dataclass
class ScalingDecision:
    actions: list[ScalingAction]
    reason: str
    estimated_lambda: float
    role_to_total_mu: dict[str, float]


class ScalingPolicy:
    def __init__(self, config: dict, policy_config: dict | None = None):
        self.config = config
        cfg = policy_config or {}
        rule_cfg = _config_section(cfg, "rule_based")
        throughput_cfg = _config_section(cfg, "throughput_optimal")
        self.enable = bool(cfg.get("enable_policy", cfg.get("enable", False)))
        self.monitor_interval_ms = int(cfg.get("monitor_interval_ms", 1000))
        self.theta_low = float(rule_cfg.get("scale_down_threshold", 0.3))
        self.theta_max = float(rule_cfg.get("scale_up_threshold", 0.85))
        self.cooldown_ms = int(cfg.get("cooldown_ms", 3000))
        self.hysteresis = float(throughput_cfg.get("min_gain", 0.05))
        self.min_awake_per_role = max(0, int(cfg.get("min_awake_per_role", 0)))
        self.full_load_mode = str(rule_cfg.get("full_load_mode", "any")).lower()
        # Prevent spontaneous shrink while an awake role still has a large waiting queue.
        self.max_waiting_queue_for_scale_down = int(rule_cfg.get("max_waiting_queue_for_scale_down", 0))

        self.cost_config = throughput_cfg

        self.last_action_time_ms: float = 0.0
        self._last_total_queue = 0
        self._last_lambda_time = time.time()
        self._lambda_ewma = 0.0
        self._lambda_ewma_alpha = float(throughput_cfg.get("lambda_ewma_alpha", 0.2))
        # Per-tick decision trace (reasons when scale conditions are not met).
        self.log_scaling_decisions = bool(cfg.get("log_scaling_decisions", True))

        self.log_prefix = "ScalingPolicy"
        psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "INFO"))
        psrl_logger.propagate = False
        psrl_logger.addHandler(FileOnlyHandler(self.config.psrl.logging_path, self.log_prefix))

    def close(self) -> None:
        """
        Release resources owned by the policy.
        """

    def _policy_log(self, event: str, **kwargs: Any) -> None:
        if self.log_scaling_decisions:
            log_policy_event(psrl_logger, event, **kwargs)

    def _policy_log_no_action(self, final_reason: str, diagnostics: list[str], detail: list[str]) -> None:
        if self.log_scaling_decisions:
            log_policy_no_action(psrl_logger, final_reason, diagnostics, detail)

    @staticmethod
    def _is_snapshot_staled(snapshot: dict, max_staleness_seconds: float = 5.0) -> bool:
        ts = snapshot.get("timestamp")
        if not ts:
            return True
        try:
            dt = datetime.fromisoformat(ts)
        except (ValueError, TypeError):
            return True
        return (datetime.now(tz=dt.tzinfo) - dt).total_seconds() > max_staleness_seconds

    @staticmethod
    def _is_signal_staled(signal: InstanceSignal, max_staleness_seconds: float = 5.0) -> bool:
        """
        Apply the snapshot freshness guard to the current signal carrier.
        """
        if not signal.is_awaken:
            return False
        return ScalingPolicy._is_snapshot_staled(
            {"timestamp": signal.snapshot_timestamp},
            max_staleness_seconds=max_staleness_seconds,
        )

    @staticmethod
    def _get_role_name(role_name: Any) -> str:
        return _role_name_to_str(role_name)

    @staticmethod
    def _group_by_role(signals: list[InstanceSignal]) -> dict[str, list[InstanceSignal]]:
        grouped: dict[str, list[InstanceSignal]] = {}
        for signal in signals:
            grouped.setdefault(ScalingPolicy._get_role_name(signal.role_name), []).append(signal)
        return grouped

    def _estimate_lambda(self, signals: list[InstanceSignal], role_to_total_mu: dict[str, float]) -> float:
        now = time.time()
        elapsed = max(now - self._last_lambda_time, 1e-6)
        current_total_queue = sum(s.running_queue_num + s.waiting_queue_num for s in signals if s.is_awaken)
        d_queue = current_total_queue - self._last_total_queue
        total_mu = sum(role_to_total_mu.values())
        raw_lambda = max(total_mu + d_queue / elapsed, 0.0)
        self._lambda_ewma = self._lambda_ewma_alpha * raw_lambda + (1 - self._lambda_ewma_alpha) * self._lambda_ewma
        self._last_total_queue = current_total_queue
        self._last_lambda_time = now
        return self._lambda_ewma

    def _build_mu_maps(
        self, signals: list[InstanceSignal]
    ) -> tuple[dict[tuple[str, str, RolloutInstanceId], float], dict[str, float]]:
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float] = {}
        role_total_mu: dict[str, float] = {}
        for signal in signals:
            mu = estimate_instance_throughput(
                role_name=signal.role_name,
                model_name=signal.model_name,
                running_queue_num=signal.running_queue_num,
                total_token_num=signal.total_token_num,
                cost_config=self.cost_config,
                config=self.config,
                fallback=signal.generation_throughput,
            )
            role_name = self._get_role_name(signal.role_name)
            key = (role_name, signal.model_name, signal.instance_id)
            instance_mu[key] = mu
            if signal.is_awaken:
                role_total_mu[role_name] = role_total_mu.get(role_name, 0.0) + mu
        return instance_mu, role_total_mu

    def _pick_scale_down_candidate(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        """
        Spontaneous shrink: only cede when KV is already low (below ``theta_low``).

        Used when a role has spare capacity (Priority 1/3), not for forced rebalance
        under mutual full load. See ``_pick_scale_down_candidate_for_bottleneck_transfer``.
        """
        awaken = [s for s in role_signals if s.is_awaken]
        if len(awaken) <= self.min_awake_per_role:
            return None
        cede_candidates = [s for s in awaken if s.kv_cache_utilization <= self.theta_low]
        if not cede_candidates:
            return None
        cede_candidates.sort(
            key=lambda s: (
                s.kv_cache_utilization,
                instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
            )
        )
        return cede_candidates[0]

    def _pick_scale_down_candidate_for_bottleneck_transfer(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        """
        Pick which awake instance to sleep when *both* roles are full-load (Priority 2).

        Does **not** require ``kv_cache_utilization <= theta_low``: mutual full load is exactly
        when we may need to sacrifice a replica to wake the other role. We pick the instance
        with the smallest estimated per-instance throughput (mu) first, then lower KV as tie-break.
        """
        awaken = [s for s in role_signals if s.is_awaken]
        if len(awaken) <= self.min_awake_per_role:
            return None
        awaken_sorted = sorted(
            awaken,
            key=lambda s: (
                instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
                s.kv_cache_utilization,
            ),
        )
        return awaken_sorted[0]

    def _pick_scale_up_candidate(
        self,
        role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        asleep = [s for s in role_signals if not s.is_awaken]
        if not asleep:
            return None
        asleep.sort(
            key=lambda s: instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
            reverse=True,
        )
        return asleep[0]

    def _pick_scale_up_candidate_on_free_gpu(
        self,
        full_role_signals: list[InstanceSignal],
        other_role_signals: list[InstanceSignal],
        instance_mu: dict[tuple[str, str, RolloutInstanceId], float],
    ) -> InstanceSignal | None:
        """
        Among asleep instances of the full-load role, return the best one whose
        GPUs are completely free of any awake instance from the other role.

        "Free" means: the candidate's gpu_keys have no intersection with the gpu_keys
        of any currently awake other-role instance. Candidates with unknown GPU
        mapping (gpu_keys is None or empty) are skipped because we cannot guarantee they are
        free, so we won't take the risk of double-occupancy.

        Returns the highest-mu free candidate, or None if no such instance exists.
        """
        other_awake_gpu_keys: set = set()
        for s in other_role_signals:
            if s.is_awaken and s.gpu_keys:
                other_awake_gpu_keys.update(s.gpu_keys)

        free_candidates: list[InstanceSignal] = []
        for s in full_role_signals:
            if s.is_awaken:
                continue
            if not s.gpu_keys:
                continue
            if not s.gpu_keys.intersection(other_awake_gpu_keys):
                free_candidates.append(s)

        if not free_candidates:
            return None
        free_candidates.sort(
            key=lambda s: instance_mu.get((self._get_role_name(s.role_name), s.model_name, s.instance_id), 0.0),
            reverse=True,
        )
        return free_candidates[0]

    @staticmethod
    def _role_full_load(role_signals: list[InstanceSignal], theta_max: float, mode: str = "all") -> bool:
        awaken = [s for s in role_signals if s.is_awaken]
        if not awaken:
            return False
        if mode == "any":
            return any(s.kv_cache_utilization >= theta_max for s in awaken)
        return all(s.kv_cache_utilization >= theta_max for s in awaken)

    @staticmethod
    def _role_has_low_load(role_signals: list[InstanceSignal], theta_low: float) -> bool:
        awaken = [s for s in role_signals if s.is_awaken]
        if not awaken:
            return False
        return any(s.kv_cache_utilization <= theta_low for s in awaken)

    def _no_action_detail_strings(
        self,
        *,
        trainer_busy: bool,
        pending_total: int,
        waiting_on: str,
        rollout_full: bool,
        rm_full: bool,
        rollout_low: bool,
        rm_low: bool,
        rollout_down_waiting: int,
        rm_down_waiting: int,
        rollout_up: InstanceSignal | None,
        rm_up: InstanceSignal | None,
        rollout_down: InstanceSignal | None,
        rm_down: InstanceSignal | None,
        rollout_free_up: InstanceSignal | None,
        rm_free_up: InstanceSignal | None,
        rollout_down_xfer: InstanceSignal | None,
        rm_down_xfer: InstanceSignal | None,
        p2_gain_rejected: float | None,
        p2_branch_notes: list[str],
    ) -> list[str]:
        """
        Human-readable reasons why lower-priority branches did not scale (for logs).
        """
        out: list[str] = []
        out.append(
            f"p-1_skip trainer_busy={trainer_busy} pending_total={pending_total} waiting_on={waiting_on!r} "
            f"(needs idle trainer + pending>0 + waiting_on rollout|reward)"
        )
        # Priority 1: RM-only-full branch
        if rm_full and not rollout_full:
            out.append(
                f"p1_rm_full_only rm_free_up={rm_free_up is not None} "
                f"rm_up={rm_up is not None} rollout_down_theta_low={rollout_down is not None} "
                f"(free_gpu tried first, fallback needs rm_up+rollout_down)"
            )
        # Priority 1: Rollout-only-full branch
        if rollout_full and not rm_full:
            out.append(
                f"p1_rollout_full_only rollout_free_up={rollout_free_up is not None} "
                f"rollout_up={rollout_up is not None} rm_down_theta_low={rm_down is not None} "
                f"(free_gpu tried first, fallback needs rollout_up+rm_down)"
            )
        if not rm_full and not rollout_full:
            out.append("p1_skip neither_side_full")
        # Priority 2
        out.append(
            f"p2_context both_full={rollout_full and rm_full} "
            f"rollout_down_xfer={rollout_down_xfer is not None} rm_down_xfer={rm_down_xfer is not None} "
            f"rm_up={rm_up is not None} rollout_up={rollout_up is not None}"
        )
        out.extend(p2_branch_notes)
        if p2_gain_rejected is not None:
            out.append(f"p2_reject best_gain={p2_gain_rejected:.6f} hysteresis={self.hysteresis}")
        out.append(
            f"p3_rollout_self_down_unmet rollout_low={rollout_low} "
            f"rollout_down_theta_low={rollout_down is not None} rm_full={rm_full} "
            f"down_waiting={rollout_down_waiting} waiting_max={self.max_waiting_queue_for_scale_down}"
        )
        out.append(
            f"p3_rm_self_down_unmet rm_low={rm_low} rm_down_theta_low={rm_down is not None} "
            f"rollout_full={rollout_full} down_waiting={rm_down_waiting} "
            f"waiting_max={self.max_waiting_queue_for_scale_down}"
        )
        return out

    def decide(
        self,
        signals: list[InstanceSignal],
        execution_in_progress: bool = False,
        router_backlog_by_role: dict[Any, int] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> ScalingDecision:
        """
        Return the next scaling decision.
        """
        raise NotImplementedError("A concrete scaling policy must implement decide.")
