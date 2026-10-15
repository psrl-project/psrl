"""
Choose scaling actions by harmonic throughput gain minus migration cost.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from psrl.trainer.ppo.utils import PSRL_Role
from psrl.utils.deployment_mode import expand_ngpus_per_node
from psrl.utils.elastic.cost_model import (
    ITLModelParams,
    _read_config_value,
    _read_nested_config_value,
    compute_itl,
    resolve_cost_model_params,
)
from psrl.utils.elastic.logging import CycleLog
from psrl.utils.elastic.overhead import interval_union_s
from psrl.workers.elastic.scaling_policy.base import (
    InstanceSignal,
    ScalingAction,
    ScalingDecision,
    ScalingPolicy,
)
from psrl.workers.elastic.scaling_policy.evaluation.cpp_candidate_evaluator import (
    CppCandidateEvaluator,
    CppCandidateEvaluatorError,
)
from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    RoleCandidatePlan,
    RoleEvaluationResult,
    RoleSnapshot,
    evaluate_role_candidate,
    prepare_role_evaluation_context,
)


@dataclass
class _ITLCandidate:
    """
    One evaluated scale-up or scale-down option for objective comparison.
    """

    action: ScalingAction
    rollout_n: int
    rm_n: int
    current_throughput: float
    next_throughput: float
    delta_throughput: float
    phi: float
    gain_l: float
    load_overrides_by_role: dict[PSRL_Role, dict[int, tuple[float, float]]] | None = None
    request_level_results_by_role: dict[PSRL_Role, RoleEvaluationResult] | None = None


class _RequestLevelCandidateEvaluationError(RuntimeError):
    """
    Abort one policy cycle when a snapshot or candidate task is invalid.
    """


@dataclass(frozen=True)
class _RouterWaitingLoad:
    """
    Router waiting queue prefix load used by the ITL objective.
    """

    request_count: float = 0.0
    token_count: float = 0.0


@dataclass(frozen=True)
class _RoleLoadSnapshot:
    """
    Per-role load snapshot used by current-state throughput calculations.
    """

    request_count: float
    token_count: float
    instance_rows: tuple[tuple[InstanceSignal, float, float], ...]


@dataclass(frozen=True)
class _WakePrefixPlan:
    """
    Deterministic wake prefix and the opposite-role instances it must evict.
    """

    wakes: tuple[InstanceSignal, ...]
    pre_sleep: tuple[InstanceSignal, ...]
    pre_wake: tuple[InstanceSignal, ...] = ()
    migration_pairs: tuple[tuple[InstanceSignal, InstanceSignal], ...] = ()


@dataclass(frozen=True)
class _WakeConflictGroup:
    """
    Ordered wake group with one deterministic opposite-role conflict set.
    """

    members: tuple[InstanceSignal, ...]
    conflicts: tuple[InstanceSignal, ...]
    conflict_mask: int
    conflict_phi: float
    conflict_requests: float
    scaling_impact: float
    train_prefix_counts: tuple[int, ...]


@dataclass(frozen=True)
class _WakePlanDPState:
    """
    Count-DP state. masks and scalar scores avoid copying signal tuples.
    """

    conflict_mask: int = 0
    conflict_phi: float = 0.0
    conflict_requests: float = 0.0
    train_count: int = 0
    path_code: int = 0


def _inv_throughput(tp: float) -> float:
    """
    Return reciprocal throughput, mapping zero to infinity and infinity to zero.
    """
    if tp <= 0.0:
        return math.inf
    if math.isinf(tp):
        return 0.0
    return 1.0 / tp


class ThroughputOptimalScalingPolicy(ScalingPolicy):
    """
    Choose scaling actions using a fitted latency model and harmonic throughput.
    """

    uses_integer_instance_ids = True

    def __init__(self, config: dict, policy_config: dict | None = None):
        """
        Initialize ITL policy knobs from ``policy_config["throughput_optimal"]``.
        """
        super().__init__(config=config, policy_config=policy_config)
        self.policy_config = policy_config or {}
        throughput_cfg = self.policy_config.get("throughput_optimal", {})
        if not isinstance(throughput_cfg, dict):
            throughput_cfg = {}
        self.throughput_cfg = throughput_cfg
        self.decision_window_s = float(throughput_cfg.get("decision_window_s", 30.0))
        self.migrate_time_s = float(throughput_cfg.get("migrate_time_s_initial", 30.0))
        self.migrate_time_ewma_alpha = float(throughput_cfg.get("migrate_time_ewma_alpha", 0.2))
        self.min_gain = float(throughput_cfg.get("min_gain", self.hysteresis))
        self.router_waiting_top_t = int(throughput_cfg.get("router_waiting_top_t", 0))
        raw_max_scale = int(throughput_cfg.get("max_scale_instances_per_action", 1))
        # -1 disables the per-action instance cap. other non-positive values still
        # clamp to 1 so a misconfigured 0 does not skip candidate enumeration.
        self.max_scale_instances_per_action = raw_max_scale if raw_max_scale == -1 else max(1, raw_max_scale)
        # GPU-bundle planning is required when role parallel sizes differ.
        self.enable_heterogeneous_parallelism_candidates = self._normalize_bool(
            throughput_cfg.get("enable_heterogeneous_parallelism_candidates", False)
        )
        self.throughput_objective = self._normalize_throughput_objective(
            throughput_cfg.get("throughput_objective", "balanced_min")
        )
        self.vllm_current_queue_scope = self._normalize_vllm_current_queue_scope(
            throughput_cfg.get("vllm_current_queue_scope", "running_waiting")
        )
        self.current_state_include_router_waiting = self._normalize_bool(
            throughput_cfg.get("current_state_include_router_waiting", True)
        )
        self.role_throughput_weight_enable = self._normalize_bool(
            throughput_cfg.get("role_throughput_weight_enable", False)
        )
        self.role_throughput_weight_basis = self._normalize_role_throughput_weight_basis(
            throughput_cfg.get("role_throughput_weight_basis", "request_count")
        )
        self.role_throughput_weight_mode = self._normalize_role_throughput_weight_mode(
            throughput_cfg.get("role_throughput_weight_mode", "share")
        )
        # Demand starvation requires scoring only the loaded role for this cycle.
        self.starvation_fallback = self._normalize_bool(throughput_cfg.get("starvation_fallback", True))
        # The objective denominator combines weighted reciprocal throughputs.
        self.harmonic_denominator_use_max = self._normalize_bool(
            throughput_cfg.get("harmonic_denominator_use_max", False)
        )
        self.enable_request_level_candidate_evaluation = self._normalize_bool(
            throughput_cfg.get("enable_request_level_candidate_evaluation", False)
        )
        # This switch gates every post-scale-up request interruption.
        self.rebalance_after_scale_up = self._normalize_bool(throughput_cfg.get("rebalance_after_scale_up", True))
        self.candidate_evaluation_max_workers = max(
            1,
            int(throughput_cfg.get("candidate_evaluation_max_workers", 8)),
        )
        self.candidate_evaluation_backend = (
            str(throughput_cfg.get("candidate_evaluation_backend", "python") or "python").strip().lower()
        )
        if self.candidate_evaluation_backend not in {"python", "cpp"}:
            raise ValueError(
                "throughput_optimal.candidate_evaluation_backend must be python or cpp, "
                f"got {self.candidate_evaluation_backend!r}."
            )
        self.candidate_evaluation_cpp_binary = throughput_cfg.get(
            "candidate_evaluation_cpp_binary",
            None,
        )
        self.candidate_evaluation_cpp_timeout_s = float(throughput_cfg.get("candidate_evaluation_cpp_timeout_s", 30.0))
        if self.candidate_evaluation_cpp_timeout_s <= 0.0:
            raise ValueError("throughput_optimal.candidate_evaluation_cpp_timeout_s must be positive.")
        self._candidate_evaluation_executor: ThreadPoolExecutor | None = None
        self._cpp_candidate_evaluator: CppCandidateEvaluator | None = None
        self.last_planner_breakdown: dict[str, float] = {
            "state_analysis_s": 0.0,
            "candidate_ordering_s": 0.0,
            "candidate_set_construction_s": 0.0,
            "simulation_input_preparation_s": 0.0,
            "candidate_evaluation_wall_s": 0.0,
            "rebalance_simulation_s": 0.0,
            "router_simulation_s": 0.0,
            "simulation_wall_s": 0.0,
            "rebalance_router_overlap_s": 0.0,
            "candidate_scoring_s": 0.0,
            "best_candidate_selection_s": 0.0,
        }
        self._cycle_log = CycleLog(self._policy_log)
        self._last_request_level_simulation_timing: dict[str, float] = {
            "simulation_input_preparation_s": 0.0,
            "candidate_evaluation_wall_s": 0.0,
            "rebalance_simulation_s": 0.0,
            "router_simulation_s": 0.0,
            "simulation_wall_s": 0.0,
            "rebalance_router_overlap_s": 0.0,
            "candidate_scoring_s": 0.0,
        }
        self._last_candidate_ordering_s = 0.0
        if self.enable_request_level_candidate_evaluation:
            self._validate_request_level_candidate_evaluation_config()
            if self.candidate_evaluation_backend == "python":
                self._candidate_evaluation_executor = ThreadPoolExecutor(
                    max_workers=self.candidate_evaluation_max_workers,
                    thread_name_prefix="itl-candidate-eval",
                )
            else:
                self._cpp_candidate_evaluator = CppCandidateEvaluator(
                    binary_path=self.candidate_evaluation_cpp_binary,
                    max_workers=self.candidate_evaluation_max_workers,
                    timeout_s=self.candidate_evaluation_cpp_timeout_s,
                )

    @staticmethod
    def _group_by_role(signals: list[InstanceSignal]) -> dict[PSRL_Role, list[InstanceSignal]]:
        """
        Group signals by the enum keys consumed by the ITL planner.
        """
        grouped: dict[PSRL_Role, list[InstanceSignal]] = {}
        for signal in signals:
            grouped.setdefault(signal.role_name, []).append(signal)
        return grouped

    def close(self) -> None:
        """
        Release candidate-evaluation workers owned by this policy.
        """
        executor = self._candidate_evaluation_executor
        self._candidate_evaluation_executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        cpp_evaluator = self._cpp_candidate_evaluator
        self._cpp_candidate_evaluator = None
        if cpp_evaluator is not None:
            cpp_evaluator.close()

    @staticmethod
    def _is_scale_up_available(signal: InstanceSignal) -> bool:
        """
        Return whether ``signal`` identifies an asleep placement.
        """
        return not signal.is_awaken

    @classmethod
    def _router_backlog_positive(cls, backlog_by_role: dict[PSRL_Role, Any] | None) -> bool:
        """
        Return whether any current router backlog has pending work.
        """
        return any(
            cls._normalize_router_waiting_load(value).request_count > 0 for value in (backlog_by_role or {}).values()
        )

    def _finish_cycle(self, outcome: str, reason: str, **extra: Any) -> None:
        """
        Flush the current ITL diagnostics through the main policy logger.
        """
        self._cycle_log.outcome(outcome, reason, **extra)
        self._cycle_log.flush()

    def _validate_request_level_candidate_evaluation_config(self) -> None:
        """
        Reject unsupported mode/strategy combinations at actor startup.
        """
        if self.router_waiting_top_t == 0:
            raise ValueError(
                "Request-level candidate evaluation requires throughput_optimal.router_waiting_top_t != 0."
            )
        rollout_method = _read_nested_config_value(
            self.config,
            "psrl",
            "rollout_coordination",
            "routing_strategy",
            "method",
            default=None,
        )
        if str(rollout_method or "").lower() != "throughput_optimal":
            raise ValueError(
                "request-level candidate evaluation currently requires rollout routing strategy 'throughput_optimal'"
            )

        reward_config = _read_config_value(self.config, "reward", {})
        manager_configs = _read_config_value(reward_config, "managers", {})
        active_names = _read_config_value(reward_config, "active_managers", [])
        reward_models = [manager_configs[name] for name in active_names if name in manager_configs]
        rm_methods: list[str] = []
        for reward_config in reward_models or ():
            method = _read_nested_config_value(
                reward_config,
                "routing_strategy",
                "method",
                default=None,
            )
            if method is not None:
                rm_methods.append(str(method).lower())
        # The RM router defaults to ITL when the shared ITL router knob is on.
        if not rm_methods:
            rm_methods = ["itl" if self._normalize_bool(self.throughput_cfg.get("rm_router_enable", False)) else ""]
        if any(method != "itl" for method in rm_methods):
            raise ValueError(
                "request-level candidate evaluation currently requires reward-model routing strategy 'itl'"
            )

    @staticmethod
    def _normalize_bool(raw: Any) -> bool:
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _normalize_throughput_objective(raw: Any) -> str:
        """
        Normalize the configured throughput objective mode.
        """
        mode = str(raw or "balanced_min").strip().lower()
        if mode in {"sum", "instance_sum", "sum_throughput", "throughput_sum"}:
            return "sum"
        return "balanced_min"

    @staticmethod
    def _normalize_vllm_current_queue_scope(raw: Any) -> str:
        """
        Normalize which vLLM queues contribute to current throughput estimates.
        """
        mode = str(raw or "running_waiting").strip().lower()
        if mode in {"running", "running_only", "vllm_running"}:
            return "running"
        return "running_waiting"

    @staticmethod
    def _normalize_role_throughput_weight_basis(raw: Any) -> str:
        """
        Normalize the per-role throughput weighting pressure basis.
        """
        mode = str(raw or "request_count").strip().lower()
        if mode in {"token_count", "tokens", "token", "total_tokens"}:
            return "token_count"
        if mode in {"none", "off", "disabled", "uniform"}:
            return "none"
        return "request_count"

    @staticmethod
    def _normalize_role_throughput_weight_mode(raw: Any) -> str:
        """
        Normalize how pressure becomes a per-role throughput divisor.
        """
        mode = str(raw or "share").strip().lower()
        if mode in {"raw", "absolute", "count"}:
            return "raw"
        return "share"

    def _vllm_current_request_load(self, signal: InstanceSignal) -> float:
        load = float(signal.running_queue_num)
        if self.vllm_current_queue_scope == "running_waiting":
            load += float(signal.waiting_queue_num)
        return load

    def _params_for_signal(self, signal: InstanceSignal) -> ITLModelParams:
        """
        Resolve ITL coefficients for the signal's role and model.
        """
        return resolve_cost_model_params(
            role_name=signal.role_name,
            model_name=signal.model_name,
            cost_config=self.throughput_cfg,
            config=self.config,
        )

    @staticmethod
    def _normalize_router_waiting_load(raw: Any) -> _RouterWaitingLoad:
        """
        Parse router waiting summary into count/token load.
        """
        if raw is None:
            return _RouterWaitingLoad()
        if isinstance(raw, dict):
            requests = raw.get("count", raw.get("request_count", raw.get("pending", 0)))
            tokens = raw.get("total_tokens", raw.get("token_count", 0))
        else:
            requests = raw
            tokens = 0
        try:
            request_count = max(0.0, float(requests))
        except (TypeError, ValueError):
            request_count = 0.0
        try:
            token_count = max(0.0, float(tokens))
        except (TypeError, ValueError):
            token_count = 0.0
        return _RouterWaitingLoad(request_count=request_count, token_count=token_count)

    def _current_router_waiting_load(self, raw: Any) -> _RouterWaitingLoad:
        """
        Router load used by current-state throughput estimates.
        """
        if not self.current_state_include_router_waiting:
            return _RouterWaitingLoad()
        return self._normalize_router_waiting_load(raw)

    def _role_total_load(
        self,
        role_signals: list[InstanceSignal],
        router_waiting_load: _RouterWaitingLoad | None = None,
    ) -> tuple[float, float]:
        """
        Return total request and token pressure for a role.
        """
        waiting = router_waiting_load or _RouterWaitingLoad()
        return (
            sum(self._vllm_current_request_load(signal) for signal in role_signals) + waiting.request_count,
            sum(float(signal.total_token_num) for signal in role_signals) + waiting.token_count,
        )

    def _role_pressure(
        self,
        role_signals: list[InstanceSignal],
        router_waiting_load: _RouterWaitingLoad | None,
    ) -> float:
        """
        Per-role pressure value used for throughput weighting.
        """
        request_count, token_count = self._role_total_load(role_signals, router_waiting_load)
        if self.role_throughput_weight_basis == "token_count":
            return max(0.0, token_count)
        if self.role_throughput_weight_basis == "none":
            return 0.0
        return max(0.0, request_count)

    def _role_throughput_weights(
        self,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad | None,
        rm_waiting: _RouterWaitingLoad | None,
    ) -> tuple[float, float]:
        """
        Return per-role throughput divisors ``(w_rollout, w_rm)``.
        """
        if not self.role_throughput_weight_enable or self.role_throughput_weight_basis == "none":
            return (0.5, 0.5)
        rollout_pressure = self._role_pressure(rollout_signals, rollout_waiting)
        rm_pressure = self._role_pressure(rm_signals, rm_waiting)
        if self.role_throughput_weight_mode == "raw":
            return (rollout_pressure if rollout_pressure > 0.0 else 1.0, rm_pressure if rm_pressure > 0.0 else 1.0)
        total = rollout_pressure + rm_pressure
        if total <= 0.0:
            return (0.5, 0.5)
        return (rollout_pressure / total, rm_pressure / total)

    def _role_throughput(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad | None = None,
    ) -> float:
        """
        Per-role throughput under balanced load across ``n_instances``.
        """
        reference = next((s for s in role_signals if s.is_awaken), role_signals[0] if role_signals else None)
        total_requests, total_tokens = self._role_total_load(role_signals, router_waiting_load)
        if total_requests <= 0.0:
            return self._idle_instance_throughput()
        if n_instances <= 0:
            return 0.0
        if reference is None:
            return 0.0
        avg_requests = total_requests / float(n_instances)
        avg_tokens = total_tokens / float(n_instances)
        itl = compute_itl(self._params_for_signal(reference), avg_tokens, avg_requests)
        return avg_requests / itl

    @staticmethod
    def _active_role_signals(
        role_signals: list[InstanceSignal],
        n_instances: int,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
    ) -> list[InstanceSignal]:
        """
        Resolve the active signal set for a hypothetical candidate state.
        """
        wake_ids = wake_instance_ids or set()
        sleep_ids = sleep_instance_ids or set()
        active = [
            signal
            for signal in role_signals
            if (signal.is_awaken or int(signal.instance_id) in wake_ids) and int(signal.instance_id) not in sleep_ids
        ]
        if len(active) < n_instances:
            active_ids = {int(signal.instance_id) for signal in active}
            active.extend(
                signal
                for signal in role_signals
                if int(signal.instance_id) not in active_ids
                and int(signal.instance_id) not in sleep_ids
                and not signal.is_awaken
            )
        return active[: max(0, int(n_instances))]

    def _role_throughput_sum(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad | None = None,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
        load_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> float:
        """
        Per-role throughput as a sum of per-instance throughput estimates.
        """
        active = self._active_role_signals(role_signals, n_instances, wake_instance_ids, sleep_instance_ids)
        total_requests, _ = self._role_total_load(role_signals, router_waiting_load)
        if total_requests <= 0.0:
            return self._idle_instance_throughput()
        if n_instances <= 0 or not active:
            return 0.0
        waiting_load = router_waiting_load or _RouterWaitingLoad()
        waiting_requests_per_instance = waiting_load.request_count / float(len(active))
        waiting_tokens_per_instance = waiting_load.token_count / float(len(active))
        throughput_sum = 0.0
        for signal in active:
            override = (load_overrides or {}).get(int(signal.instance_id))
            if override is None:
                request_load = self._vllm_current_request_load(signal)
                token_load = float(signal.total_token_num)
            else:
                request_load, token_load = override
            request_load += waiting_requests_per_instance
            token_load += waiting_tokens_per_instance
            if request_load <= 0.0:
                throughput_sum += self._idle_instance_throughput()
                continue
            itl = compute_itl(self._params_for_signal(signal), token_load, request_load)
            throughput_sum += request_load / itl
        return throughput_sum

    def _current_role_load_snapshot(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad | None = None,
    ) -> _RoleLoadSnapshot:
        """
        Build current per-instance load rows, distributing router waiting load.
        """
        active = self._active_role_signals(role_signals, n_instances)
        waiting = router_waiting_load or _RouterWaitingLoad()
        if not active:
            return _RoleLoadSnapshot(waiting.request_count, waiting.token_count, tuple())
        waiting_requests_per_instance = waiting.request_count / float(len(active))
        waiting_tokens_per_instance = waiting.token_count / float(len(active))
        rows = tuple(
            (
                signal,
                self._vllm_current_request_load(signal) + waiting_requests_per_instance,
                float(signal.total_token_num) + waiting_tokens_per_instance,
            )
            for signal in active
        )
        return _RoleLoadSnapshot(
            request_count=sum(row[1] for row in rows),
            token_count=sum(row[2] for row in rows),
            instance_rows=rows,
        )

    @staticmethod
    def _throughput_delta(next_throughput: float, current_throughput: float) -> float:
        """
        Compute throughput delta while keeping all-idle ``inf -> inf`` transitions stable.
        """
        if math.isinf(next_throughput) and math.isinf(current_throughput) and next_throughput == current_throughput:
            return 0.0
        return next_throughput - current_throughput

    def _windowed_throughput_gain(self, delta_throughput: float, rollout_n: int, rm_n: int) -> float:
        """
        Compute the throughput part of L for the configured objective.
        """
        if self.throughput_objective == "sum":
            return self.decision_window_s * delta_throughput
        return self.decision_window_s * delta_throughput * (rollout_n + rm_n)

    def record_migration_observation(self, sleep_time_s: float, wake_time_s: float) -> None:
        """
        Update migration-time estimate from observed SLEEP and WAKE_UP durations.
        """
        observed = max(0.0, float(sleep_time_s)) + max(0.0, float(wake_time_s))
        if observed <= 0.0:
            return
        alpha = min(1.0, max(0.0, float(self.migrate_time_ewma_alpha)))
        self.migrate_time_s = alpha * observed + (1.0 - alpha) * self.migrate_time_s
        self._policy_log(
            "itl_migration_observation",
            observed_s=f"{observed:.3f}",
            sleep_s=f"{float(sleep_time_s):.3f}",
            wake_s=f"{float(wake_time_s):.3f}",
            ewma_s=f"{self.migrate_time_s:.3f}",
        )

    def _current_role_throughputs(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        rollout_n: int,
        rm_n: int,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
    ) -> dict[PSRL_Role, float]:
        """
        Compute per-role throughput from the current per-instance load.
        """
        backlog_map = router_backlog_by_role or {}
        return {
            PSRL_Role.Rollout: self._role_throughput_sum(
                grouped.get(PSRL_Role.Rollout, []),
                rollout_n,
                self._current_router_waiting_load(backlog_map.get(PSRL_Role.Rollout)),
            ),
            PSRL_Role.RewardModel: self._role_throughput_sum(
                grouped.get(PSRL_Role.RewardModel, []),
                rm_n,
                self._current_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel)),
            ),
        }

    def _instance_phi(self, signal: InstanceSignal, delta_requests: float) -> float:
        """
        Routing penalty for moving ``delta_requests`` off an instance.
        """
        if delta_requests <= 0.0:
            return 0.0
        itl = compute_itl(self._params_for_signal(signal), signal.total_token_num, signal.running_queue_num)
        return self.migrate_time_s / itl * delta_requests

    def _sleep_phi(self, victims: list[InstanceSignal]) -> float:
        """
        Phi for putting instances to sleep: re-route all their running requests.
        """
        return sum(self._instance_phi(victim, float(victim.running_queue_num)) for victim in victims)

    @staticmethod
    def _bundle_conflict_free(awake_after: list[InstanceSignal]) -> bool:
        """
        Return False if two awake instances would share the same bundle key.
        """
        occupied: set[tuple[str, int]] = set()
        for signal in awake_after:
            if not signal.gpu_keys:
                continue
            if occupied.intersection(signal.gpu_keys):
                return False
            occupied.update(signal.gpu_keys)
        return True

    @staticmethod
    def _signal_entry(signal: InstanceSignal) -> dict[str, Any]:
        """
        Identify a solver selected instance for cross role transitions.
        """
        return {
            "role_name": signal.role_name,
            "model_name": signal.model_name,
            "instance_id": int(signal.instance_id),
        }

    @staticmethod
    def _wake_conflicts(wakes: list[InstanceSignal], other_signals: list[InstanceSignal]) -> list[InstanceSignal]:
        """
        Return awake instances from the other role whose bundles overlap ``wakes``.
        """
        other_awake = [s for s in other_signals if s.is_awaken]
        return [
            s
            for s in other_awake
            if any(wake.gpu_keys and s.gpu_keys and wake.gpu_keys.intersection(s.gpu_keys) for wake in wakes)
        ]

    @staticmethod
    def _trainer_pool_is_available(trainer_waiting_hint: dict[str, Any] | None) -> bool:
        """
        Return True when elastic replicas may use trainer-pool devices.
        """
        hint = trainer_waiting_hint or {}
        return not bool(hint.get("trainer_busy", True))

    def _signal_pool_available(
        self,
        signal: InstanceSignal,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> bool:
        """
        Return whether a signal's pool can be used for elastic candidates.
        """
        _ = signal, trainer_waiting_hint
        return True

    @staticmethod
    def _pool_wake_priority() -> int:
        """
        Return the stable priority used when pool metadata is unavailable.
        """
        return 0

    @staticmethod
    def _bundle_conflicts_with_occupied(signal: InstanceSignal, occupied: set[tuple[str, int]]) -> bool:
        if not signal.gpu_keys:
            return False
        return bool(occupied.intersection(signal.gpu_keys))

    def _pick_first_conflict_free_batch(
        self,
        ordered: list[InstanceSignal],
        num_instances: int,
        awake_after_sleep: list[InstanceSignal],
    ) -> list[InstanceSignal] | None:
        """
        Pick the first priority-ordered wake batch without bundle conflicts.

        This intentionally avoids enumerating all ``combinations(n, k)``. Large
        elastic pools may request k=16..32 wakes from 32 candidates. exhaustive
        combination search is intractable and can stall the monitor loop.
        """
        if num_instances <= 0:
            return []
        occupied: set[tuple[str, int]] = set()
        for signal in awake_after_sleep:
            if signal.gpu_keys:
                occupied.update(signal.gpu_keys)
        batch: list[InstanceSignal] = []
        for signal in ordered:
            if self._bundle_conflicts_with_occupied(signal, occupied):
                continue
            batch.append(signal)
            if signal.gpu_keys:
                occupied.update(signal.gpu_keys)
            if len(batch) >= num_instances:
                return batch
        return None

    def _pick_first_batch_with_eviction(
        self,
        ordered: list[InstanceSignal],
        num_instances: int,
        other_awake_after_sleep: list[InstanceSignal],
        other_signals: list[InstanceSignal],
    ) -> tuple[list[InstanceSignal], list[InstanceSignal]] | None:
        """
        Pick a priority-ordered wake batch and its cross-role eviction set.

        The ordered list already encodes shared-pool-first and instance-id
        tie-breaks. We keep the batch itself bundle-conflict-free, then evict
        the awake instances from the other role that overlap selected wakes.
        """
        if num_instances <= 0:
            return [], []
        batch: list[InstanceSignal] = []
        occupied: set[tuple[str, int]] = set()
        for signal in ordered:
            if self._bundle_conflicts_with_occupied(signal, occupied):
                continue
            batch.append(signal)
            if signal.gpu_keys:
                occupied.update(signal.gpu_keys)
            if len(batch) >= num_instances:
                conflicts = self._wake_conflicts(batch, other_signals)
                remaining_other = len(other_awake_after_sleep) - len(conflicts)
                if remaining_other < self.min_awake_per_role:
                    return None
                surviving = [s for s in other_awake_after_sleep if s not in conflicts]
                if not self._bundle_conflict_free(surviving + batch):
                    return None
                return batch, conflicts
        return None

    def _elastic_device_capacity(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> int:
        """
        Estimate device capacity available to Rollout and RewardModel replicas.
        """
        gpu_keys = {
            key
            for role_signals in grouped.values()
            for signal in role_signals
            if self._signal_pool_available(signal, trainer_waiting_hint)
            for key in (signal.gpu_keys or frozenset())
        }
        if gpu_keys:
            return len(gpu_keys)
        elastic_cfg = _read_nested_config_value(self.config, "psrl", "deployment", "elastic", default={})
        shared_nodes = int(_read_config_value(elastic_cfg, "shared_nnodes", 1) or 1)
        shared_gpu_spec = expand_ngpus_per_node(
            _read_config_value(elastic_cfg, "shared_ngpus_per_node", 1),
            shared_nodes,
            "psrl.deployment.elastic.shared_ngpus_per_node",
        )
        capacity = max(0, sum(shared_gpu_spec))
        if bool(_read_config_value(elastic_cfg, "enable_trainer_pool", False)) and self._trainer_pool_is_available(
            trainer_waiting_hint
        ):
            trainer_nodes = int(_read_nested_config_value(self.config, "trainer", "nnodes", default=0) or 0)
            trainer_gpus = int(_read_nested_config_value(self.config, "trainer", "n_gpus_per_node", default=0) or 0)
            capacity += max(0, trainer_nodes * trainer_gpus)
        return capacity

    @staticmethod
    def _role_device_size(role_signals: list[InstanceSignal]) -> int:
        """
        Return a conservative per-instance device size for a role.
        """
        sizes = [len(signal.gpu_keys) for signal in role_signals if signal.gpu_keys]
        return max(sizes) if sizes else 1

    def _candidate_device_usage(
        self,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_n: int,
        rm_n: int,
    ) -> int:
        """
        Estimate devices occupied by a candidate role-count pair.
        """
        return max(0, int(rollout_n)) * self._role_device_size(rollout_signals) + max(
            0, int(rm_n)
        ) * self._role_device_size(rm_signals)

    def _estimated_instance_throughput(self, signal: InstanceSignal) -> float:
        """
        Estimate current per-instance throughput for shrink victim ordering.
        """
        if signal.generation_throughput > 0.0:
            return float(signal.generation_throughput)
        request_load = self._vllm_current_request_load(signal)
        if request_load <= 0.0:
            return 0.0
        itl = compute_itl(self._params_for_signal(signal), float(signal.total_token_num), request_load)
        return request_load / itl

    def _pick_low_throughput_victims(
        self,
        role_signals: list[InstanceSignal],
        num_instances: int,
        ordered_victims_by_model: dict[str, list[InstanceSignal]] | None = None,
    ) -> list[InstanceSignal] | None:
        """
        Pick same-model awake victims with the lowest current throughput.
        """
        if num_instances <= 0:
            return []
        by_model = ordered_victims_by_model
        if by_model is None:
            by_model = {}
            for signal in role_signals:
                if signal.is_awaken:
                    by_model.setdefault(signal.model_name, []).append(signal)
        candidate_groups: list[list[InstanceSignal]] = []
        for signals in by_model.values():
            if len(signals) < num_instances:
                continue
            if ordered_victims_by_model is None:
                signals = sorted(
                    signals,
                    key=lambda item: (
                        self._estimated_instance_throughput(item),
                        self._instance_phi(item, float(item.running_queue_num)),
                        int(item.instance_id),
                    ),
                )
            candidate_groups.append(signals[:num_instances])
        if not candidate_groups:
            return None
        return min(
            candidate_groups,
            key=lambda items: (
                sum(self._estimated_instance_throughput(item) for item in items),
                self._sleep_phi(items),
                [int(item.instance_id) for item in items],
            ),
        )

    def _pick_wake_batch(
        self,
        *,
        role_signals: list[InstanceSignal],
        num_instances: int,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        sleep_victims: list[InstanceSignal],
        trainer_waiting_hint: dict[str, Any] | None = None,
        ordered_wakeable_by_model: dict[str, list[InstanceSignal]] | None = None,
    ) -> list[InstanceSignal] | None:
        """
        Pick same-model wake targets that fit after planned sleeps.
        """
        if num_instances <= 0:
            return []
        excluded = {(victim.role_name, victim.model_name, int(victim.instance_id)) for victim in sleep_victims}
        awake_after_sleep = [
            signal
            for signals in grouped.values()
            for signal in signals
            if signal.is_awaken and (signal.role_name, signal.model_name, int(signal.instance_id)) not in excluded
        ]
        by_model = ordered_wakeable_by_model
        if by_model is None:
            wakeable = [
                signal
                for signal in role_signals
                if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint)
            ]
            by_model = {}
            for signal in wakeable:
                by_model.setdefault(signal.model_name, []).append(signal)
        batches: list[list[InstanceSignal]] = []
        for signals in by_model.values():
            if len(signals) < num_instances:
                continue
            if ordered_wakeable_by_model is None:
                # Share-pool devices first. train-pool devices only fill the rest.
                signals = sorted(
                    signals,
                    key=lambda item: (self._pool_wake_priority(), int(item.instance_id)),
                )
            batch = self._pick_first_conflict_free_batch(signals, num_instances, awake_after_sleep)
            if batch is not None:
                batches.append(batch)
        if not batches:
            return None

        # Prefer batches with the fewest train-pool instances (share pool first),
        # then by pool priority and instance id for a stable tie-break.
        def _batch_key(items: list[InstanceSignal]) -> tuple[Any, ...]:
            train_count = 0
            priority_keys = [(self._pool_wake_priority(), int(it.instance_id)) for it in items]
            return (train_count, *priority_keys)

        return min(batches, key=_batch_key)

    def _pick_wake_batch_with_eviction(
        self,
        *,
        target_signals: list[InstanceSignal],
        other_signals: list[InstanceSignal],
        num_instances: int,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        sleep_victims: list[InstanceSignal],
        trainer_waiting_hint: dict[str, Any] | None = None,
        ordered_wakeable_by_model: dict[str, list[InstanceSignal]] | None = None,
    ) -> tuple[list[InstanceSignal], list[InstanceSignal]] | None:
        """
        Pick a wake batch, evicting cross-role bundle conflicts if needed.

        Returns ``(wakes, evicted_other)`` where ``evicted_other`` are awake
        instances on the other role whose bundles overlap the chosen wakes and
        must be slept first. Tries a conflict-free batch first. only when none
        exists does it fall back to evicting the other role. Eviction is rejected
        if it would drop the other role below ``min_awake_per_role``.
        """
        wakes = self._pick_wake_batch(
            role_signals=target_signals,
            num_instances=num_instances,
            grouped=grouped,
            sleep_victims=sleep_victims,
            trainer_waiting_hint=trainer_waiting_hint,
            ordered_wakeable_by_model=ordered_wakeable_by_model,
        )
        if wakes:
            return wakes, []

        if num_instances <= 0:
            return [], []
        excluded = {(victim.role_name, victim.model_name, int(victim.instance_id)) for victim in sleep_victims}
        other_awake_after_sleep = [
            s for s in other_signals if s.is_awaken and (s.role_name, s.model_name, int(s.instance_id)) not in excluded
        ]
        by_model = ordered_wakeable_by_model
        if by_model is None:
            wakeable = [
                s
                for s in target_signals
                if self._is_scale_up_available(s) and self._signal_pool_available(s, trainer_waiting_hint)
            ]
            by_model = {}
            for signal in wakeable:
                by_model.setdefault(signal.model_name, []).append(signal)
        best: tuple[list[InstanceSignal], list[InstanceSignal]] | None = None
        best_key: tuple[Any, ...] | None = None
        for signals in by_model.values():
            if len(signals) < num_instances:
                continue
            if ordered_wakeable_by_model is None:
                # Share-pool devices first. train-pool devices only fill the rest.
                signals = sorted(
                    signals,
                    key=lambda item: (self._pool_wake_priority(), int(item.instance_id)),
                )
            pick = self._pick_first_batch_with_eviction(
                signals,
                num_instances,
                other_awake_after_sleep,
                other_signals,
            )
            if pick is None:
                continue
            batch, conflicts = pick
            train_count = 0
            priority_keys = [(self._pool_wake_priority(), int(it.instance_id)) for it in batch]
            key = (len(conflicts), train_count, priority_keys)
            if best_key is None or key < best_key:
                best = (batch, conflicts)
                best_key = key
        return best

    def _role_count_bounds(
        self,
        role_signals: list[InstanceSignal],
        current_n: int,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> tuple[int, int]:
        """
        Return feasible min/max awake counts for one role.
        """
        wakeable_n = sum(
            1
            for signal in role_signals
            if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint)
        )
        return self.min_awake_per_role, current_n + wakeable_n

    def _selection_orders_for_role(
        self,
        role_signals: list[InstanceSignal],
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> tuple[dict[str, list[InstanceSignal]], dict[str, list[InstanceSignal]]]:
        """
        Precompute deterministic wake/sleep priority orders for one decide cycle.
        """
        wakeable_by_model: dict[str, list[InstanceSignal]] = {}
        victims_by_model: dict[str, list[InstanceSignal]] = {}
        for signal in role_signals:
            if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint):
                wakeable_by_model.setdefault(signal.model_name, []).append(signal)
            if signal.is_awaken:
                victims_by_model.setdefault(signal.model_name, []).append(signal)

        for model_name, signals in list(wakeable_by_model.items()):
            wakeable_by_model[model_name] = sorted(
                signals,
                key=lambda item: (self._pool_wake_priority(), int(item.instance_id)),
            )
        for model_name, signals in list(victims_by_model.items()):
            victims_by_model[model_name] = sorted(
                signals,
                key=lambda item: (
                    self._estimated_instance_throughput(item),
                    self._instance_phi(item, float(item.running_queue_num)),
                    int(item.instance_id),
                ),
            )
        return wakeable_by_model, victims_by_model

    @staticmethod
    def _signal_key(signal: InstanceSignal) -> tuple[str, str, int]:
        return (str(signal.role_name), signal.model_name, int(signal.instance_id))

    @staticmethod
    def _bundle_sort_key(signal: InstanceSignal) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(signal.gpu_keys or ()))

    def _use_heterogeneous_wake_plans(
        self,
        target_signals: list[InstanceSignal],
        other_signals: list[InstanceSignal],
    ) -> bool:
        """
        Return whether this role pair should use heterogeneous placement planning.
        """
        if not self.enable_heterogeneous_parallelism_candidates:
            return False
        all_signals = target_signals + other_signals
        if not target_signals or not other_signals or any(not signal.gpu_keys for signal in all_signals):
            return False
        target_sizes = {len(signal.gpu_keys or ()) for signal in target_signals}
        other_sizes = {len(signal.gpu_keys or ()) for signal in other_signals}
        all_sizes = target_sizes | other_sizes
        if any(size <= 0 or size & (size - 1) for size in all_sizes):
            return False
        return target_sizes != other_sizes

    def _pick_heterogeneous_migration_pairs(
        self,
        *,
        wakes: tuple[InstanceSignal, ...],
        conflicts: tuple[InstanceSignal, ...],
        target_signals: list[InstanceSignal],
        other_signals: list[InstanceSignal],
        trainer_waiting_hint: dict[str, Any] | None,
        ordered_sources: tuple[InstanceSignal, ...] | None = None,
        ordered_targets_by_shape: dict[tuple[str, int], tuple[InstanceSignal, ...]] | None = None,
    ) -> tuple[tuple[InstanceSignal, InstanceSignal], ...]:
        """
        Move conflicting opposite-role instances onto currently free equal-size placements.
        """
        if not wakes or not conflicts:
            return ()

        occupied: set[tuple[str, int]] = set()
        for signal in target_signals + other_signals:
            if signal.is_awaken and signal.gpu_keys:
                occupied.update(signal.gpu_keys)
        reserved = {bundle_key for wake in wakes for bundle_key in (wake.gpu_keys or ())}
        pairs: list[tuple[InstanceSignal, InstanceSignal]] = []
        used_targets: set[tuple[str, str, int]] = set()

        if ordered_targets_by_shape is None:
            mutable_targets_by_shape: dict[tuple[str, int], list[InstanceSignal]] = {}
            for signal in other_signals:
                if (
                    self._is_scale_up_available(signal)
                    and signal.gpu_keys
                    and self._signal_pool_available(signal, trainer_waiting_hint)
                ):
                    mutable_targets_by_shape.setdefault(
                        (signal.model_name, len(signal.gpu_keys)),
                        [],
                    ).append(signal)
            ordered_targets_by_shape = {
                shape: tuple(
                    sorted(
                        candidates,
                        key=lambda item: (
                            self._pool_wake_priority(),
                            self._bundle_sort_key(item),
                            int(item.instance_id),
                        ),
                    )
                )
                for shape, candidates in mutable_targets_by_shape.items()
            }

        if ordered_sources is None:
            sources = sorted(
                conflicts,
                key=lambda item: (
                    item.model_name,
                    len(item.gpu_keys or ()),
                    self._bundle_sort_key(item),
                    int(item.instance_id),
                ),
            )
        else:
            conflict_keys = {self._signal_key(signal) for signal in conflicts}
            sources = [signal for signal in ordered_sources if self._signal_key(signal) in conflict_keys]

        target_cursors: dict[tuple[str, int], int] = {}
        for source in sources:
            source_size = len(source.gpu_keys or ())
            shape = (source.model_name, source_size)
            candidates = ordered_targets_by_shape.get(shape, ())
            cursor = target_cursors.get(shape, 0)
            while cursor < len(candidates):
                target = candidates[cursor]
                cursor += 1
                if (
                    self._signal_key(target) in used_targets
                    or target.gpu_keys.intersection(occupied)
                    or target.gpu_keys.intersection(reserved)
                ):
                    continue
                break
            else:
                target_cursors[shape] = cursor
                continue
            target_cursors[shape] = cursor
            pairs.append((source, target))
            used_targets.add(self._signal_key(target))
            reserved.update(target.gpu_keys or ())
        return tuple(pairs)

    def _heterogeneous_wake_prefix_plans_for_role(
        self,
        *,
        target_signals: list[InstanceSignal],
        other_signals: list[InstanceSignal],
        current_other_n: int,
        max_batch: int,
        trainer_waiting_hint: dict[str, Any] | None,
    ) -> dict[int, _WakePrefixPlan]:
        """
        Build minimum-Phi plans for each wake count using concrete GPU conflicts.

        The ordering pass is ``O(N log N)``. For different-size bundles the
        count DP has one state per reachable wake count and performs bounded
        group transitions, ``O(N^2)`` in the worst case. Conflict unions are
        represented by integer masks, so transitions do not copy or sort signal
        tuples. Plans are materialized once per reachable count, which keeps
        candidate construction ``O(N^2)`` (and ``O(N)`` for a single count).
        """
        other_awake = [signal for signal in other_signals if signal.is_awaken]
        other_awake.sort(key=lambda item: (self._bundle_sort_key(item), int(item.instance_id)))
        other_by_bundle: dict[tuple[str, int], list[InstanceSignal]] = {}
        for signal in other_awake:
            for bundle_key in signal.gpu_keys or ():
                other_by_bundle.setdefault(bundle_key, []).append(signal)

        victim_index = {self._signal_key(signal): index for index, signal in enumerate(other_awake)}
        victim_phi = tuple(self._instance_phi(signal, float(signal.running_queue_num)) for signal in other_awake)
        victim_requests = tuple(float(signal.running_queue_num) for signal in other_awake)

        def conflicts_for(signal: InstanceSignal) -> tuple[InstanceSignal, ...]:
            conflict_keys: set[tuple[str, str, int]] = set()
            for bundle_key in signal.gpu_keys or ():
                conflict_keys.update(self._signal_key(other) for other in other_by_bundle.get(bundle_key, ()))
            return tuple(
                other_awake[victim_index[key]]
                for key in sorted(
                    conflict_keys,
                    key=lambda item: (
                        self._bundle_sort_key(other_awake[victim_index[item]]),
                        int(item[2]),
                    ),
                )
            )

        wakeable_by_model: dict[str, list[InstanceSignal]] = {}
        for signal in target_signals:
            if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint):
                wakeable_by_model.setdefault(signal.model_name, []).append(signal)

        best_plans: dict[int, _WakePrefixPlan] = {}
        best_plan_keys: dict[int, tuple[Any, ...]] = {}
        max_group_size = max(
            (len(signals) for signals in wakeable_by_model.values()),
            default=0,
        )
        # The radix must exceed every group size to preserve prefix tie-breaking order.
        path_base = max_group_size + 2

        def state_key(state: _WakePlanDPState) -> tuple[Any, ...]:
            return (
                state.conflict_phi,
                state.conflict_requests,
                state.conflict_mask.bit_count(),
                state.train_count,
                state.path_code,
            )

        def add_conflicts(state: _WakePlanDPState, group: _WakeConflictGroup) -> _WakePlanDPState:
            new_mask = group.conflict_mask & ~state.conflict_mask
            added_phi = 0.0
            added_requests = 0.0
            while new_mask:
                bit = new_mask & -new_mask
                index = bit.bit_length() - 1
                added_phi += victim_phi[index]
                added_requests += victim_requests[index]
                new_mask ^= bit
            return _WakePlanDPState(
                conflict_mask=state.conflict_mask | group.conflict_mask,
                conflict_phi=state.conflict_phi + added_phi,
                conflict_requests=state.conflict_requests + added_requests,
                train_count=state.train_count,
                path_code=state.path_code,
            )

        def decode_group_takes(path_code: int, group_count: int) -> list[int]:
            takes = [0] * group_count
            for index in range(group_count - 1, -1, -1):
                digit = path_code % path_base
                takes[index] = path_base - 1 - digit
                path_code //= path_base
            return takes

        wake_impact_by_shape: dict[tuple[str, str, int], float] = {}

        def estimated_wake_impact(signal: InstanceSignal) -> float:
            shape_key = (
                str(signal.role_name),
                signal.model_name,
                len(signal.gpu_keys or ()),
            )
            impact = wake_impact_by_shape.get(shape_key)
            if impact is None:
                params = self._params_for_signal(signal)
                impact = 1.0 / compute_itl(params, 0.0, 1.0)
                wake_impact_by_shape[shape_key] = impact
            return impact

        ordered_migration_sources = tuple(
            sorted(
                other_awake,
                key=lambda item: (
                    item.model_name,
                    len(item.gpu_keys or ()),
                    self._bundle_sort_key(item),
                    int(item.instance_id),
                ),
            )
        )
        mutable_migration_targets: dict[tuple[str, int], list[InstanceSignal]] = {}
        for signal in other_signals:
            if (
                self._is_scale_up_available(signal)
                and signal.gpu_keys
                and self._signal_pool_available(signal, trainer_waiting_hint)
            ):
                mutable_migration_targets.setdefault(
                    (signal.model_name, len(signal.gpu_keys)),
                    [],
                ).append(signal)
        ordered_migration_targets = {
            shape: tuple(
                sorted(
                    candidates,
                    key=lambda item: (
                        self._pool_wake_priority(),
                        self._bundle_sort_key(item),
                        int(item.instance_id),
                    ),
                )
            )
            for shape, candidates in mutable_migration_targets.items()
        }

        for model_signals in wakeable_by_model.values():
            grouped_members: dict[tuple[tuple[str, str, int], ...], list[InstanceSignal]] = {}
            conflicts_by_group: dict[tuple[tuple[str, str, int], ...], tuple[InstanceSignal, ...]] = {}
            for signal in model_signals:
                conflicts = conflicts_for(signal)
                group_key = tuple(self._signal_key(item) for item in conflicts)
                grouped_members.setdefault(group_key, []).append(signal)
                conflicts_by_group[group_key] = conflicts

            groups: list[_WakeConflictGroup] = []
            for group_key, members in grouped_members.items():
                ordered_members = tuple(
                    sorted(
                        members,
                        key=lambda item: (
                            self._pool_wake_priority(),
                            self._bundle_sort_key(item),
                            int(item.instance_id),
                        ),
                    )
                )
                conflicts = conflicts_by_group[group_key]
                conflict_mask = 0
                conflict_phi = 0.0
                conflict_requests = 0.0
                for victim in conflicts:
                    index = victim_index[self._signal_key(victim)]
                    conflict_mask |= 1 << index
                    conflict_phi += victim_phi[index]
                    conflict_requests += victim_requests[index]
                train_prefix_counts = [0]
                for member in ordered_members:
                    train_prefix_counts.append(train_prefix_counts[-1] + self._pool_wake_priority())
                groups.append(
                    _WakeConflictGroup(
                        members=ordered_members,
                        conflicts=conflicts,
                        conflict_mask=conflict_mask,
                        conflict_phi=conflict_phi,
                        conflict_requests=conflict_requests,
                        scaling_impact=sum(estimated_wake_impact(member) for member in ordered_members),
                        train_prefix_counts=tuple(train_prefix_counts),
                    )
                )

            groups.sort(
                key=lambda group: (
                    group.conflict_phi,
                    group.conflict_requests,
                    -group.scaling_impact,
                    len(group.conflicts),
                    self._pool_wake_priority(),
                    self._bundle_sort_key(group.members[0]),
                    int(group.members[0].instance_id),
                )
            )

            # Each conflict group contributes a prefix of members.
            # the conflict mask charges a shared victim only once.
            states: dict[int, _WakePlanDPState] = {0: _WakePlanDPState()}
            for group in groups:
                next_states: dict[int, _WakePlanDPState] = {}
                for current_count, state in states.items():
                    max_take = min(len(group.members), max_batch - current_count)
                    opened_state = add_conflicts(state, group)
                    for take in range(max_take + 1):
                        next_count = current_count + take
                        candidate_state = opened_state if take else state
                        candidate_state = _WakePlanDPState(
                            conflict_mask=candidate_state.conflict_mask,
                            conflict_phi=candidate_state.conflict_phi,
                            conflict_requests=candidate_state.conflict_requests,
                            train_count=state.train_count + group.train_prefix_counts[take],
                            path_code=state.path_code * path_base + (path_base - 1 - take),
                        )
                        existing = next_states.get(next_count)
                        if existing is None or state_key(candidate_state) < state_key(existing):
                            next_states[next_count] = candidate_state
                states = next_states

            for batch_size, state in states.items():
                if batch_size <= 0:
                    continue
                takes = decode_group_takes(state.path_code, len(groups))
                wakes = tuple(
                    member for group, take in zip(groups, takes, strict=True) for member in group.members[:take]
                )
                # A candidate must be physically feasible for the target role as
                # well as for the opposite-role eviction set.
                if not self._bundle_conflict_free(list(wakes)):
                    continue
                conflicts = tuple(
                    other_awake[index] for index in range(len(other_awake)) if state.conflict_mask & (1 << index)
                )
                migration_pairs = self._pick_heterogeneous_migration_pairs(
                    wakes=wakes,
                    conflicts=conflicts,
                    target_signals=target_signals,
                    other_signals=other_signals,
                    trainer_waiting_hint=trainer_waiting_hint,
                    ordered_sources=ordered_migration_sources,
                    ordered_targets_by_shape=ordered_migration_targets,
                )
                if current_other_n - len(conflicts) + len(migration_pairs) < self.min_awake_per_role:
                    continue
                plan = _WakePrefixPlan(
                    wakes=wakes,
                    pre_sleep=conflicts,
                    pre_wake=tuple(target for _, target in migration_pairs),
                    migration_pairs=migration_pairs,
                )
                key = (
                    state.conflict_phi,
                    -len(plan.pre_wake),
                    state.conflict_requests,
                    state.conflict_mask.bit_count(),
                    state.train_count,
                    tuple(int(item.instance_id) for item in plan.wakes),
                )
                if batch_size not in best_plan_keys or key < best_plan_keys[batch_size]:
                    best_plans[batch_size] = plan
                    best_plan_keys[batch_size] = key
        return best_plans

    def _wake_prefix_plans_for_role(
        self,
        *,
        target_signals: list[InstanceSignal],
        other_signals: list[InstanceSignal],
        current_other_n: int,
        max_batch: int,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> dict[int, _WakePrefixPlan]:
        """
        Build deterministic wake prefixes for one scale-up side.

        A count delta maps to exactly one concrete wake/presleep choice: sort all
        wakeable target instances once, then take the first feasible prefix. Free
        devices come before occupied devices. among free devices, shared-pool
        instances come before train-pool instances. occupied devices prefer the
        ones whose opposite-role eviction has lower migration cost.
        """
        if max_batch <= 0:
            return {}
        if self._use_heterogeneous_wake_plans(target_signals, other_signals):
            return self._heterogeneous_wake_prefix_plans_for_role(
                target_signals=target_signals,
                other_signals=other_signals,
                current_other_n=current_other_n,
                max_batch=max_batch,
                trainer_waiting_hint=trainer_waiting_hint,
            )

        other_awake = [signal for signal in other_signals if signal.is_awaken]
        other_by_bundle: dict[tuple[str, int], list[InstanceSignal]] = {}
        for signal in other_awake:
            for bundle_key in signal.gpu_keys or ():
                other_by_bundle.setdefault(bundle_key, []).append(signal)

        def conflicts_for(signal: InstanceSignal) -> tuple[InstanceSignal, ...]:
            conflicts: dict[tuple[str, str, int], InstanceSignal] = {}
            for bundle_key in signal.gpu_keys or ():
                for other in other_by_bundle.get(bundle_key, ()):
                    conflicts[self._signal_key(other)] = other
            return tuple(sorted(conflicts.values(), key=lambda item: int(item.instance_id)))

        def wake_key(signal: InstanceSignal) -> tuple[Any, ...]:
            conflicts = conflicts_for(signal)
            free_device = 0 if not conflicts else 1
            conflict_phi = sum(self._instance_phi(item, float(item.running_queue_num)) for item in conflicts)
            conflict_requests = sum(float(item.running_queue_num) for item in conflicts)
            return (
                free_device,
                self._pool_wake_priority(),
                conflict_phi,
                conflict_requests,
                len(conflicts),
                int(signal.instance_id),
            )

        wakeable_by_model: dict[str, list[InstanceSignal]] = {}
        for signal in target_signals:
            if self._is_scale_up_available(signal) and self._signal_pool_available(signal, trainer_waiting_hint):
                wakeable_by_model.setdefault(signal.model_name, []).append(signal)
        if not wakeable_by_model:
            return {}

        def plan_key(plan: _WakePrefixPlan) -> tuple[Any, ...]:
            train_count = 0
            pre_sleep_phi = sum(self._instance_phi(item, float(item.running_queue_num)) for item in plan.pre_sleep)
            pre_sleep_requests = sum(float(item.running_queue_num) for item in plan.pre_sleep)
            return (
                pre_sleep_phi,
                pre_sleep_requests,
                len(plan.pre_sleep),
                train_count,
                [int(signal.instance_id) for signal in plan.wakes],
            )

        best_plans: dict[int, _WakePrefixPlan] = {}
        best_keys: dict[int, tuple[Any, ...]] = {}
        for model_signals in wakeable_by_model.values():
            selected: list[InstanceSignal] = []
            selected_target_bundles: set[tuple[str, int]] = set()
            pre_sleep_by_key: dict[tuple[str, str, int], InstanceSignal] = {}

            for signal in sorted(model_signals, key=wake_key):
                if signal.gpu_keys and selected_target_bundles.intersection(signal.gpu_keys):
                    continue
                signal_conflicts = conflicts_for(signal)
                next_pre_sleep = dict(pre_sleep_by_key)
                for victim in signal_conflicts:
                    next_pre_sleep[self._signal_key(victim)] = victim
                if current_other_n - len(next_pre_sleep) < self.min_awake_per_role:
                    continue

                selected.append(signal)
                if signal.gpu_keys:
                    selected_target_bundles.update(signal.gpu_keys)
                pre_sleep_by_key = next_pre_sleep
                batch_size = len(selected)
                if batch_size <= max_batch:
                    plan = _WakePrefixPlan(
                        wakes=tuple(selected),
                        pre_sleep=tuple(pre_sleep_by_key.values()),
                    )
                    key = plan_key(plan)
                    if batch_size not in best_keys or key < best_keys[batch_size]:
                        best_plans[batch_size] = plan
                        best_keys[batch_size] = key
                if batch_size >= max_batch:
                    break

        return best_plans

    def _build_rebalanced_count_candidate(
        self,
        *,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        rollout_n: int,
        rm_n: int,
        current_rollout_n: int,
        current_rm_n: int,
        current_throughput: float,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
        victim_orders_by_role: dict[PSRL_Role, dict[str, list[InstanceSignal]]] | None = None,
        wake_prefix_plans_by_role: dict[PSRL_Role, dict[int, _WakePrefixPlan]] | None = None,
        evaluate_candidate: bool = True,
    ) -> _ITLCandidate | None:
        """
        Build one candidate from target role counts after planned rebalancing.
        """
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_delta = int(rollout_n) - int(current_rollout_n)
        rm_delta = int(rm_n) - int(current_rm_n)
        if rollout_delta == 0 and rm_delta == 0:
            return None
        sleep_victims: list[InstanceSignal] = []
        wakes: list[InstanceSignal] = []
        if rollout_delta < 0:
            victims = self._pick_low_throughput_victims(
                rollout_signals,
                -rollout_delta,
                (victim_orders_by_role or {}).get(PSRL_Role.Rollout),
            )
            if victims is None:
                return None
            sleep_victims.extend(victims)
        if rm_delta < 0:
            victims = self._pick_low_throughput_victims(
                rm_signals,
                -rm_delta,
                (victim_orders_by_role or {}).get(PSRL_Role.RewardModel),
            )
            if victims is None:
                return None
            sleep_victims.extend(victims)
        if rollout_delta > 0 and rm_delta > 0:
            return None
        evicted_other: list[InstanceSignal] = []
        migration_wakes: list[InstanceSignal] = []
        migration_pairs: list[tuple[InstanceSignal, InstanceSignal]] = []
        if rollout_delta > 0:
            plan = (wake_prefix_plans_by_role or {}).get(PSRL_Role.Rollout, {}).get(rollout_delta)
            if plan is None:
                return None
            wakes = list(plan.wakes)
            evicted_other = list(plan.pre_sleep)
            migration_wakes = list(plan.pre_wake)
            migration_pairs = list(plan.migration_pairs)
            action = ScalingAction(
                action_type="scale_up",
                role_name=PSRL_Role.Rollout,
                model_name=wakes[0].model_name,
                num_instances=len(wakes),
                preferred_instance_ids=[int(wake.instance_id) for wake in wakes],
                reason="itl_objective_scale_up",
                pre_sleep_other_preferred=[self._signal_entry(victim) for victim in sleep_victims] or None,
                pre_wake_other_preferred=[self._signal_entry(target) for target in migration_wakes] or None,
            )
        elif rm_delta > 0:
            plan = (wake_prefix_plans_by_role or {}).get(PSRL_Role.RewardModel, {}).get(rm_delta)
            if plan is None:
                return None
            wakes = list(plan.wakes)
            evicted_other = list(plan.pre_sleep)
            migration_wakes = list(plan.pre_wake)
            migration_pairs = list(plan.migration_pairs)
            action = ScalingAction(
                action_type="scale_up",
                role_name=PSRL_Role.RewardModel,
                model_name=wakes[0].model_name,
                num_instances=len(wakes),
                preferred_instance_ids=[int(wake.instance_id) for wake in wakes],
                reason="itl_objective_scale_up",
                pre_sleep_other_preferred=[self._signal_entry(victim) for victim in sleep_victims] or None,
                pre_wake_other_preferred=[self._signal_entry(target) for target in migration_wakes] or None,
            )
        else:
            if not sleep_victims:
                return None
            target_role = sleep_victims[0].role_name
            target_model = sleep_victims[0].model_name
            if any(victim.role_name != target_role or victim.model_name != target_model for victim in sleep_victims):
                return None
            action = ScalingAction(
                action_type="scale_down",
                role_name=target_role,
                model_name=target_model,
                num_instances=len(sleep_victims),
                preferred_instance_ids=[int(victim.instance_id) for victim in sleep_victims],
                reason="itl_objective_scale_down",
            )
        # Cross-role evictions change the victim role count before scoring.
        if evicted_other:
            sleep_victims.extend(evicted_other)
            if evicted_other[0].role_name == PSRL_Role.Rollout:
                rollout_n += len(migration_wakes) - len(evicted_other)
            else:
                rm_n += len(migration_wakes) - len(evicted_other)
            if action.pre_sleep_other_preferred is None:
                action.pre_sleep_other_preferred = []
            action.pre_sleep_other_preferred.extend(self._signal_entry(v) for v in evicted_other)
            if rollout_n < self.min_awake_per_role or rm_n < self.min_awake_per_role:
                return None
        # Score the exact hypothetical active set for the summed throughput estimate.
        wake_ids_by_role: dict[PSRL_Role, set[int]] = {}
        sleep_ids_by_role: dict[PSRL_Role, set[int]] = {}
        if wakes:
            wake_ids_by_role[wakes[0].role_name] = {int(wake.instance_id) for wake in wakes}
        for migration_wake in migration_wakes:
            wake_ids_by_role.setdefault(migration_wake.role_name, set()).add(int(migration_wake.instance_id))
        for victim in sleep_victims:
            sleep_ids_by_role.setdefault(victim.role_name, set()).add(int(victim.instance_id))
        load_overrides_by_role: dict[PSRL_Role, dict[int, tuple[float, float]]] = {}
        for source, target in migration_pairs:
            load_overrides_by_role.setdefault(target.role_name, {})[int(target.instance_id)] = (
                self._vllm_current_request_load(source),
                float(source.total_token_num),
            )
        phi = self._sleep_phi(sleep_victims)
        if evaluate_candidate:
            next_throughput = self._system_throughput(
                grouped,
                rollout_n,
                rm_n,
                router_backlog_by_role,
                wake_ids_by_role=wake_ids_by_role,
                sleep_ids_by_role=sleep_ids_by_role,
                load_overrides_by_role=load_overrides_by_role,
            )
            delta_throughput = self._throughput_delta(next_throughput, current_throughput)
            gain_l = self._windowed_throughput_gain(delta_throughput, rollout_n, rm_n) - phi
        else:
            # Request-level mode first constructs the exact same action/placement
            # plan, then fills these score fields from the immutable snapshots.
            next_throughput = current_throughput
            delta_throughput = 0.0
            gain_l = -phi
        return _ITLCandidate(
            action,
            rollout_n,
            rm_n,
            current_throughput,
            next_throughput,
            delta_throughput,
            phi,
            gain_l,
            load_overrides_by_role or None,
        )

    def _effective_max_scale_instances_per_action(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
    ) -> int:
        """
        Return the per-action instance cap used by candidate enumeration.

        ``max_scale_instances_per_action=-1`` disables the cap: the returned
        value is the largest role instance count so wake/sleep prefixes are
        bounded only by available instances.

        Args:
            grouped (dict[PSRL_Role, list[InstanceSignal]]): Current instances
                keyed by role.

        Returns:
            int: Positive instance cap to pass into count enumeration and wake
                prefix planning.
        """
        configured = int(self.max_scale_instances_per_action)
        if configured == -1:
            return max(1, max((len(signals) for signals in grouped.values()), default=0))
        return max(1, configured)

    def _enumerate_candidates(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
    ) -> list[_ITLCandidate]:
        """
        Build count-based candidates scored after post-action rebalancing.
        """
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        current_rollout_n = sum(1 for s in rollout_signals if s.is_awaken)
        current_rm_n = sum(1 for s in rm_signals if s.is_awaken)
        candidate_ordering_started_s = time.monotonic()
        _, rollout_victim_order = self._selection_orders_for_role(
            rollout_signals,
            trainer_waiting_hint,
        )
        _, rm_victim_order = self._selection_orders_for_role(
            rm_signals,
            trainer_waiting_hint,
        )
        self._last_candidate_ordering_s = time.monotonic() - candidate_ordering_started_s
        victim_orders_by_role = {
            PSRL_Role.Rollout: rollout_victim_order,
            PSRL_Role.RewardModel: rm_victim_order,
        }
        role_throughputs = self._current_role_throughputs(
            grouped,
            current_rollout_n,
            current_rm_n,
            router_backlog_by_role,
        )
        # Current and candidate states must use the same load and objective definitions.
        backlog_map = router_backlog_by_role or {}
        current_throughput = self._system_throughput(
            grouped,
            current_rollout_n,
            current_rm_n,
            router_backlog_by_role,
        )
        candidates: list[_ITLCandidate] = []
        max_batch = self._effective_max_scale_instances_per_action(grouped)
        # Bottleneck weights include all outstanding demand, including router backlog.
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        w_rollout, w_rm = self._role_throughput_weights(rollout_signals, rm_signals, rollout_waiting, rm_waiting)
        rollout_bottleneck = (
            role_throughputs[PSRL_Role.Rollout] / w_rollout if w_rollout > 0.0 else role_throughputs[PSRL_Role.Rollout]
        )
        rm_bottleneck = (
            role_throughputs[PSRL_Role.RewardModel] / w_rm if w_rm > 0.0 else role_throughputs[PSRL_Role.RewardModel]
        )
        bottleneck_roles = self._enumerate_bottleneck_roles(
            grouped=grouped,
            role_throughputs=role_throughputs,
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
            w_rollout=w_rollout,
            w_rm=w_rm,
            rollout_bottleneck=rollout_bottleneck,
            rm_bottleneck=rm_bottleneck,
            router_backlog_by_role=router_backlog_by_role,
        )
        capacity = self._elastic_device_capacity(grouped, trainer_waiting_hint)
        rollout_min_n, rollout_max_n = self._role_count_bounds(
            rollout_signals,
            current_rollout_n,
            trainer_waiting_hint,
        )
        rm_min_n, rm_max_n = self._role_count_bounds(
            rm_signals,
            current_rm_n,
            trainer_waiting_hint,
        )
        wake_prefix_plans_by_role = {
            PSRL_Role.Rollout: self._wake_prefix_plans_for_role(
                target_signals=rollout_signals,
                other_signals=rm_signals,
                current_other_n=current_rm_n,
                max_batch=max_batch,
                trainer_waiting_hint=trainer_waiting_hint,
            ),
            PSRL_Role.RewardModel: self._wake_prefix_plans_for_role(
                target_signals=rm_signals,
                other_signals=rollout_signals,
                current_other_n=current_rollout_n,
                max_batch=max_batch,
                trainer_waiting_hint=trainer_waiting_hint,
            ),
        }
        seen_counts: set[tuple[int, int]] = set()

        for bottleneck_role in bottleneck_roles:
            if bottleneck_role == PSRL_Role.Rollout:
                min_target = max(rollout_min_n, current_rollout_n - max_batch)
                max_target = min(rollout_max_n, current_rollout_n + max_batch)
                target_counts = [n for n in range(min_target, max_target + 1) if n != current_rollout_n]
                for rollout_n in target_counts:
                    rm_n = current_rm_n
                    if rollout_n < current_rollout_n:
                        if rm_n < rm_min_n or rm_n > rm_max_n:
                            continue
                        if (
                            capacity > 0
                            and self._candidate_device_usage(rollout_signals, rm_signals, rollout_n, rm_n) > capacity
                        ):
                            continue
                    count_key = (rollout_n, rm_n)
                    if count_key in seen_counts:
                        continue
                    seen_counts.add(count_key)
                    candidate = self._build_rebalanced_count_candidate(
                        grouped=grouped,
                        rollout_n=rollout_n,
                        rm_n=rm_n,
                        current_rollout_n=current_rollout_n,
                        current_rm_n=current_rm_n,
                        current_throughput=current_throughput,
                        router_backlog_by_role=router_backlog_by_role,
                        trainer_waiting_hint=trainer_waiting_hint,
                        victim_orders_by_role=victim_orders_by_role,
                        wake_prefix_plans_by_role=wake_prefix_plans_by_role,
                        evaluate_candidate=not self.enable_request_level_candidate_evaluation,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
            else:
                min_target = max(rm_min_n, current_rm_n - max_batch)
                max_target = min(rm_max_n, current_rm_n + max_batch)
                target_counts = [n for n in range(min_target, max_target + 1) if n != current_rm_n]
                for rm_n in target_counts:
                    rollout_n = current_rollout_n
                    if rm_n < current_rm_n:
                        if rollout_n < rollout_min_n or rollout_n > rollout_max_n:
                            continue
                        if (
                            capacity > 0
                            and self._candidate_device_usage(rollout_signals, rm_signals, rollout_n, rm_n) > capacity
                        ):
                            continue
                    count_key = (rollout_n, rm_n)
                    if count_key in seen_counts:
                        continue
                    seen_counts.add(count_key)
                    candidate = self._build_rebalanced_count_candidate(
                        grouped=grouped,
                        rollout_n=rollout_n,
                        rm_n=rm_n,
                        current_rollout_n=current_rollout_n,
                        current_rm_n=current_rm_n,
                        current_throughput=current_throughput,
                        router_backlog_by_role=router_backlog_by_role,
                        trainer_waiting_hint=trainer_waiting_hint,
                        victim_orders_by_role=victim_orders_by_role,
                        wake_prefix_plans_by_role=wake_prefix_plans_by_role,
                        evaluate_candidate=not self.enable_request_level_candidate_evaluation,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
        return candidates

    @staticmethod
    def _request_level_snapshot_for_role(
        snapshots_by_role: dict[PSRL_Role, Any] | dict[str, Any] | None,
        role_name: PSRL_Role,
    ) -> Any:
        snapshots = snapshots_by_role or {}
        for key in (
            role_name,
            role_name.name,
            role_name.name.lower(),
            str(role_name),
        ):
            if key in snapshots:
                return snapshots[key]
        return None

    def _prepare_request_level_snapshot(
        self,
        *,
        role_name: PSRL_Role,
        raw_snapshot: Any,
        role_signals: list[InstanceSignal],
    ) -> RoleSnapshot:
        if raw_snapshot is None or isinstance(raw_snapshot, Exception):
            raise _RequestLevelCandidateEvaluationError(f"missing request-level snapshot for {role_name.name}")
        try:
            snapshot = (
                raw_snapshot if isinstance(raw_snapshot, RoleSnapshot) else RoleSnapshot.from_mapping(raw_snapshot)
            )
        except Exception as exc:
            raise _RequestLevelCandidateEvaluationError(
                f"invalid request-level snapshot for {role_name.name}: {exc}"
            ) from exc

        expected_strategy = "throughput_optimal" if role_name == PSRL_Role.Rollout else "itl"
        if snapshot.strategy != expected_strategy:
            raise _RequestLevelCandidateEvaluationError(
                f"snapshot strategy for {role_name.name} is {snapshot.strategy!r}, expected {expected_strategy!r}"
            )
        signal_by_id = {int(signal.instance_id): signal for signal in role_signals}
        prepared_instances = []
        for instance in snapshot.instances:
            signal = signal_by_id.get(instance.instance_id)
            if signal is None:
                raise _RequestLevelCandidateEvaluationError(
                    f"snapshot instance {instance.instance_id} has no {role_name.name} signal"
                )
            params = self._params_for_signal(signal)
            prepared_instances.append(
                replace(
                    instance,
                    is_awake=bool(signal.is_awaken),
                    throughput_params=(params.A, params.B, params.C, params.D),
                )
            )
        if set(signal_by_id) != {instance.instance_id for instance in prepared_instances}:
            missing = sorted(set(signal_by_id) - {instance.instance_id for instance in prepared_instances})
            raise _RequestLevelCandidateEvaluationError(
                f"request-level snapshot for {role_name.name} is missing instances {missing}"
            )
        return replace(
            snapshot,
            role=role_name.name,
            instances=tuple(prepared_instances),
            queue_scope=self.vllm_current_queue_scope,
        )

    def _evaluate_request_level_candidates(
        self,
        *,
        candidates: list[_ITLCandidate],
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
        request_level_snapshots_by_role: dict[PSRL_Role, Any] | dict[str, Any] | None,
    ) -> float:
        """
        Evaluate candidate scores using one immutable snapshot baseline.
        """
        evaluation_started_s = time.monotonic()
        self._last_request_level_simulation_timing = {
            "simulation_input_preparation_s": 0.0,
            "candidate_evaluation_wall_s": 0.0,
            "rebalance_simulation_s": 0.0,
            "router_simulation_s": 0.0,
            "simulation_wall_s": 0.0,
            "rebalance_router_overlap_s": 0.0,
            "candidate_scoring_s": 0.0,
        }
        snapshots = {
            role_name: self._prepare_request_level_snapshot(
                role_name=role_name,
                raw_snapshot=self._request_level_snapshot_for_role(
                    request_level_snapshots_by_role,
                    role_name,
                ),
                role_signals=grouped.get(role_name, []),
            )
            for role_name in (PSRL_Role.Rollout, PSRL_Role.RewardModel)
        }
        candidate_plans: list[dict[PSRL_Role, RoleCandidatePlan]] = []
        for index, candidate in enumerate(candidates):
            wake_by_role, sleep_by_role = self._candidate_wake_sleep_by_role(candidate)
            candidate_plans.append(
                {
                    role_name: RoleCandidatePlan(
                        wake_instance_ids=frozenset(wake_by_role.get(role_name, set())),
                        sleep_instance_ids=frozenset(sleep_by_role.get(role_name, set())),
                        primary_scale_up=(
                            candidate.action.action_type == "scale_up" and candidate.action.role_name == role_name
                        ),
                    )
                    for role_name in snapshots
                }
            )

        logical_task_count = 2 * (len(candidates) + 1)
        task_count = logical_task_count
        deduplicated_task_count = 0
        timing_semantics = "direct_stage_wall_union"
        candidate_scoring_started_s: float | None = None

        if self.candidate_evaluation_backend == "cpp":
            cpp_evaluator = self._cpp_candidate_evaluator
            if cpp_evaluator is None:
                raise CppCandidateEvaluatorError("C++ request-level candidate evaluator was not initialized")
            cpp_started_s = time.monotonic()
            batch = cpp_evaluator.evaluate(
                snapshots={role_name.name: snapshot for role_name, snapshot in snapshots.items()},
                candidate_plans=[
                    {role_name.name: plan for role_name, plan in plans.items()} for plans in candidate_plans
                ],
            )
            cpp_backend_wall_s = time.monotonic() - cpp_started_s
            self._last_request_level_simulation_timing.update(
                {
                    "simulation_input_preparation_s": (
                        cpp_started_s - evaluation_started_s + max(0.0, cpp_backend_wall_s - batch.evaluation_wall_s)
                    ),
                    "candidate_evaluation_wall_s": batch.evaluation_wall_s,
                    "rebalance_simulation_s": batch.rebalance_simulation_s,
                    "router_simulation_s": batch.router_simulation_s,
                    "simulation_wall_s": batch.simulation_wall_s,
                    "rebalance_router_overlap_s": batch.rebalance_router_overlap_s,
                }
            )
            candidate_scoring_started_s = time.monotonic()
            baseline_results = {PSRL_Role[role_name]: result for role_name, result in batch.baseline_results.items()}
            candidate_results = [
                {PSRL_Role[role_name]: result for role_name, result in results.items()}
                for results in batch.candidate_results
            ]
            logical_task_count = batch.logical_task_count
            task_count = batch.unique_task_count
            deduplicated_task_count = batch.deduplicated_task_count
        else:
            executor = self._candidate_evaluation_executor
            if executor is None:
                raise _RequestLevelCandidateEvaluationError(
                    "Python request-level candidate executor was not initialized"
                )
            try:
                contexts = {
                    role_name: prepare_role_evaluation_context(snapshot) for role_name, snapshot in snapshots.items()
                }
            except Exception as exc:
                raise _RequestLevelCandidateEvaluationError(
                    f"failed to prepare request-level evaluation context: {exc}"
                ) from exc

            simulation_started_s = time.monotonic()
            self._last_request_level_simulation_timing.update(
                {"simulation_input_preparation_s": simulation_started_s - evaluation_started_s}
            )
            plan_futures: dict[
                tuple[PSRL_Role, RoleCandidatePlan],
                Future,
            ] = {}
            plan_labels: dict[
                tuple[PSRL_Role, RoleCandidatePlan],
                tuple[str, int | None],
            ] = {}

            def submit_once(
                role_name: PSRL_Role,
                plan: RoleCandidatePlan,
                *,
                kind: str,
                index: int | None,
            ) -> tuple[PSRL_Role, RoleCandidatePlan]:
                key = (role_name, plan)
                if key not in plan_futures:
                    plan_futures[key] = executor.submit(
                        evaluate_role_candidate,
                        contexts[role_name],
                        plan,
                    )
                    plan_labels[key] = (kind, index)
                return key

            baseline_plan = RoleCandidatePlan()
            baseline_keys = {
                role_name: submit_once(
                    role_name,
                    baseline_plan,
                    kind="baseline",
                    index=None,
                )
                for role_name in contexts
            }
            candidate_plan_keys = [
                {
                    role_name: submit_once(
                        role_name,
                        plans[role_name],
                        kind="candidate",
                        index=index,
                    )
                    for role_name in contexts
                }
                for index, plans in enumerate(candidate_plans)
            ]

            results_by_plan: dict[
                tuple[PSRL_Role, RoleCandidatePlan],
                RoleEvaluationResult,
            ] = {}
            try:
                for key, future in plan_futures.items():
                    role_name, _ = key
                    kind, index = plan_labels[key]
                    try:
                        result = future.result()
                    except Exception as exc:
                        raise RuntimeError(f"kind={kind} candidate={index} role={role_name.name}: {exc}") from exc
                    results_by_plan[key] = result
            except Exception as exc:
                for future in plan_futures.values():
                    future.cancel()
                raise _RequestLevelCandidateEvaluationError(f"candidate-role evaluation failed: {exc}") from exc
            simulation_elapsed_s = time.monotonic() - simulation_started_s
            rebalance_intervals = [
                (result.rebalance_started_s, result.rebalance_finished_s)
                for result in results_by_plan.values()
                if result.rebalance_started_s is not None and result.rebalance_finished_s is not None
            ]
            router_intervals = [
                (result.router_started_s, result.router_finished_s)
                for result in results_by_plan.values()
                if result.router_started_s is not None and result.router_finished_s is not None
            ]
            rebalance_wall_s = interval_union_s(rebalance_intervals)
            router_wall_s = interval_union_s(router_intervals)
            simulation_wall_s = interval_union_s(rebalance_intervals + router_intervals)
            self._last_request_level_simulation_timing.update(
                {
                    "candidate_evaluation_wall_s": simulation_elapsed_s,
                    "rebalance_simulation_s": rebalance_wall_s,
                    "router_simulation_s": router_wall_s,
                    "simulation_wall_s": simulation_wall_s,
                    "rebalance_router_overlap_s": max(
                        0.0,
                        rebalance_wall_s + router_wall_s - simulation_wall_s,
                    ),
                }
            )

            candidate_scoring_started_s = time.monotonic()
            baseline_results = {role_name: results_by_plan[key] for role_name, key in baseline_keys.items()}
            candidate_results = [
                {role_name: results_by_plan[key] for role_name, key in role_keys.items()}
                for role_keys in candidate_plan_keys
            ]
            task_count = len(plan_futures)
            deduplicated_task_count = logical_task_count - task_count

        if candidate_scoring_started_s is None:
            candidate_scoring_started_s = time.monotonic()
        expected_roles = {PSRL_Role.Rollout, PSRL_Role.RewardModel}
        if set(baseline_results) != expected_roles or any(
            set(results) != expected_roles for results in candidate_results
        ):
            raise _RequestLevelCandidateEvaluationError("candidate-role evaluation returned an incomplete result set")

        baseline_throughput = self._combine_request_level_role_throughputs(
            grouped,
            {role: result.throughput for role, result in baseline_results.items()},
            router_backlog_by_role,
        )
        for candidate, results in zip(candidates, candidate_results, strict=True):
            next_throughput = self._combine_request_level_role_throughputs(
                grouped,
                {role: result.throughput for role, result in results.items()},
                router_backlog_by_role,
            )
            delta_throughput = self._throughput_delta(
                next_throughput,
                baseline_throughput,
            )
            candidate.current_throughput = baseline_throughput
            candidate.next_throughput = next_throughput
            candidate.delta_throughput = delta_throughput
            candidate.gain_l = (
                self._windowed_throughput_gain(
                    delta_throughput,
                    candidate.rollout_n,
                    candidate.rm_n,
                )
                - candidate.phi
            )
            candidate.request_level_results_by_role = results
            if candidate.action.action_type == "scale_up":
                primary_result = results[candidate.action.role_name]
                candidate.action.planned_request_migrations = [
                    move.as_dict() for move in primary_result.rebalance_moves
                ]
        self._last_request_level_simulation_timing["candidate_scoring_s"] = (
            time.monotonic() - candidate_scoring_started_s
        )
        self._policy_log(
            "request_level_candidate_evaluation",
            backend=self.candidate_evaluation_backend,
            candidates=len(candidates),
            logical_tasks=logical_task_count,
            tasks=task_count,
            deduplicated_tasks=deduplicated_task_count,
            max_workers=self.candidate_evaluation_max_workers,
            elapsed_s=f"{time.monotonic() - evaluation_started_s:.6f}",
            simulation_input_preparation_s=(
                f"{self._last_request_level_simulation_timing['simulation_input_preparation_s']:.6f}"
            ),
            candidate_evaluation_wall_s=(
                f"{self._last_request_level_simulation_timing['candidate_evaluation_wall_s']:.6f}"
            ),
            rebalance_simulation_s=f"{self._last_request_level_simulation_timing['rebalance_simulation_s']:.6f}",
            router_simulation_s=f"{self._last_request_level_simulation_timing['router_simulation_s']:.6f}",
            simulation_wall_s=f"{self._last_request_level_simulation_timing['simulation_wall_s']:.6f}",
            rebalance_router_overlap_s=(
                f"{self._last_request_level_simulation_timing['rebalance_router_overlap_s']:.6f}"
            ),
            candidate_scoring_s=f"{self._last_request_level_simulation_timing['candidate_scoring_s']:.6f}",
            timing_semantics=timing_semantics,
        )
        return baseline_throughput

    @staticmethod
    def _format_throughput_value(value: float) -> str:
        """
        Format a throughput for logs, preserving the +inf (idle) bottleneck case.
        """
        if math.isinf(value):
            return "inf"
        return f"{value:.6f}"

    def _instance_throughput_from_load(self, signal: InstanceSignal, request_load: float, token_load: float) -> float:
        """
        Estimate one instance's throughput from a hypothetical request/token load.
        """
        if request_load <= 0.0:
            return self._idle_instance_throughput()
        itl = compute_itl(self._params_for_signal(signal), token_load, request_load)
        return request_load / itl

    def _role_throughput_for_objective(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
        load_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> float:
        """
        Per-role throughput under the active objective (matches ``_system_throughput``).
        """
        if self.throughput_objective == "sum":
            return self._role_throughput_sum(
                role_signals,
                n_instances,
                router_waiting_load,
                wake_instance_ids,
                sleep_instance_ids,
                load_overrides,
            )
        return self._role_throughput(role_signals, n_instances, router_waiting_load)

    def _role_instance_load_rows(
        self,
        role_signals: list[InstanceSignal],
        n_instances: int,
        router_waiting_load: _RouterWaitingLoad,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
        load_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> list[tuple[int, float, float, float]]:
        """
        Per-instance ``(id, requests, tokens, throughput)`` rows under the active objective.

        The ``sum`` objective reports each active instance's own running load plus an
        even share of router backlog. the balanced objective reports the averaged
        per-instance load (identical across instances), matching how the role/system
        throughput is actually scored.
        """
        active = sorted(
            self._active_role_signals(role_signals, n_instances, wake_instance_ids, sleep_instance_ids),
            key=lambda item: int(item.instance_id),
        )
        if not active:
            return []
        rows: list[tuple[int, float, float, float]] = []
        if self.throughput_objective == "sum":
            waiting_requests_per_instance = router_waiting_load.request_count / float(len(active))
            waiting_tokens_per_instance = router_waiting_load.token_count / float(len(active))
            for signal in active:
                override = (load_overrides or {}).get(int(signal.instance_id))
                if override is None:
                    request_load = self._vllm_current_request_load(signal)
                    token_load = float(signal.total_token_num)
                else:
                    request_load, token_load = override
                request_load += waiting_requests_per_instance
                token_load += waiting_tokens_per_instance
                rows.append(
                    (
                        int(signal.instance_id),
                        request_load,
                        token_load,
                        self._instance_throughput_from_load(signal, request_load, token_load),
                    )
                )
            return rows
        total_requests, total_tokens = self._role_total_load(role_signals, router_waiting_load)
        avg_requests = total_requests / float(n_instances) if n_instances > 0 else 0.0
        avg_tokens = total_tokens / float(n_instances) if n_instances > 0 else 0.0
        avg_throughput = (
            self._instance_throughput_from_load(active[0], avg_requests, avg_tokens)
            if avg_requests > 0.0
            else self._idle_instance_throughput()
        )
        for signal in active:
            rows.append((int(signal.instance_id), avg_requests, avg_tokens, avg_throughput))
        return rows

    @staticmethod
    def _format_instance_load_rows(rows: list[tuple[int, float, float, float]]) -> str:
        """
        Render per-instance load rows for the decision cycle log.
        """
        if not rows:
            return "[]"
        parts = [
            f"id{instance_id}:req={request_load:.1f},tok={token_load:.0f},"
            f"tp={ThroughputOptimalScalingPolicy._format_throughput_value(throughput)}"
            for instance_id, request_load, token_load, throughput in rows
        ]
        return "[" + "; ".join(parts) + "]"

    def _log_role_state(
        self,
        cycle_log: Any,
        label: str,
        role_signals: list[InstanceSignal],
        n_instances: int,
        load_waiting: _RouterWaitingLoad,
        router_backlog: _RouterWaitingLoad,
        wake_instance_ids: set[int] | None = None,
        sleep_instance_ids: set[int] | None = None,
        load_overrides: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        """
        Log one role's instance count, per-instance load, router backlog and role throughput.

        ``load_waiting`` drives the per-instance/role throughput estimates (subject to
        ``current_state_include_router_waiting`` for the current state), while
        ``router_backlog`` is always the real backlog reported for visibility.
        """
        rows = self._role_instance_load_rows(
            role_signals,
            n_instances,
            load_waiting,
            wake_instance_ids,
            sleep_instance_ids,
            load_overrides,
        )
        total_requests, total_tokens = self._role_total_load(role_signals, load_waiting)
        role_throughput = self._role_throughput_for_objective(
            role_signals,
            n_instances,
            load_waiting,
            wake_instance_ids,
            sleep_instance_ids,
            load_overrides,
        )
        cycle_log.kv(
            **{
                f"{label}_n": n_instances,
                f"{label}_total_req": f"{total_requests:.1f}",
                f"{label}_total_tok": f"{total_tokens:.0f}",
                f"{label}_role_tp": self._format_throughput_value(role_throughput),
                f"{label}_router_req": f"{router_backlog.request_count:.0f}",
                f"{label}_router_tok": f"{router_backlog.token_count:.0f}",
            }
        )
        cycle_log.note(f"{label}_instances={self._format_instance_load_rows(rows)}")

    @staticmethod
    def _candidate_wake_sleep_by_role(
        candidate: _ITLCandidate,
    ) -> tuple[dict[PSRL_Role, set[int]], dict[PSRL_Role, set[int]]]:
        """
        Derive per-role wake/sleep instance id sets from a candidate's action.
        """
        wake_by_role: dict[PSRL_Role, set[int]] = {}
        sleep_by_role: dict[PSRL_Role, set[int]] = {}
        action = candidate.action
        if action.action_type == "scale_up":
            if action.preferred_instance_ids:
                wake_by_role[action.role_name] = {int(i) for i in action.preferred_instance_ids}
            for entry in action.pre_wake_other_preferred or []:
                wake_by_role.setdefault(entry["role_name"], set()).add(int(entry["instance_id"]))
            for entry in action.pre_sleep_other_preferred or []:
                sleep_by_role.setdefault(entry["role_name"], set()).add(int(entry["instance_id"]))
        elif action.preferred_instance_ids:
            sleep_by_role[action.role_name] = {int(i) for i in action.preferred_instance_ids}
        return wake_by_role, sleep_by_role

    def _log_candidate_state(
        self,
        cycle_log: Any,
        index: int,
        candidate: _ITLCandidate,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad,
        rm_waiting: _RouterWaitingLoad,
    ) -> None:
        """
        Log one candidate's action, estimated per-side load/throughput, Phi and L.
        """
        action = candidate.action
        wake_by_role, sleep_by_role = self._candidate_wake_sleep_by_role(candidate)
        cycle_log.note(
            f"candidate[{index}] {action.action_type} {action.role_name.name}/{action.model_name} "
            f"num_instances={action.num_instances} preferred={action.preferred_instance_ids} "
            f"pre_wake={action.pre_wake_other_preferred} pre_sleep={action.pre_sleep_other_preferred}"
        )
        cycle_log.kv(
            rollout_n=candidate.rollout_n,
            rm_n=candidate.rm_n,
            next_throughput=self._format_throughput_value(candidate.next_throughput),
            delta_throughput=f"{candidate.delta_throughput:.6f}",
            Phi=f"{candidate.phi:.6f}",
            L=f"{candidate.gain_l:.6f}",
        )
        if candidate.request_level_results_by_role:
            for role_name in (PSRL_Role.Rollout, PSRL_Role.RewardModel):
                result = candidate.request_level_results_by_role.get(role_name)
                if result is None:
                    continue
                label = "rollout" if role_name == PSRL_Role.Rollout else "rm"
                cycle_log.kv(
                    **{
                        f"{label}_request_level_tp": self._format_throughput_value(result.throughput),
                        f"{label}_routed": result.routed_count,
                        f"{label}_unrouted": result.unrouted_count,
                        f"{label}_rebalance_planned": len(result.rebalance_moves),
                    }
                )
                cycle_log.note(
                    f"{label}_request_level_instances={result.instance_throughputs} "
                    f"moves={[move.as_dict() for move in result.rebalance_moves]}"
                )
        self._log_role_state(
            cycle_log,
            "rollout_est",
            rollout_signals,
            candidate.rollout_n,
            rollout_waiting,
            rollout_waiting,
            wake_by_role.get(PSRL_Role.Rollout),
            sleep_by_role.get(PSRL_Role.Rollout),
            (candidate.load_overrides_by_role or {}).get(PSRL_Role.Rollout),
        )
        self._log_role_state(
            cycle_log,
            "rm_est",
            rm_signals,
            candidate.rm_n,
            rm_waiting,
            rm_waiting,
            wake_by_role.get(PSRL_Role.RewardModel),
            sleep_by_role.get(PSRL_Role.RewardModel),
            (candidate.load_overrides_by_role or {}).get(PSRL_Role.RewardModel),
        )

    def decide(
        self,
        signals: list[InstanceSignal],
        execution_in_progress: bool = False,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        trainer_waiting_hint: dict[str, Any] | None = None,
        pending_scale_up_by_role: dict[PSRL_Role, int] | None = None,
        request_level_snapshots_by_role: dict[PSRL_Role, Any] | dict[str, Any] | None = None,
    ) -> ScalingDecision:
        """
        Pick the scale action with highest ``gain_l`` if it exceeds ``min_gain``.
        """
        _ = pending_scale_up_by_role
        if self.enable_request_level_candidate_evaluation and request_level_snapshots_by_role is not None:
            # Queue pressure and candidate simulation must describe the same immutable requests.
            router_backlog_by_role = {}
            for role in (PSRL_Role.Rollout, PSRL_Role.RewardModel):
                raw = self._request_level_snapshot_for_role(request_level_snapshots_by_role, role)
                if raw is None:
                    continue
                snapshot = raw if isinstance(raw, RoleSnapshot) else RoleSnapshot.from_mapping(raw)
                router_backlog_by_role[role] = {
                    "count": len(snapshot.pending_requests),
                    "total_tokens": sum(request.seq_len for request in snapshot.pending_requests),
                }
        self.last_planner_breakdown = {
            "state_analysis_s": 0.0,
            "candidate_ordering_s": 0.0,
            "candidate_set_construction_s": 0.0,
            "simulation_input_preparation_s": 0.0,
            "candidate_evaluation_wall_s": 0.0,
            "rebalance_simulation_s": 0.0,
            "router_simulation_s": 0.0,
            "simulation_wall_s": 0.0,
            "rebalance_router_overlap_s": 0.0,
            "candidate_scoring_s": 0.0,
            "best_candidate_selection_s": 0.0,
        }
        self._last_candidate_ordering_s = 0.0
        state_analysis_started_s = time.monotonic()
        cy = self._cycle_log
        cy.start()
        cy.section("inputs")
        cy.kv(n_signals=len(signals), execution_in_progress=execution_in_progress)
        if not self.enable:
            self._finish_cycle("skipped", "policy_disabled")
            return ScalingDecision(actions=[], reason="policy_disabled", estimated_lambda=0.0, role_to_total_mu={})
        if not signals:
            self._finish_cycle("skipped", "empty_signals")
            return ScalingDecision(actions=[], reason="empty_signals", estimated_lambda=0.0, role_to_total_mu={})
        if execution_in_progress:
            self._finish_cycle("skipped", "decision_execution_in_progress")
            return ScalingDecision(
                actions=[], reason="decision_execution_in_progress", estimated_lambda=0.0, role_to_total_mu={}
            )
        grouped = self._group_by_role(signals)
        _, role_total_mu = self._build_mu_maps(signals)
        estimated_lambda = self._estimate_lambda(signals, role_total_mu)
        now_ms = time.time() * 1000
        if now_ms - self.last_action_time_ms < self.cooldown_ms:
            self._finish_cycle("skipped", "cooldown")
            return ScalingDecision(actions=[], reason="cooldown", estimated_lambda=0.0, role_to_total_mu=role_total_mu)
        stale_count = sum(1 for signal in signals if self._is_signal_staled(signal))
        all_stale = stale_count == len(signals)
        backlog_positive = self._router_backlog_positive(router_backlog_by_role)
        if all_stale and not backlog_positive:
            self._finish_cycle("skipped", "all_signals_stale", stale_count=stale_count)
            return ScalingDecision(
                actions=[],
                reason="all_signals_stale",
                estimated_lambda=estimated_lambda,
                role_to_total_mu=role_total_mu,
            )
        if all_stale and backlog_positive:
            cy.note(
                "all instance snapshots are stale, but router backlog is positive; continuing for backlog-driven wake"
            )

        self.last_planner_breakdown["state_analysis_s"] = time.monotonic() - state_analysis_started_s
        candidate_construction_started_s = time.monotonic()
        candidates = self._enumerate_candidates(grouped, router_backlog_by_role, trainer_waiting_hint)
        candidate_construction_elapsed_s = time.monotonic() - candidate_construction_started_s
        self.last_planner_breakdown["candidate_ordering_s"] = self._last_candidate_ordering_s
        self.last_planner_breakdown["candidate_set_construction_s"] = max(
            0.0,
            candidate_construction_elapsed_s - self._last_candidate_ordering_s,
        )
        request_level_current_throughput: float | None = None
        if self.enable_request_level_candidate_evaluation:
            try:
                request_level_current_throughput = self._evaluate_request_level_candidates(
                    candidates=candidates,
                    grouped=grouped,
                    router_backlog_by_role=router_backlog_by_role,
                    request_level_snapshots_by_role=request_level_snapshots_by_role,
                )
                self.last_planner_breakdown.update(self._last_request_level_simulation_timing)
            except CppCandidateEvaluatorError as exc:
                cy.section("request_level_candidate_evaluation")
                cy.kv(
                    enabled=True,
                    status="fatal",
                    error=str(exc),
                )
                self._finish_cycle(
                    "error",
                    "cpp_candidate_evaluator_failed",
                    error=str(exc),
                )
                raise
            except _RequestLevelCandidateEvaluationError as exc:
                cy.section("request_level_candidate_evaluation")
                cy.kv(
                    enabled=True,
                    status="failed",
                    error=str(exc),
                )
                self._finish_cycle(
                    "no_action",
                    "request_level_candidate_evaluation_failed",
                    error=str(exc),
                )
                return ScalingDecision(
                    actions=[],
                    reason="request_level_candidate_evaluation_failed",
                    estimated_lambda=estimated_lambda,
                    role_to_total_mu=role_total_mu,
                )
        best_candidate_selection_started_s = time.monotonic()
        candidates.sort(key=lambda item: item.gain_l, reverse=True)
        best = candidates[0] if candidates else None
        self.last_planner_breakdown["best_candidate_selection_s"] = (
            time.monotonic() - best_candidate_selection_started_s
        )
        current_rollout_n = sum(1 for s in grouped.get(PSRL_Role.Rollout, []) if s.is_awaken)
        current_rm_n = sum(1 for s in grouped.get(PSRL_Role.RewardModel, []) if s.is_awaken)
        # Same objective-aware path used inside ``_enumerate_candidates`` so the
        # logged baseline matches the baseline used in candidate ``delta_throughput``.
        current_throughput = (
            request_level_current_throughput
            if request_level_current_throughput is not None
            else self._system_throughput(
                grouped,
                current_rollout_n,
                current_rm_n,
                router_backlog_by_role,
            )
        )
        cy.section("itl_objective")
        cy.kv(
            current_throughput=f"{current_throughput:.8f}",
            candidates=len(candidates),
            decision_window_s=f"{self.decision_window_s:.3f}",
            migrate_time_s=f"{self.migrate_time_s:.3f}",
            min_gain=f"{self.min_gain:.8f}",
            throughput_objective=self.throughput_objective,
            vllm_current_queue_scope=self.vllm_current_queue_scope,
            max_scale_instances_per_action=self.max_scale_instances_per_action,
            enable_heterogeneous_parallelism_candidates=self.enable_heterogeneous_parallelism_candidates,
            enable_request_level_candidate_evaluation=self.enable_request_level_candidate_evaluation,
            candidate_evaluation_max_workers=self.candidate_evaluation_max_workers,
            router_waiting_top_t=self.router_waiting_top_t,
            current_state_include_router_waiting=self.current_state_include_router_waiting,
            role_throughput_weight_enable=self.role_throughput_weight_enable,
            role_throughput_weight_basis=self.role_throughput_weight_basis,
            role_throughput_weight_mode=self.role_throughput_weight_mode,
            **self._objective_log_extras(grouped, router_backlog_by_role),
        )

        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        backlog_map = router_backlog_by_role or {}
        # Real router backlog (always shown) vs the current-state load fed into
        # throughput (zeroed when current_state_include_router_waiting is off).
        rollout_backlog = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_backlog = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        rollout_current_waiting = self._current_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_current_waiting = self._current_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        cy.section("current_state")
        self._log_role_state(
            cy, "rollout", rollout_signals, current_rollout_n, rollout_current_waiting, rollout_backlog
        )
        self._log_role_state(cy, "rm", rm_signals, current_rm_n, rm_current_waiting, rm_backlog)
        cy.kv(system_throughput=self._format_throughput_value(current_throughput))
        cy.section("candidates")
        for index, candidate in enumerate(candidates):
            self._log_candidate_state(cy, index, candidate, rollout_signals, rm_signals, rollout_backlog, rm_backlog)

        cy.section("itl_decision")
        if best is None:
            self._finish_cycle(
                "no_action", "itl_no_feasible_candidate", current_throughput=f"{current_throughput:.8f}"
            )
            return ScalingDecision(
                actions=[],
                reason="itl_no_feasible_candidate",
                estimated_lambda=estimated_lambda,
                role_to_total_mu=role_total_mu,
            )
        cy.kv(
            best_action=best.action.action_type,
            best_role=best.action.role_name.name,
            best_model=best.action.model_name,
            best_num_instances=best.action.num_instances,
            preferred=best.action.preferred_instance_ids,
            pre_wake=best.action.pre_wake_other_preferred,
            pre_sleep=best.action.pre_sleep_other_preferred,
            planned_request_migrations=best.action.planned_request_migrations,
            next_rollout_n=best.rollout_n,
            next_rm_n=best.rm_n,
            next_throughput=f"{best.next_throughput:.8f}",
            delta_throughput=f"{best.delta_throughput:.8f}",
            Phi=f"{best.phi:.8f}",
            L=f"{best.gain_l:.8f}",
        )
        if best.gain_l <= self.min_gain:
            self._finish_cycle(
                "no_action",
                "itl_gain_below_min_gain",
                best_L=f"{best.gain_l:.8f}",
                min_gain=f"{self.min_gain:.8f}",
            )
            return ScalingDecision(
                actions=[],
                reason="itl_gain_below_min_gain",
                estimated_lambda=estimated_lambda,
                role_to_total_mu=role_total_mu,
            )
        self.last_action_time_ms = now_ms
        reason = f"itl_best_{best.action.action_type}_{best.action.role_name.name}"
        best.action.reason = reason
        self._finish_cycle(
            "action",
            reason,
            current_throughput=f"{best.current_throughput:.8f}",
            next_throughput=f"{best.next_throughput:.8f}",
            delta_throughput=f"{best.delta_throughput:.8f}",
            Phi=f"{best.phi:.8f}",
            L=f"{best.gain_l:.8f}",
        )
        return ScalingDecision(
            actions=[best.action],
            reason=reason,
            estimated_lambda=estimated_lambda,
            role_to_total_mu=role_total_mu,
        )

    @staticmethod
    def _idle_instance_throughput() -> float:
        """
        Return zero throughput for an instance with no outstanding demand.
        """
        return 0.0

    def _weighted_system_throughput(
        self,
        rollout_tp: float,
        rm_tp: float,
        w_rollout: float,
        w_rm: float,
    ) -> float:
        """
        Combine role throughputs using pressure-weighted harmonic terms.

        `harmonic_denominator_use_max` selects the maximum reciprocal term.
        A zero-throughput role collapses the harmonic objective to zero.
        """
        wr = w_rollout if w_rollout > 0.0 else 1.0
        wm = w_rm if w_rm > 0.0 else 1.0
        rollout_term = wr * _inv_throughput(rollout_tp)
        rm_term = wm * _inv_throughput(rm_tp)
        if self.harmonic_denominator_use_max:
            denom = max(rollout_term, rm_term)
        else:
            denom = rollout_term + rm_term
        if math.isinf(denom):
            # At least one role has zero throughput -> harmonic mean is 0.
            return 0.0
        if denom == 0.0:
            # Both roles idle (infinite throughput) -> objective is unbounded.
            return math.inf
        return (wr + wm) / denom

    def _system_throughput(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        rollout_n: int,
        rm_n: int,
        router_backlog_by_role: dict[PSRL_Role, Any] | None = None,
        wake_ids_by_role: dict[PSRL_Role, set[int]] | None = None,
        sleep_ids_by_role: dict[PSRL_Role, set[int]] | None = None,
        load_overrides_by_role: dict[PSRL_Role, dict[int, tuple[float, float]]] | None = None,
    ) -> float:
        """
        Score the hypothetical active set using harmonic throughput.

        When exactly one role is demand-starved, score the loaded role
        using its balanced batch-drain rate.
        """
        backlog_map = router_backlog_by_role or {}
        wake_map = wake_ids_by_role or {}
        sleep_map = sleep_ids_by_role or {}
        load_override_map = load_overrides_by_role or {}
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        if self.throughput_objective == "sum":
            rollout_tp = self._role_throughput_sum(
                rollout_signals,
                rollout_n,
                rollout_waiting,
                wake_map.get(PSRL_Role.Rollout),
                sleep_map.get(PSRL_Role.Rollout),
                load_override_map.get(PSRL_Role.Rollout),
            )
            rm_tp = self._role_throughput_sum(
                rm_signals,
                rm_n,
                rm_waiting,
                wake_map.get(PSRL_Role.RewardModel),
                sleep_map.get(PSRL_Role.RewardModel),
                load_override_map.get(PSRL_Role.RewardModel),
            )
        else:
            rollout_tp = self._role_throughput(rollout_signals, rollout_n, rollout_waiting)
            rm_tp = self._role_throughput(rm_signals, rm_n, rm_waiting)
        w_rollout, w_rm = self._role_throughput_weights(rollout_signals, rm_signals, rollout_waiting, rm_waiting)
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role is None:
            return self._weighted_system_throughput(rollout_tp, rm_tp, w_rollout, w_rm)
        # Under starvation, estimate how added capacity drains the loaded batch.
        if starved_role == PSRL_Role.Rollout:
            return self._degraded_loaded_throughput(rm_signals, rm_n, rm_waiting, w_rm)
        return self._degraded_loaded_throughput(rollout_signals, rollout_n, rollout_waiting, w_rollout)

    def _combine_request_level_role_throughputs(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        role_throughputs: dict[PSRL_Role, float],
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> float:
        """
        Apply harmonic/starvation math to request-level role simulations.
        """
        backlog_map = router_backlog_by_role or {}
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        w_rollout, w_rm = self._role_throughput_weights(
            rollout_signals,
            rm_signals,
            rollout_waiting,
            rm_waiting,
        )
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role == PSRL_Role.Rollout:
            divisor = w_rm if w_rm > 0.0 else 1.0
            return role_throughputs[PSRL_Role.RewardModel] / divisor
        if starved_role == PSRL_Role.RewardModel:
            divisor = w_rollout if w_rollout > 0.0 else 1.0
            return role_throughputs[PSRL_Role.Rollout] / divisor
        return self._weighted_system_throughput(
            role_throughputs[PSRL_Role.Rollout],
            role_throughputs[PSRL_Role.RewardModel],
            w_rollout,
            w_rm,
        )

    def _degraded_loaded_throughput(
        self,
        loaded_signals: list[InstanceSignal],
        loaded_n: int,
        loaded_waiting: _RouterWaitingLoad,
        w_loaded: float,
    ) -> float:
        """
        Return the balanced batch-drain rate divided by the loaded role weight.
        """
        if loaded_n <= 0:
            return 0.0
        per_instance_tp = self._role_throughput(loaded_signals, loaded_n, loaded_waiting)
        if math.isinf(per_instance_tp) or math.isnan(per_instance_tp):
            return per_instance_tp
        total_drain = per_instance_tp * float(loaded_n)
        divisor = w_loaded if w_loaded > 0.0 else 1.0
        return total_drain / divisor

    def _role_request_pressure(
        self,
        role_signals: list[InstanceSignal],
        waiting: _RouterWaitingLoad,
    ) -> float:
        """
        Total outstanding request pressure for a role (in-instance + router).
        """
        in_instance = sum(self._vllm_current_request_load(signal) for signal in role_signals if signal.is_awaken)
        return in_instance + float(waiting.request_count)

    def _role_is_demand_starved(
        self,
        role_signals: list[InstanceSignal],
        waiting: _RouterWaitingLoad,
    ) -> bool:
        """
        Return whether awake capacity has no in-instance demand or router backlog.
        """
        awake_n = sum(1 for signal in role_signals if signal.is_awaken)
        if awake_n <= 0:
            return False
        return self._role_request_pressure(role_signals, waiting) <= 0.0

    def _detect_starved_role(
        self,
        *,
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad,
        rm_waiting: _RouterWaitingLoad,
    ) -> PSRL_Role | None:
        """
        Return the single demand-starved role, or ``None`` if not degraded.

        Only triggers when exactly one role is starved. If both roles are idle
        the system has no work at all (leave the objective as-is). if neither is
        starved the normal harmonic objective applies.
        """
        if not self.starvation_fallback:
            return None
        rollout_starved = self._role_is_demand_starved(rollout_signals, rollout_waiting)
        rm_starved = self._role_is_demand_starved(rm_signals, rm_waiting)
        if rollout_starved and not rm_starved:
            return PSRL_Role.Rollout
        if rm_starved and not rollout_starved:
            return PSRL_Role.RewardModel
        return None

    def _enumerate_bottleneck_roles(
        self,
        *,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        role_throughputs: dict[PSRL_Role, float],
        rollout_signals: list[InstanceSignal],
        rm_signals: list[InstanceSignal],
        rollout_waiting: _RouterWaitingLoad,
        rm_waiting: _RouterWaitingLoad,
        w_rollout: float,
        w_rm: float,
        rollout_bottleneck: float,
        rm_bottleneck: float,
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> list[PSRL_Role]:
        """
        Select the pressure-weighted bottleneck, or the loaded role under starvation.
        """
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role == PSRL_Role.Rollout:
            return [PSRL_Role.RewardModel]
        if starved_role == PSRL_Role.RewardModel:
            return [PSRL_Role.Rollout]
        _ = grouped, rollout_signals, rm_signals, rollout_waiting, rm_waiting, router_backlog_by_role
        if rm_bottleneck < rollout_bottleneck:
            return [PSRL_Role.RewardModel]
        if rm_bottleneck == rollout_bottleneck:
            return [PSRL_Role.Rollout, PSRL_Role.RewardModel]
        return [PSRL_Role.Rollout]

    def _objective_log_extras(
        self,
        grouped: dict[PSRL_Role, list[InstanceSignal]],
        router_backlog_by_role: dict[PSRL_Role, Any] | None,
    ) -> dict[str, Any]:
        """
        Surface starvation-fallback state in the ``itl_objective`` log section.
        """
        backlog_map = router_backlog_by_role or {}
        rollout_signals = grouped.get(PSRL_Role.Rollout, [])
        rm_signals = grouped.get(PSRL_Role.RewardModel, [])
        rollout_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.Rollout))
        rm_waiting = self._normalize_router_waiting_load(backlog_map.get(PSRL_Role.RewardModel))
        starved_role = self._detect_starved_role(
            rollout_signals=rollout_signals,
            rm_signals=rm_signals,
            rollout_waiting=rollout_waiting,
            rm_waiting=rm_waiting,
        )
        if starved_role is None:
            return {"starvation_fallback": self.starvation_fallback, "degraded_mode": False}
        rollout_pressure = self._role_request_pressure(rollout_signals, rollout_waiting)
        rm_pressure = self._role_request_pressure(rm_signals, rm_waiting)
        return {
            "starvation_fallback": self.starvation_fallback,
            "degraded_mode": True,
            "starved_role": starved_role.name,
            "rollout_pressure": f"{rollout_pressure:.1f}",
            "rm_pressure": f"{rm_pressure:.1f}",
        }
