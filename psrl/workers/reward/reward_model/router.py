import asyncio
import heapq
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import deque
from math import ceil
from typing import Any

import numpy as np
import ray
from omegaconf import DictConfig
from verl import DataProto

from psrl.utils.cost_model_path import require_cost_model_json_path
from psrl.utils.elastic.cost_model import (
    ITLModelParams,
    compute_itl,
    resolve_cost_model_params,
)
from psrl.utils.elastic.diagnostics import log_elastic_rm_backlog_diag
from psrl.utils.elastic.overhead import RequestMigrationOverheadTracker
from psrl.utils.logger import DualOutputHandler
from psrl.utils.rollout.request_id import canonical_psrl_request_id
from psrl.workers.elastic.candidate_routing import select_itl_candidate
from psrl.workers.gen.stats_collector import EngineStats

psrl_logger = logging.getLogger("reward_model_router")
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------
_RM_ROUTE_STRATEGY_REGISTRY: dict[str, type["RewardModelRouteStrategyBase"]] = {}


def register_rm_route_strategy(name: str):
    """Register a reward-model route strategy class."""

    def decorator(cls: type["RewardModelRouteStrategyBase"]):
        if name in _RM_ROUTE_STRATEGY_REGISTRY:
            raise ValueError(f"Reward-model route strategy {name!r} is already registered.")
        _RM_ROUTE_STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def get_rm_route_strategy_class(name: str) -> type["RewardModelRouteStrategyBase"]:
    if name not in _RM_ROUTE_STRATEGY_REGISTRY:
        raise ValueError(
            f"Reward-model route strategy {name!r} is not registered. "
            f"Available strategies: {list(_RM_ROUTE_STRATEGY_REGISTRY.keys())}."
        )
    return _RM_ROUTE_STRATEGY_REGISTRY[name]


def list_available_rm_route_strategies() -> list[str]:
    return list(_RM_ROUTE_STRATEGY_REGISTRY.keys())


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def _request_token_num(request: DataProto) -> int:
    """Best-effort token count for a queued single reward request (module-level)."""
    batch = getattr(request, "batch", None)
    if batch is None:
        batch = {}
    attention_mask = batch.get("attention_mask", None)
    if attention_mask is not None:
        if hasattr(attention_mask, "sum"):
            return max(0, int(attention_mask.sum().item()))
        return max(0, int(np.asarray(attention_mask).sum()))
    input_ids = batch.get("input_ids", None)
    if input_ids is not None:
        shape = getattr(input_ids, "shape", None)
        if shape is not None and len(shape) > 0:
            return max(0, int(shape[-1]))
        return max(0, len(input_ids))
    return 0


# ---------------------------------------------------------------------------
# Strategy base + implementations
# ---------------------------------------------------------------------------
class RewardModelRouteStrategyBase(ABC):
    """Base class for reward-model routing strategies.

    Reward-model routing mirrors rollout routing but does not group or order
    candidates by model version. Callers pass only the currently available
    replica candidates and optional load hints.
    """

    def __init__(self, n_instances: int, strategy_kwargs: dict | None = None):
        self.n_instances = int(n_instances)
        self.strategy_kwargs = strategy_kwargs or {}
        self.logger = self.strategy_kwargs.get("logger", logging.getLogger(__file__))
        self.logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))
        self.instance_to_engine_status = {
            i: EngineStats(
                instance_id=i,
                model_version=0,
                snapshot=EngineStats.get_default_snapshot(),
            )
            for i in range(self.n_instances)
        }

    @abstractmethod
    def route(
        self,
        request: DataProto,
        candidates: list[int] | None = None,
        route_kwargs: dict | None = None,
    ) -> int | None:
        pass

    def update_instance_to_engine_status(self, instance_to_engine_status: dict[int, EngineStats]) -> None:
        for instance_id, engine_status in instance_to_engine_status.items():
            self.instance_to_engine_status[int(instance_id)] = engine_status

    def update_instance_loads(self, instance_to_load: dict[int, int]) -> None:
        """Update local load counters from router-side probes."""

    def push_request(self, request: DataProto, instance_id: int) -> None:
        """Record a routed request."""

    def pop_request(self, request: DataProto, instance_id: int) -> None:
        """Record a finished request."""

    def force_route_unchecked(self, request: DataProto, instance_id: int) -> int:
        """Record a simulation-selected route without applying admission rules."""
        return int(instance_id)

    def calculate_routing_benefit(self, request: DataProto, instance_id: int) -> float:
        return 1.0


@register_rm_route_strategy("random")
class RandomRewardModelRouteStrategy(RewardModelRouteStrategyBase):
    """Randomly select one available reward-model replica."""

    def route(
        self,
        request: DataProto,
        candidates: list[int] | None = None,
        route_kwargs: dict | None = None,
    ) -> int | None:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if not candidates:
            return None
        return int(np.random.choice(candidates))


@register_rm_route_strategy("round_robin")
class RoundRobinRewardModelRouteStrategy(RewardModelRouteStrategyBase):
    """Route requests in round-robin order across available replicas."""

    def __init__(self, n_instances: int, strategy_kwargs: dict | None = None):
        super().__init__(n_instances, strategy_kwargs)
        self.curr_idx = 0

    def route(
        self,
        request: DataProto,
        candidates: list[int] | None = None,
        route_kwargs: dict | None = None,
    ) -> int | None:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if not candidates:
            return None
        idx = int(candidates[self.curr_idx % len(candidates)])
        self.curr_idx = (self.curr_idx + 1) % len(candidates)
        return idx


@register_rm_route_strategy("request_num_balance")
class RequestNumBalanceRewardModelRouteStrategy(RewardModelRouteStrategyBase):
    """Route to the replica with the fewest active requests."""

    def __init__(self, n_instances: int, strategy_kwargs: dict | None = None):
        super().__init__(n_instances, strategy_kwargs)
        self.max_concurrent_seqs_per_instance = int(
            self.strategy_kwargs.get("max_concurrent_seqs_per_instance", 2**31 - 1)
        )
        self.instance_request_counts = {i: 0 for i in range(self.n_instances)}

    def route(
        self,
        request: DataProto,
        candidates: list[int] | None = None,
        route_kwargs: dict | None = None,
    ) -> int | None:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if not candidates:
            return None
        active_loads = (route_kwargs or {}).get("active_loads", {})
        for instance_id, load in active_loads.items():
            self.instance_request_counts[int(instance_id)] = max(
                int(load),
                self.instance_request_counts.get(int(instance_id), 0),
            )
        selected = min(candidates, key=lambda idx: (self.instance_request_counts[int(idx)], int(idx)))
        if self.instance_request_counts[int(selected)] >= self.max_concurrent_seqs_per_instance:
            return None
        self.instance_request_counts[int(selected)] += 1
        return int(selected)

    def force_route_unchecked(self, request: DataProto, instance_id: int) -> int:
        instance_id = int(instance_id)
        self.instance_request_counts[instance_id] += 1
        return instance_id

    def update_instance_loads(self, instance_to_load: dict[int, int]) -> None:
        for instance_id, load in instance_to_load.items():
            self.instance_request_counts[int(instance_id)] = max(0, int(load))

    def pop_request(self, request: DataProto, instance_id: int) -> None:
        instance_id = int(instance_id)
        self.instance_request_counts[instance_id] = max(0, self.instance_request_counts[instance_id] - 1)

    def calculate_routing_benefit(self, request: DataProto, instance_id: int) -> float:
        if self.instance_request_counts[int(instance_id)] >= self.max_concurrent_seqs_per_instance:
            return 0.0
        return 1.0


@register_rm_route_strategy("itl")
class ITLBalanceRewardModelRouteStrategy(RewardModelRouteStrategyBase):
    """Route to the replica with the smallest predicted ITL after admission.

    Mirrors the legacy base-router ITL fast path: estimate
    ``compute_itl(params, total_token_num=0, running_queue_num=load + 1)`` for
    each candidate and pick the minimum, optionally filtered by an ITL cap.
    """

    def __init__(self, n_instances: int, strategy_kwargs: dict | None = None):
        super().__init__(n_instances, strategy_kwargs)
        self._itl_params = self.strategy_kwargs.get("itl_model_params", None)
        self._itl_max_itl = self.strategy_kwargs.get("itl_max_itl", None)

    def route(
        self,
        request: DataProto,
        candidates: list[int] | None = None,
        route_kwargs: dict | None = None,
    ) -> int | None:
        active_loads = (route_kwargs or {}).get("active_loads", {})
        if candidates is None:
            candidates = list(active_loads.keys())
        if not candidates:
            return None

        def _next_itl(instance_id: int, load: int) -> float:
            if self._itl_params is None:
                return float(load + 1)
            return compute_itl(
                self._itl_params,
                total_token_num=0.0,
                running_queue_num=load + 1,
            )

        best_idx = select_itl_candidate(
            candidates=candidates,
            active_loads=active_loads,
            next_itl=_next_itl,
            max_itl=self._itl_max_itl,
        )
        if best_idx is None:
            self.logger.warning(
                "[router] No reward workers under ITL threshold %s; active_loads=%s.",
                self._itl_max_itl,
                active_loads,
            )
            return None
        return best_idx


class CostModelBasedRewardModelRouteStrategy(RewardModelRouteStrategyBase):
    """Cost-model based reward-model route strategy."""

    def __init__(self, n_instances: int, strategy_kwargs: dict | None = None):
        super().__init__(n_instances, strategy_kwargs)
        required_keys = (
            "cost_model_path",
            "instance_to_tp_pp",
            "max_num_waiting_reqs_after_preemption",
            "max_concurrent_seqs_per_instance",
            "delta_throughput_threshold",
            "max_prompt_length",
            "request_budget",
            "instance_to_max_model_len",
        )
        missing = [key for key in required_keys if key not in self.strategy_kwargs]
        if missing:
            raise ValueError(f"Missing reward-model routing strategy kwargs: {missing}.")

        cost_model_path = self.strategy_kwargs["cost_model_path"]
        model_name = self.strategy_kwargs.get("model_name", "")
        resolved_cost_model_path = require_cost_model_json_path(cost_model_path, model_name)
        with open(resolved_cost_model_path, encoding="utf-8") as f:
            self.cost_model = json.load(f)

        self.instance_to_tp_pp = self.strategy_kwargs["instance_to_tp_pp"]
        for tp_pp in self.instance_to_tp_pp.values():
            if tp_pp not in self.cost_model:
                raise ValueError(f"tp_pp {tp_pp!r} is not in cost model.")

        self.max_num_waiting_reqs_after_preemption = int(
            self.strategy_kwargs["max_num_waiting_reqs_after_preemption"]
        )
        self.max_concurrent_seqs_per_instance = int(self.strategy_kwargs["max_concurrent_seqs_per_instance"])
        self.delta_throughput_threshold = float(self.strategy_kwargs["delta_throughput_threshold"])
        self.logging_interval_s = float(self.strategy_kwargs.get("logging_interval_in_ms", 1000)) / 1000.0
        self._last_route_reject_log_ts = 0.0
        self.max_prompt_length = int(self.strategy_kwargs["max_prompt_length"])
        self.request_budget = int(self.strategy_kwargs["request_budget"])
        self.instance_to_max_model_len = {
            int(instance_id): int(max_model_len)
            for instance_id, max_model_len in self.strategy_kwargs["instance_to_max_model_len"].items()
        }
        self.instance_to_request_num = {i: 0 for i in range(self.n_instances)}
        self.instance_to_running_request_num = {i: 0 for i in range(self.n_instances)}
        self.instance_to_waiting_request_num = {i: 0 for i in range(self.n_instances)}
        self.instance_to_token_num = {i: 0 for i in range(self.n_instances)}
        # Cost-model strategies need engine-status snapshots (token / waiting
        # queue) to estimate throughput correctly. Until the coordinator pushes
        # real snapshots via ``update_instance_to_engine_status``, fall back to
        # load-based eager dispatch so requests are never stranded in the router
        # queue (see ``_route_blind``).
        self._has_engine_status = False

    def _get_request_token_num(self, request: DataProto, log_len: bool = False) -> int:
        non_tensor_batch = getattr(request, "non_tensor_batch", {}) or {}
        raw_prompt_ids = non_tensor_batch.get("raw_prompt_ids")
        if raw_prompt_ids is not None:
            prompt_ids = raw_prompt_ids[0] if len(raw_prompt_ids) > 0 else []
            response_len = non_tensor_batch.get("response_unpadded_len", [0])
            if hasattr(response_len, "tolist"):
                response_len = response_len.tolist()
            if isinstance(response_len, (list, tuple, np.ndarray)):
                response_len = response_len[0] if len(response_len) > 0 else 0
            token_num = len(prompt_ids) + int(response_len)
        else:
            token_num = _request_token_num(request)
        if log_len:
            self.logger.info("Reward request token num is %d.", token_num)
        return max(0, int(token_num))

    def _can_run_directly(self, request: DataProto, instance_id: int) -> bool:
        if self.instance_to_waiting_request_num[int(instance_id)] > 0:
            return False
        new_token_num = self.instance_to_token_num[int(instance_id)] + self._get_request_token_num(request)
        return new_token_num <= self.instance_to_max_model_len[int(instance_id)]

    def _estimate_latency(self, instance_id: int, request_num: int, token_num: int) -> float:
        tp_pp = self.instance_to_tp_pp[int(instance_id)]
        cost_model = self.cost_model[tp_pp]
        return (
            cost_model["attn_latency_b"]
            + cost_model["attn_latency_k"] * token_num
            + max(
                cost_model["other_threshold"],
                cost_model["other_latency_b"] + cost_model["other_latency_k"] * request_num,
            )
        )

    def _estimate_curr_throughput(self, instance_id: int) -> float:
        instance_id = int(instance_id)
        request_num = self.instance_to_running_request_num[instance_id]
        if request_num <= 0:
            return 0.0
        return request_num / self._estimate_latency(
            instance_id,
            request_num,
            self.instance_to_token_num[instance_id],
        )

    def _estimate_curr_throughput_after_route_request(self, request: DataProto, instance_id: int) -> float:
        instance_id = int(instance_id)
        request_num = self.instance_to_running_request_num[instance_id] + 1
        token_num = self.instance_to_token_num[instance_id] + self._get_request_token_num(request)
        return request_num / self._estimate_latency(instance_id, request_num, token_num)

    def _estimate_baseline_delta_throughput(self, request: DataProto, instance_id: int) -> float:
        return 1.0 / self._estimate_latency(int(instance_id), 1, self._get_request_token_num(request))

    def _format_candidate_state(self, candidates: list[int]) -> str:
        parts = []
        for idx in candidates:
            idx = int(idx)
            parts.append(
                (
                    f"{idx}:req={self.instance_to_request_num.get(idx, 0)},"
                    f"run={self.instance_to_running_request_num.get(idx, 0)},"
                    f"wait={self.instance_to_waiting_request_num.get(idx, 0)},"
                    f"tok={self.instance_to_token_num.get(idx, 0)},"
                    f"max={self.instance_to_max_model_len.get(idx, 0)}"
                )
            )
        return "; ".join(parts)

    def _maybe_log_route_reject(self, reason: str, **kwargs: Any) -> None:
        now = time.monotonic()
        if now - self._last_route_reject_log_ts < self.logging_interval_s:
            return
        self._last_route_reject_log_ts = now
        details = ", ".join(f"{key}={value}" for key, value in kwargs.items())
        self.logger.warning("[router][throughput_optimal] route reject: reason=%s, %s", reason, details)

    def obtain_instance_token_num_from_engine_status(self, instance_id: int, engine_status: EngineStats) -> int:
        return ceil(engine_status.get_kv_cache_utilization() * self.instance_to_max_model_len[int(instance_id)])

    def update_instance_to_engine_status(self, instance_to_engine_status: dict[int, EngineStats]) -> None:
        super().update_instance_to_engine_status(instance_to_engine_status)
        if instance_to_engine_status:
            self._has_engine_status = True
        for instance_id, engine_stats in instance_to_engine_status.items():
            instance_id = int(instance_id)
            scheduler_stats = engine_stats.snapshot.get("scheduler_stats", {})
            self.instance_to_request_num[instance_id] = engine_stats.get_waiting_and_running_queue_size()
            self.instance_to_running_request_num[instance_id] = int(scheduler_stats.get("num_running_reqs", 0))
            self.instance_to_waiting_request_num[instance_id] = int(scheduler_stats.get("num_waiting_reqs", 0))
            self.instance_to_token_num[instance_id] = self.obtain_instance_token_num_from_engine_status(
                instance_id,
                engine_stats,
            )

    def update_instance_loads(self, instance_to_load: dict[int, int]) -> None:
        for instance_id, load in instance_to_load.items():
            instance_id = int(instance_id)
            load = max(0, int(load))
            self.instance_to_request_num[instance_id] = load
            self.instance_to_running_request_num[instance_id] = load

    def route(
        self,
        request: DataProto,
        candidates: list[int] | None = None,
        route_kwargs: dict | None = None,
    ) -> int | None:
        raise RuntimeError("CostModelBasedRewardModelRouteStrategy is abstract.")

    def push_request(self, request: DataProto, instance_id: int) -> None:
        instance_id = int(instance_id)
        self.instance_to_request_num[instance_id] += 1
        self.instance_to_running_request_num[instance_id] += 1
        # Only maintain a router-side token estimate when no engine-status
        # snapshots are available. Once the coordinator pushes real snapshots
        # (``_has_engine_status``), ``instance_to_token_num`` is the engine
        # truth refreshed every coordinator sync; accumulating per-dispatch
        # token estimates here would inflate it within a greedy batch and make
        # ``_can_run_directly`` falsely refuse dispatch while the real worker
        # still has kv-cache headroom.
        if not self._has_engine_status:
            self.instance_to_token_num[instance_id] += self._get_request_token_num(request)

    def force_route_unchecked(self, request: DataProto, instance_id: int) -> int:
        instance_id = int(instance_id)
        self.push_request(request, instance_id)
        return instance_id

    def pop_request(self, request: DataProto, instance_id: int) -> None:
        instance_id = int(instance_id)
        self.instance_to_request_num[instance_id] = max(0, self.instance_to_request_num[instance_id] - 1)
        self.instance_to_running_request_num[instance_id] = max(0, self.instance_to_running_request_num[instance_id] - 1)
        if not self._has_engine_status:
            self.instance_to_token_num[instance_id] = max(
                0,
                self.instance_to_token_num[instance_id] - self._get_request_token_num(request),
            )


@register_rm_route_strategy("throughput_optimal")
class ThroughputOptimalRewardModelRouteStrategy(CostModelBasedRewardModelRouteStrategy):
    """Route each reward request to the replica with best throughput gain."""

    def route(
        self,
        request: DataProto,
        candidates: list[int] | None = None,
        route_kwargs: dict | None = None,
    ) -> int | None:
        if candidates is None:
            candidates = list(range(self.n_instances))
        if not candidates:
            return None
        if not self._has_engine_status:
            return self._route_blind(request, candidates, route_kwargs)

        best_candidate = None
        best_delta_throughput = float("-inf")
        request_token_num = self._get_request_token_num(request)
        waiting_blocked = 0
        token_blocked = 0
        for candidate in candidates:
            candidate = int(candidate)
            if self.instance_to_waiting_request_num[candidate] > 0:
                waiting_blocked += 1
                continue
            if self.instance_to_token_num[candidate] + request_token_num > self.instance_to_max_model_len[candidate]:
                token_blocked += 1
                continue
            delta_throughput = (
                self._estimate_curr_throughput_after_route_request(request, candidate)
                - self._estimate_curr_throughput(candidate)
            )
            if delta_throughput > best_delta_throughput:
                best_delta_throughput = delta_throughput
                best_candidate = candidate

        if best_candidate is None:
            self._maybe_log_route_reject(
                "no_direct_candidate",
                request_token_num=request_token_num,
                candidate_count=len(candidates),
                waiting_blocked=waiting_blocked,
                token_blocked=token_blocked,
                candidates=self._format_candidate_state(candidates),
            )
            return None

        threshold = self._estimate_baseline_delta_throughput(request, best_candidate) * self.delta_throughput_threshold
        best_request_num = self.instance_to_request_num[best_candidate]
        if (
            best_delta_throughput >= threshold
            and best_request_num < self.max_concurrent_seqs_per_instance
        ):
            self.push_request(request, best_candidate)
            return best_candidate
        reason = (
            "marginal_throughput_below_threshold"
            if best_delta_throughput < threshold
            else "max_concurrent_seqs_cap"
        )
        self._maybe_log_route_reject(
            reason,
            best_candidate=best_candidate,
            best_delta_throughput=best_delta_throughput,
            threshold=threshold,
            delta_throughput_threshold=self.delta_throughput_threshold,
            request_token_num=request_token_num,
            request_num=best_request_num,
            running=self.instance_to_running_request_num[best_candidate],
            waiting=self.instance_to_waiting_request_num[best_candidate],
            token_num=self.instance_to_token_num[best_candidate],
            new_token_num=self.instance_to_token_num[best_candidate] + request_token_num,
            max_model_len=self.instance_to_max_model_len[best_candidate],
            max_concurrent_seqs_per_instance=self.max_concurrent_seqs_per_instance,
            candidate_count=len(candidates),
            candidates=self._format_candidate_state(candidates),
        )
        return None

    def _route_blind(
        self,
        request: DataProto,
        candidates: list[int],
        route_kwargs: dict | None = None,
    ) -> int | None:
        """Load-based eager dispatch used until engine-status snapshots arrive.

        Without token/waiting-queue snapshots the throughput model is unreliable,
        so we degrade to "pick the least-loaded candidate under the concurrency
        cap" and always dispatch. This keeps the router from stranding requests
        while the coordinator's status broadcast is not yet connected (or while
        ``status_collection.enable`` is False).
        """
        active_loads = (route_kwargs or {}).get("active_loads", {})
        best_idx: int | None = None
        best_key: tuple | None = None
        for idx in candidates:
            idx = int(idx)
            if self.instance_to_request_num[idx] >= self.max_concurrent_seqs_per_instance:
                continue
            load = int(active_loads.get(idx, self.instance_to_request_num[idx]))
            key = (load, idx)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        if best_idx is None:
            return None
        self.push_request(request, best_idx)
        return best_idx

    def calculate_routing_benefit(self, request: DataProto, instance_id: int) -> float:
        instance_id = int(instance_id)
        if not self._can_run_directly(request, instance_id):
            return 0.0
        delta_throughput = (
            self._estimate_curr_throughput_after_route_request(request, instance_id)
            - self._estimate_curr_throughput(instance_id)
        )
        baseline = self._estimate_baseline_delta_throughput(request, instance_id) * self.delta_throughput_threshold
        if delta_throughput < baseline:
            return 0.0
        if self.instance_to_request_num[instance_id] >= self.max_concurrent_seqs_per_instance:
            return 0.0
        return delta_throughput


@register_rm_route_strategy("throughput_optimal_with_budget")
class ThroughputOptimalWithBudgetRewardModelRouteStrategy(ThroughputOptimalRewardModelRouteStrategy):
    """Throughput-optimal reward-model routing with per-request token budget."""

    def _get_request_token_num(self, request: DataProto, log_len: bool = False) -> int:
        non_tensor_batch = getattr(request, "non_tensor_batch", {}) or {}
        raw_prompt_ids = non_tensor_batch.get("raw_prompt_ids")
        if raw_prompt_ids is None:
            return super()._get_request_token_num(request, log_len=log_len)
        prompt_ids = raw_prompt_ids[0] if len(raw_prompt_ids) > 0 else []
        response_len = non_tensor_batch.get("response_unpadded_len", [0])
        if hasattr(response_len, "tolist"):
            response_len = response_len.tolist()
        if isinstance(response_len, (list, tuple, np.ndarray)):
            response_len = response_len[0] if len(response_len) > 0 else 0
        return len(prompt_ids) + ceil((int(response_len) + 1) / self.request_budget) * self.request_budget

    def obtain_instance_token_num_from_engine_status(self, instance_id: int, engine_status: EngineStats) -> int:
        prompt_token_nums = engine_status.get_req_id_to_prompt_token_num().values()
        response_token_nums = engine_status.get_req_id_to_response_token_num().values()
        token_num = 0
        for prompt_token_num, response_token_num in zip(prompt_token_nums, response_token_nums, strict=False):
            token_num += int(prompt_token_num)
            token_num += ceil((int(response_token_num) + 1) / self.request_budget) * self.request_budget
        return token_num


# ---------------------------------------------------------------------------
# Router actor
# ---------------------------------------------------------------------------
@ray.remote(concurrency_groups={"control": 5})
class PSRL_RewardModelRouter:
    """Queue-based reward-model router with pluggable routing strategies.

    The router accepts Ray ActorHandles, queues incoming requests, and forwards
    each to a replica chosen by the configured ``RewardModelRouteStrategyBase``.

    NOTE(zyf): will enable to use http-server instead of ray actor in the future.
    """

    def __init__(
        self,
        worker_handles: list[ray.actor.ActorHandle],
        worker_groups: list[Any] | None,
        config: DictConfig,
        reward_model_config: DictConfig | None = None,
        retry_delay: float = 2.0,
        verbose: bool = False,
        route_strategy_config: dict | DictConfig | None = None,
    ) -> None:
        self.config = config
        self.verbose = verbose
        self.worker_handles = worker_handles
        self.worker_groups = worker_groups
        self.reward_model_config = reward_model_config
        self.rollout_name = str(reward_model_config.rollout.name).lower() if reward_model_config is not None else "vllm"
        self.request_counts = {i: 0 for i in range(len(worker_handles))}
        self.paused_worker_indices: set[int] = set()
        self.retry_delay = retry_delay
        self.request_futures: dict[str, asyncio.Future] = {}
        self.routing_status_update_queue: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self.worker_probe_backoff_until: dict[int, float] = {}
        # Min-heap items: (missing_flag, buffer_id, fifo_seq, request_key, request).
        # missing_flag: 0 if buffer_id present (routed first), 1 if absent (demoted).
        # buffer_id: normalized id for ordering (smaller first); 0 when missing_flag==1.
        # fifo_seq: monotonic tie-break for FIFO among equal priority.
        self.requests_to_route: list[tuple[int, int, int, str, DataProto]] = []
        self.routing_lock = asyncio.Lock()
        self._is_routing = False
        self._interrupt_routing = False
        self.scheduler_task: asyncio.Task | None = None
        self._request_key_counter = 0
        self._waiting_seq_counter = 0
        self._pending_count = 0
        self.instance_to_engine_status: dict[int, EngineStats] = {}

        raw_cap = None
        if self.reward_model_config is not None:
            raw_cap = self.reward_model_config.get("max_concurrent_requests_per_instance", None)
        self.max_concurrent_requests_per_instance: int | None = None
        if raw_cap is not None:
            try:
                cap = int(raw_cap)
                if cap > 0:
                    self.max_concurrent_requests_per_instance = cap
            except (TypeError, ValueError):
                self.max_concurrent_requests_per_instance = None
        self.worker_active_task_timeout_s = self._positive_float_config(
            "router_active_task_timeout_s", 1.0
        )
        self.worker_probe_backoff_s = self._positive_float_config(
            "router_worker_probe_backoff_s", max(1.0, self.worker_active_task_timeout_s)
        )
        self.load_cache_ttl_s = self._positive_float_config("router_load_cache_ttl_s", 0.05)

        elastic_cfg = self._get_elastic_config()
        cost_cfg = elastic_cfg.get("throughput_optimal", {}) if hasattr(elastic_cfg, "get") else {}
        # OmegaConf may keep this nested section as DictConfig.
        if not hasattr(cost_cfg, "get"):
            cost_cfg = {}
        variant = str(elastic_cfg.get("scaling_policy_variant", "throughput-optimal")).lower()
        self._itl_router_enable = bool(cost_cfg.get("rm_router_enable", variant == "throughput-optimal"))
        raw_max_itl = cost_cfg.get("rm_router_max_itl", None)
        self._itl_router_max_itl: float | None = None
        if raw_max_itl is not None:
            try:
                max_itl = float(raw_max_itl)
                if max_itl > 0:
                    self._itl_router_max_itl = max_itl
            except (TypeError, ValueError):
                self._itl_router_max_itl = None
        self._itl_router_params: ITLModelParams | None = resolve_cost_model_params(
            role_name="RewardModel",
            model_name=self._reward_model_name(),
            cost_config=cost_cfg,
            config=self.config,
        )

        # Load cache for active task counts.
        self._load_cache: dict[int, int] = {}
        self._load_cache_ts: float = -1.0

        # Strategy init.
        self.route_strategy_config = route_strategy_config or _cfg_get(
            reward_model_config, "routing_strategy", None
        )
        self.waiting_admission_cap: int | None = None
        raw_waiting_admission_cap = _cfg_get(
            self.route_strategy_config,
            "max_num_waiting_reqs_after_preemption",
            None,
        )
        if raw_waiting_admission_cap is not None:
            try:
                waiting_admission_cap = int(raw_waiting_admission_cap)
                if waiting_admission_cap > 0:
                    self.waiting_admission_cap = waiting_admission_cap
            except (TypeError, ValueError):
                self.waiting_admission_cap = None
        self.route_strategy = self._init_route_strategy()
        self._request_to_strategy_instance: dict[str, int] = {}
        self._migration_overhead = RequestMigrationOverheadTracker()
        self._candidate_evaluation_inflight_requests: dict[str, DataProto] = {}
        self._uid_to_inflight_instance: dict[str, int] = {}
        self._planned_migration_destinations: dict[str, int] = {}
        self._planned_migration_counters = {
            "planned": 0,
            "accepted": 0,
            "skipped": 0,
            "forced": 0,
            "fallback": 0,
        }
        self._exclusive_rebalance_migration_queue_enabled = bool(
            itl_cfg.get("exclusive_rebalance_migration_queue", False)
        )
        self._rebalance_requests_to_route: deque[tuple[str, DataProto]] = deque()
        self._rebalance_pending_request_ids: set[str] = set()
        self._rebalance_queued_request_keys: set[str] = set()
        self._rebalance_dispatching_request_keys: set[str] = set()

        # Logger.
        self.log_prefix = "RewardModelRouter"
        psrl_logger.addHandler(DualOutputHandler(self.config.psrl.logging_path, self.log_prefix))
        psrl_logger.info(
            (
                "RewardModelRouter initialized with %d workers "
                "(max_concurrent_requests_per_instance=%s, strategy=%s, itl_router_enable=%s, "
                "rm_router_max_itl=%s, waiting_admission_cap=%s, load_cache_ttl_s=%.4f)"
            ),
            len(worker_handles),
            self.max_concurrent_requests_per_instance,
            getattr(self.route_strategy, "__class__", type(None)).__name__,
            self._itl_router_enable,
            self._itl_router_max_itl,
            self.waiting_admission_cap,
            self.load_cache_ttl_s,
        )

    # ---- config helpers --------------------------------------------------
    def _positive_float_config(self, key: str, default: float) -> float:
        if self.reward_model_config is None:
            return float(default)
        try:
            value = float(self.reward_model_config.get(key, default))
        except (TypeError, ValueError):
            return float(default)
        return value if value > 0 else float(default)

    def _get_elastic_config(self) -> dict[str, Any]:
        try:
            cfg = self.config.psrl.deployment.elastic
        except AttributeError:
            return {}
        if hasattr(cfg, "items"):
            return dict(cfg.items())
        return cfg if isinstance(cfg, dict) else {}

    def _reward_model_name(self) -> str:
        if self.reward_model_config is None:
            return "reward_model"
        return str(
            self.reward_model_config.get(
                "reward_model_name",
                self.reward_model_config.get("model", {}).get("path", "reward_model"),
            )
        )

    # ---- strategy init ---------------------------------------------------
    def _init_route_strategy(self) -> RewardModelRouteStrategyBase:
        method = _cfg_get(self.route_strategy_config, "method", None)
        if method is None:
            method = "itl" if self._itl_router_enable else "request_num_balance"
        method = str(method).lower()
        n_instances = len(self.worker_handles)
        strategy_kwargs = self._build_strategy_kwargs(n_instances)
        try:
            strategy_cls = get_rm_route_strategy_class(method)
            strategy = strategy_cls(n_instances, strategy_kwargs)
            psrl_logger.info("Initialized reward-model route strategy: %s.", method)
            return strategy
        except Exception as exc:
            psrl_logger.warning("Reward-model route strategy error: %s.", exc)
            psrl_logger.warning("Falling back to reward-model round_robin strategy.")
            return RoundRobinRewardModelRouteStrategy(n_instances, strategy_kwargs)

    def _build_strategy_kwargs(self, n_instances: int) -> dict[str, Any]:
        rollout_config = _cfg_get(self.reward_model_config, "rollout", {})
        model_config = _cfg_get(self.reward_model_config, "model", {})
        model_path = str(_cfg_get(model_config, "path", self._reward_model_name()))
        model_name = str(_cfg_get(self.reward_model_config, "reward_model_name", os.path.basename(model_path)))
        tp = int(_cfg_get(rollout_config, "tensor_model_parallel_size", 1) or 1)
        pp = int(_cfg_get(rollout_config, "pipeline_model_parallel_size", 1) or 1)
        max_model_len = int(
            _cfg_get(
                rollout_config,
                "max_model_len",
                _cfg_get(self.config.data, "max_prompt_length", 4096)
                + _cfg_get(self.route_strategy_config, "request_budget", 1024),
            )
            or 4096
        )
        max_concurrency = self.max_concurrent_requests_per_instance
        if max_concurrency is None:
            max_concurrency = int(_cfg_get(self.route_strategy_config, "max_concurrent_seqs_per_instance", 2**31 - 1))
        return {
            "logging_interval_in_ms": _cfg_get(self.route_strategy_config, "logging_interval_in_ms", 1000),
            "cost_model_path": _cfg_get(self.route_strategy_config, "cost_model_path", ""),
            "model_name": model_name,
            "instance_to_tp_pp": {i: f"TP{tp}_PP{pp}" for i in range(n_instances)},
            "max_num_waiting_reqs_after_preemption": _cfg_get(
                self.route_strategy_config,
                "max_num_waiting_reqs_after_preemption",
                0,
            ),
            "balanced_concurrent_seqs_per_instance": max_concurrency,
            "max_concurrent_seqs_per_instance": max_concurrency,
            "delta_throughput_threshold": _cfg_get(self.route_strategy_config, "delta_throughput_threshold", 0.0),
            "max_prompt_length": _cfg_get(self.config.data, "max_prompt_length", 4096),
            "request_budget": _cfg_get(self.route_strategy_config, "request_budget", 1024),
            "snapshot_staleness_threshold_in_ms": _cfg_get(
                self.route_strategy_config,
                "snapshot_staleness_threshold_in_ms",
                100,
            ),
            "instance_to_max_model_len": {i: max_model_len for i in range(n_instances)},
            "itl_model_params": self._itl_router_params,
            "itl_max_itl": self._itl_router_max_itl,
            "logger": psrl_logger,
        }

    # ---- public entry points --------------------------------------------
    async def generate(self, request: DataProto) -> DataProto | None:
        """Route one request through the queue-based scheduler."""
        return await self.generate_async(request)

    async def generate_async(self, request: DataProto) -> DataProto | None:
        """Queue a request and wait for the routing result."""
        if self.scheduler_task is None:
            self.scheduler_task = asyncio.create_task(self._routing_loop())
            self.scheduler_task.add_done_callback(lambda task: task.result())
            psrl_logger.info("[router] Started reward routing loop")

        request_key = self._build_request_key(request)
        result_future = asyncio.get_running_loop().create_future()
        self.request_futures[request_key] = result_future
        self._enqueue_request(request_key, request)
        try:
            return await result_future
        finally:
            self.request_futures.pop(request_key, None)

    # ---- routing loop ----------------------------------------------------
    async def _routing_loop(self):
        """Continuously route queued requests using a request-driven loop.

        Per tick: peek the head request, obtain a (cached) load snapshot, then
        greedily dispatch as many queued requests as the snapshot allows before
        re-probing. Backoff only when no candidate can accept the head request.
        """
        while True:
            if self._interrupt_routing:
                self._is_routing = False
                await self._wait_for_routing_update()
                continue

            if self._exclusive_rebalance_active():
                dispatched_any = await self._dispatch_exclusive_rebalance_requests()
                if not dispatched_any:
                    self._is_routing = False
                    await self._wait_for_routing_update(timeout_s=self.worker_probe_backoff_s)
                else:
                    await asyncio.sleep(0)
                continue

            if not self.requests_to_route:
                self._is_routing = False
                await self._wait_for_routing_update()
                continue

            active_loads = await self._get_cached_active_loads()
            if not active_loads:
                self._is_routing = False
                await self._wait_for_routing_update(timeout_s=self.worker_probe_backoff_s)
                continue

            # Refresh strategy counters once per probe snapshot.
            self.route_strategy.update_instance_loads(active_loads)

            dispatched_any = False
            while (
                self.requests_to_route
                and not self._interrupt_routing
                and not self._exclusive_rebalance_active()
            ):
                peeked_key, peeked_request = self._peek_request()
                worker_idx = self._select_worker_for_request(peeked_request, active_loads)
                if worker_idx is None:
                    break
                # Commit dequeue only after a successful selection.
                dequeued = self._dequeue_request()
                if dequeued is None:
                    self._release_strategy_worker(peeked_key, peeked_request, worker_idx)
                    break
                request_key, request = dequeued
                self._request_to_strategy_instance[request_key] = int(worker_idx)
                self._candidate_evaluation_inflight_requests[request_key] = request
                for request_uid in self._request_uid_values(request):
                    self._uid_to_inflight_instance[request_uid] = int(worker_idx)
                self._is_routing = True
                inflight_at_dispatch = active_loads.get(int(worker_idx), 0)
                task = asyncio.create_task(
                    self._route_single_request(request_key, request, worker_idx, inflight_at_dispatch)
                )
                task.add_done_callback(lambda t: t.result())
                # Optimistic reservation bump so the next greedy pick's cap
                # filter and strategy see the just-dispatched in-flight count.
                active_loads[int(worker_idx)] = inflight_at_dispatch + 1
                dispatched_any = True
                await asyncio.sleep(0)

            if not dispatched_any:
                self._is_routing = False
                await self._wait_for_routing_update(timeout_s=self.worker_probe_backoff_s)
                continue

    async def _route_single_request(
        self,
        request_key: str,
        request: DataProto,
        worker_idx: int,
        inflight_at_dispatch: int,
    ):
        """Route one request to a selected worker; requeue when needed."""
        if request_key not in self.request_futures:
            self._release_strategy_worker(request_key, request, worker_idx)
            for request_uid in self._request_uid_values(request):
                self._migration_overhead.discard(request_uid)
                self._planned_migration_destinations.pop(request_uid, None)
                self._finish_exclusive_rebalance_request(request_uid, reason="request_missing")
            getattr(self, "_rebalance_dispatching_request_keys", set()).discard(request_key)
            return

        request_uids = self._format_request_uids(request)
        if self._interrupt_routing:
            self._release_strategy_worker(request_key, request, worker_idx)
            getattr(self, "_rebalance_dispatching_request_keys", set()).discard(request_key)
            request_uid_values = self._request_uid_values(request)
            exclusive_pending_uids = {
                canonical_psrl_request_id(request_uid)
                for request_uid in request_uid_values
                if canonical_psrl_request_id(request_uid)
                in getattr(self, "_rebalance_pending_request_ids", set())
            }
            exclusive_enqueued = self._enqueue_exclusive_rebalance_request(request_key, request)
            if not exclusive_enqueued:
                for request_uid in request_uid_values:
                    if canonical_psrl_request_id(request_uid) in exclusive_pending_uids:
                        self._migration_overhead.discard(request_uid)
                self._enqueue_request(request_key, request)
            psrl_logger.debug(
                "[router] Routing is interrupted, Request %s, requeueing original request.",
                request_uids,
            )
            return

        worker_handle = self.worker_handles[worker_idx]
        request_uid_values = self._request_uid_values(request)
        is_exclusive_redispatch = request_key in getattr(self, "_rebalance_dispatching_request_keys", set())
        for request_uid in request_uid_values:
            dispatch_overhead = self._migration_overhead.mark_dispatched(request_uid, worker_idx)
            if dispatch_overhead is not None:
                if is_exclusive_redispatch:
                    self._log_migration_interrupt(request_uid, dispatch_overhead)
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=request_migration_dispatch "
                    "scope=post_scale_up_rebalance role=RewardModel migration_id=%s decision_id=%s "
                    "request_id=%s source_instance=%s destination_instance=%s selected_count=%s "
                    "interrupt_s=%.6f network_s=%.6f abort_to_redispatch_s=%.6f "
                    "interrupt_scope=abort_to_requeue network_scope=requeue_to_redispatch",
                    dispatch_overhead.get("migration_id"),
                    dispatch_overhead.get("decision_id"),
                    request_uid,
                    dispatch_overhead.get("source_instance_id"),
                    dispatch_overhead.get("destination_instance_id"),
                    dispatch_overhead.get("selected_count"),
                    dispatch_overhead["interrupt_s"],
                    dispatch_overhead["network_s"],
                    dispatch_overhead["abort_to_redispatch_s"],
                )
        if request_key in getattr(self, "_rebalance_dispatching_request_keys", set()):
            for request_uid in request_uid_values:
                self._finish_exclusive_rebalance_request(request_uid, reason="redispatched")
            self._rebalance_dispatching_request_keys.discard(request_key)
        try:
            result = await self._generate_with_worker(worker_idx, worker_handle, request)
        finally:
            self._release_strategy_worker(request_key, request, worker_idx)

        if result is None:
            psrl_logger.debug("Request %s interrupted or unavailable, requeueing original request.", request_uids)
            exclusive_pending_uids = {
                canonical_psrl_request_id(request_uid)
                for request_uid in request_uid_values
                if canonical_psrl_request_id(request_uid)
                in getattr(self, "_rebalance_pending_request_ids", set())
            }
            exclusive_enqueued = self._enqueue_exclusive_rebalance_request(request_key, request)
            for request_uid in request_uid_values:
                if exclusive_enqueued:
                    self._mark_migration_requeued(request_uid, defer_log=True)
                elif canonical_psrl_request_id(request_uid) in exclusive_pending_uids:
                    self._migration_overhead.discard(request_uid)
                else:
                    self._mark_migration_requeued(request_uid)
            if not exclusive_enqueued:
                self._enqueue_request(request_key, request)
            await asyncio.sleep(self.retry_delay)
            return

        interrupted = False
        try:
            interrupted = bool(result.non_tensor_batch.get("interrupted", [False])[0])
        except Exception:
            interrupted = False
        if interrupted:
            psrl_logger.debug("Request %s interrupted, requeueing partial output for continuation.", request_uids)
            exclusive_pending_uids = {
                canonical_psrl_request_id(request_uid)
                for request_uid in request_uid_values
                if canonical_psrl_request_id(request_uid)
                in getattr(self, "_rebalance_pending_request_ids", set())
            }
            exclusive_enqueued = self._enqueue_exclusive_rebalance_request(request_key, result)
            for request_uid in request_uid_values:
                if exclusive_enqueued:
                    self._mark_migration_requeued(request_uid, defer_log=True)
                elif canonical_psrl_request_id(request_uid) in exclusive_pending_uids:
                    self._migration_overhead.discard(request_uid)
                else:
                    self._mark_migration_requeued(request_uid)
            if not exclusive_enqueued:
                self._enqueue_request(request_key, result)
            await asyncio.sleep(self.retry_delay)
            return

        for request_uid in request_uid_values:
            migration_overhead, completion_status = self._migration_overhead.complete_with_status(
                request_uid,
                result,
            )
            if completion_status == "completed" and migration_overhead is not None:
                psrl_logger.info(
                    "[ELASTIC_OVERHEAD] operation=post_scale_up_rebalance "
                    "scope=post_scale_up_rebalance role=RewardModel migration_id=%s decision_id=%s "
                    "request_id=%s source_instance=%s destination_instance=%s selected_count=%s "
                    "interrupt_s=%.6f network_s=%.6f abort_to_redispatch_s=%.6f "
                    "reprefill_s=%.6f migration_s=%.6f "
                    "interrupt_scope=abort_to_requeue network_scope=requeue_to_redispatch "
                    "reprefill_scope=vllm_scheduled_to_first_token",
                    migration_overhead.get("migration_id"),
                    migration_overhead.get("decision_id"),
                    request_uid,
                    migration_overhead.get("source_instance_id"),
                    migration_overhead.get("destination_instance_id"),
                    migration_overhead.get("selected_count"),
                    migration_overhead["interrupt_s"],
                    migration_overhead["network_s"],
                    migration_overhead["abort_to_redispatch_s"],
                    migration_overhead["reprefill_s"],
                    migration_overhead["migration_s"],
                )
            elif completion_status == "missing_vllm_prefill" and migration_overhead is not None:
                psrl_logger.warning(
                    "RM migration completion has no vLLM prefill metric: "
                    "request=%s migration_id=%s network_s=%.6f; dropping terminal tracker state.",
                    request_uid,
                    migration_overhead.get("migration_id"),
                    migration_overhead["network_s"],
                )
                self._migration_overhead.discard(request_uid)

        if psrl_logger.isEnabledFor(logging.DEBUG):
            psrl_logger.debug("[router] Reward request %s finished on worker %d", request_uids, worker_idx)
        for request_uid in request_uid_values:
            self._planned_migration_destinations.pop(request_uid, None)
            self._finish_exclusive_rebalance_request(request_uid, reason="terminal")
        self._set_result(request_key, result)
        return

    async def _generate_with_worker(self, worker_idx: int, worker_handle, request: DataProto) -> DataProto | None:
        if self.rollout_name in ("transformers", "hf"):
            if self.worker_groups is None:
                raise RuntimeError("Transformers reward routing requires worker_groups for all-rank execution.")
            results = await asyncio.gather(*self.worker_groups[worker_idx].execute_all_async("generate_async", request))
            return self._select_rank_zero_result(results)
        return await worker_handle.generate_async.remote(request)

    @staticmethod
    def _select_rank_zero_result(results) -> DataProto | None:
        for result in results:
            if result is not None:
                return result
        return None

    # ---- worker selection + load cache ----------------------------------
    async def _get_worker_active_task_num(self, worker_idx: int) -> int:
        """Fetch the reward worker's active task number (one Ray remote)."""
        worker_handle = self.worker_handles[int(worker_idx)]
        return int(
            await asyncio.wait_for(
                worker_handle.get_active_task_num.remote(),
                timeout=self.worker_active_task_timeout_s,
            )
        )

    def _available_indices(self) -> list[int]:
        now = time.monotonic()
        return [
            idx
            for idx in self.request_counts
            if idx not in self.paused_worker_indices
            and self.worker_probe_backoff_until.get(idx, 0.0) <= now
        ]

    async def _get_cached_active_loads(self) -> dict[int, int]:
        """Return active loads for available workers, using a short-TTL cache.

        Cached values are reconciled with current router reservations
        (``request_counts``) so the cap filter and strategies always observe
        in-flight dispatches even when the probe snapshot is slightly stale.
        """
        available_indices = self._available_indices()
        if not available_indices:
            return {}

        now = time.monotonic()
        cache_fresh = (now - self._load_cache_ts) < self.load_cache_ttl_s
        if (
            cache_fresh
            and self._load_cache
            and all(idx in self._load_cache for idx in available_indices)
        ):
            return {
                idx: max(0, self._load_cache[idx], self.request_counts.get(idx, 0))
                for idx in available_indices
            }

        loads = await self._get_available_active_loads(available_indices)
        if loads:
            self._load_cache = dict(loads)
            self._load_cache_ts = now
        return loads

    async def _get_available_active_loads(self, available_indices: list[int]) -> dict[int, int]:
        load_results = await asyncio.gather(
            *(self._get_worker_active_task_num(idx) for idx in available_indices),
            return_exceptions=True,
        )
        active_loads: dict[int, int] = {}
        for idx, result in zip(available_indices, load_results, strict=True):
            if isinstance(result, Exception):
                self.worker_probe_backoff_until[idx] = time.monotonic() + self.worker_probe_backoff_s
                psrl_logger.warning(
                    "[router] Excluding worker %d because active task count fetch failed: %s; backoff=%.3fs",
                    idx,
                    result,
                    self.worker_probe_backoff_s,
                )
                continue
            active_loads[idx] = max(0, int(result), self.request_counts.get(idx, 0))
        return active_loads

    def _select_worker_for_request(
        self, request: DataProto, active_loads: dict[int, int]
    ) -> int | None:
        """Select a worker for a specific request using the active strategy."""
        now = time.monotonic()
        candidates_loads = {
            idx: load
            for idx, load in active_loads.items()
            if idx not in self.paused_worker_indices
            and self.worker_probe_backoff_until.get(idx, 0.0) <= now
        }
        if self.waiting_admission_cap is not None:
            candidates_loads = {
                idx: load
                for idx, load in candidates_loads.items()
                if self._can_admit_without_exceeding_waiting_cap(idx, load)
            }
        if not candidates_loads:
            return None
        if self.max_concurrent_requests_per_instance is not None:
            candidates_loads = {
                idx: load
                for idx, load in candidates_loads.items()
                if load < self.max_concurrent_requests_per_instance
            }
        if not candidates_loads:
            return None

        request_uids = self._request_uid_values(request)
        planned_destinations = {
            self._planned_migration_destinations[request_uid]
            for request_uid in request_uids
            if request_uid in self._planned_migration_destinations
        }
        if planned_destinations:
            destination = next(iter(planned_destinations)) if len(planned_destinations) == 1 else None
            for request_uid in request_uids:
                self._planned_migration_destinations.pop(request_uid, None)
            if destination is not None and destination in candidates_loads:
                self._record_forced_route(request, destination)
                self._planned_migration_counters["forced"] += len(request_uids)
                psrl_logger.info(
                    "Forced planned RM migration: request=%s destination=%s",
                    self._format_request_uids(request),
                    destination,
                )
                return int(destination)
            self._planned_migration_counters["fallback"] += len(request_uids)
            psrl_logger.info(
                "Planned RM migration target invalid; request=%s destinations=%s candidates=%s. "
                "Falling back immediately.",
                self._format_request_uids(request),
                sorted(planned_destinations),
                sorted(candidates_loads),
            )

        worker_idx = self.route_strategy.route(
            request,
            candidates=list(candidates_loads.keys()),
            route_kwargs={"active_loads": candidates_loads},
        )
        if worker_idx is None:
            return None
        worker_idx = int(worker_idx)
        self.request_counts[worker_idx] += 1
        return worker_idx

    def _can_admit_without_exceeding_waiting_cap(self, instance_id: int, active_load: int) -> bool:
        """Keep newly dispatched work within the latest running plus waiting budget."""
        if self.waiting_admission_cap is None:
            return True

        engine_status = self.instance_to_engine_status.get(int(instance_id))
        if engine_status is None:
            running = 0
            waiting = 0
        else:
            scheduler_stats = engine_status.snapshot.get("scheduler_stats", {})
            running = max(0, int(scheduler_stats.get("num_running_reqs", 0)))
            waiting = max(0, int(scheduler_stats.get("num_waiting_reqs", 0)))

        if waiting >= self.waiting_admission_cap:
            return False
        return int(active_load) < running + self.waiting_admission_cap

    # ---- release / result ------------------------------------------------
    def _release_worker(self, worker_idx: int) -> None:
        self.request_counts[worker_idx] = max(0, self.request_counts[worker_idx] - 1)
        self._invalidate_load_cache()
        self._signal_routing_update()

    def _release_strategy_worker(self, request_key: str, request: DataProto, worker_idx: int) -> None:
        worker_idx = int(worker_idx)
        self.route_strategy.pop_request(request, worker_idx)
        self._request_to_strategy_instance.pop(request_key, None)
        self._candidate_evaluation_inflight_requests.pop(request_key, None)
        for request_uid in self._request_uid_values(request):
            self._uid_to_inflight_instance.pop(request_uid, None)
        self._release_worker(worker_idx)

    def _set_result(self, request_key: str, result: DataProto | None):
        request_future = self.request_futures.get(request_key, None)
        if request_future is None or request_future.done():
            return
        request_future.set_result(result)

    def _invalidate_load_cache(self) -> None:
        self._load_cache = {}
        self._load_cache_ts = -1.0

    # ---- queue helpers ---------------------------------------------------
    def _build_request_key(self, request: DataProto) -> str:
        uid_repr = self._format_request_uids(request)
        request_key = f"{uid_repr}#{self._request_key_counter}"
        self._request_key_counter += 1
        return request_key

    def _enqueue_request(self, request_key: str, request: DataProto) -> None:
        buffer_id = self._extract_buffer_id(request)
        missing_flag = 1 if buffer_id is None else 0
        normalized_buffer_id = buffer_id if buffer_id is not None else 0
        item = (
            missing_flag,
            normalized_buffer_id,
            self._waiting_seq_counter,
            request_key,
            request,
        )
        self._waiting_seq_counter += 1
        heapq.heappush(self.requests_to_route, item)
        self._pending_count += 1
        self._signal_routing_update()

    def _exclusive_rebalance_active(self) -> bool:
        return bool(
            getattr(self, "_exclusive_rebalance_migration_queue_enabled", False)
            and getattr(self, "_rebalance_pending_request_ids", set())
        )

    def _start_exclusive_rebalance(self, instance_to_uids: dict, migration_context: dict) -> None:
        if not getattr(self, "_exclusive_rebalance_migration_queue_enabled", False):
            return
        request_ids = {
            canonical_psrl_request_id(uid)
            for uids in (instance_to_uids or {}).values()
            for uid in (uids if isinstance(uids, (list, tuple, set)) else [uids])
            if uid is not None
        }
        # Legacy ratio-based rebalance has no simulated destination. Keep its
        # existing routing behavior even when the exclusive queue is enabled.
        request_ids.intersection_update(self._planned_migration_destinations)
        if not request_ids:
            return
        self._rebalance_pending_request_ids.update(request_ids)
        psrl_logger.info(
            "Exclusive RM rebalance started: migration_id=%s pending=%d normal_pending=%d",
            migration_context.get("migration_id"),
            len(self._rebalance_pending_request_ids),
            self._pending_count,
        )
        self._signal_routing_update()

    def _finish_exclusive_rebalance_request(self, request_uid: Any, *, reason: str) -> None:
        request_uid = canonical_psrl_request_id(request_uid)
        if request_uid not in getattr(self, "_rebalance_pending_request_ids", set()):
            return
        self._rebalance_pending_request_ids.discard(request_uid)
        self._planned_migration_destinations.pop(request_uid, None)
        psrl_logger.info(
            "Exclusive RM rebalance request settled: request=%s reason=%s remaining=%d",
            request_uid,
            reason,
            len(self._rebalance_pending_request_ids),
        )
        if not self._rebalance_pending_request_ids:
            psrl_logger.info("Exclusive RM rebalance completed; resuming normal routing.")
        self._signal_routing_update()

    def _enqueue_exclusive_rebalance_request(self, request_key: str, request: DataProto) -> bool:
        if not self._exclusive_rebalance_active():
            return False
        pending_uids = [
            canonical_psrl_request_id(uid)
            for uid in self._request_uid_values(request)
            if canonical_psrl_request_id(uid) in self._rebalance_pending_request_ids
        ]
        if not pending_uids:
            return False
        destinations = {
            self._planned_migration_destinations[uid]
            for uid in pending_uids
            if uid in self._planned_migration_destinations
        }
        if len(destinations) != 1 or any(
            uid not in self._planned_migration_destinations for uid in pending_uids
        ):
            psrl_logger.error(
                "Exclusive RM rebalance cannot resolve one simulated destination for request=%s; "
                "returning it to normal routing. destinations=%s",
                self._format_request_uids(request),
                sorted(destinations),
            )
            for uid in pending_uids:
                self._finish_exclusive_rebalance_request(uid, reason="missing_or_conflicting_destination")
            return False
        if request_key in self._rebalance_queued_request_keys:
            return True
        self._rebalance_requests_to_route.append((request_key, request))
        self._rebalance_queued_request_keys.add(request_key)
        psrl_logger.info(
            "Queued interrupted RM request for exclusive rebalance: request=%s destination=%s fifo_depth=%d",
            self._format_request_uids(request),
            next(iter(destinations)),
            len(self._rebalance_requests_to_route),
        )
        self._signal_routing_update()
        return True

    def _exclusive_rebalance_destination(self, request: DataProto) -> int | None:
        pending_uids = [
            canonical_psrl_request_id(uid)
            for uid in self._request_uid_values(request)
            if canonical_psrl_request_id(uid) in self._rebalance_pending_request_ids
        ]
        destinations = {
            self._planned_migration_destinations[uid]
            for uid in pending_uids
            if uid in self._planned_migration_destinations
        }
        if len(destinations) != 1:
            return None
        destination = int(next(iter(destinations)))
        if destination not in self.request_counts or destination in self.paused_worker_indices:
            return None
        return destination

    def _record_forced_route(self, request: DataProto, destination: int) -> None:
        destination = int(destination)
        self.request_counts[destination] += 1
        force_route = getattr(self.route_strategy, "force_route_unchecked", None)
        if force_route is not None:
            force_route(request, destination)

    async def _dispatch_exclusive_rebalance_requests(self) -> bool:
        if self._rebalance_dispatching_request_keys:
            return False
        dispatched_any = False
        while self._rebalance_requests_to_route and not self._interrupt_routing:
            request_key, request = self._rebalance_requests_to_route[0]
            if request_key not in self.request_futures:
                self._rebalance_requests_to_route.popleft()
                self._rebalance_queued_request_keys.discard(request_key)
                for request_uid in self._request_uid_values(request):
                    self._migration_overhead.discard(request_uid)
                    self._finish_exclusive_rebalance_request(request_uid, reason="request_missing")
                continue
            destination = self._exclusive_rebalance_destination(request)
            if destination is None:
                request_uid_values = self._request_uid_values(request)
                planned_destinations = {
                    self._planned_migration_destinations[request_uid]
                    for request_uid in request_uid_values
                    if request_uid in self._planned_migration_destinations
                }
                self._rebalance_requests_to_route.popleft()
                self._rebalance_queued_request_keys.discard(request_key)
                self._planned_migration_counters["fallback"] += len(request_uid_values)
                for request_uid in request_uid_values:
                    self._migration_overhead.discard(request_uid)
                    self._finish_exclusive_rebalance_request(request_uid, reason="fallback_normal_routing")

                try:
                    active_loads = await self._get_cached_active_loads()
                    fallback_destination = self._select_worker_for_request(request, active_loads)
                except Exception:
                    self._enqueue_request(request_key, request)
                    psrl_logger.exception(
                        "Exclusive RM rebalance normal fallback failed; request=%s destinations=%s "
                        "requeued to normal routing; migration timing discarded.",
                        self._format_request_uids(request),
                        sorted(planned_destinations),
                    )
                    dispatched_any = True
                    continue
                if fallback_destination is None:
                    self._enqueue_request(request_key, request)
                    psrl_logger.warning(
                        "Exclusive RM rebalance destination unreachable; request=%s destinations=%s "
                        "normal strategy found no route, requeued to normal routing.",
                        self._format_request_uids(request),
                        sorted(planned_destinations),
                    )
                    dispatched_any = True
                    continue

                self._request_to_strategy_instance[request_key] = int(fallback_destination)
                self._candidate_evaluation_inflight_requests[request_key] = request
                for request_uid in request_uid_values:
                    self._uid_to_inflight_instance[request_uid] = int(fallback_destination)
                inflight_at_dispatch = active_loads.get(int(fallback_destination), 0)
                task = asyncio.create_task(
                    self._route_single_request(
                        request_key,
                        request,
                        int(fallback_destination),
                        inflight_at_dispatch,
                    )
                )
                task.add_done_callback(lambda future: future.result())
                self._is_routing = True
                psrl_logger.warning(
                    "Exclusive RM rebalance destination unreachable; request=%s destinations=%s "
                    "fell back to normal destination=%s; migration timing discarded.",
                    self._format_request_uids(request),
                    sorted(planned_destinations),
                    fallback_destination,
                )
                return True
            self._rebalance_requests_to_route.popleft()
            self._rebalance_queued_request_keys.discard(request_key)
            self._rebalance_dispatching_request_keys.add(request_key)
            inflight_at_dispatch = self.request_counts.get(destination, 0)
            self._record_forced_route(request, destination)
            self._planned_migration_counters["forced"] += len(self._request_uid_values(request))
            self._request_to_strategy_instance[request_key] = destination
            self._candidate_evaluation_inflight_requests[request_key] = request
            for request_uid in self._request_uid_values(request):
                self._uid_to_inflight_instance[request_uid] = destination
            task = asyncio.create_task(
                self._route_single_request(request_key, request, destination, inflight_at_dispatch)
            )
            task.add_done_callback(lambda future: future.result())
            self._is_routing = True
            dispatched_any = True
            # Preserve FIFO dispatch order through the asynchronous worker
            # handoff; the next item is released after redispatch settles.
            break
        return dispatched_any

    def _dequeue_request(self) -> tuple[str, DataProto] | None:
        if not self.requests_to_route:
            return None
        _, _, _, request_key, request = heapq.heappop(self.requests_to_route)
        self._pending_count = max(0, self._pending_count - 1)
        return request_key, request

    def _peek_request(self) -> tuple[str, DataProto] | None:
        if not self.requests_to_route:
            return None
        _, _, _, request_key, request = self.requests_to_route[0]
        return request_key, request

    # ---- control API -----------------------------------------------------
    @ray.method(concurrency_group="control")
    def pause_instances(self, instance_ids: list[int]):
        for instance_id in instance_ids:
            self.paused_worker_indices.add(int(instance_id))
        self._invalidate_load_cache()

    @ray.method(concurrency_group="control")
    def resume_instances(self, instance_ids: list[int]):
        for instance_id in instance_ids:
            instance_id = int(instance_id)
            self.paused_worker_indices.discard(instance_id)
            self.worker_probe_backoff_until.pop(instance_id, None)
        self._invalidate_load_cache()
        self._signal_routing_update()

    @ray.method(concurrency_group="control")
    def mark_migration_requests(self, instance_to_uids: dict, migration_context: dict) -> None:
        """Start distributed overhead tracking before selected requests are aborted."""
        self._migration_overhead.mark_batch(instance_to_uids, migration_context)
        self._start_exclusive_rebalance(instance_to_uids, migration_context)

    @ray.method(concurrency_group="control")
    async def wait_for_exclusive_rebalance(self) -> None:
        """Wait until all accepted exclusive-rebalance requests are redispatched."""
        while self._exclusive_rebalance_active():
            await asyncio.sleep(max(0.01, min(self.retry_delay, 0.1)))

    @staticmethod
    def _log_migration_interrupt(request_uid: Any, overhead: dict[str, Any]) -> None:
        psrl_logger.info(
            "[ELASTIC_OVERHEAD] operation=request_migration_interrupt "
            "scope=post_scale_up_rebalance role=RewardModel migration_id=%s decision_id=%s "
            "request_id=%s source_instance=%s selected_count=%s interrupt_s=%.6f "
            "interrupt_scope=abort_to_requeue",
            overhead.get("migration_id"),
            overhead.get("decision_id"),
            request_uid,
            overhead.get("source_instance_id"),
            overhead.get("selected_count"),
            overhead["interrupt_s"],
        )

    def _mark_migration_requeued(self, request_uid: Any, *, defer_log: bool = False) -> None:
        overhead = self._migration_overhead.mark_requeued(request_uid)
        if overhead is None or defer_log:
            return
        self._log_migration_interrupt(request_uid, overhead)

    @ray.method(concurrency_group="control")
    def prepare_request_migrations(
        self,
        request_migrations: list[dict],
        migration_context: dict | None = None,
    ) -> dict:
        """Validate a plan and atomically arm migration tracking before abort."""
        instance_to_uids: dict[int, list[str]] = {}
        accepted = 0
        skipped = 0
        skip_reasons: dict[str, int] = {}
        skip_samples: dict[str, list[str]] = {}

        def record_skip(reason: str, request_id: str) -> None:
            nonlocal skipped
            skipped += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            samples = skip_samples.setdefault(reason, [])
            if len(samples) < 3:
                samples.append(request_id)

        for migration in request_migrations or []:
            engine_request_id = str(migration.get("request_id", ""))
            logical_request_uid = canonical_psrl_request_id(engine_request_id)
            try:
                source = int(migration["source_instance_id"])
                destination = int(migration["destination_instance_id"])
            except (KeyError, TypeError, ValueError):
                record_skip("invalid_migration", engine_request_id)
                continue
            if not engine_request_id:
                record_skip("invalid_request_id", engine_request_id)
                continue
            current_source = self._uid_to_inflight_instance.get(logical_request_uid)
            if current_source is None:
                record_skip("request_not_inflight", engine_request_id)
                continue
            if current_source != source:
                record_skip("source_changed", engine_request_id)
                continue
            if destination not in self.request_counts:
                record_skip("invalid_destination", engine_request_id)
                continue
            self._planned_migration_destinations[logical_request_uid] = destination
            # Preserve the scheduler/vLLM ID for worker abort. The destination
            # intent is consumed by logical UID when the partial request returns.
            instance_to_uids.setdefault(source, []).append(engine_request_id)
            accepted += 1
        self._planned_migration_counters["accepted"] += accepted
        self._planned_migration_counters["skipped"] += skipped
        self._planned_migration_counters["planned"] += len(request_migrations or [])
        if migration_context is not None:
            self._migration_overhead.mark_batch(instance_to_uids, migration_context)
            self._start_exclusive_rebalance(instance_to_uids, migration_context)
        psrl_logger.info(
            "Prepared RM request migrations: planned=%d accepted=%d skipped=%d "
            "skip_reasons=%s skip_samples=%s sources=%s",
            len(request_migrations or []),
            accepted,
            skipped,
            skip_reasons,
            skip_samples,
            sorted(instance_to_uids),
        )
        return {
            "instance_to_uids": instance_to_uids,
            "planned": len(request_migrations or []),
            "accepted": accepted,
            "skipped": skipped,
            "skip_reasons": skip_reasons,
        }

    @ray.method(concurrency_group="control")
    def is_routing(self) -> bool:
        return self._is_routing

    @ray.method(concurrency_group="control")
    def get_pending_request_count(self) -> int:
        """Return waiting-queue depth for reward routing (elastic_rm backlog signal)."""
        t0 = time.monotonic()
        log_elastic_rm_backlog_diag(psrl_logger, "stage=RewardModelRouter_enter")
        n = int(self._pending_count) + len(self._rebalance_requests_to_route)
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelRouter_exit pending=%d body_s=%.6f",
            n,
            time.monotonic() - t0,
        )
        return n

    @ray.method(concurrency_group="control")
    def get_pending_request_summary(self, top_t: int | None = None) -> dict[str, int]:
        """Return count and token load for the leading reward waiting requests."""
        t0 = time.monotonic()
        log_elastic_rm_backlog_diag(psrl_logger, "stage=RewardModelRouter_summary_enter")
        exclusive_requests = [request for _, request in self._rebalance_requests_to_route]
        total_pending = int(self._pending_count) + len(exclusive_requests)
        limit = total_pending if top_t is None else max(0, min(total_pending, int(top_t)))
        if limit == 0:
            requests: list[DataProto] = []
        else:
            requests = exclusive_requests[:limit]
            normal_limit = limit - len(requests)
            if normal_limit > 0:
                leading = heapq.nsmallest(
                    normal_limit,
                    self.requests_to_route,
                    key=lambda item: item[:3],
                )
                requests.extend(item[4] for item in leading)
        summary = {
            "pending": total_pending,
            "count": len(requests),
            "total_tokens": sum(self._request_token_num(request) for request in requests),
        }
        log_elastic_rm_backlog_diag(
            psrl_logger,
            "stage=RewardModelRouter_summary_exit pending=%d count=%d total_tokens=%d body_s=%.6f",
            summary["pending"],
            summary["count"],
            summary["total_tokens"],
            time.monotonic() - t0,
        )
        return summary

    @ray.method(concurrency_group="control")
    async def get_candidate_evaluation_snapshot(self, top_t: int | None = None) -> dict:
        """Return a compact, immutable-by-convention request/router snapshot."""
        active_loads = await self._get_cached_active_loads()
        exclusive_items = [
            (-1, 0, route_order, request_key, request)
            for route_order, (request_key, request) in enumerate(self._rebalance_requests_to_route)
        ]
        total_pending = int(self._pending_count) + len(exclusive_items)
        limit = total_pending if top_t is None else max(0, min(total_pending, int(top_t)))
        leading = exclusive_items[:limit]
        normal_limit = limit - len(leading)
        if normal_limit > 0:
            leading.extend(
                heapq.nsmallest(
                    normal_limit,
                    self.requests_to_route,
                    key=lambda item: item[:3],
                )
            )
        pending_rows = [
            {
                "request_id": self._format_request_uids(item[4]),
                "seq_len": self._request_token_num(item[4]),
                "source_instance_id": None,
                "is_waiting": True,
                "route_order": route_order,
                "routing_priority": list(item[:3]),
            }
            for route_order, item in enumerate(leading)
        ]

        request_by_uid: dict[str, DataProto] = {}
        for request in self._candidate_evaluation_inflight_requests.values():
            for request_uid in self._request_uid_values(request):
                request_by_uid[request_uid] = request

        instances: list[dict] = []
        inflight_route_order = total_pending
        for instance_id in sorted(self.request_counts):
            engine_status = self.instance_to_engine_status.get(instance_id)
            scheduler_stats = (
                engine_status.snapshot.get("scheduler_stats", {})
                if engine_status is not None
                else {}
            )
            prompt_map = {
                str(request_id): int(value)
                for request_id, value in (
                    scheduler_stats.get("req_id_to_prompt_token_num", {}) or {}
                ).items()
            }
            response_map = {
                str(request_id): int(value)
                for request_id, value in (
                    scheduler_stats.get("req_id_to_response_token_num", {}) or {}
                ).items()
            }
            waiting_ids = {
                str(request_id)
                for request_id in scheduler_stats.get("req_id_in_waiting", []) or []
            }
            request_rows = []
            for request_uid in sorted(set(prompt_map) | set(response_map)):
                request = request_by_uid.get(canonical_psrl_request_id(request_uid))
                buffer_id = self._extract_buffer_id(request) if request is not None else None
                request_rows.append(
                    {
                        "request_id": request_uid,
                        "seq_len": (
                            self._request_token_num(request)
                            if request is not None
                            else prompt_map.get(request_uid, 0)
                            + response_map.get(request_uid, 0)
                        ),
                        "source_instance_id": instance_id,
                        "is_waiting": request_uid in waiting_ids,
                        "route_order": inflight_route_order,
                        "routing_priority": [
                            1 if buffer_id is None else 0,
                            0 if buffer_id is None else buffer_id,
                            self._waiting_seq_counter + inflight_route_order,
                        ],
                    }
                )
                inflight_route_order += 1
            instances.append(
                {
                    "instance_id": instance_id,
                    "is_awake": instance_id not in self.paused_worker_indices,
                    "model_version": (
                        int(engine_status.model_version) if engine_status is not None else 0
                    ),
                    "requests": request_rows,
                    "route_request_count": max(
                        self.request_counts.get(instance_id, 0),
                        active_loads.get(instance_id, 0),
                    ),
                    "running_count": int(scheduler_stats.get("num_running_reqs", 0)),
                    "waiting_count": int(scheduler_stats.get("num_waiting_reqs", 0)),
                    "token_count": sum(prompt_map.values()) + sum(response_map.values()),
                }
            )
        return {
            "role": "RewardModel",
            "strategy": (
                "itl"
                if isinstance(self.route_strategy, ITLBalanceRewardModelRouteStrategy)
                else self.route_strategy.__class__.__name__
            ),
            "instances": instances,
            "pending_requests": pending_rows,
            "pending_total": total_pending,
            "max_concurrent_requests": self.max_concurrent_requests_per_instance,
            "waiting_admission_cap": self.waiting_admission_cap,
            "itl_max_itl": self._itl_router_max_itl,
            "migration_counters": dict(self._planned_migration_counters),
        }

    @ray.method(concurrency_group="control")
    async def interrupt_routing(self):
        """Pause routing (used during coordinated transitions)."""
        async with self.routing_lock:
            self._interrupt_routing = True
        self._signal_routing_update()

    @ray.method(concurrency_group="control")
    async def resume_routing(self):
        """Resume routing and wake waiting routing loop."""
        async with self.routing_lock:
            self._interrupt_routing = False
        self._signal_routing_update()

    @ray.method(concurrency_group="control")
    def update_instance_status(self, instance_to_engine_status: dict[int, EngineStats]) -> None:
        """Update route-strategy engine status snapshots."""
        self.instance_to_engine_status.update(
            {
                int(instance_id): engine_status
                for instance_id, engine_status in instance_to_engine_status.items()
            }
        )
        self.route_strategy.update_instance_to_engine_status(instance_to_engine_status)
        self._invalidate_load_cache()
        self._signal_routing_update()

    # ---- wake/sleep helpers ---------------------------------------------
    async def _wait_for_routing_update(self, timeout_s: float | None = None) -> None:
        """Wait for a routing-state change without dropping wakeups."""
        try:
            if timeout_s is None:
                await self.routing_status_update_queue.get()
            else:
                await asyncio.wait_for(self.routing_status_update_queue.get(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return

    def _signal_routing_update(self) -> None:
        """Wake the routing loop, keeping at most one pending notification."""
        if not hasattr(self, "routing_status_update_queue"):
            return
        try:
            self.routing_status_update_queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    # ---- request introspection -----------------------------------------
    @staticmethod
    def _request_uid_values(request: DataProto) -> list[str]:
        uid_value = request.non_tensor_batch.get("uid")
        if uid_value is None:
            return []
        if hasattr(uid_value, "tolist"):
            uid_value = uid_value.tolist()
        if isinstance(uid_value, (list, tuple)):
            return [str(uid) for uid in uid_value]
        return [str(uid_value)]

    @staticmethod
    def _format_request_uids(request: DataProto) -> str:
        uid_values = PSRL_RewardModelRouter._request_uid_values(request)
        if not uid_values:
            return "unknown"
        return ",".join(uid_values)

    @staticmethod
    def _extract_buffer_id(request: DataProto) -> int | None:
        non_tensor_batch = getattr(request, "non_tensor_batch", {}) or {}
        candidate_keys = ("buffer_id", "waiting_buffer_id", "train_buffer_id")

        def _normalize_int(v: Any) -> int | None:
            if hasattr(v, "tolist"):
                v = v.tolist()
            if isinstance(v, (list, tuple)):
                if not v:
                    return None
                v = v[0]
            try:
                return int(v)
            except (TypeError, ValueError):
                return None

        for key in candidate_keys:
            value = non_tensor_batch.get(key, None)
            normalized = _normalize_int(value)
            if normalized is not None:
                return normalized

        extra_info = non_tensor_batch.get("extra_info", None)
        if isinstance(extra_info, dict):
            normalized = _normalize_int(extra_info.get("buffer_id", None))
            if normalized is not None:
                return normalized
        return None

    @staticmethod
    def _request_token_num(request: DataProto) -> int:
        return _request_token_num(request)


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------
def launch_router_process(
    worker_handles: list[ray.actor.ActorHandle],
    worker_groups: list[Any] | None,
    config: DictConfig,
    reward_model_config: DictConfig | None = None,
    max_attempts: int = 3,
    retry_delay: float = 2.0,
    verbose: bool = False,
    max_concurrency: int = 1,
    route_strategy_config: dict | DictConfig | None = None,
) -> ray.actor.ActorHandle:
    """Launch the reward-model router as a Ray actor.

    Args:
        worker_handles: List of RewardModelWorker Ray handles.
        worker_groups: Worker groups for all-rank (transformers) execution.
        config: Full PSRL Hydra config for the router actor.
        max_attempts: Unused; kept for API compatibility.
        retry_delay: Delay between retries (in seconds).
        verbose: Enable verbose logging.
        max_concurrency: Ray actor max concurrency.
        route_strategy_config: Optional RM-specific routing config override.

    Returns:
        Ray ActorHandle for the router.
    """
    del max_attempts
    router_handle = PSRL_RewardModelRouter.options(max_concurrency=max_concurrency).remote(
        worker_handles=worker_handles,
        worker_groups=worker_groups,
        config=config,
        reward_model_config=reward_model_config,
        retry_delay=retry_delay,
        verbose=verbose,
        route_strategy_config=route_strategy_config,
    )
    psrl_logger.info("RewardModelRouter launched as Ray actor")
    return router_handle
