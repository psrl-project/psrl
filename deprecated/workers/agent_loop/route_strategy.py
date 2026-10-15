# TODO(zyf): check the newest data-tructs used instead of `DataProto`

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from math import ceil

import numpy as np

from psrl.utils.cost_model_path import resolve_cost_model_json_path
from psrl.workers.elastic.candidate_routing import (
    select_throughput_optimal_candidate,
)
from psrl.workers.gen.stats_collector import EngineStats
from psrl.workers.gen.utils import RolloutInstanceId, TokenInput

_ROUTE_STRATEGY_REGISTRY: dict[str, type["RouteStrategyBase"]] = {}


def register_route_strategy(name: str):
    """Register a route strategy class with the given name.

    Args:
        name (str): Name to register the strategy under.

    Returns:
        function: Decorator function for registering the class.
    """

    def decorator(cls: type["RouteStrategyBase"]):
        if name in _ROUTE_STRATEGY_REGISTRY:
            raise ValueError(f"Route strategy '{name}' is already registered")
        _ROUTE_STRATEGY_REGISTRY[name] = cls
        return cls

    return decorator


def get_route_strategy_class(name: str) -> type["RouteStrategyBase"]:
    """Get the route strategy class by name.

    Args:
        name (str): Name of the route strategy.

    Returns:
        Type[RouteStrategyBase]: The route strategy class.

    Raises:
        ValueError: If the strategy name is not registered.
    """
    if name not in _ROUTE_STRATEGY_REGISTRY:
        raise ValueError(
            f"Route strategy '{name}' is not registered. Available strategies: {list(_ROUTE_STRATEGY_REGISTRY.keys())}"
        )
    return _ROUTE_STRATEGY_REGISTRY[name]


def list_available_route_strategies() -> list:
    """Get a list of all available route strategy names.

    Returns:
        list: List of registered strategy names.
    """
    return list(_ROUTE_STRATEGY_REGISTRY.keys())


class RouteStrategyBase(ABC):
    def __init__(self, strategy_kwargs: dict | None = None):
        self.rollout_instances = set()
        self.strategy_kwargs = strategy_kwargs
        self.instance_to_engine_status: dict[RolloutInstanceId, EngineStats] = {}

        # Track the last clock of each instance
        self.instance_to_time_record: dict[RolloutInstanceId, datetime] = {}

        self.logger = strategy_kwargs.get("logger", logging.getLogger(__file__))
        self.logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

    def add_worker(self, instance_id: RolloutInstanceId, **kwargs):
        """Add a worker instance to the route strategy."""
        self.rollout_instances.add(instance_id)
        self.instance_to_time_record[instance_id] = datetime.now()
        self.instance_to_engine_status[instance_id] = EngineStats(
            instance_id=instance_id,
            model_version=kwargs.get("model_version", 0),
            snapshot=EngineStats.get_default_snapshot(),
        )

    def update_config(self, **kwargs):
        """Update the configuration of the route strategy.

        Args:
            **kwargs: Additional keyword arguments for updating the configuration.
        """
        for key, value in kwargs.items():
            self.strategy_kwargs[key] = value

    def update_instance_to_engine_status(self, instance_to_engine_status: dict[RolloutInstanceId, EngineStats]):
        """Update the engine status with latest information from coordinator.

        Args:
            instance_to_engine_status (dict[RolloutInstanceId, EngineStats]): Latest engine status information.
        """
        self.instance_to_engine_status.update(instance_to_engine_status)

    def is_staled(self, instance_id: RolloutInstanceId, engine_status: EngineStats) -> bool:
        """Check if the engine status is stale.

        Args:
            instance_id (RolloutInstanceId): The index of the worker instance.
            engine_status (EngineStats): The engine status.
        """
        snapshot_time = datetime.fromisoformat(engine_status.snapshot["timestamp"])
        return self.instance_to_time_record[instance_id] - snapshot_time > timedelta(
            milliseconds=self.strategy_kwargs.get("snapshot_staleness_threshold_in_ms", 100)
        )

    @abstractmethod
    def route(
        self,
        request: TokenInput,
        candidates: list[RolloutInstanceId] | None = None,
        route_kwargs: dict | None = None,
    ) -> RolloutInstanceId | None:
        """Route a request to a specific worker instance.

        Args:
            request (TokenInput): The request to route.
            candidates (Optional[List[int]]): List of candidate worker indices if any.
            route_kwargs (Optional[dict]): Additional keyword arguments for routing.

        Returns:
            Optional[int]: Index of the selected worker instance. None if no instance is selected.
        """
        pass

    def push_request(self, request: TokenInput, instance_id: RolloutInstanceId):
        """Push a request to a specific worker instance.

        Args:
            request (TokenInput): The request to push.
            instance_id (int): The index of the worker instance to push the request to.
        """
        self.instance_to_time_record[instance_id] = datetime.now()

    def force_route_unchecked(self, request: TokenInput, instance_id: RolloutInstanceId) -> RolloutInstanceId:
        """Record a simulation-selected route without applying admission rules."""
        return int(instance_id)

    def pop_request(self, request: TokenInput, instance_id: RolloutInstanceId):
        """Pop a request from a specific worker instance.

        Args:
            request (TokenInput): The request to pop.
            instance_id (int): The index of the worker instance to pop the request from.
        """
        self.instance_to_time_record[instance_id] = datetime.now()

    def calculate_routing_benefit(self, request: TokenInput, instance_id: RolloutInstanceId) -> float:
        """Calculate the routing benefit of routing a request to a specific
        worker instance.

        Args:
            request (TokenInput): The request to calculate the routing benefit for.
            instance_id (RolloutInstanceId): The index of the worker instance to calculate
                the routing benefit for.
        """
        # By default, we return 1 as the routing benefit (meaning it is
        # beneficial to route the request to the instance)
        # Subclasses can override this method to implement their own routing
        # benefit calculation logic
        return 1


@register_route_strategy("random")
class RandomRouteStrategy(RouteStrategyBase):
    """Randomly selects a worker instance for each request."""

    def __init__(self, strategy_kwargs: dict | None = None):
        super().__init__(strategy_kwargs)
        self.logger.info("Initialized RandomRouteStrategy")

    def route(
        self,
        request: TokenInput,
        candidates: list[RolloutInstanceId] | None = None,
        route_kwargs: dict | None = None,
    ) -> RolloutInstanceId | None:
        if candidates is None:
            candidates = self.rollout_instances
        if len(candidates) == 0:
            return None
        candidates_idx = list(range(len(candidates)))
        instance_idx = np.random.choice(candidates_idx)
        return candidates[instance_idx]


@register_route_strategy("round_robin")
class RoundRobinRouteStrategy(RouteStrategyBase):
    """Routes requests in a round-robin fashion across worker instances."""

    def __init__(self, strategy_kwargs: dict | None = None):
        super().__init__(strategy_kwargs)
        self.curr_idx = 0
        self.logger.info("Initialized RoundRobinRouteStrategy")

    def route(
        self,
        request: TokenInput,
        candidates: list[RolloutInstanceId] | None = None,
        route_kwargs: dict | None = None,
    ) -> RolloutInstanceId | None:
        if candidates is None:
            candidates = self.rollout_instances
        if len(candidates) == 0:
            return None
        # choose from candidates in a round-robin manner
        idx = self.curr_idx
        idx = candidates[idx % len(candidates)]
        self.curr_idx = (self.curr_idx + 1) % len(candidates)
        return idx


@register_route_strategy("request_num_balance")
class RequestNumBalanceRouteStrategy(RouteStrategyBase):
    """Routes requests to the worker instance with the fewest pending requests."""

    def __init__(self, strategy_kwargs: dict | None = None):
        super().__init__(strategy_kwargs)
        assert "balanced_concurrent_seqs_per_instance" in strategy_kwargs, (
            "balanced_concurrent_seqs_per_instance is required for RequestNumBalanceRouteStrategy"
        )
        assert "max_concurrent_seqs_per_instance" in strategy_kwargs, (
            "max_concurrent_seqs_per_instance is required for RequestNumBalanceRouteStrategy"
        )
        self.balanced_concurrent_seqs_per_instance = strategy_kwargs.get("balanced_concurrent_seqs_per_instance", 64)
        self.max_concurrent_seqs_per_instance = strategy_kwargs.get("max_concurrent_seqs_per_instance", 64)
        self.instance_request_counts: dict[RolloutInstanceId, int] = {}
        self.logger.info("Initialized RequestNumBalanceRouteStrategy.")

    def add_worker(self, instance_id: RolloutInstanceId, **kwargs):
        super().add_worker(instance_id, **kwargs)
        self.instance_request_counts[instance_id] = 0

    def route(
        self,
        request: TokenInput,
        candidates: list[RolloutInstanceId] | None = None,
        route_kwargs: dict | None = None,
    ) -> RolloutInstanceId | None:
        if candidates is None:
            candidates = self.rollout_instances
        if len(candidates) == 0:
            return None
        idx = np.argmin([self.instance_request_counts[i] for i in candidates])

        self.logger.debug(
            f"Routing request {request.request_id} among "
            f"candidates {candidates} with all instance workloads "
            f"{self.instance_request_counts}, selected instance "
            f"{candidates[idx]} with workload "
            f"{self.instance_request_counts[candidates[idx]]}"
        )

        if self.instance_request_counts[candidates[idx]] >= self.max_concurrent_seqs_per_instance:
            # if self.instance_request_counts[candidates[idx]] >= self.max_concurrent_seqs_per_instance:
            # Avoid overload, return None (currently not route to any instance)
            return None
        self.instance_request_counts[candidates[idx]] += 1
        return candidates[idx]

    def force_route_unchecked(self, request: TokenInput, instance_id: RolloutInstanceId) -> RolloutInstanceId:
        self.instance_request_counts[instance_id] += 1
        return instance_id

    def update_instance_to_engine_status(self, instance_to_engine_status: dict[RolloutInstanceId, EngineStats]):
        super().update_instance_to_engine_status(instance_to_engine_status)
        for instance_id, engine_stats in instance_to_engine_status.items():
            self.instance_request_counts[instance_id] = engine_stats.get_waiting_and_running_queue_size()

    def push_request(self, request: TokenInput, instance_id: RolloutInstanceId):
        super().push_request(request, instance_id)
        # Do not update the request count here, it has been updated when the request is routed to the instance

    def pop_request(self, request: TokenInput, instance_id: RolloutInstanceId):
        super().pop_request(request, instance_id)
        self.instance_request_counts[instance_id] = max(0, self.instance_request_counts[instance_id] - 1)

    def is_staled(self, instance_id: RolloutInstanceId, engine_status: EngineStats) -> bool:
        if super().is_staled(instance_id, engine_status):
            return True
        if engine_status.get_waiting_and_running_queue_size() != self.instance_request_counts[instance_id]:
            self.logger.debug(
                f"Instance {instance_id} engine queue size "
                f"{engine_status.get_waiting_and_running_queue_size()} "
                f"differs from router recorded request count "
                f"{self.instance_request_counts[instance_id]} "
                f"(expected due to snapshot lag, not gating engine-status resync)"
            )
        return False

    def calculate_routing_benefit(self, request: TokenInput, instance_id: RolloutInstanceId) -> float:
        if self.instance_request_counts[instance_id] >= min(
            self.max_concurrent_seqs_per_instance,
            self.balanced_concurrent_seqs_per_instance,
        ):
            # Avoid overload, return 0 as the routing benefit
            return 0
        return 1


class CostModelBasedRouteStrategy(RouteStrategyBase):
    """Routes requests to the worker instance based on the cost model."""

    def __init__(self, strategy_kwargs: dict | None = None):
        super().__init__(strategy_kwargs)
        assert "logging_interval_in_ms" in strategy_kwargs, (
            "logging_interval_in_ms is required for CostModelBasedRouteStrategy"
        )
        assert "cost_model_path" in strategy_kwargs, "cost_model_path is required for CostModelBasedRouteStrategy"
        assert "instance_to_tp_pp" in strategy_kwargs, "instance_to_tp_pp is required for CostModelBasedRouteStrategy"
        assert "max_num_waiting_reqs_after_preemption" in strategy_kwargs, (
            "max_num_waiting_reqs_after_preemption is required for CostModelBasedRouteStrategy"
        )
        assert "balanced_concurrent_seqs_per_instance" in strategy_kwargs, (
            "balanced_concurrent_seqs_per_instance is required for CostModelBasedRouteStrategy"
        )
        assert "max_concurrent_seqs_per_instance" in strategy_kwargs, (
            "max_concurrent_seqs_per_instance is required for CostModelBasedRouteStrategy"
        )
        assert "delta_throughput_threshold" in strategy_kwargs, (
            "delta_throughput_threshold is required for CostModelBasedRouteStrategy"
        )
        assert "max_prompt_length" in strategy_kwargs, "max_prompt_length is required for CostModelBasedRouteStrategy"
        assert "request_budget" in strategy_kwargs, "request_budget is required for CostModelBasedRouteStrategy"

        self.last_logging_time = {}
        self.logging_interval_in_ms = strategy_kwargs["logging_interval_in_ms"]
        cost_model_path = strategy_kwargs["cost_model_path"]
        model_name = strategy_kwargs.get("model_name", "")
        resolved_cost_model_path = resolve_cost_model_json_path(cost_model_path, model_name)
        assert resolved_cost_model_path is not None, (
            f"cost_model_path {cost_model_path} does not resolve to a JSON for model {model_name!r}"
        )
        with open(resolved_cost_model_path, encoding="utf-8") as f:
            self.cost_model = json.load(f)
        self.instance_to_tp_pp = strategy_kwargs["instance_to_tp_pp"]
        for tp_pp in self.instance_to_tp_pp.values():
            assert tp_pp in self.cost_model, f"tp_pp {tp_pp} is not in cost model"
            cost_model = self.cost_model[tp_pp]
            assert "other_threshold" in cost_model, "other_threshold is required in cost model"
            assert "other_latency_b" in cost_model, "other_latency_b is required in cost model"
            assert "other_latency_k" in cost_model, "other_latency_k is required in cost model"
            assert "attn_latency_b" in cost_model, "attn_latency_b is required in cost model"
            assert "attn_latency_k" in cost_model, "attn_latency_k is required in cost model"
        self.max_num_waiting_reqs_after_preemption = strategy_kwargs["max_num_waiting_reqs_after_preemption"]
        self.balanced_concurrent_seqs_per_instance = strategy_kwargs["balanced_concurrent_seqs_per_instance"]
        self.max_concurrent_seqs_per_instance = strategy_kwargs["max_concurrent_seqs_per_instance"]
        self.delta_throughput_threshold = strategy_kwargs["delta_throughput_threshold"]
        self.max_prompt_length = strategy_kwargs["max_prompt_length"]
        self.request_budget = strategy_kwargs["request_budget"]
        self.instance_to_max_model_len = strategy_kwargs.get("instance_to_max_model_len", {})

        self.instance_to_request_num = {}
        self.instance_to_running_request_num = {}
        self.instance_to_waiting_request_num = {}
        self.instance_to_token_num = {}

    def add_worker(self, instance_id: RolloutInstanceId, **kwargs):
        assert "max_model_len" in kwargs, "max_model_len is required for CostModelBasedRouteStrategy"
        super().add_worker(instance_id, **kwargs)
        self.instance_to_request_num[instance_id] = 0
        self.instance_to_running_request_num[instance_id] = 0
        self.instance_to_waiting_request_num[instance_id] = 0
        self.instance_to_token_num[instance_id] = 0
        self.last_logging_time[instance_id] = time.time()
        self.instance_to_max_model_len[instance_id] = kwargs["max_model_len"]

    def _get_request_token_num(self, request: TokenInput, log_len: bool = False) -> int:
        input_ids = request.input_ids
        response_token_num = request.cu_response_len if request.cu_response_len is not None else 0
        prompt_token_num = len(input_ids) - response_token_num
        if log_len:  # For debug
            self.logger.info(
                f"Request {request.request_id} has "
                f"{prompt_token_num} prompt tokens and "
                f"{response_token_num} response tokens"
            )
        return prompt_token_num + response_token_num

    def force_route_unchecked(self, request: TokenInput, instance_id: RolloutInstanceId) -> RolloutInstanceId:
        self.instance_to_request_num[instance_id] += 1
        self.instance_to_running_request_num[instance_id] += 1
        self.instance_to_token_num[instance_id] += self._get_request_token_num(request)
        return instance_id

    def _can_run_directly(self, request: TokenInput, instance_id: RolloutInstanceId) -> bool:
        if self.instance_to_waiting_request_num[instance_id] > 0:
            return False
        # The request will be put into the waiting queue if the token number
        # exceeds the max kv cache capacity
        new_token_num = self.instance_to_token_num[instance_id] + self._get_request_token_num(request)
        # if new_token_num > self.instance_to_max_model_len[instance_id] +
        # (self.max_prompt_length + self.request_budget) *
        # self.max_num_waiting_reqs_after_preemption:
        if new_token_num > self.instance_to_max_model_len[instance_id]:
            return False
        return True

    # Estimate the latency of a request with given request number and token number
    # The latency is estimated in seconds
    def _estimate_latency(self, instance_id: RolloutInstanceId, request_num: int, token_num: int) -> float:
        tp_pp = self.instance_to_tp_pp[instance_id]
        cost_model = self.cost_model[tp_pp]
        other_threshold = cost_model["other_threshold"]
        other_latency_b = cost_model["other_latency_b"]
        other_latency_k = cost_model["other_latency_k"]
        attn_latency_b = cost_model["attn_latency_b"]
        attn_latency_k = cost_model["attn_latency_k"]
        return (
            attn_latency_b
            + attn_latency_k * token_num
            + max(other_threshold, other_latency_b + other_latency_k * request_num)
        )

    def _estimate_curr_latency_after_route_request(self, request: TokenInput, instance_id: RolloutInstanceId) -> float:
        new_running_request_num = self.instance_to_running_request_num[instance_id] + 1
        new_token_num = self.instance_to_token_num[instance_id] + self._get_request_token_num(request)
        return self._estimate_latency(instance_id, new_running_request_num, new_token_num)

    def _estimate_curr_throughput(self, instance_id: RolloutInstanceId) -> float:
        return self.instance_to_running_request_num[instance_id] / self._estimate_latency(
            instance_id,
            self.instance_to_running_request_num[instance_id],
            self.instance_to_token_num[instance_id],
        )

    def _estimate_curr_throughput_after_route_request(
        self, request: TokenInput, instance_id: RolloutInstanceId
    ) -> float:
        new_running_request_num = self.instance_to_running_request_num[instance_id] + 1
        new_token_num = self.instance_to_token_num[instance_id] + self._get_request_token_num(request)
        return new_running_request_num / self._estimate_latency(instance_id, new_running_request_num, new_token_num)

    def route(
        self,
        request: TokenInput,
        candidates: list[RolloutInstanceId] | None = None,
        route_kwargs: dict | None = None,
    ) -> RolloutInstanceId | None:
        raise RuntimeError(
            "CostModelBasedRouteStrategy is an abstract class, please use "
            "other inherited classes which implement the route method"
        )

    def obtain_instance_token_num_from_engine_status(
        self, instance_id: RolloutInstanceId, engine_status: EngineStats
    ) -> int:
        # Faster way to get the token number from the engine status
        return ceil(engine_status.get_kv_cache_utilization() * self.instance_to_max_model_len[instance_id])
        # Slower way to get the token number from the engine status
        # return engine_status.get_total_token_num()

    def update_instance_to_engine_status(self, instance_to_engine_status: dict[RolloutInstanceId, EngineStats]):
        super().update_instance_to_engine_status(instance_to_engine_status)
        for instance_id, engine_stats in instance_to_engine_status.items():
            self.instance_to_request_num[instance_id] = engine_stats.get_waiting_and_running_queue_size()
            self.instance_to_running_request_num[instance_id] = engine_stats.snapshot.get("scheduler_stats", {}).get(
                "num_running_reqs", 0
            )
            self.instance_to_waiting_request_num[instance_id] = engine_stats.snapshot.get("scheduler_stats", {}).get(
                "num_waiting_reqs", 0
            )
            if time.time() - self.last_logging_time[instance_id] >= self.logging_interval_in_ms / 1000:
                generation_throughput = engine_stats.get_generation_throughput()
                kv_cache_usage = engine_stats.snapshot.get("scheduler_stats", {}).get("kv_cache_usage", 0.0)
                new_token_num = ceil(kv_cache_usage * self.instance_to_max_model_len[instance_id])
                actual_latency = (
                    self.instance_to_running_request_num[instance_id] / generation_throughput
                    if generation_throughput > 0
                    else float("nan")
                )
                estimated_latency = self._estimate_latency(
                    instance_id,
                    self.instance_to_running_request_num[instance_id],
                    self.instance_to_token_num[instance_id],
                )
                self.logger.debug(
                    f"Router collected engine status for instance {instance_id}: "
                    f"request num {self.instance_to_request_num[instance_id]}, "
                    f"token num change from {self.instance_to_token_num[instance_id]} "
                    f"to {new_token_num}, "
                    f"actual generation latency {actual_latency}, "
                    f"estimated generation latency {estimated_latency}, "
                    f"actual generation throughput {generation_throughput}, "
                    f"estimated generation throughput "
                    f"{self._estimate_curr_throughput(instance_id)}"
                )
                self.last_logging_time[instance_id] = time.time()
            self.instance_to_token_num[instance_id] = self.obtain_instance_token_num_from_engine_status(
                instance_id, engine_stats
            )

    def push_request(self, request: TokenInput, instance_id: RolloutInstanceId):
        super().push_request(request, instance_id)
        # Do not update the request count here, it has been updated when the request is routed to the instance

    def pop_request(self, request: TokenInput, instance_id: RolloutInstanceId):
        super().pop_request(request, instance_id)
        _old_request_num = self.instance_to_request_num[instance_id]
        _new_request_num = max(0, _old_request_num - 1)
        self.instance_to_request_num[instance_id] = _new_request_num
        if _old_request_num <= 0:
            self.logger.warning(
                f"request {request.non_tensor_batch['uid'][0]} finished from "
                f"instance {instance_id}, but instance_to_request_num was already "
                f"{_old_request_num}; clamped to 0 to avoid negative drift "
                f"(inflight count may be desynced from engine)"
            )
        else:
            self.logger.debug(
                f"request {request.non_tensor_batch['uid'][0]} finished from "
                f"instance {instance_id}, instance_to_request_num turning from "
                f"{_old_request_num} to {_new_request_num}"
            )
        self.instance_to_running_request_num[instance_id] = max(
            0, self.instance_to_running_request_num[instance_id] - 1
        )
        self.instance_to_token_num[instance_id] = max(
            0, self.instance_to_token_num[instance_id] - self._get_request_token_num(request)
        )

    def is_staled(self, instance_id: RolloutInstanceId, engine_status: EngineStats) -> bool:
        if super().is_staled(instance_id, engine_status):
            return True
        if engine_status.get_waiting_and_running_queue_size() != self.instance_to_request_num[instance_id]:
            self.logger.debug(
                f"Instance {instance_id} engine queue size "
                f"{engine_status.get_waiting_and_running_queue_size()} "
                f"differs from router recorded request count "
                f"{self.instance_to_request_num[instance_id]} "
                f"(expected due to snapshot lag, not gating engine-metric resync)"
            )
        return False


@register_route_strategy("throughput_optimal")
class ThroughputOptimalRouteStrategy(CostModelBasedRouteStrategy):
    """Routes requests to the worker instance with optimal throughput."""

    def __init__(self, strategy_kwargs: dict | None = None):
        super().__init__(strategy_kwargs)
        self.logger.info("Initialized ThroughputOptimalRouteStrategy")

    # We regard the baseline delta throughput (also the max delta throughput)
    # as the throughput after routing a request to an empty instance
    def _estimate_baseline_delta_throughput(self, request: TokenInput, instance_id: RolloutInstanceId) -> float:
        return 1 / self._estimate_latency(instance_id, 1, self._get_request_token_num(request))

    def force_route(self, request: TokenInput, instance_id: RolloutInstanceId) -> RolloutInstanceId | None:
        """Commit a prepared migration to one target after hard-cap checks."""
        if not self._can_run_directly(request, instance_id):
            return None
        if self.instance_to_request_num[instance_id] >= self.max_concurrent_seqs_per_instance:
            return None
        return self.force_route_unchecked(request, instance_id)

    def route(
        self,
        request: TokenInput,
        candidates: list[RolloutInstanceId] | None = None,
        route_kwargs: dict | None = None,
    ) -> RolloutInstanceId:
        if candidates is None:
            candidates = self.rollout_instances
        if len(candidates) == 0:
            # self.logger.info(f"No candidates available for request {request.request_id}")
            return None
        candidates_group_by_priority = []

        # Sort candidates by indicator
        assert "candidate_indicator_list" in route_kwargs, (
            "candidate_indicator_list is required for CostModelBasedRouteStrategy"
        )
        candidate_indicator_list = route_kwargs["candidate_indicator_list"]
        assert len(candidate_indicator_list) == len(candidates), (
            f"The number of candidates and candidate indicator list must "
            f"be the same, but have {len(candidates)} candidates and "
            f"{len(candidate_indicator_list)} candidate indicator list"
        )
        shared_selection = select_throughput_optimal_candidate(
            candidates=candidates,
            candidate_indicators=candidate_indicator_list,
            can_run_directly=lambda instance_id: self._can_run_directly(
                request,
                instance_id,
            ),
            current_throughput=self._estimate_curr_throughput,
            next_throughput=lambda instance_id: (
                self._estimate_curr_throughput_after_route_request(
                    request,
                    instance_id,
                )
            ),
            baseline_delta_throughput=lambda instance_id: (
                self._estimate_baseline_delta_throughput(request, instance_id)
            ),
            within_request_cap=lambda instance_id: (
                self.instance_to_request_num[instance_id]
                < self.max_concurrent_seqs_per_instance
            ),
            delta_throughput_threshold=self.delta_throughput_threshold,
        )
        if shared_selection is None:
            return None
        indicator_to_candidates = {}
        for candidate, indicator in zip(candidates, candidate_indicator_list):
            if indicator not in indicator_to_candidates:
                indicator_to_candidates[indicator] = []
            indicator_to_candidates[indicator].append(candidate)
        # Sort indicators from smallest to largest
        for indicator in sorted(indicator_to_candidates.keys()):
            candidates_group_by_priority.append((indicator, indicator_to_candidates[indicator]))

        # Process each group of candidates
        # from highest priority to lowest priority
        for indicator, candidates in candidates_group_by_priority:
            # Calculate baseline threshold
            # (use first candidate's baseline as reference)
            baseline_delta_throughput = self._estimate_baseline_delta_throughput(request, candidates[0])
            threshold = baseline_delta_throughput * self.delta_throughput_threshold
            best_candidate = None
            best_delta_throughput = float("-inf")

            # Find the candidate with maximum delta_throughput
            # in this version group
            for candidate in candidates:
                if not self._can_run_directly(request, candidate):
                    continue
                estimated_curr_throughput = self._estimate_curr_throughput(candidate)
                estimated_curr_throughput_after_route_request = self._estimate_curr_throughput_after_route_request(
                    request, candidate
                )
                delta_throughput = estimated_curr_throughput_after_route_request - estimated_curr_throughput
                if delta_throughput > best_delta_throughput:
                    best_delta_throughput = delta_throughput
                    best_candidate = candidate

            if best_candidate is None:
                if request.rollout_instance_id is not None:
                    version_info = request.version_tag if request.version_tag != -1 else "None (retry request)"
                    self.logger.debug(
                        f"No candidate in group {candidates} with indicator "
                        f"{indicator} meets the condition for request "
                        f"{request.request_id} from rollout "
                        f"instance "
                        f"{request.rollout_instance_id} "
                        f"with version {version_info}, because none of the "
                        f"candidates can run directly (kv_cache is full), "
                        f"waiting request num is "
                        f"{self.instance_to_waiting_request_num}, running "
                        f"request num is "
                        f"{self.instance_to_running_request_num}, token num "
                        f"is {self.instance_to_token_num}, request token num "
                        f"is {self._get_request_token_num(request)}, max "
                        f"model len is {self.instance_to_max_model_len}"
                    )
                else:
                    self.logger.debug(
                        f"No candidate in group {candidates} with indicator "
                        f"{indicator} meets the condition for request "
                        f"{request.request_id}, because none "
                        f"of the candidates can run directly (kv_cache is "
                        f"full), waiting request num is "
                        f"{self.instance_to_waiting_request_num}, running "
                        f"request num is "
                        f"{self.instance_to_running_request_num}, token num "
                        f"is {self.instance_to_token_num}, request token num "
                        f"is {self._get_request_token_num(request)}, max "
                        f"model len is {self.instance_to_max_model_len}"
                    )
                continue

            # If this group's best delta_throughput meets the threshold
            # Return it
            if (
                best_delta_throughput >= threshold
                and self.instance_to_request_num[best_candidate] < self.max_concurrent_seqs_per_instance
            ):
                assert best_candidate == shared_selection, (
                    "shared throughput-optimal selector diverged from live route path: "
                    f"shared={shared_selection}, live={best_candidate}"
                )
                # if best_delta_throughput >= threshold:
                _old_request_num = self.instance_to_request_num[best_candidate]
                self.instance_to_request_num[best_candidate] += 1
                self.logger.debug(
                    f"request {request.non_tensor_batch['uid'][0]} decided to "
                    f"route to instance {best_candidate}, "
                    f"instance_to_request_num turning from {_old_request_num} "
                    f"to {self.instance_to_request_num[best_candidate]}"
                )
                self.instance_to_running_request_num[best_candidate] += 1
                self.instance_to_token_num[best_candidate] += self._get_request_token_num(request)
                if request.rollout_instance_id is not None:
                    version_info = request.version_tag if request.version_tag != -1 else "None (retry request)"
                    self.logger.debug(
                        f"Candidate in group {candidates} with indicator "
                        f"{indicator} meets the condition for partial "
                        f"rollout request "
                        f"{request.request_id} from rollout "
                        f"instance "
                        f"{request.rollout_instance_id} "
                        f"with version {version_info}, best candidate "
                        f"{best_candidate} delta throughput: "
                        f"{best_delta_throughput}, baseline delta "
                        f"throughput: {baseline_delta_throughput}, "
                        f"threshold: {threshold}, its current request_num is "
                        f"{self.instance_to_request_num[best_candidate]} and "
                        f"token_num is "
                        f"{self.instance_to_token_num[best_candidate]}"
                    )
                else:
                    self.logger.debug(
                        f"Candidate in group {candidates} with indicator "
                        f"{indicator} meets the condition for request "
                        f"{request.request_id}, best "
                        f"candidate {best_candidate} delta throughput: "
                        f"{best_delta_throughput}, baseline delta "
                        f"throughput: {baseline_delta_throughput}, "
                        f"threshold: {threshold}, its current request_num is "
                        f"{self.instance_to_request_num[best_candidate]} and "
                        f"token_num is "
                        f"{self.instance_to_token_num[best_candidate]}"
                    )
                return best_candidate
            else:
                if request.rollout_instance_id is not None:
                    version_info = request.version_tag if request.version_tag != -1 else "None (retry request)"
                    self.logger.debug(
                        f"No candidate in group {candidates} with indicator "
                        f"{indicator} meets the condition for partial "
                        f"rollout request "
                        f"{request.request_id} from rollout "
                        f"instance "
                        f"{request.rollout_instance_id} "
                        f"with version {version_info}, best candidate "
                        f"{best_candidate} delta throughput: "
                        f"{best_delta_throughput}, baseline delta "
                        f"throughput: {baseline_delta_throughput}, "
                        f"threshold: {threshold}, its current request_num is "
                        f"{self.instance_to_request_num[best_candidate]} and "
                        f"token_num is "
                        f"{self.instance_to_token_num[best_candidate]}"
                    )
                else:
                    self.logger.debug(
                        f"No candidate in group {candidates} with indicator "
                        f"{indicator} meets the condition for request "
                        f"{request.request_id}, best "
                        f"candidate {best_candidate} delta throughput: "
                        f"{best_delta_throughput}, baseline delta "
                        f"throughput: {baseline_delta_throughput}, "
                        f"threshold: {threshold}, its current request_num is "
                        f"{self.instance_to_request_num[best_candidate]} and "
                        f"token_num is "
                        f"{self.instance_to_token_num[best_candidate]}"
                    )

        # If all groups' best delta_throughput are below threshold, return None
        return None

    def calculate_routing_benefit(self, request: TokenInput, instance_id: RolloutInstanceId) -> float:
        if not self._can_run_directly(request, instance_id):
            # If the request cannot be run directly, return 0 as the routing
            # benefit
            return 0
        estimated_curr_throughput = self._estimate_curr_throughput(instance_id)
        estimated_curr_throughput_after_route_request = self._estimate_curr_throughput_after_route_request(
            request, instance_id
        )
        delta_throughput = estimated_curr_throughput_after_route_request - estimated_curr_throughput
        baseline_delta_throughput = (
            self._estimate_baseline_delta_throughput(request, instance_id) * self.delta_throughput_threshold
        )
        return (
            delta_throughput
            if delta_throughput >= baseline_delta_throughput
            and self.instance_to_request_num[instance_id] < self.max_concurrent_seqs_per_instance
            else 0
        )
        # return delta_throughput if delta_throughput >=
        # baseline_delta_throughput else 0


@register_route_strategy("throughput_optimal_with_budget")
class ThroughputOptimalWithBudgetRouteStrategy(ThroughputOptimalRouteStrategy):
    """Routes requests to the worker instance with optimal throughput
    (considering the budget for each request)."""

    def __init__(self, strategy_kwargs: dict | None = None):
        super().__init__(strategy_kwargs)
        self.logger.info("Initialized ThroughputOptimalWithBudgetRouteStrategy")

    def _get_request_token_num(self, request: TokenInput, log_len: bool = False) -> int:
        input_ids = request.input_ids
        response_token_num = request.cu_response_len if request.cu_response_len is not None else 0
        prompt_token_num = len(input_ids) - response_token_num
        if log_len:  # For debug
            self.logger.info(
                f"Request {request.request_id} has "
                f"{prompt_token_num} prompt tokens and "
                f"{response_token_num} response tokens"
            )
        # Adjust the response token number to the nearest multiple of the
        # budget for each request
        return prompt_token_num + ceil((response_token_num + 1) / self.request_budget) * self.request_budget

    def obtain_instance_token_num_from_engine_status(
        self, instance_id: RolloutInstanceId, engine_status: EngineStats
    ) -> int:
        all_request_prompt_token_num = engine_status.get_req_id_to_prompt_token_num().values()
        all_request_response_token_num = engine_status.get_req_id_to_response_token_num().values()
        assert len(all_request_prompt_token_num) == len(all_request_response_token_num), (
            f"The number of prompts and responses must be the same, but have "
            f"{len(all_request_prompt_token_num)} prompts and "
            f"{len(all_request_response_token_num)} responses"
        )
        token_num_with_budget = 0
        for request_prompt_token_num, request_response_token_num in zip(
            all_request_prompt_token_num, all_request_response_token_num
        ):
            token_num_with_budget += request_prompt_token_num
            token_num_with_budget += ceil((request_response_token_num + 1) / self.request_budget) * self.request_budget
        return token_num_with_budget

    """
    def _get_actual_request_token_num(self, request: TokenInput) -> int:
        input_ids = request.input_ids
        response_token_num = request.cu_response_len if request.cu_response_len is not None else 0
        prompt_token_num = len(input_ids) - response_token_num
        return prompt_token_num + response_token_num
    
    def pop_request(self, request: TokenInput, instance_id: RolloutInstanceId):
        RouteStrategyBase.pop_request(self, request, instance_id)
        self.instance_to_request_num[instance_id] -= 1
        self.instance_to_running_request_num[instance_id] -= 1
        self.instance_to_token_num[instance_id] -= self._get_actual_request_token_num(request)
    """
