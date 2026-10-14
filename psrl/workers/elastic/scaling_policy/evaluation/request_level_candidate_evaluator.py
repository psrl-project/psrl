"""
Pure request-level simulation for ITL elastic-scaling candidates.

The evaluator consumes compact router snapshots and never mutates router or
worker state. It intentionally supports only the two strategies used by the
mode5 request-level policy: rollout ``throughput_optimal`` and reward-model
``itl``.
"""

from __future__ import annotations

import heapq
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from psrl.workers.elastic.candidate_routing import (
    group_candidates_by_indicator,
    priority_key,
    select_itl_candidate,
    select_throughput_optimal_from_groups,
)


@dataclass(frozen=True)
class RequestSnapshot:
    request_id: str
    seq_len: int
    source_instance_id: int | None = None
    is_waiting: bool = False
    movable: bool = True
    route_order: int = 0
    routing_priority: tuple[Any, ...] = ()
    eligible_instance_ids: tuple[int, ...] | None = None
    fallback_instance_ids: tuple[int, ...] | None = None
    candidate_priorities: tuple[tuple[int, Any], ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RequestSnapshot:
        eligible = raw.get("eligible_instance_ids")
        fallback = raw.get("fallback_instance_ids")
        priorities = raw.get("candidate_priorities") or ()
        routing_priority = raw.get("routing_priority") or ()
        if not isinstance(routing_priority, (list, tuple)):
            routing_priority = (routing_priority,)
        return cls(
            request_id=str(raw.get("request_id", raw.get("uid", ""))),
            seq_len=max(0, int(raw.get("seq_len", raw.get("token_count", 0)) or 0)),
            source_instance_id=(None if raw.get("source_instance_id") is None else int(raw["source_instance_id"])),
            is_waiting=bool(raw.get("is_waiting", False)),
            movable=bool(raw.get("movable", True)),
            route_order=int(raw.get("route_order", 0) or 0),
            routing_priority=tuple(routing_priority),
            eligible_instance_ids=(None if eligible is None else tuple(int(item) for item in eligible)),
            fallback_instance_ids=(None if fallback is None else tuple(int(item) for item in fallback)),
            candidate_priorities=tuple((int(item[0]), item[1]) for item in priorities if len(item) >= 2),
        )


@dataclass(frozen=True)
class InstanceSnapshot:
    instance_id: int
    is_awake: bool
    model_version: int = 0
    requests: tuple[RequestSnapshot, ...] = ()
    route_request_count: int = 0
    running_count: int = 0
    waiting_count: int = 0
    token_count: int = 0
    max_model_len: int = 2**63 - 1
    throughput_params: tuple[float, float, float, float] = (0.0, 1.0, 1.0, 0.0)
    route_cost_params: tuple[float, float, float, float, float] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> InstanceSnapshot:
        throughput_params = raw.get("throughput_params") or (0.0, 1.0, 1.0, 0.0)
        route_cost = raw.get("route_cost_params")
        return cls(
            instance_id=int(raw["instance_id"]),
            is_awake=bool(raw.get("is_awake", raw.get("available", False))),
            model_version=int(raw.get("candidate_model_version", raw.get("model_version", 0)) or 0),
            requests=tuple(RequestSnapshot.from_mapping(item) for item in raw.get("requests", ())),
            route_request_count=max(
                0,
                int(raw.get("route_request_count", raw.get("request_count", 0)) or 0),
            ),
            running_count=max(0, int(raw.get("running_count", 0) or 0)),
            waiting_count=max(0, int(raw.get("waiting_count", 0) or 0)),
            token_count=max(0, int(raw.get("token_count", 0) or 0)),
            max_model_len=max(0, int(raw.get("max_model_len", 2**63 - 1) or 0)),
            throughput_params=tuple(float(value) for value in throughput_params),
            route_cost_params=(None if route_cost is None else tuple(float(value) for value in route_cost)),
        )


@dataclass(frozen=True)
class RoleSnapshot:
    role: str
    strategy: str
    instances: tuple[InstanceSnapshot, ...]
    pending_requests: tuple[RequestSnapshot, ...] = ()
    queue_scope: str = "running"
    max_concurrent_requests: int | None = None
    delta_throughput_threshold: float = 0.0
    itl_max_itl: float | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RoleSnapshot:
        max_concurrent = raw.get("max_concurrent_requests")
        max_itl = raw.get("itl_max_itl")
        return cls(
            role=str(raw.get("role", "")),
            strategy=str(raw.get("strategy", "")).lower(),
            instances=tuple(
                sorted(
                    (InstanceSnapshot.from_mapping(item) for item in raw.get("instances", ())),
                    key=lambda item: item.instance_id,
                )
            ),
            pending_requests=tuple(RequestSnapshot.from_mapping(item) for item in raw.get("pending_requests", ())),
            queue_scope=str(raw.get("queue_scope", "running")).lower(),
            max_concurrent_requests=(None if max_concurrent is None else int(max_concurrent)),
            delta_throughput_threshold=float(raw.get("delta_throughput_threshold", 0.0) or 0.0),
            itl_max_itl=None if max_itl is None else float(max_itl),
        )


@dataclass(frozen=True)
class RequestMigration:
    request_id: str
    source_instance_id: int
    destination_instance_id: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_instance_id": self.source_instance_id,
            "destination_instance_id": self.destination_instance_id,
        }


@dataclass(frozen=True)
class RoleCandidatePlan:
    wake_instance_ids: frozenset[int] = frozenset()
    sleep_instance_ids: frozenset[int] = frozenset()
    primary_scale_up: bool = False


@dataclass(frozen=True)
class RoleEvaluationResult:
    throughput: float
    routed_count: int
    unrouted_count: int
    rebalance_moves: tuple[RequestMigration, ...]
    instance_throughputs: tuple[tuple[int, float], ...]
    rebalance_simulation_s: float = field(default=0.0, compare=False)
    router_simulation_s: float = field(default=0.0, compare=False)
    other_simulation_s: float = field(default=0.0, compare=False)
    rebalance_started_s: float | None = field(default=None, compare=False)
    rebalance_finished_s: float | None = field(default=None, compare=False)
    router_started_s: float | None = field(default=None, compare=False)
    router_finished_s: float | None = field(default=None, compare=False)


@dataclass(frozen=True)
class _PreparedRequest:
    snapshot: RequestSnapshot
    route_key: tuple[Any, ...]
    eligible_groups: tuple[tuple[int, ...], ...]
    fallback_groups: tuple[tuple[int, ...], ...] | None


@dataclass(frozen=True)
class _PreparedInstance:
    snapshot: InstanceSnapshot
    visible_request_count: int
    visible_token_count: int
    movable_requests: tuple[_PreparedRequest, ...]
    reroute_requests: tuple[_PreparedRequest, ...]


@dataclass(frozen=True)
class RoleEvaluationContext:
    """
    Immutable indexes and request ordering shared by all plans for one role.
    """

    snapshot: RoleSnapshot
    instance_by_id: Mapping[int, _PreparedInstance]
    all_instance_ids: tuple[int, ...]
    before_awake: frozenset[int]
    pending_requests: tuple[_PreparedRequest, ...]


@dataclass
class _InstanceState:
    prepared: _PreparedInstance
    throughput_request_count: int
    throughput_token_count: int
    route_request_count: int
    running_count: int
    waiting_count: int
    token_count: int

    @property
    def snapshot(self) -> InstanceSnapshot:
        return self.prepared.snapshot


def _request_id_key(request_id: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(request_id), request_id)
    except (TypeError, ValueError):
        return (1, request_id, request_id)


def _request_route_key(request: RequestSnapshot) -> tuple[Any, ...]:
    priority = request.routing_priority or (request.route_order,)
    return (
        tuple(priority_key(item) for item in priority),
        request.route_order,
        _request_id_key(request.request_id),
    )


def _prepare_request(
    request: RequestSnapshot,
    all_instance_ids: tuple[int, ...],
    *,
    prepare_rollout_groups: bool,
) -> _PreparedRequest:
    if not prepare_rollout_groups:
        return _PreparedRequest(
            snapshot=request,
            route_key=_request_route_key(request),
            eligible_groups=(),
            fallback_groups=None,
        )
    priority_by_id = dict(request.candidate_priorities)
    eligible_ids = all_instance_ids if request.eligible_instance_ids is None else request.eligible_instance_ids
    eligible_groups = group_candidates_by_indicator(
        eligible_ids,
        [priority_by_id.get(instance_id, 0) for instance_id in eligible_ids],
    )
    fallback_groups = None
    if request.fallback_instance_ids is not None:
        fallback_groups = group_candidates_by_indicator(
            request.fallback_instance_ids,
            [priority_by_id.get(instance_id, 0) for instance_id in request.fallback_instance_ids],
        )
    return _PreparedRequest(
        snapshot=request,
        route_key=_request_route_key(request),
        eligible_groups=eligible_groups,
        fallback_groups=fallback_groups,
    )


def prepare_role_evaluation_context(
    snapshot: RoleSnapshot | Mapping[str, Any],
) -> RoleEvaluationContext:
    """
    Index and order one role snapshot once for all candidate evaluations.
    """
    if not isinstance(snapshot, RoleSnapshot):
        snapshot = RoleSnapshot.from_mapping(snapshot)
    if snapshot.strategy not in {"throughput_optimal", "itl"}:
        raise ValueError(f"unsupported request-level routing strategy: {snapshot.strategy!r}")

    all_instance_ids = tuple(sorted(instance.instance_id for instance in snapshot.instances))
    prepare_rollout_groups = snapshot.strategy == "throughput_optimal"
    instance_by_id: dict[int, _PreparedInstance] = {}
    for instance in snapshot.instances:
        prepared_requests = tuple(
            _prepare_request(
                request,
                all_instance_ids,
                prepare_rollout_groups=prepare_rollout_groups,
            )
            for request in instance.requests
        )
        visible_requests = tuple(
            request
            for request in prepared_requests
            if snapshot.queue_scope != "running" or not request.snapshot.is_waiting
        )
        instance_by_id[instance.instance_id] = _PreparedInstance(
            snapshot=instance,
            visible_request_count=len(visible_requests),
            visible_token_count=sum(max(0, request.snapshot.seq_len) for request in visible_requests),
            movable_requests=tuple(
                sorted(
                    (request for request in visible_requests if request.snapshot.movable),
                    key=lambda item: (
                        item.snapshot.seq_len,
                        _request_id_key(item.snapshot.request_id),
                    ),
                )
            ),
            reroute_requests=tuple(
                sorted(
                    (request for request in prepared_requests if request.snapshot.movable),
                    key=lambda item: item.route_key,
                )
            ),
        )

    pending_requests = tuple(
        sorted(
            (
                _prepare_request(
                    request,
                    all_instance_ids,
                    prepare_rollout_groups=prepare_rollout_groups,
                )
                for request in snapshot.pending_requests
            ),
            key=lambda item: item.route_key,
        )
    )
    return RoleEvaluationContext(
        snapshot=snapshot,
        instance_by_id=instance_by_id,
        all_instance_ids=all_instance_ids,
        before_awake=frozenset(instance.instance_id for instance in snapshot.instances if instance.is_awake),
        pending_requests=pending_requests,
    )


def _compute_itl(
    params: tuple[float, float, float, float],
    tokens: int,
    requests: int,
) -> float:
    a, b, c, d = params
    return max(a * max(0, tokens) + max(b, c * max(0, requests)) + d, 1e-9)


def _throughput_for_load(state: _InstanceState, requests: int, tokens: int) -> float:
    if requests <= 0:
        return 0.0
    return requests / _compute_itl(
        state.snapshot.throughput_params,
        tokens,
        requests,
    )


def _instance_throughput(state: _InstanceState) -> float:
    return _throughput_for_load(
        state,
        state.throughput_request_count,
        state.throughput_token_count,
    )


def _route_latency(state: _InstanceState, running: int, tokens: int) -> float:
    params = state.snapshot.route_cost_params
    if params is None:
        return _compute_itl(state.snapshot.throughput_params, tokens, running)
    other_threshold, other_b, other_k, attn_b, attn_k = params
    return max(
        attn_b + attn_k * tokens + max(other_threshold, other_b + other_k * running),
        1e-9,
    )


def _rollout_can_run_directly(
    state: _InstanceState,
    request: RequestSnapshot,
) -> bool:
    if state.waiting_count > 0:
        return False
    if state.token_count + request.seq_len > state.snapshot.max_model_len:
        return False
    return True


def _filter_active_groups(
    groups: Sequence[Sequence[int]],
    active_ids: frozenset[int],
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        filtered
        for group in groups
        if (filtered := tuple(instance_id for instance_id in group if instance_id in active_ids))
    )


def _rollout_select(
    context: RoleEvaluationContext,
    states: dict[int, _InstanceState],
    active_ids: frozenset[int],
    request: _PreparedRequest,
) -> int | None:
    groups = _filter_active_groups(request.eligible_groups, active_ids)
    if not groups and request.fallback_groups is not None:
        groups = _filter_active_groups(request.fallback_groups, active_ids)
    if not groups:
        return None
    request_snapshot = request.snapshot

    def current_throughput(instance_id: int) -> float:
        state = states[instance_id]
        return state.running_count / _route_latency(
            state,
            state.running_count,
            state.token_count,
        )

    def next_throughput(instance_id: int) -> float:
        state = states[instance_id]
        next_running = state.running_count + 1
        next_tokens = state.token_count + request_snapshot.seq_len
        return next_running / _route_latency(state, next_running, next_tokens)

    return select_throughput_optimal_from_groups(
        candidate_groups=groups,
        can_run_directly=lambda instance_id: _rollout_can_run_directly(states[instance_id], request_snapshot),
        current_throughput=current_throughput,
        next_throughput=next_throughput,
        baseline_delta_throughput=lambda instance_id: (
            1.0 / _route_latency(states[instance_id], 1, request_snapshot.seq_len)
        ),
        within_request_cap=lambda instance_id: (
            context.snapshot.max_concurrent_requests is None
            or states[instance_id].route_request_count < context.snapshot.max_concurrent_requests
        ),
        delta_throughput_threshold=context.snapshot.delta_throughput_threshold,
    )


def _rm_can_admit(snapshot: RoleSnapshot, state: _InstanceState) -> bool:
    load = state.route_request_count
    if snapshot.max_concurrent_requests is not None and load >= snapshot.max_concurrent_requests:
        return False
    return True


class _RMSelectorHeap:
    """
    Candidate-local RM selector updated only for the chosen instance.
    """

    def __init__(self, snapshot: RoleSnapshot, states: dict[int, _InstanceState]):
        self.snapshot = snapshot
        self.states = states
        self.versions = {instance_id: 0 for instance_id in states}
        self.heap: list[tuple[float, int, int, int]] = []
        for instance_id in sorted(states):
            self._push(instance_id)

    def _key(self, instance_id: int) -> tuple[float, int, int]:
        state = self.states[instance_id]
        load = state.route_request_count
        return (
            _compute_itl(state.snapshot.throughput_params, 0, load + 1),
            load,
            instance_id,
        )

    def _push(self, instance_id: int) -> None:
        state = self.states[instance_id]
        if not _rm_can_admit(self.snapshot, state):
            return
        estimated_itl, load, _ = self._key(instance_id)
        if self.snapshot.itl_max_itl is not None and estimated_itl > self.snapshot.itl_max_itl:
            return
        heapq.heappush(
            self.heap,
            (estimated_itl, load, instance_id, self.versions[instance_id]),
        )

    def _peek_global(self) -> int | None:
        while self.heap:
            _, _, instance_id, version = self.heap[0]
            if version != self.versions[instance_id]:
                heapq.heappop(self.heap)
                continue
            return instance_id
        return None

    def select(self, request: RequestSnapshot) -> int | None:
        if request.eligible_instance_ids is None:
            return self._peek_global()
        candidate_ids = [
            instance_id
            for instance_id in request.eligible_instance_ids
            if instance_id in self.states and _rm_can_admit(self.snapshot, self.states[instance_id])
        ]
        return select_itl_candidate(
            candidates=candidate_ids,
            active_loads={instance_id: self.states[instance_id].route_request_count for instance_id in candidate_ids},
            next_itl=lambda instance_id, load: _compute_itl(
                self.states[instance_id].snapshot.throughput_params,
                0,
                load + 1,
            ),
            max_itl=self.snapshot.itl_max_itl,
        )

    def instance_changed(self, instance_id: int) -> None:
        self.versions[instance_id] += 1
        self._push(instance_id)


def _dispatch(
    context: RoleEvaluationContext,
    states: dict[int, _InstanceState],
    active_ids: frozenset[int],
    request: _PreparedRequest,
    rm_selector: _RMSelectorHeap | None,
) -> bool:
    if context.snapshot.strategy == "throughput_optimal":
        destination = _rollout_select(context, states, active_ids, request)
    elif context.snapshot.strategy == "itl":
        if rm_selector is None:
            raise RuntimeError("RM selector heap was not initialized")
        destination = rm_selector.select(request.snapshot)
    else:
        raise ValueError(f"unsupported request-level routing strategy: {context.snapshot.strategy!r}")
    if destination is None:
        return False

    state = states[destination]
    state.throughput_request_count += 1
    state.throughput_token_count += request.snapshot.seq_len
    state.route_request_count += 1
    if context.snapshot.strategy == "throughput_optimal":
        state.running_count += 1
        state.token_count += request.snapshot.seq_len
    else:
        # SMG reserves worker load at dispatch, before scheduler observations arrive.
        assert rm_selector is not None, "RM selector must be initialized before updating its heap."
        rm_selector.instance_changed(destination)
    return True


def _peek_valid_donor(
    donor_heap: list[tuple[float, int, int]],
    versions: Mapping[int, int],
    movable_requests: Mapping[int, tuple[_PreparedRequest, ...]],
    cursors: dict[int, int],
    migrated_request_ids: set[str],
) -> int | None:
    while donor_heap:
        _, instance_id, version = donor_heap[0]
        requests = movable_requests[instance_id]
        cursor = cursors[instance_id]
        while cursor < len(requests) and requests[cursor].snapshot.request_id in migrated_request_ids:
            cursor += 1
        cursors[instance_id] = cursor
        if version != versions[instance_id] or cursor >= len(requests):
            heapq.heappop(donor_heap)
            continue
        return instance_id
    return None


def _peek_valid_destination_excluding(
    destination_heap: list[tuple[float, int, int]],
    versions: Mapping[int, int],
    excluded_instance_id: int,
) -> int | None:
    held: list[tuple[float, int, int]] = []
    destination: int | None = None
    while destination_heap:
        entry = destination_heap[0]
        _, instance_id, version = entry
        if version != versions[instance_id]:
            heapq.heappop(destination_heap)
            continue
        if instance_id == excluded_instance_id:
            held.append(heapq.heappop(destination_heap))
            continue
        destination = instance_id
        break
    for entry in held:
        heapq.heappush(destination_heap, entry)
    return destination


def _rebalance(
    context: RoleEvaluationContext,
    states: dict[int, _InstanceState],
    donor_ids: set[int],
) -> tuple[RequestMigration, ...]:
    if len(states) <= 1:
        return ()

    movable_requests = {instance_id: context.instance_by_id[instance_id].movable_requests for instance_id in donor_ids}
    cursors = {instance_id: 0 for instance_id in donor_ids}
    versions = {instance_id: 0 for instance_id in states}
    donor_heap = [
        (-_instance_throughput(states[instance_id]), instance_id, 0)
        for instance_id in sorted(donor_ids)
        if movable_requests[instance_id]
    ]
    destination_heap = [(_instance_throughput(state), instance_id, 0) for instance_id, state in states.items()]
    heapq.heapify(donor_heap)
    heapq.heapify(destination_heap)

    moves: list[RequestMigration] = []
    migrated_request_ids: set[str] = set()
    while donor_heap:
        donor_id = _peek_valid_donor(
            donor_heap,
            versions,
            movable_requests,
            cursors,
            migrated_request_ids,
        )
        if donor_id is None:
            break
        destination_id = _peek_valid_destination_excluding(
            destination_heap,
            versions,
            donor_id,
        )
        if destination_id is None:
            break

        request = movable_requests[donor_id][cursors[donor_id]]
        request_snapshot = request.snapshot
        donor = states[donor_id]
        destination = states[destination_id]
        donor_throughput = _instance_throughput(donor)
        destination_throughput = _instance_throughput(destination)
        next_donor_throughput = _throughput_for_load(
            donor,
            donor.throughput_request_count - 1,
            donor.throughput_token_count - request_snapshot.seq_len,
        )
        next_destination_throughput = _throughput_for_load(
            destination,
            destination.throughput_request_count + 1,
            destination.throughput_token_count + request_snapshot.seq_len,
        )
        current_pair_throughput = donor_throughput + destination_throughput
        next_pair_throughput = next_donor_throughput + next_destination_throughput
        if next_pair_throughput <= current_pair_throughput:
            break

        cursors[donor_id] += 1
        migrated_request_ids.add(request_snapshot.request_id)
        donor.throughput_request_count -= 1
        donor.throughput_token_count -= request_snapshot.seq_len
        donor.route_request_count = max(0, donor.route_request_count - 1)
        donor.running_count = max(
            0,
            donor.running_count - (0 if request_snapshot.is_waiting else 1),
        )
        donor.waiting_count = max(
            0,
            donor.waiting_count - (1 if request_snapshot.is_waiting else 0),
        )
        donor.token_count = max(0, donor.token_count - request_snapshot.seq_len)

        destination.throughput_request_count += 1
        destination.throughput_token_count += request_snapshot.seq_len
        destination.route_request_count += 1
        destination.running_count += 1
        destination.token_count += request_snapshot.seq_len
        moves.append(
            RequestMigration(
                request_id=request_snapshot.request_id,
                source_instance_id=donor_id,
                destination_instance_id=destination_id,
            )
        )

        changed_ids = {donor_id, destination_id}
        for instance_id in changed_ids:
            versions[instance_id] += 1
            version = versions[instance_id]
            throughput = _instance_throughput(states[instance_id])
            heapq.heappush(
                destination_heap,
                (throughput, instance_id, version),
            )
            if instance_id in donor_ids:
                requests = movable_requests[instance_id]
                cursor = cursors[instance_id]
                while cursor < len(requests) and requests[cursor].snapshot.request_id in migrated_request_ids:
                    cursor += 1
                cursors[instance_id] = cursor
                if cursor < len(requests):
                    heapq.heappush(
                        donor_heap,
                        (-throughput, instance_id, version),
                    )
    return tuple(moves)


def _iter_requests_to_route(
    context: RoleEvaluationContext,
    sleep_instance_ids: frozenset[int],
) -> tuple[Iterator[_PreparedRequest], int]:
    streams: list[Sequence[_PreparedRequest]] = []
    total = 0
    for instance_id in sorted(sleep_instance_ids):
        instance = context.instance_by_id.get(instance_id)
        if instance is None:
            raise ValueError(f"sleep instance {instance_id} is absent from snapshot")
        if instance.reroute_requests:
            streams.append(instance.reroute_requests)
            total += len(instance.reroute_requests)
    if context.pending_requests:
        streams.append(context.pending_requests)
        total += len(context.pending_requests)
    if not streams:
        return iter(()), 0
    if len(streams) == 1:
        return iter(streams[0]), total
    return heapq.merge(*streams, key=lambda item: item.route_key), total


def evaluate_role_candidate(
    snapshot: RoleEvaluationContext | RoleSnapshot | Mapping[str, Any],
    plan: RoleCandidatePlan,
) -> RoleEvaluationResult:
    """
    Evaluate one role for one candidate using candidate-local load state.
    """
    evaluation_started_s = time.monotonic()
    context = snapshot if isinstance(snapshot, RoleEvaluationContext) else prepare_role_evaluation_context(snapshot)
    before_awake = context.before_awake
    active_ids = frozenset((before_awake | plan.wake_instance_ids) - plan.sleep_instance_ids)
    if not active_ids.issubset(context.instance_by_id):
        raise ValueError(
            f"candidate active instances are invalid: active={sorted(active_ids)} "
            f"known={list(context.all_instance_ids)}"
        )

    states: dict[int, _InstanceState] = {}
    for instance_id in sorted(active_ids):
        prepared = context.instance_by_id[instance_id]
        instance = prepared.snapshot
        was_awake = instance_id in before_awake
        states[instance_id] = _InstanceState(
            prepared=prepared,
            throughput_request_count=prepared.visible_request_count,
            throughput_token_count=prepared.visible_token_count,
            route_request_count=instance.route_request_count if was_awake else 0,
            running_count=instance.running_count if was_awake else 0,
            waiting_count=instance.waiting_count if was_awake else 0,
            token_count=instance.token_count if was_awake else 0,
        )

    moves: tuple[RequestMigration, ...] = ()
    rebalance_started_s: float | None = None
    rebalance_finished_s: float | None = None
    if plan.primary_scale_up:
        rebalance_started_s = time.monotonic()
        moves = _rebalance(context, states, set(before_awake & active_ids))
        rebalance_finished_s = time.monotonic()
    rebalance_simulation_s = (
        rebalance_finished_s - rebalance_started_s
        if rebalance_started_s is not None and rebalance_finished_s is not None
        else 0.0
    )

    router_started_s = time.monotonic()
    requests_to_route, total_to_route = _iter_requests_to_route(
        context,
        plan.sleep_instance_ids,
    )
    if not active_ids:
        router_finished_s = time.monotonic()
        router_simulation_s = router_finished_s - router_started_s
        return RoleEvaluationResult(
            throughput=0.0,
            routed_count=0,
            unrouted_count=total_to_route,
            rebalance_moves=moves,
            instance_throughputs=(),
            rebalance_simulation_s=rebalance_simulation_s,
            router_simulation_s=router_simulation_s,
            other_simulation_s=max(
                0.0,
                time.monotonic() - evaluation_started_s - rebalance_simulation_s - router_simulation_s,
            ),
            rebalance_started_s=rebalance_started_s,
            rebalance_finished_s=rebalance_finished_s,
            router_started_s=router_started_s,
            router_finished_s=router_finished_s,
        )
    rm_selector = _RMSelectorHeap(context.snapshot, states) if context.snapshot.strategy == "itl" else None
    routed = 0
    for request in requests_to_route:
        if _dispatch(context, states, active_ids, request, rm_selector):
            routed += 1

    instance_throughputs = tuple(
        (instance_id, _instance_throughput(states[instance_id])) for instance_id in sorted(states)
    )
    router_finished_s = time.monotonic()
    router_simulation_s = router_finished_s - router_started_s
    return RoleEvaluationResult(
        throughput=sum(value for _, value in instance_throughputs),
        routed_count=routed,
        unrouted_count=total_to_route - routed,
        rebalance_moves=moves,
        instance_throughputs=instance_throughputs,
        rebalance_simulation_s=rebalance_simulation_s,
        router_simulation_s=router_simulation_s,
        other_simulation_s=max(
            0.0,
            time.monotonic() - evaluation_started_s - rebalance_simulation_s - router_simulation_s,
        ),
        rebalance_started_s=rebalance_started_s,
        rebalance_finished_s=rebalance_finished_s,
        router_started_s=router_started_s,
        router_finished_s=router_finished_s,
    )
