"""
Translate native router queues and scheduler observations into solver inputs.
"""

from functools import lru_cache

from psrl.utils.cost_model_path import load_cost_model_json
from psrl.workers.elastic.scaling_policy.evaluation.request_level_candidate_evaluator import (
    InstanceSnapshot,
    RequestSnapshot,
    RoleSnapshot,
)


@lru_cache(maxsize=32)
def _routing_cost_params(path, model, tp, pp):
    payload = load_cost_model_json(path, model)
    key = f"TP{tp}_PP{pp}"
    if payload is None or key not in payload:
        raise ValueError(f"Routing cost model is missing: model={model!r}, parallelism={key!r}.")
    row = payload[key]
    return tuple(
        float(row[key])
        for key in (
            "other_threshold",
            "other_latency_b",
            "other_latency_k",
            "attn_latency_b",
            "attn_latency_k",
        )
    )


def build_role_snapshot(
    role,
    model,
    signals,
    engine_stats,
    pending,
    identity_map,
    routing_config,
    max_model_len,
    tp,
    pp,
    wake_model_version,
):
    """
    Preserve queue order, version eligibility and routing costs for evaluation.
    """
    rollout = role.name == "Rollout"
    versions = {}
    for signal in signals:
        iid = identity_map.decode(signal.instance_id)[2]
        if signal.is_awaken:
            snapshot = engine_stats.get(iid)
            if not snapshot or not isinstance(snapshot.get("elastic_model_version"), int):
                raise ValueError(f"Awake engine version is missing: model={model!r}, instance={iid!r}.")
            versions[signal.instance_id] = snapshot["elastic_model_version"]
        else:
            # Sleeping rollout replicas pull committed PS weights before becoming routing candidates.
            versions[signal.instance_id] = int(wake_model_version)
    cost_params = _routing_cost_params(routing_config.cost_model_path, model, tp, pp)
    instances = []
    for signal in signals:
        solver_id = signal.instance_id
        _, _, iid = identity_map.decode(solver_id)
        stats = engine_stats.get(iid, {}).get("scheduler_stats", {}) if signal.is_awaken else {}
        prompts = stats.get("req_id_to_prompt_token_num", {})
        responses = stats.get("req_id_to_response_token_num", {})
        waiting = set(stats.get("req_id_in_waiting", []))
        requests = tuple(
            RequestSnapshot(
                request_id=str(uid),
                seq_len=int(tokens) + int(responses.get(uid, 0)),
                source_instance_id=solver_id,
                is_waiting=uid in waiting,
                movable=rollout,
                eligible_instance_ids=tuple(i for i, v in versions.items() if v >= versions[solver_id]),
                candidate_priorities=tuple((i, v) for i, v in versions.items()),
            )
            for uid, tokens in prompts.items()
        )
        observed = int(stats.get("num_running_reqs", 0)) + int(stats.get("num_waiting_reqs", 0))
        if len(requests) != observed:
            raise ValueError(f"Incomplete scheduler request snapshot: model={model!r}, instance={iid!r}.")
        instances.append(
            InstanceSnapshot(
                instance_id=solver_id,
                is_awake=signal.is_awaken,
                model_version=versions[solver_id],
                requests=requests,
                route_request_count=observed,
                running_count=signal.running_queue_num,
                waiting_count=signal.waiting_queue_num,
                token_count=sum(request.seq_len for request in requests),
                max_model_len=max_model_len,
                route_cost_params=cost_params,
            )
        )
    pending_requests = []
    for request in pending:
        version = int(request["version_tag"])
        eligible = tuple(
            iid
            for iid, candidate_version in versions.items()
            if not rollout
            or (candidate_version >= version and (version >= 0 or candidate_version in request["reservable_versions"]))
        )
        priorities = []
        for iid, candidate_version in versions.items():
            version_priority = -candidate_version if version < 0 else candidate_version
            if rollout and routing_config.get("candidate_sort_indicator", "version") == "reserve_capability":
                indicator = request["reserve_indicators"][candidate_version]
                priorities.append((iid, (indicator, version_priority)))
            else:
                priorities.append((iid, version_priority))
        fallback = None
        pinned = request.get("pinned_instance")
        if rollout and pinned and routing_config.get("enable_group_sticky", False):
            pinned_id = identity_map.encode(role, model, (str(pinned[0]), int(pinned[1])))
            if pinned_id in eligible:
                fallback, eligible = eligible, (pinned_id,)
        pending_requests.append(
            RequestSnapshot(
                request_id=request["request_id"],
                seq_len=int(request["seq_len"]),
                route_order=int(request["route_order"]),
                eligible_instance_ids=eligible,
                fallback_instance_ids=fallback,
                candidate_priorities=tuple(priorities),
            )
        )
    return RoleSnapshot(
        role=role.name,
        strategy=routing_config.method,
        instances=tuple(instances),
        pending_requests=tuple(pending_requests),
        max_concurrent_requests=routing_config.get("max_concurrent_seqs_per_instance"),
        delta_throughput_threshold=float(routing_config.get("delta_throughput_threshold", 0.0)),
        itl_max_itl=routing_config.get("itl_max_itl"),
    )
