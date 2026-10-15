import argparse
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from psrl.utils.cost_model_path import require_cost_model_json_path

psrl_logger = logging.getLogger(__name__)

WORKERS_UPDATE_STATS_PATH = "/workers/update_stats"
WORKERS_UPDATE_WEIGHT_VERSION_PATH = "/workers/update_weight_version"
WORKERS_STATS_PATH = "/workers/stats"
WORKERS_PAUSE_PATH = "/workers/pause"
WORKERS_RESUME_PATH = "/workers/resume"
ROUTING_LOOP_STATUS_PATH = "/routing_loop/status"
TITO_SESSIONS_PATH = "tito/sessions"
TRAJECTORY_ID_STRATEGIES = frozenset({"auto", "manual"})

CACHE_AWARE_METHODS = frozenset({"cache_aware", "cache_aware_v1"})


class ReplicaRoutingStatus(str, Enum):
    """Routing eligibility state for one rollout replica."""

    ACTIVE = "active"
    PAUSED = "paused"


def is_cache_aware_method(method: str) -> bool:
    return method in CACHE_AWARE_METHODS


def cfg_get(config: Any, path: str, default: Any = None) -> Any:
    node = config
    for part in path.split("."):
        if node is None or not hasattr(node, part):
            return default
        node = getattr(node, part)
    return node if node is not None else default


def get_trajectory_id_strategy(config: Any) -> str:
    """Return the validated TITO trajectory ID strategy configured for PSRL."""
    strategy = str(cfg_get(config, "psrl.rollout_gateway.trajectory_id_strategy", "manual")).lower()
    if strategy not in TRAJECTORY_ID_STRATEGIES:
        choices = ", ".join(sorted(TRAJECTORY_ID_STRATEGIES))
        raise ValueError(f"Invalid trajectory_id_strategy {strategy!r}; expected one of: {choices}.")
    return strategy


def _cache_aware_cfg(config: Any, key: str, default: Any = None) -> Any:
    nested = cfg_get(config, f"psrl.rollout_coordination.routing_strategy.cache_aware_policy.{key}", None)
    if nested is not None:
        return nested
    return default


def _first_nonempty(*values: Any) -> str:
    for value in values:
        if value:
            return str(value)
    return ""


def resolve_router_cost_model_path(cost_model_path: Any, model_name: Any) -> str | None:
    """
    Resolve a router cost-model JSON from a file path or a directory.
    """
    if not cost_model_path:
        return None
    resolved = require_cost_model_json_path(str(cost_model_path), str(model_name or ""))
    psrl_logger.info(f"Resolved cost_model_path {cost_model_path!r} for model {model_name!r} to {resolved!r}.")
    return resolved


def build_rollout_router_args(config: Any, host: str, port: int, ps_manager_addr: str):
    from smg.launch_router import RouterArgs

    routing_method = str(cfg_get(config, "psrl.rollout_coordination.routing_strategy.method", "request_num_balance"))
    request_budget = int(cfg_get(config, "psrl.rollout_coordination.routing_strategy.request_budget", 1024))
    enable_group_sticky = bool(cfg_get(config, "psrl.rollout_coordination.routing_strategy.enable_group_sticky", True))

    # KV-cache transfer when a request is re-routed to a different instance
    # (hint != selected). Independent of coordinator-side imbalance migration.
    kv_transfer_enable = bool(cfg_get(config, "psrl.rollout_coordination.routing_strategy.kv_transfer.enable", False))
    kv_transfer_mode = str(
        cfg_get(config, "psrl.rollout_coordination.routing_strategy.kv_transfer.transfer_mode", "async")
    )
    kv_transfer_timeout_ms = int(
        cfg_get(config, "psrl.rollout_coordination.routing_strategy.kv_transfer.transfer_timeout_ms", 30000)
    )

    psrl_logger.info(
        "[sticky] SMG router args: enable_group_sticky=%s, kv_transfer_enable=%s, kv_transfer_mode=%s",
        enable_group_sticky,
        kv_transfer_enable,
        kv_transfer_mode,
    )

    model_path = cfg_get(config, "train_actor_rollout_ref.model.path", None)
    cost_model_path = resolve_router_cost_model_path(
        cfg_get(config, "psrl.rollout_coordination.routing_strategy.cost_model_path", None),
        _first_nonempty(cfg_get(config, "gen_actor_rollout_ref.model.path", None), model_path),
    )

    cli_args = argparse.Namespace(
        host=host,
        port=port,
        dp_aware=True,
        connection_mode="grpc",
        pd_disaggregation=False,
        prefill=None,
        decode=None,
        policy=routing_method,
        prefill_policy=None,
        decode_policy=None,
        disable_retries=True,
        cache_threshold=float(_cache_aware_cfg(config, "cache_threshold", 0.3)),
        gpu_overlap_weight=float(_cache_aware_cfg(config, "gpu_overlap_weight", 1.0)),
        lmcache_overlap_weight=float(_cache_aware_cfg(config, "lmcache_overlap_weight", 0.5)),
        balance_abs_threshold=int(_cache_aware_cfg(config, "balance_abs_threshold", 64)),
        balance_rel_threshold=float(_cache_aware_cfg(config, "balance_rel_threshold", 1.5)),
        balance_token_usage_threshold=float(_cache_aware_cfg(config, "balance_token_usage_threshold", 1.0)),
        overload_token_usage_threshold=float(_cache_aware_cfg(config, "overload_token_usage_threshold", 1.0)),
        # This flag makes the always-on admission gate reject instances with a
        # nonempty waiting queue.
        enable_kv_admission_control=bool(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.admission_reject_on_waiting", False)
        ),
        kv_capacity_threshold=float(_cache_aware_cfg(config, "kv_capacity_threshold", 1.0)),
        eviction_interval_secs=int(_cache_aware_cfg(config, "eviction_interval_secs", 60)),
        max_tree_size=int(_cache_aware_cfg(config, "max_tree_size", 2**26)),
        block_size=int(_cache_aware_cfg(config, "block_size", 16)),
        max_concurrent_seqs_per_instance=int(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.max_concurrent_seqs_per_instance", 1024)
        ),
        cost_model_path=cost_model_path,
        max_num_waiting_reqs_after_preemption=int(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.max_num_waiting_reqs_after_preemption", 1000)
        ),
        delta_throughput_threshold=float(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.delta_throughput_threshold", 0.5)
        ),
        max_prompt_length=int(
            cfg_get(config, "data.max_prompt_length", cfg_get(config, "rollout.prompt_length", 8192))
        ),
        request_budget=request_budget,
        enable_routing_loop=True,
        routing_loop_check_interval_ms=int(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.check_interval_in_ms", 10)
        ),
        routing_loop_request_sort_key=str(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.request_sort_indicator", "short_length")
        ),
        routing_loop_multi_priority_queue=bool(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.enable_multi_priority_queue", False)
        ),
        routing_loop_dispatch_batch_size=1,
        worker_selection_strategy="psrl",
        psrl_ps_manager_addr=ps_manager_addr,
        psrl_candidate_sort_key=str(
            cfg_get(config, "psrl.rollout_coordination.routing_strategy.candidate_sort_indicator", "version")
        ),
        psrl_enable_group_sticky=enable_group_sticky,
        psrl_kv_transfer_enable=kv_transfer_enable,
        psrl_kv_transfer_mode=kv_transfer_mode,
        psrl_kv_transfer_timeout_ms=kv_transfer_timeout_ms,
        enable_tito=True,
        tito_debug=bool(cfg_get(config, "psrl.rollout_gateway.tito_debug", False)),
        tito_gc_threshold=cfg_get(config, "psrl.rollout_gateway.tito_gc_threshold", None),
        trajectory_id_strategy=get_trajectory_id_strategy(config),
        multimodal_tensor_transport=str(
            cfg_get(config, "psrl.rollout_gateway.multimodal_tensor_transport", "auto")
        ).lower(),
        multimodal_shm_min_bytes=int(cfg_get(config, "psrl.rollout_gateway.multimodal_shm_min_bytes", 64 * 1024)),
        model_path=model_path,
        chat_template=cfg_get(config, "gen_actor_rollout_ref.rollout.chat_template", None),
        service_discovery=False,
        prometheus_port=None,
        request_timeout_secs=2**64 - 1,
        log_level="warn",
        log_dir=cfg_get(config, "psrl.logging_path", None),
        api_key=None,
        disable_health_check=True,
    )
    return RouterArgs.from_cli_args(cli_args, use_router_prefix=False)


def build_reward_router_args(config: Any, host: str, port: int, prometheus_port: int | None, *, rm_config: Any):
    from smg.launch_router import RouterArgs

    model_path = cfg_get(rm_config, "model.path", None)
    if not model_path:
        raise ValueError("Reward gateway requires model.path to load its tokenizer.")
    tokenizer_path = cfg_get(rm_config, "model.tokenizer_path", model_path)
    routing_method = str(cfg_get(rm_config, "routing_strategy.method", "round_robin"))
    max_concurrent_seqs = int(cfg_get(rm_config, "routing_strategy.max_concurrent_seqs_per_instance", 1024))
    request_limit = cfg_get(rm_config, "max_concurrent_requests_per_instance", None)
    if request_limit is not None:
        max_concurrent_seqs = min(max_concurrent_seqs, int(request_limit))
    if max_concurrent_seqs <= 0:
        raise ValueError("Reward routing concurrency must be positive.")
    cost_model_path = cfg_get(rm_config, "routing_strategy.cost_model_path", None)
    if routing_method in {"itl", "throughput_optimal", "throughput_optimal_with_budget"} and not cost_model_path:
        raise ValueError(f"Reward routing policy {routing_method!r} requires routing_strategy.cost_model_path.")
    cost_model_path = resolve_router_cost_model_path(
        cost_model_path,
        _first_nonempty(cfg_get(rm_config, "reward_model_name", None), model_path),
    )

    cli_args = argparse.Namespace(
        host=host,
        port=port,
        dp_aware=True,
        connection_mode="grpc",
        pd_disaggregation=False,
        prefill=None,
        decode=None,
        policy=routing_method,
        prefill_policy=None,
        decode_policy=None,
        disable_retries=True,
        max_concurrent_seqs_per_instance=max_concurrent_seqs,
        cost_model_path=cost_model_path,
        itl_max_latency=cfg_get(config, "psrl.deployment.elastic.throughput_optimal.rm_router_max_itl", None),
        max_num_waiting_reqs_after_preemption=int(
            cfg_get(rm_config, "routing_strategy.max_num_waiting_reqs_after_preemption", 1000)
        ),
        delta_throughput_threshold=float(cfg_get(rm_config, "routing_strategy.delta_throughput_threshold", 0.5)),
        max_prompt_length=int(cfg_get(rm_config, "rollout.max_model_len", 32768)),
        request_budget=int(cfg_get(rm_config, "routing_strategy.request_budget", 1024)),
        enable_routing_loop=True,
        routing_loop_check_interval_ms=int(cfg_get(rm_config, "routing_strategy.check_interval_in_ms", 10)),
        routing_loop_request_sort_key=str(cfg_get(rm_config, "routing_strategy.request_sort_indicator", "small_id")),
        routing_loop_multi_priority_queue=bool(
            cfg_get(rm_config, "routing_strategy.enable_multi_priority_queue", False)
        ),
        routing_loop_dispatch_batch_size=1,
        worker_selection_strategy="naive",
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        chat_template=cfg_get(rm_config, "rollout.chat_template", None),
        # The gateway registers its tokenizer under `reward_model_name` before accepting requests.
        disable_tokenizer_autoload=True,
        service_discovery=False,
        prometheus_port=prometheus_port,
        request_timeout_secs=2**64 - 1,
        log_level="warn",
        log_dir=cfg_get(config, "psrl.logging_path", None),
        api_key=None,
        disable_health_check=True,
    )
    return RouterArgs.from_cli_args(cli_args, use_router_prefix=False)


def build_worker_registration_payload(
    *,
    url: str,
    model_id: str,
    role: str,
    fuse_rollout_with_validate: bool,
    max_model_len: int,
    dp_size: int,
    tp_size: int,
    pp_size: int,
    kv_block_size: int | None = None,
    lmcache_instance_id: str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    if role in ("rollout", "validate"):
        request_scope = "all" if fuse_rollout_with_validate else ("validate" if role == "validate" else "train")
    elif role.startswith("reward_model_"):
        request_scope = "train"
    else:
        raise ValueError(f"Unsupported inference worker role {role!r}.")
    labels = {
        "psrl_request_scope": request_scope,
        "max_model_len": str(max_model_len),
        "dp_size": str(dp_size),
        "tp_size": str(tp_size),
        "pp_size": str(pp_size),
    }
    # KV block size lets SMG's event-driven router seed a provisional block size
    # before the first KV event arrives (kv_event_monitor falls back to this).
    if kv_block_size:
        labels["kv_block_size"] = str(kv_block_size)
    # SMG addresses cross-instance transfers by LMCache instance ID. The source
    # servicer resolves per-rank peer URLs from its registry.
    if lmcache_instance_id:
        labels["lmcache_instance_id"] = lmcache_instance_id
    payload = {
        "url": url,
        "worker_type": "regular",
        "connection_mode": "grpc",
        "runtime_type": "vllm",
        "models": [{"id": model_id}],
        "labels": labels,
    }
    # A stable worker ID aligns SMG route traces with local statistics files.
    if worker_id is not None:
        payload["id"] = worker_id
    return payload


def build_pause_resume_payload(instance_ids: list[Any]) -> list[dict[str, Any]]:
    payload = []
    for instance_id in instance_ids:
        if isinstance(instance_id, tuple):
            base_worker_id, dp_rank = instance_id
            payload.append({"base_worker_id": base_worker_id, "dp_rank": dp_rank})
        else:
            payload.append({"base_worker_id": instance_id})
    return payload


def build_weight_version_updates(instance_ids: list[tuple[str, int]], weight_version: int) -> list[dict[str, Any]]:
    return [
        {"worker_id": base_worker_id, "dp_rank": dp_rank, "weight_version": weight_version}
        for base_worker_id, dp_rank in instance_ids
    ]


def build_worker_stats_update(worker_id: str, dp_rank: int, snapshot: dict[str, Any]) -> dict[str, Any]:
    scheduler_stats = snapshot.get("scheduler_stats", {}) if isinstance(snapshot, dict) else {}
    return {
        "worker_id": worker_id,
        "dp_rank": dp_rank,
        "timestamp": _parse_timestamp(snapshot.get("timestamp") if isinstance(snapshot, dict) else None),
        "scheduler_stats": {
            "req_id_to_prompt_token_num": scheduler_stats.get("req_id_to_prompt_token_num", {}),
            "req_id_to_response_token_num": scheduler_stats.get("req_id_to_response_token_num", {}),
            "num_running_reqs": int(scheduler_stats.get("num_running_reqs", 0)),
            "num_waiting_reqs": int(scheduler_stats.get("num_waiting_reqs", 0)),
            "kv_cache_usage": float(scheduler_stats.get("kv_cache_usage", 0.0)),
        },
    }


def _parse_timestamp(value):
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat().replace("+00:00", "Z")
        except ValueError:
            pass
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
