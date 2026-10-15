"""
Resolve and evaluate fitted per-instance latency models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from psrl.utils.cost_model_path import load_cost_model_json


@dataclass(frozen=True)
class ITLModelParams:
    """
    Coefficients for the per-instance ITL latency model.
    """

    A: float = 0.0
    B: float = 1.0
    C: float = 1.0
    D: float = 0.0


def compute_itl(params: ITLModelParams, total_token_num: float, running_queue_num: float) -> float:
    """
    Compute per-instance ITL latency proxy in seconds.
    """
    x = max(0.0, float(running_queue_num))
    total_tokens = max(0.0, float(total_token_num))
    return max(float(params.A) * total_tokens + max(float(params.B), float(params.C) * x) + float(params.D), 1e-9)


def compute_itl_delta(params: ITLModelParams, running_queue_num: float, total_token_num: float = 0.0) -> float:
    """
    Compute the ITL increment from routing one more request to an instance.
    """
    x = max(0.0, float(running_queue_num))
    total_tokens = max(0.0, float(total_token_num))
    current = float(params.A) * total_tokens + max(float(params.B), float(params.C) * x) + float(params.D)
    next_value = float(params.A) * total_tokens + max(float(params.B), float(params.C) * (x + 1.0)) + float(params.D)
    return next_value - current


def compute_throughput(
    params: ITLModelParams,
    total_token_num: float,
    running_queue_num: float,
) -> float:
    """
    Estimate instance throughput as queue length divided by ITL.

    Idle instances use the reciprocal of the intercept ITL so a sleeper still
    has a positive service-rate prior.
    """
    queue = max(0.0, float(running_queue_num))
    latency = compute_itl(params, total_token_num, queue)
    if queue <= 0.0:
        return 1.0 / latency
    return queue / latency


def _cost_config_has_model(cost_config: dict[str, Any] | None) -> bool:
    if not cost_config:
        return False
    if cost_config.get("cost_model_path"):
        return True
    inline = cost_config.get("model_params")
    return isinstance(inline, dict) and bool(inline)


_TP_PP_KEY_RE = re.compile(r"^TP\d+_PP\d+$", re.IGNORECASE)


def format_tp_pp_key(tensor_parallel: int = 1, pipeline_parallel: int = 1) -> str:
    """
    Format a cost-model JSON bucket key such as ``TP1_PP1``.
    """
    return f"TP{int(tensor_parallel)}_PP{int(pipeline_parallel)}"


def _read_nested_config_value(config: Any, *keys: str, default: Any = None) -> Any:
    current = config
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if not hasattr(current, key):
            return default
        current = getattr(current, key)
    return current


def _read_config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except TypeError:
            pass
    return getattr(config, key, default)


def resolve_tp_pp_for_role_model(config: Any, role_name: Any, model_name: str) -> str | None:
    """
    Derive ``TP{tp}_PP{pp}`` from rollout parallel settings for a role/model pair.
    """
    if config is None:
        return None
    role_key = getattr(role_name, "name", str(role_name)).lower()
    if role_key == "rollout":
        rollout_cfg = _read_nested_config_value(config, "gen_actor_rollout_ref", "rollout")
        if rollout_cfg is None:
            return None
        tp = int(_read_config_value(rollout_cfg, "tensor_model_parallel_size", 1) or 1)
        pp = int(_read_config_value(rollout_cfg, "pipeline_model_parallel_size", 1) or 1)
        return format_tp_pp_key(tp, pp)
    if role_key == "rewardmodel":
        reward_config = _read_config_value(config, "reward")
        manager_configs = _read_config_value(reward_config, "managers", {})
        active_names = _read_config_value(reward_config, "active_managers", [])
        reward_models = [manager_configs[name] for name in active_names if name in manager_configs]
        for reward_cfg in reward_models:
            reward_model_name = _read_config_value(reward_cfg, "reward_model_name")
            model_path = _read_nested_config_value(reward_cfg, "model", "path")
            stem = str(model_path).rstrip("/").split("/")[-1] if model_path else None
            if reward_model_name not in (None, model_name) and stem != model_name:
                continue
            rollout_cfg = _read_config_value(reward_cfg, "rollout", reward_cfg)
            tp = int(_read_config_value(rollout_cfg, "tensor_model_parallel_size", 1) or 1)
            pp = int(_read_config_value(rollout_cfg, "pipeline_model_parallel_size", 1) or 1)
            return format_tp_pp_key(tp, pp)
    return None


def _lookup_tp_pp_params_entry(payload: dict[str, Any], tp_pp: str | None) -> dict[str, Any] | None:
    if tp_pp and isinstance(payload.get(tp_pp), dict):
        return payload[tp_pp]
    for key in sorted(payload):
        if _TP_PP_KEY_RE.match(str(key)) and isinstance(payload[key], dict):
            return payload[key]
    return None


def _lookup_params_entry(
    payload: dict[str, Any],
    candidates: tuple[str, ...],
    role_key: str,
    model_name: str,
    *,
    tp_pp: str | None = None,
) -> dict[str, Any] | None:
    """
    Return the first matching parameter dict inside a cost-model payload.
    """
    tp_pp_entry = _lookup_tp_pp_params_entry(payload, tp_pp)
    if tp_pp_entry is not None:
        return tp_pp_entry
    for key in candidates:
        entry = payload.get(key)
        if isinstance(entry, dict):
            nested = _lookup_tp_pp_params_entry(entry, tp_pp)
            return nested or entry
    roles = payload.get("roles")
    if isinstance(roles, dict):
        for role_candidate in (role_key, role_key.lower()):
            role_entry = roles.get(role_candidate)
            if not isinstance(role_entry, dict):
                continue
            model_entry = role_entry.get(model_name)
            if isinstance(model_entry, dict):
                nested = _lookup_tp_pp_params_entry(model_entry, tp_pp)
                return nested or model_entry
            if all(k in role_entry for k in ("A", "B", "C", "D")):
                return role_entry
    models = payload.get("models")
    if isinstance(models, dict):
        model_entry = models.get(model_name)
        if isinstance(model_entry, dict):
            nested = _lookup_tp_pp_params_entry(model_entry, tp_pp)
            return nested or model_entry
    if all(k in payload for k in ("A", "B", "C", "D")):
        return payload
    return None


def _coerce_params(entry: dict[str, Any]) -> ITLModelParams | None:
    """
    Parse A/B/C/D or legacy latency field names from a config dict.
    """

    def _get(primary: str, legacy: str | None = None, default: float = 0.0) -> float:
        if primary in entry:
            return float(entry[primary])
        if legacy is not None and legacy in entry:
            return float(entry[legacy])
        return float(default)

    try:
        return ITLModelParams(
            A=_get("A", "attn_latency_k", 0.0),
            B=_get("B", "other_threshold", 1.0),
            C=_get("C", "other_latency_k", 1.0),
            D=_get("D", "attn_latency_b", 0.0),
        )
    except (TypeError, ValueError):
        return None


def resolve_cost_model_params(
    role_name: Any,
    model_name: str,
    cost_config: dict[str, Any] | None = None,
    tp_pp: str | None = None,
    config: Any = None,
) -> ITLModelParams:
    """
    Resolve cost-model ITL coefficients for a role/model pair.

    ``cost_config["cost_model_path"]`` may be a JSON file or a directory of
    ``{stem}.json`` files. ``cost_config["model_params"]`` is an inline override.
    """
    cost_cfg = cost_config or {}
    resolved_tp_pp = tp_pp or resolve_tp_pp_for_role_model(config, role_name, model_name)
    payloads: list[dict[str, Any]] = []
    inline = cost_cfg.get("model_params", {})
    if isinstance(inline, dict):
        payloads.append(inline)
    cost_model_path = cost_cfg.get("cost_model_path")
    if cost_model_path:
        loaded = load_cost_model_json(str(cost_model_path), model_name)
        if loaded is not None:
            payloads.append(loaded)

    role_key = getattr(role_name, "name", str(role_name))
    role_key_lower = role_key.lower()
    candidates = (
        f"{role_key}:{model_name}",
        f"{role_key_lower}:{model_name}",
        model_name,
        role_key,
        role_key_lower,
        "default",
    )
    for payload in payloads:
        entry = _lookup_params_entry(payload, candidates, role_key, model_name, tp_pp=resolved_tp_pp)
        if entry is None:
            continue
        params = _coerce_params(entry)
        if params is not None:
            return params
    return ITLModelParams()


def estimate_instance_throughput(
    role_name: Any,
    model_name: str,
    running_queue_num: float,
    total_token_num: float,
    cost_config: dict[str, Any] | None = None,
    tp_pp: str | None = None,
    config: Any = None,
    fallback: float | None = None,
) -> float:
    """
    Estimate per-instance throughput from the cost model.

    Args:
        role_name: Rollout or reward-model role used to pick TP/PP.
        model_name (str): Model name used to resolve ``cost_model_path``.
        running_queue_num (float): Current running-queue length.
        total_token_num (float): Current token occupancy.
        cost_config (dict[str, Any] | None): Section with ``cost_model_path`` and optional ``model_params``.
        tp_pp (str | None): Explicit ``TP{n}_PP{n}`` bucket. Derived from ``config`` when omitted.
        config (Any): Full trainer config used to derive TP/PP when needed.
        fallback (float | None): Runtime throughput used when no cost model is configured.

    Returns:
        float: Estimated requests per second.
    """
    if not _cost_config_has_model(cost_config):
        return max(0.0, float(fallback or 0.0))
    params = resolve_cost_model_params(
        role_name=role_name,
        model_name=model_name,
        cost_config=cost_config,
        tp_pp=tp_pp,
        config=config,
    )
    return compute_throughput(params, total_token_num, running_queue_num)
