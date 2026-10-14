"""
AIRS-Bench reward: the published normalized score.

The task metric itself is computed inside the episode, because MLGym runs each
task's own `evaluate.py` in the sandbox when the agent submits. This module only
converts that raw metric into a bounded, cross-task comparable reward using the
formula from the AIRS-Bench paper.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# Distance floor used at the optimum, where the log would otherwise diverge.
PHI_EPS = 1e-12


def phi(score: float, optimal: float, eps: float = PHI_EPS) -> float:
    """
    Apply the AIRS-Bench non-linear transform to one raw metric value.

    The transform takes the negative base 10 logarithm of the distance from the
    optimum. The distance is floored at `eps` so the result stays finite.

    Args:
        score (float): Raw metric value.
        optimal (float): Best achievable value for this metric.
        eps (float): Distance floor.

    Returns:
        float: Transformed value, larger meaning better.
    """
    distance = max(abs(score - optimal), eps)
    return -math.log10(distance)


def normalized_score(score: float, sota: float, worst: float, optimal: float) -> float:
    """
    Convert one raw metric value into a reward in [0, 1].

    Zero corresponds to the worst score observed across the published agents, and
    one corresponds to human SOTA. Values outside that band are clipped, so beating
    SOTA yields exactly 1.0.

    Args:
        score (float): Raw metric value achieved by the agent.
        sota (float): Published SOTA value for this task.
        worst (float): Worst observed value for this task.
        optimal (float): Best achievable value for this metric.

    Returns:
        float: Reward in [0, 1].
    """
    phi_score = phi(score, optimal)
    phi_sota = phi(sota, optimal)
    phi_worst = phi(worst, optimal)

    denominator = phi_sota - phi_worst
    if abs(denominator) < PHI_EPS:
        # NOTE(claude): A degenerate range means SOTA and the worst observation are
        # indistinguishable after the transform, so the only meaningful question is
        # whether the agent reached SOTA at all.
        psrl_logger.warning(f"Using a binary reward for degenerate range sota={sota!r}, worst={worst!r}.")
        return 1.0 if phi_score >= phi_sota else 0.0

    return min(1.0, max(0.0, (phi_score - phi_worst) / denominator))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict,
    **kwargs: Any,
) -> dict:
    """
    Score one AIRS-Bench trajectory.

    The agent may submit more than once, because MLGym appends one entry to
    `AgentInfo.score` per submit. The final submission is the agent's answer, so it
    is the one scored. Rewarding the best of several submissions would instead
    reward shotgunning.

    Args:
        data_source (str): Dataset tag, always `airs_bench` for this recipe.
        solution_str (str): Unused. The metric comes from in-episode evaluation.
        ground_truth (Any): Unused. Hidden labels live inside the sandbox.
        extra_info (dict): Per-row constants plus the episode outcome fields
            `airs_scores`, `airs_submit_count`, and `airs_validate_count`.
        **kwargs (Any): Ignored framework extras.

    Returns:
        dict: `score` in [0, 1] plus `reward_extra_info` diagnostics.
    """
    task_id = extra_info.get("airs_task_id", "unknown")
    metric_key = extra_info.get("metric", "")
    diagnostics: dict[str, Any] = {
        "airs_task_id": task_id,
        "airs_metric": metric_key,
        "airs_raw_metric": None,
        "airs_submit_count": int(extra_info.get("airs_submit_count", 0) or 0),
        "airs_validate_count": int(extra_info.get("airs_validate_count", 0) or 0),
        "airs_zero_reason": "",
    }

    def zero(reason: str) -> dict:
        diagnostics["airs_zero_reason"] = reason
        psrl_logger.info(f"Scored task={task_id!r} at 0.0. Reason: {reason!r}.")
        return {"score": 0.0, "reward_extra_info": diagnostics}

    eval_error = extra_info.get("airs_eval_error", "")
    if eval_error:
        diagnostics["airs_eval_error"] = eval_error
        return zero("eval_error")

    scores = extra_info.get("airs_scores") or []
    if not scores:
        return zero("no_submission")

    last_score = scores[-1]
    if not isinstance(last_score, dict) or metric_key not in last_score:
        diagnostics["airs_available_keys"] = sorted(last_score) if isinstance(last_score, dict) else []
        return zero("metric_key_missing")

    try:
        raw_metric = float(last_score[metric_key])
    except (TypeError, ValueError):
        return zero("invalid_score")
    if not math.isfinite(raw_metric):
        return zero("invalid_score")

    diagnostics["airs_raw_metric"] = raw_metric
    try:
        reward = normalized_score(
            raw_metric,
            sota=float(extra_info["sota_score"]),
            worst=float(extra_info["estimated_worst_score"]),
            optimal=float(extra_info["optimal_score"]),
        )
    except (KeyError, ValueError, TypeError):
        return zero("invalid_score")
    diagnostics["airs_normalized_score"] = reward
    return {"score": reward, "reward_extra_info": diagnostics}
