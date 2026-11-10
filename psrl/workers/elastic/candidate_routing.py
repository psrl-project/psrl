"""
Side-effect-free selectors shared by live routers and candidate simulation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Real
from typing import Any


def priority_key(value: Any) -> Any:
    """
    Normalize JSON/list priorities while preserving numeric ordering.
    """
    if isinstance(value, list):
        return tuple(priority_key(item) for item in value)
    if isinstance(value, tuple):
        return tuple(priority_key(item) for item in value)
    if value is None:
        return (0, "")
    if isinstance(value, Real):
        return (1, float(value))
    return (2, str(value))


def resolve_candidate_model_versions(
    current_versions: Mapping[int, int],
    sleeping_instance_ids: Iterable[int],
    current_ps_model_version: int,
) -> dict[int, int]:
    """
    Resolve the model version each instance would have in a candidate.

    A sleeping rollout instance is synchronized to the current PS version when
    it is woken. Awake instances keep their currently loaded version.
    """
    sleeping_ids = {int(instance_id) for instance_id in sleeping_instance_ids}
    ps_version = int(current_ps_model_version)
    return {
        int(instance_id): (ps_version if int(instance_id) in sleeping_ids else int(model_version))
        for instance_id, model_version in current_versions.items()
    }


def select_throughput_optimal_candidate(
    *,
    candidates: Sequence[int],
    candidate_indicators: Sequence[Any],
    can_run_directly: Callable[[int], bool],
    current_throughput: Callable[[int], float],
    next_throughput: Callable[[int], float],
    baseline_delta_throughput: Callable[[int], float],
    within_request_cap: Callable[[int], bool],
    delta_throughput_threshold: float,
) -> int | None:
    """
    Mirror throughput-optimal grouping, threshold and stable tie semantics.
    """
    if len(candidates) != len(candidate_indicators):
        raise ValueError("candidates and candidate_indicators must have the same length")
    groups = group_candidates_by_indicator(candidates, candidate_indicators)
    return select_throughput_optimal_from_groups(
        candidate_groups=groups,
        can_run_directly=can_run_directly,
        current_throughput=current_throughput,
        next_throughput=next_throughput,
        baseline_delta_throughput=baseline_delta_throughput,
        within_request_cap=within_request_cap,
        delta_throughput_threshold=delta_throughput_threshold,
    )


def group_candidates_by_indicator(
    candidates: Sequence[int],
    candidate_indicators: Sequence[Any],
) -> tuple[tuple[int, ...], ...]:
    """
    Return stable candidate groups ordered by normalized indicator.
    """
    if len(candidates) != len(candidate_indicators):
        raise ValueError("candidates and candidate_indicators must have the same length")
    groups: dict[Any, list[int]] = {}
    for candidate, indicator in zip(candidates, candidate_indicators, strict=True):
        groups.setdefault(priority_key(indicator), []).append(int(candidate))
    return tuple(tuple(groups[indicator]) for indicator in sorted(groups))


def select_throughput_optimal_from_groups(
    *,
    candidate_groups: Sequence[Sequence[int]],
    can_run_directly: Callable[[int], bool],
    current_throughput: Callable[[int], float],
    next_throughput: Callable[[int], float],
    baseline_delta_throughput: Callable[[int], float],
    within_request_cap: Callable[[int], bool],
    delta_throughput_threshold: float,
) -> int | None:
    """
    Select from already ordered groups without regrouping or sorting.
    """
    for group in candidate_groups:
        if not group:
            continue
        threshold = baseline_delta_throughput(group[0]) * float(delta_throughput_threshold)
        best_candidate: int | None = None
        best_delta = float("-inf")
        for candidate in group:
            if not can_run_directly(candidate):
                continue
            delta = next_throughput(candidate) - current_throughput(candidate)
            # Strict comparison preserves input-order tie-breaking.
            if delta > best_delta:
                best_delta = delta
                best_candidate = candidate
        if best_candidate is not None and best_delta >= threshold and within_request_cap(best_candidate):
            return best_candidate
    return None


def select_itl_candidate(
    *,
    candidates: Iterable[int],
    active_loads: Mapping[int, int],
    next_itl: Callable[[int, int], float],
    max_itl: float | None,
) -> int | None:
    """
    Return the minimum ``(next_itl, load, instance_id)`` candidate.
    """
    best_key: tuple[float, int, int] | None = None
    best_instance: int | None = None
    for raw_instance_id in candidates:
        instance_id = int(raw_instance_id)
        load = int(active_loads.get(instance_id, 0))
        estimated_itl = float(next_itl(instance_id, load))
        if max_itl is not None and estimated_itl > max_itl:
            continue
        key = (estimated_itl, load, instance_id)
        if best_key is None or key < best_key:
            best_key = key
            best_instance = instance_id
    return best_instance
