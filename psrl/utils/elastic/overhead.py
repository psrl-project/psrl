"""
Helpers for elastic scaling and request-migration overhead logging.
"""

from __future__ import annotations

import time
from typing import Any


def extract_vllm_reprefill_s(result: Any) -> float | None:
    """
    Return vLLM scheduled-to-first-token time from a generated DataProto.
    """
    if result is None:
        return None
    meta_info = getattr(result, "meta_info", {}) or {}
    metrics = meta_info.get("vllm_metrics")
    if metrics is None:
        return None
    if hasattr(metrics, "tolist"):
        metrics = metrics.tolist()
    if isinstance(metrics, (list, tuple)):
        metrics = metrics[0] if metrics else None
    if metrics is None:
        return None

    if isinstance(metrics, dict):
        if metrics.get("prefill_time") is not None:
            try:
                return max(0.0, float(metrics["prefill_time"]))
            except (TypeError, ValueError):
                return None
        scheduled_ts = metrics.get("scheduled_ts", 0.0)
        first_token_ts = metrics.get("first_token_ts", 0.0)
    else:
        if getattr(metrics, "prefill_time", None) is not None:
            try:
                return max(0.0, float(metrics.prefill_time))
            except (TypeError, ValueError):
                return None
        scheduled_ts = getattr(metrics, "scheduled_ts", 0.0)
        first_token_ts = getattr(metrics, "first_token_ts", 0.0)
    try:
        scheduled_ts = float(scheduled_ts)
        first_token_ts = float(first_token_ts)
    except (TypeError, ValueError):
        return None
    if scheduled_ts <= 0.0 or first_token_ts < scheduled_ts:
        return None
    return first_token_ts - scheduled_ts


class RequestMigrationOverheadTracker:
    """
    Track non-overlapping interrupt, redispatch, and re-prefill overhead.
    """

    def __init__(self) -> None:
        self._requests: dict[str, dict[str, Any]] = {}

    def mark_batch(self, instance_to_uids: dict, context: dict | None) -> None:
        if not context:
            return
        selected_count = max(1, int(context.get("selected_count", 1)))
        started_s = time.monotonic()
        for source_instance_id, uids in instance_to_uids.items():
            normalized_uids = uids if isinstance(uids, (list, tuple, set)) else [uids]
            for uid in normalized_uids:
                if uid is None:
                    continue
                request_context = {
                    **context,
                    "source_instance_id": int(source_instance_id),
                    "selected_count": selected_count,
                    "abort_started_s": started_s,
                    "interrupt_completed_s": None,
                    "interrupt_s": None,
                    "requeued": False,
                    "network_s": None,
                }
                self._requests[str(uid)] = request_context

    def _lookup_key(self, request_id: Any) -> str | None:
        """
        Resolve a router logical ID to its unique scheduler request ID.

        vLLM scheduler stats expose IDs such as ``<logical-id>-<uuid>``, while
        router callbacks retain only ``<logical-id>``. Prefer exact matching.
        use the prefix form only when it identifies one tracked request.
        """
        request_key = str(request_id)
        if request_key in self._requests:
            return request_key
        matches = [key for key in self._requests if key.startswith(f"{request_key}-")]
        if not matches:
            return None
        # Select the newest migration when repeated dispatch gives a request multiple scheduler IDs.
        return max(
            matches,
            key=lambda key: float(self._requests[key].get("abort_started_s", 0.0)),
        )

    def mark_requeued(self, request_id: Any) -> dict[str, Any] | None:
        key = self._lookup_key(request_id)
        context = self._requests.get(key) if key is not None else None
        if context is None or context["requeued"]:
            return None
        interrupted_s = time.monotonic()
        context["requeued"] = True
        context["interrupt_completed_s"] = interrupted_s
        context["interrupt_s"] = max(
            0.0,
            interrupted_s - context["abort_started_s"],
        )
        return dict(context)

    def mark_dispatched(
        self,
        request_id: Any,
        destination_instance_id: int,
    ) -> dict[str, Any] | None:
        key = self._lookup_key(request_id)
        context = self._requests.get(key) if key is not None else None
        if context is None or not context["requeued"] or context["network_s"] is not None:
            return None
        dispatched_s = time.monotonic()
        context["destination_instance_id"] = int(destination_instance_id)
        context["network_s"] = max(
            0.0,
            dispatched_s - context["interrupt_completed_s"],
        )
        context["abort_to_redispatch_s"] = max(
            0.0,
            dispatched_s - context["abort_started_s"],
        )
        return dict(context)

    def complete_with_status(
        self,
        request_id: Any,
        result: Any,
    ) -> tuple[dict[str, Any] | None, str]:
        """
        Complete a tracked migration and explain why no sample was emitted.
        """
        key = self._lookup_key(request_id)
        context = self._requests.get(key) if key is not None else None
        if context is None:
            return None, "not_tracked"
        if context["network_s"] is None:
            return dict(context), "not_redispatched"
        reprefill_s = extract_vllm_reprefill_s(result)
        if reprefill_s is None:
            return dict(context), "missing_vllm_prefill"
        self.discard(request_id)
        context["reprefill_s"] = reprefill_s
        context["migration_s"] = context["interrupt_s"] + context["network_s"] + reprefill_s
        return context, "completed"

    def complete(self, request_id: Any, result: Any) -> dict[str, Any] | None:
        context, status = self.complete_with_status(request_id, result)
        return context if status == "completed" else None

    def discard(self, request_id: Any) -> None:
        """
        Drop terminal tracking state that cannot produce a complete sample.
        """
        request_key = str(request_id)
        matching_keys = [key for key in self._requests if key == request_key or key.startswith(f"{request_key}-")]
        for key in matching_keys:
            self._requests.pop(key, None)


def interval_union_s(intervals: list[tuple[float, float]]) -> float:
    """
    Return wall time covered by at least one monotonic-clock interval.
    """
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return 0.0
    current_start, current_end = valid[0]
    total = 0.0
    for start, end in valid[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start
