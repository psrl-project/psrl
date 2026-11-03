"""Reward function that carries the verifier verdict into the PSRL reward loop."""

from __future__ import annotations

from typing import Any


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any = "",
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Return the score the agent loop recorded for one episode.

    The verifier call happens in the episode finalizer, which owns the proof,
    the receipt and the ungraded decision. This function only projects that
    metadata into the reward loop. It never calls the verifier and it never
    turns an infrastructure failure into a training zero.

    Args:
        data_source (str): Dataset name. Unused.
        solution_str (str): Decoded trajectory text. Unused.
        ground_truth (Any): Ground truth. Unused because the source has no solutions.
        extra_info (dict[str, Any] | None): Merged `extra_info` and `agent_reward_info`.
        **kwargs (Any): Framework arguments. Ignored.

    Returns:
        dict[str, Any]: A reward dictionary with ``score`` and diagnostics.
    """
    info = extra_info or {}
    verdict = str(info.get("verifier_verdict") or "unknown")
    graded = bool(info.get("verifier_graded", False))
    verifier_error = info.get("verifier_error")
    if verifier_error or not graded:
        return {
            "score": 0.0,
            "acc": 0.0,
            "verifier_verdict": "error" if verifier_error else verdict,
            "verifier_reason": "",
            "verifier_error": str(verifier_error or "ungraded"),
        }
    score = float(info.get("reward_score", 0.0))
    score = min(1.0, max(0.0, score))
    return {
        "score": score,
        "acc": score,
        "verifier_verdict": verdict,
        "verifier_reason": str(info.get("verifier_reason") or "")[:2000],
        "verifier_run_id": str(info.get("verifier_run_id") or ""),
        "proof_sha256": str(info.get("proof_sha256") or ""),
        "verifier_error": "",
    }
