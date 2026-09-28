"""
SciBuddy reward extraction.

SciBuddy verifiers write a scalar reward to /logs/verifier/reward.txt (0 or 1).
Harbor converts that to a `reward` key in the trial's reward dict.
This module provides the offline scoring function for batch rollout output.
"""

from __future__ import annotations


def compute_score(data_source: str, solution: str, ground_truth: str, extra_info: dict) -> dict:
    """Compute the offline reward for a recorded SciBuddy episode.

    The reward comes from Harbor's verifier output, captured in
    `extra_fields.harbor_rewards` by the agent loop. This function applies
    the same extraction the agent loop uses at runtime so offline scoring
    is consistent.

    Args:
        data_source: Dataset identifier (unused, kept for API compatibility).
        solution: Model response text (unused, verifier outcome is in extra_info).
        ground_truth: Ground truth (unused, verifier is the authority).
        extra_info: Merged dict of extra_info and extra_fields from the rollout record.

    Returns:
        dict with a `score` key in [0.0, 1.0].
    """
    harbor_rewards = extra_info.get("harbor_rewards") or {}
    score = float(harbor_rewards.get("reward", 0.0))
    return {"score": score, "reward": score}
