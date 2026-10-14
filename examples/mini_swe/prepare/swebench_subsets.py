"""
SWE-bench / SWE-smith-py subset sampling utilities.

Provides repo-balanced subsampling so that no single repository dominates the
training distribution. All sampling is deterministic given a fixed seed.
"""

from __future__ import annotations

import logging
import os
import random
import re
from collections import defaultdict
from typing import Any

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


# --- Image name helpers ---


def get_swebench_image_name(swe_problem: dict[str, Any]) -> str:
    """
    Compute the Docker image name for a SWE-bench or SWE-smith SWE problem.

    For SWE-smith-py rows the `image_name` field is already populated.
    For SWE-bench Verified / Lite / Full it must be derived from `instance_id`
    using the `__` → `_1776_` escape rule.

    Args:
        swe_problem (dict[str, Any]): A single dataset row describing one
            SWE problem.

    Returns:
        str: Fully-qualified Docker image name (lowercase, no tag if already
            present in the field, otherwise appended with `:latest`).
    """
    image_name: str | None = swe_problem.get("image_name") or swe_problem.get("docker_image")
    if image_name:
        return image_name.lower()
    swe_problem_id: str = swe_problem["instance_id"]
    id_safe = swe_problem_id.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_safe}:latest".lower()


def get_repo_key(swe_problem: dict[str, Any]) -> str:
    """
    Extract the canonical repository key from a SWE problem row.

    SWE-smith rows have `repo` like ``swesmith/oauthlib__oauthlib.1fd52536``.
    SWE-bench rows have `repo` like ``django/django``.
    We normalise to a shorter stable key for grouping.

    Args:
        swe_problem (dict[str, Any]): A single dataset row.

    Returns:
        str: Short repository key used for balanced sampling.
    """
    repo: str = swe_problem.get("repo", swe_problem.get("instance_id", "unknown"))
    # For SWE-smith: "swesmith/oauthlib__oauthlib.1fd52536" → "oauthlib__oauthlib"
    # Drop the "swesmith/" prefix and the commit-hash suffix ".xxxxxxxx".
    repo = repo.removeprefix("swesmith/")
    repo = re.sub(r"\.[0-9a-f]{8}$", "", repo)
    return repo


# --- Subset sampling ---


def repo_balanced_sample(
    swe_problems: list[dict[str, Any]],
    *,
    total: int,
    per_repo_k: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Return a repo-balanced subset of at most `total` SWE problems.

    Algorithm:
      1. Group all SWE problems by their repo key.
      2. If `per_repo_k` is given, cap each group at that size.
      3. Round-robin across groups (alphabetical order for determinism) until
         `total` SWE problems are collected or all groups are exhausted.

    Args:
        swe_problems (list[dict[str, Any]]): Full pool of dataset rows.
        total (int): Maximum number of SWE problems to return.
        per_repo_k (int | None): Hard cap per repo before round-robin.
            If None, no per-repo cap is applied (only the global `total` cap).
        seed (int): Random seed for within-group shuffling.

    Returns:
        list[dict[str, Any]]: Balanced subset, length <= `total`.
    """
    assert total > 0, "Total must be a positive integer."

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prob in swe_problems:
        groups[get_repo_key(prob)].append(prob)

    # Shuffle within each group.
    for key in groups:
        rng.shuffle(groups[key])

    # Apply per-repo cap if requested.
    if per_repo_k is not None:
        assert per_repo_k > 0, "per_repo_k must be a positive integer."
        groups = {k: v[:per_repo_k] for k, v in groups.items()}

    # Round-robin in alphabetical key order for full determinism.
    sorted_keys = sorted(groups.keys())
    pools = [groups[k] for k in sorted_keys]
    result: list[dict[str, Any]] = []
    i = 0
    while len(result) < total and any(pools):
        pool = pools[i % len(pools)]
        if pool:
            result.append(pool.pop(0))
        i += 1

    psrl_logger.info(f"repo_balanced_sample: requested={total}, returned={len(result)}, repos={len(sorted_keys)}.")
    return result


def filter_by_spec(
    swe_problems: list[dict[str, Any]],
    spec: str,
) -> list[dict[str, Any]]:
    """
    Filter SWE problems by Python slice syntax or an `instance_id` regex.

    Args:
        swe_problems (list[dict[str, Any]]): Full pool of dataset rows.
        spec (str): Filter specification string.

    Returns:
        list[dict[str, Any]]: Filtered list.
    """
    if not spec:
        return swe_problems

    # Slice spec: digits, colon, optional digits (and optional second colon + step).
    if re.match(r"^-?\d*:-?\d*(:-?\d*)?$", spec):
        parts = [int(x) if x else None for x in spec.split(":")]
        return swe_problems[slice(*parts)]

    # Regex spec: match against instance_id (the HF dataset field name).
    pattern = re.compile(spec)
    filtered = [prob for prob in swe_problems if pattern.search(prob["instance_id"])]
    psrl_logger.info(f"Filtered by regex={spec!r}. Input count: {len(swe_problems)}. Output count: {len(filtered)}.")
    return filtered
