"""Resolve cost-model JSON paths from a file, directory, or model name."""

from __future__ import annotations

import json
import os
import re
from typing import Any


def _dedupe_stems(stems: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for stem in stems:
        if not stem or stem in seen:
            continue
        seen.add(stem)
        ordered.append(stem)
    return ordered


def _extract_model_size(parts: list[str]) -> str | None:
    for part in reversed(parts):
        m = re.match(r"^(\d+)\s*b", part, re.IGNORECASE)
        if m:
            return f"{m.group(1)}b"
    return None


def _extract_qwen_family(lower: str, parts: list[str]) -> str | None:
    if any("moe" in p for p in parts):
        return "qwen_moe"
    if "qwen2.5" in lower or (len(parts) >= 2 and parts[0] == "qwen2" and parts[1] == "5"):
        return "qwen2.5"
    if lower.startswith("qwen3") or any(p.startswith("qwen3") for p in parts):
        return "qwen3"
    if any(p.startswith("qwen") for p in parts):
        return "qwen"
    return None


def model_name_to_cost_model_stems(model_name: str) -> list[str]:
    """
    Map a HuggingFace-style model name to candidate cost-model JSON stems.

    Unified rule: ``{family}_{size}``, where family is one of
    ``qwen2.5``, ``qwen3``, ``qwen_moe``, or ``qwen``.

    Examples:
        ``Qwen2.5-7B`` -> ``qwen2.5_7b``
        ``Qwen3-8B`` -> ``qwen3_8b``
        ``Qwen-MoE-30B-A3B`` -> ``qwen_moe_30b``
        ``Qwen-14B`` -> ``qwen_14b``
    """
    name = str(model_name).strip()
    if not name:
        return []

    lower = name.lower()
    parts = [p for p in re.split(r"[-_.\s/]+", lower) if p]
    stems: list[str] = []

    model_basename = lower.rstrip("/").rsplit("/", 1)[-1]
    if model_basename == "qwen2.5-32b":
        stems.append("qwen_32b")
    elif model_basename == "qwen3.5-122b-a10b":
        stems.append("qwen3.5_122b_a10b")
    else:
        family = _extract_qwen_family(lower, parts)
        size = _extract_model_size(parts)
        if family and size:
            stems.append(f"{family}_{size}")

    norm = re.sub(r"[^a-z0-9.]+", "_", lower).strip("_")
    if norm:
        stems.append(norm)

    return _dedupe_stems(stems)


def _lookup_cost_model_json_in_dir(directory: str, model_name: str) -> str | None:
    for stem in model_name_to_cost_model_stems(model_name):
        candidate = os.path.join(directory, f"{stem}.json")
        if os.path.isfile(candidate):
            return candidate

    norm_model = re.sub(r"[^a-z0-9]+", "", model_name.lower())
    if not norm_model:
        return None

    best_path: str | None = None
    best_score = 0
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith(".json") or "deprecated" in entry.lower():
            continue
        stem = entry[:-5]
        norm_stem = re.sub(r"[^a-z0-9]+", "", stem.lower())
        if not norm_stem:
            continue
        if norm_stem in norm_model or norm_model in norm_stem:
            score = len(norm_stem)
            if score > best_score:
                best_score = score
                best_path = os.path.join(directory, entry)
    return best_path


def resolve_cost_model_json_path(cost_model_path: str, model_name: str) -> str | None:
    """
    Resolve a cost-model JSON file from a JSON path or a directory.

    A JSON file path is returned as-is. A directory is searched for ``{stem}.json``
    using :func:`model_name_to_cost_model_stems`, then the best filename match.
    """
    if not cost_model_path:
        return None

    path = os.path.expanduser(str(cost_model_path))
    if os.path.isdir(path):
        return _lookup_cost_model_json_in_dir(path, model_name)
    return path


def require_cost_model_json_path(cost_model_path: str, model_name: str) -> str:
    """
    Resolve a cost-model JSON path and raise if a directory has no matching file.

    Args:
        cost_model_path (str): JSON file path, or a directory of cost-model files.
        model_name (str): Model name used to pick ``{stem}.json`` in a directory.

    Returns:
        str: Resolved JSON file path.

    Raises:
        ValueError: If the path is empty, or a directory has no matching file.
    """
    resolved = resolve_cost_model_json_path(cost_model_path, model_name)
    if not resolved:
        raise ValueError(f"cost_model_path {cost_model_path!r} does not resolve for model {model_name!r}.")
    return resolved


def load_cost_model_json(cost_model_path: str, model_name: str) -> dict[str, Any] | None:
    """Load a cost-model JSON dict from a file path or model-named file in a directory."""
    json_path = resolve_cost_model_json_path(cost_model_path, model_name)
    if not json_path:
        return None
    try:
        with open(json_path, encoding="utf-8") as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None
