import hashlib
import json
import logging
import time
from collections.abc import Mapping
from typing import Any

import torch
from omegaconf import OmegaConf

_PRECISION_SENSITIVE_PATTERNS = ("linear_attn.norm.weight", "A_log")


def resolve_weight_fingerprint_options(
    psrl_config: Any,
    *,
    flow: str,
    model_version: int,
) -> dict[str, Any] | None:
    """Resolve the configured fingerprint mode for one model-version event."""
    if not weight_fingerprint_flow_enabled(psrl_config, flow=flow):
        return None
    config = OmegaConf.select(psrl_config, "nixl.weight_verification", default=None)
    assert config is not None, "Enabled weight fingerprinting requires a weight_verification configuration."

    full_every = int(config.get("full_every_n_versions", 0))
    sample_every = int(config.get("sample_every_n_versions", 1))
    if full_every > 0 and model_version % full_every == 0:
        mode = "full"
    elif sample_every > 0 and model_version % sample_every == 0:
        mode = "sampled"
    else:
        return None

    sample_count = int(config.get("sample_count_per_tensor", 16))
    chunk_bytes = int(config.get("full_hash_chunk_bytes", 64 * 1024**2))
    if sample_count <= 0:
        raise ValueError("nixl.weight_verification.sample_count_per_tensor must be positive")
    if chunk_bytes <= 0:
        raise ValueError("nixl.weight_verification.full_hash_chunk_bytes must be positive")
    return {
        "mode": mode,
        "sample_count": sample_count,
        "chunk_bytes": chunk_bytes,
        "include_tensor_digests": bool(config.get("include_tensor_digests", True)),
        "fail_on_mismatch": bool(config.get("fail_on_mismatch", False)),
    }


def weight_fingerprint_flow_enabled(psrl_config: Any, *, flow: str) -> bool:
    config = OmegaConf.select(psrl_config, "nixl.weight_verification", default=None)
    return bool(config is not None and config.get("enable", False) and config.get(flow, False))


def _logical_name(key: str, shard_idx: tuple[int, ...]) -> str:
    return f"{key}|{tuple(shard_idx)}"


def _mapping_metadata(
    tensor_mapping: Mapping[tuple[str, tuple[int, ...]], torch.Tensor],
) -> tuple[str, str]:
    mapping_hasher = hashlib.sha256()
    address_hasher = hashlib.sha256()
    for (key, shard_idx), tensor in sorted(tensor_mapping.items(), key=lambda item: item[0]):
        metadata = (
            f"{_logical_name(key, shard_idx)}\0{tensor.dtype}\0{tuple(tensor.shape)}\0"
            f"{tuple(tensor.stride())}\0{tensor.storage_offset()}\0{tensor.numel()}"
        ).encode()
        mapping_hasher.update(metadata)
        address_hasher.update(metadata)
        address_hasher.update(f"\0{tensor.data_ptr()}".encode())
    return mapping_hasher.hexdigest(), address_hasher.hexdigest()


def tensor_mapping_signature(
    tensor_mapping: Mapping[tuple[str, tuple[int, ...]], torch.Tensor],
) -> dict[str, Any]:
    if not tensor_mapping:
        raise RuntimeError("NIXL tensor mapping is empty; register tensors before fingerprinting")
    mapping_digest, address_digest = _mapping_metadata(tensor_mapping)
    return {
        "mapping_digest": mapping_digest,
        "address_digest": address_digest,
        "logical_tensor_count": len(tensor_mapping),
        "total_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensor_mapping.values()),
    }


def _sample_positions(numel: int, sample_count: int) -> list[int]:
    if numel <= sample_count:
        return list(range(numel))
    if sample_count == 1:
        return [0]
    return [index * (numel - 1) // (sample_count - 1) for index in range(sample_count)]


def _sample_tensor(tensor: torch.Tensor, positions: list[int]) -> torch.Tensor:
    detached = tensor.detach()
    if detached.ndim == 0:
        return detached.reshape(1)
    if detached.is_contiguous():
        index = torch.tensor(positions, device=detached.device, dtype=torch.long)
        return detached.reshape(-1).index_select(0, index)

    coordinates: list[list[int]] = [[] for _ in detached.shape]
    for position in positions:
        remainder = position
        coordinate = [0] * detached.ndim
        for dim in range(detached.ndim - 1, -1, -1):
            coordinate[dim] = remainder % detached.shape[dim]
            remainder //= detached.shape[dim]
        for dim, value in enumerate(coordinate):
            coordinates[dim].append(value)
    indices = tuple(torch.tensor(values, device=detached.device, dtype=torch.long) for values in coordinates)
    return detached[indices]


def _tensor_content_digest(
    key: str,
    shard_idx: tuple[int, ...],
    tensor: torch.Tensor,
    *,
    mode: str,
    sample_count: int,
    chunk_bytes: int,
) -> str:
    metadata = (f"{_logical_name(key, shard_idx)}\0{tensor.dtype}\0{tuple(tensor.shape)}\0{tensor.numel()}").encode()
    hasher = hashlib.sha256(metadata)
    if mode == "sampled":
        positions = _sample_positions(tensor.numel(), sample_count)
        hasher.update(json.dumps(positions, separators=(",", ":")).encode())
        sampled = _sample_tensor(tensor, positions).contiguous().view(torch.uint8).reshape(-1).cpu()
        hasher.update(sampled.numpy().tobytes())
        return hasher.hexdigest()
    if mode != "full":
        raise ValueError(f"Unsupported weight fingerprint mode: {mode!r}")

    tensor_bytes = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    for offset in range(0, tensor_bytes.numel(), chunk_bytes):
        length = min(chunk_bytes, tensor_bytes.numel() - offset)
        cpu_chunk = tensor_bytes.narrow(0, offset, length).cpu()
        hasher.update(cpu_chunk.numpy().tobytes())
    return hasher.hexdigest()


def fingerprint_tensor_mapping(
    tensor_mapping: Mapping[tuple[str, tuple[int, ...]], torch.Tensor],
    *,
    mode: str = "sampled",
    sample_count: int = 16,
    chunk_bytes: int = 64 * 1024**2,
) -> dict[str, Any]:
    """Fingerprint the logical contents of a NIXL registered-tensor mapping."""
    if not tensor_mapping:
        raise RuntimeError("NIXL tensor mapping is empty; register tensors before fingerprinting")
    if sample_count <= 0 or chunk_bytes <= 0:
        raise ValueError("sample_count and chunk_bytes must be positive")

    started_at = time.perf_counter()
    global_hasher = hashlib.sha256()
    tensor_digests: dict[str, str] = {}
    for (key, shard_idx), tensor in sorted(tensor_mapping.items(), key=lambda item: item[0]):
        digest = _tensor_content_digest(
            key,
            shard_idx,
            tensor,
            mode=mode,
            sample_count=sample_count,
            chunk_bytes=chunk_bytes,
        )
        logical_name = _logical_name(key, shard_idx)
        tensor_digests[logical_name] = digest
        global_hasher.update(logical_name.encode())
        global_hasher.update(bytes.fromhex(digest))

    result = tensor_mapping_signature(tensor_mapping)
    result.update(
        {
            "mode": mode,
            "sample_count_per_tensor": sample_count if mode == "sampled" else None,
            "digest": global_hasher.hexdigest(),
            "tensor_digests": tensor_digests,
            "elapsed_s": time.perf_counter() - started_at,
        }
    )
    return result


def compare_weight_fingerprints(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    expected_tensors = expected.get("tensor_digests", {})
    actual_tensors = actual.get("tensor_digests", {})
    differing = sorted(
        key
        for key in expected_tensors.keys() | actual_tensors.keys()
        if expected_tensors.get(key) != actual_tensors.get(key)
    )
    return {
        "match": expected.get("digest") == actual.get("digest"),
        "mapping_match": expected.get("mapping_digest") == actual.get("mapping_digest"),
        "address_match": expected.get("address_digest") == actual.get("address_digest"),
        "differing_tensor_count": len(differing),
        "differing_tensors": differing,
    }


def fingerprint_log_record(
    fingerprint: Mapping[str, Any],
    *,
    flow: str,
    stage: str,
    role: str,
    model_version: int,
    rank: int,
    include_tensor_digests: bool,
    instance_id: int | None = None,
    tp_rank: int | None = None,
    client_name: str | None = None,
) -> dict[str, Any]:
    record = {
        "flow": flow,
        "stage": stage,
        "role": role,
        "model_version": int(model_version),
        "rank": int(rank),
        "instance_id": instance_id,
        "tp_rank": tp_rank,
        "client_name": client_name,
        **fingerprint,
    }
    if not include_tensor_digests:
        record.pop("tensor_digests", None)
    return record


def precision_sensitive_parameter_stats(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, dict[str, object]]:
    """Summarize small tensors that are sensitive to parameter-sync transforms."""
    stats: dict[str, dict[str, object]] = {}
    for pattern in _PRECISION_SENSITIVE_PATTERNS:
        matches = [(key, tensor) for key, tensor in state_dict.items() if pattern in key]
        if not matches:
            continue
        values = torch.cat([tensor.detach().float().reshape(-1) for _, tensor in matches])
        stats[pattern] = {
            "tensor_count": len(matches),
            "element_count": values.numel(),
            "dtypes": sorted({str(tensor.dtype) for _, tensor in matches}),
            "min": values.min().item(),
            "max": values.max().item(),
            "mean": values.mean().item(),
        }
    return stats


def report_precision_sensitive_parameters(
    state_dict: dict[str, torch.Tensor],
    *,
    logger: logging.Logger,
    role: str,
    rank: int,
    pull_times: int,
    tp_rank: int | None = None,
) -> dict[str, dict[str, object]]:
    """Log precision stats for GDN tensors after a weight pull."""
    stats = precision_sensitive_parameter_stats(state_dict)
    if not stats:
        return stats
    if tp_rank is None:
        logger.info(
            "[WEIGHT_SYNC_PRECISION] role=%s rank=%d pull=%d stats=%s.",
            role,
            rank,
            pull_times,
            stats,
        )
    else:
        logger.info(
            "[WEIGHT_SYNC_PRECISION] role=%s rank=%d tp_rank=%d pull=%d stats=%s.",
            role,
            rank,
            tp_rank,
            pull_times,
            stats,
        )
    return stats


def assert_positive_standard_gamma_mean(
    stats: dict[str, dict[str, object]],
    *,
    rank: int,
) -> None:
    """Reject a pull whose linear-attn gamma still looks zero-centered."""
    gamma_stats = stats.get("linear_attn.norm.weight")
    if gamma_stats is None or gamma_stats["mean"] > 0:
        return
    raise RuntimeError(
        "Qwen3.5 actor weight pull produced non-positive standard gamma mean: "
        f"rank={rank}, stats={gamma_stats!r}. "
        "The FSDP/HF actor must not apply Megatron zero-centered-gamma conversion."
    )
