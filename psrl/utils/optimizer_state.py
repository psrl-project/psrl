"""Small optimizer-state checks used at trainer sleep boundaries."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch


def _iter_tensors(value: Any, path: str) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            yield from _iter_tensors(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            yield from _iter_tensors(nested, f"{path}[{index}]")


def _iter_optimizer_state_tensors(optimizer) -> Iterator[tuple[str, torch.Tensor]]:
    """Cover torch optimizers plus Megatron wrappers without importing Megatron."""
    chained_optimizers = getattr(optimizer, "chained_optimizers", None)
    if chained_optimizers is not None:
        for index, nested_optimizer in enumerate(chained_optimizers):
            for path, tensor in _iter_optimizer_state_tensors(nested_optimizer):
                yield f"chained_optimizers[{index}].{path}", tensor
        return

    state_owner = getattr(optimizer, "optimizer", optimizer)
    state = getattr(state_owner, "state", None)
    if state is not None:
        for state_index, state_value in enumerate(state.values()):
            yield from _iter_tensors(state_value, f"state[{state_index}]")

    master_shards = getattr(optimizer, "shard_fp32_from_float16_groups", None)
    if master_shards is not None:
        yield from _iter_tensors(master_shards, "shard_fp32_from_float16_groups")


def summarize_optimizer_state_devices(optimizer) -> dict[str, Any]:
    """Summarize tensor placement without reading optimizer tensor contents."""
    device_tensor_counts: dict[str, int] = {}
    non_cpu_tensors: list[dict[str, Any]] = []
    tensor_count = 0
    total_bytes = 0

    if optimizer is not None:
        for path, tensor in _iter_optimizer_state_tensors(optimizer):
            device = str(tensor.device)
            tensor_count += 1
            total_bytes += tensor.numel() * tensor.element_size()
            device_tensor_counts[device] = device_tensor_counts.get(device, 0) + 1
            if tensor.device.type != "cpu":
                non_cpu_tensors.append(
                    {
                        "path": path,
                        "device": device,
                        "shape": tuple(tensor.shape),
                        "dtype": str(tensor.dtype),
                    }
                )

    return {
        "tensor_count": tensor_count,
        "total_bytes": total_bytes,
        "device_tensor_counts": device_tensor_counts,
        "non_cpu_tensor_count": len(non_cpu_tensors),
        "non_cpu_tensors": non_cpu_tensors,
    }


def assert_optimizer_state_offloaded(optimizer, *, stage: str, rank: int) -> dict[str, Any]:
    """Fail before TMS sleep if any persistent optimizer tensor is not on CPU."""
    summary = summarize_optimizer_state_devices(optimizer)
    if summary["non_cpu_tensor_count"]:
        samples = summary["non_cpu_tensors"][:16]
        raise RuntimeError(
            "Optimizer offload invariant failed before trainer sleep/wake: "
            f"stage={stage}, rank={rank}, non_cpu_tensor_count={summary['non_cpu_tensor_count']}, "
            f"samples={samples}."
        )
    return summary
