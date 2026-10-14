"""Save and load per-rank Megatron checkpoints with plain tensors.

Sharded wrappers are replaced with payloads before `torch.save` to avoid DCP
metadata gathering. Save and load must use identical parallel configurations.
"""

import json
import logging
import os

import torch
from megatron.core import mpu
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedBase,
)

logger = logging.getLogger(__name__)

_METADATA_FILE = "parallel_config.json"
_KNOWN_FORMATS = frozenset({"per_rank_plain_tensors", "per_rank_torch_save"})


def _assert_no_sharded_objects(obj, _path="root"):
    """Recursively assert that *obj* contains no Megatron checkpoint wrappers.

    Raises ``AssertionError`` with a descriptive path on the first violation.
    Call after ``_extract_plain_state_dict`` (save path) and after
    ``_unwrap_sharded_state_dict`` (load path) to catch any missed wrappers.
    """
    if isinstance(obj, (ShardedBase, LocalNonpersistentObject)):
        raise AssertionError(
            f"Unexpected sharded wrapper at {_path!r}: {type(obj).__name__}. "
            f"All wrappers should have been extracted before this point."
        )
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_sharded_objects(v, f"{_path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _assert_no_sharded_objects(v, f"{_path}[{i}]")


def _extract_plain_state_dict(obj):
    """Replace Megatron checkpoint wrappers with their payloads before saving."""
    if isinstance(obj, dict):
        return {k: _extract_plain_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_extract_plain_state_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_extract_plain_state_dict(v) for v in obj)
    # `ShardedTensorFactory` subclasses `ShardedBase`, so this branch handles both.
    if isinstance(obj, ShardedBase):
        return obj.data
    if isinstance(obj, LocalNonpersistentObject):
        return obj.obj
    return obj


def _unwrap_sharded_state_dict(obj):
    """Unwrap checkpoint wrappers, including shard lists emitted by `apply_factories`."""
    if isinstance(obj, dict):
        return {k: _unwrap_sharded_state_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        # Dense factory outputs use shard lists that merge with `torch.cat`.
        if obj and all(isinstance(x, ShardedBase) for x in obj):
            return torch.cat([x.data for x in obj])
        return [_unwrap_sharded_state_dict(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_unwrap_sharded_state_dict(v) for v in obj)
    if isinstance(obj, ShardedBase):
        return obj.data
    if isinstance(obj, LocalNonpersistentObject):
        return obj.obj
    return obj


def save_megatron_checkpoint(sharded_state_dict, ckpt_path, async_save=False):
    """Save a Megatron sharded state dict using per-rank ``torch.save``.

    Args:
        sharded_state_dict: Megatron sharded state dict (may contain
            ``ShardedTensorFactory``, ``ShardedTensor``, ``ShardedObject``,
            ``LocalNonpersistentObject``).
        ckpt_path (str): Directory to save checkpoint files into.
        async_save (bool): Unused and retained for API compatibility.
    """
    assert not async_save, "async_save is not supported by save_megatron_checkpoint"

    plain_state_dict = _extract_plain_state_dict(sharded_state_dict)
    _assert_no_sharded_objects(plain_state_dict)

    os.makedirs(ckpt_path, exist_ok=True)
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    save_path = os.path.join(ckpt_path, f"rank_{rank}.pt")
    torch.save(plain_state_dict, save_path)
    assert os.path.exists(save_path), f"torch.save appeared to succeed, but path={save_path!r} was not found on disk."

    if rank == 0:
        metadata = {
            "format": "per_rank_plain_tensors",
            "world_size": world_size,
            "tp_size": mpu.get_tensor_model_parallel_world_size(),
            "pp_size": mpu.get_pipeline_model_parallel_world_size(),
            "dp_size": mpu.get_data_parallel_world_size(),
        }
        with open(os.path.join(ckpt_path, _METADATA_FILE), "w") as f:
            json.dump(metadata, f, indent=2)

    torch.distributed.barrier()
    logger.info("[Rank %d] Saved per-rank checkpoint to %s", rank, save_path)
    return None


def load_megatron_checkpoint(sharded_state_dict, ckpt_dir):  # noqa: ARG001
    """Load a per-rank Megatron checkpoint from *ckpt_dir*.

    Validates the parallel config stored in ``parallel_config.json`` and
    asserts that the loaded state dict contains no residual Megatron wrappers.

    Args:
        sharded_state_dict: Unused and retained for API compatibility with
            ``load_dist_checkpointing``.
        ckpt_dir (str): Directory containing ``rank_<N>.pt`` files and
            ``parallel_config.json``.

    Returns:
        State dict with plain tensors and objects (no Megatron wrappers).
    """
    rank = torch.distributed.get_rank()
    rank_path = os.path.join(ckpt_dir, f"rank_{rank}.pt")

    assert os.path.exists(rank_path), (
        f"Per-rank checkpoint not found at {rank_path!r}. "
        f"Only the per-rank format saved by save_megatron_checkpoint is supported."
    )

    metadata_path = os.path.join(ckpt_dir, _METADATA_FILE)
    assert os.path.exists(metadata_path), (
        f"Metadata file not found at {metadata_path!r}. Checkpoint directory may be corrupt or incomplete."
    )

    with open(metadata_path) as f:
        metadata = json.load(f)

    fmt = metadata.get("format")
    assert fmt in _KNOWN_FORMATS, f"Unknown checkpoint format {fmt!r}. Expected one of {sorted(_KNOWN_FORMATS)}."

    saved_ws = metadata.get("world_size")
    current_ws = torch.distributed.get_world_size()
    assert saved_ws == current_ws, (
        f"Checkpoint world_size={saved_ws} does not match current world_size={current_ws}. "
        f"Cannot load per-rank checkpoint with a different parallel config."
    )

    raw_state_dict = torch.load(rank_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_sharded_state_dict(raw_state_dict)
    _assert_no_sharded_objects(state_dict)

    logger.info("[Rank %d] Loaded per-rank checkpoint from %s", rank, rank_path)
    return state_dict
