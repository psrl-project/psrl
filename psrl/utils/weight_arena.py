"""Pack model weight storages into a bounded number of real allocations."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass, replace

import torch
from torch import nn

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor


def gb_to_bytes(value: int | float | str, *, field_name: str, allow_zero: bool = False) -> int:
    """Convert a GB config value to binary bytes."""
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative number, got {value!r}.")
    try:
        gb = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number, got {value!r}.") from exc
    size_bytes = gb * 1024**3
    minimum_invalid = gb < 0 if allow_zero else gb <= 0
    if not math.isfinite(gb) or minimum_invalid or not size_bytes.is_integer():
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field_name} must be a {qualifier} GB value resolving to whole bytes, got {value!r}.")
    return int(size_bytes)


@dataclass(frozen=True)
class WeightArenaStats:
    arena_count: int
    unique_storage_count: int
    tensor_binding_count: int
    total_storage_bytes: int
    total_arena_bytes: int
    largest_storage_bytes: int
    pack_seconds: float
    tms_source_reserved_bytes: int = 0
    tms_source_active_bytes: int = 0
    tms_source_active_allocations: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class WeightArenaPlacement:
    arena_index: int
    offset_bytes: int
    nbytes: int
    binding_names: tuple[str, ...]


@dataclass
class WeightArenaHandle:
    arenas: tuple[torch.Tensor, ...]
    placements: tuple[WeightArenaPlacement, ...]
    stats: WeightArenaStats
    base_addresses: tuple[int, ...]

    def assert_virtual_addresses_unchanged(self) -> tuple[int, ...]:
        current_addresses = tuple(arena.data_ptr() for arena in self.arenas)
        if current_addresses != self.base_addresses:
            raise RuntimeError(
                "Weight arena virtual addresses changed after initialization: "
                f"expected {self.base_addresses}, got {current_addresses}."
            )
        return current_addresses

    def assert_module_weights_in_arena(self, module: nn.Module) -> None:
        """Verify that every non-empty parameter/buffer still uses an arena."""
        self.assert_virtual_addresses_unchanged()
        arena_storages = {
            (str(arena.device), arena.untyped_storage().data_ptr(), arena.untyped_storage().nbytes())
            for arena in self.arenas
        }
        outside_bindings: list[str] = []
        binding_count = 0
        for name, tensor in _collect_module_bindings(module):
            binding_count += 1
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in arena_storages:
                outside_bindings.append(name)
        if outside_bindings:
            raise RuntimeError(
                "Model weights escaped the weight arena after materialization: "
                f"count={len(outside_bindings)} first_bindings={outside_bindings[:8]}."
            )
        # Empty tensors have no registered storage and are intentionally not
        # returned by _collect_module_bindings; a loader may materialize one
        # later without changing the NIXL-visible weight set. New non-empty
        # bindings still cannot be introduced without being detected above.
        if binding_count > self.stats.tensor_binding_count:
            raise RuntimeError(
                "Model weight binding count changed after arena materialization: "
                f"expected {self.stats.tensor_binding_count}, got {binding_count}."
            )


@dataclass
class FSDP2WeightArenaHandle(WeightArenaHandle):
    fsdp_param_placements: tuple[tuple[object, int, int], ...]
    _post_load_hooks_removed: bool = False

    def prepare_state_dict_load(self) -> None:
        if self._post_load_hooks_removed:
            raise RuntimeError("FSDP2 weight arena post-load hooks were already removed.")
        for fsdp_param, _, _ in self.fsdp_param_placements:
            hook = getattr(fsdp_param, "_post_load_hook_handle", None)
            if hook is None:
                raise RuntimeError(
                    f"FSDP2 parameter {getattr(fsdp_param, '_param_fqn', '<unknown>')} "
                    "is missing its state-dict post-load hook."
                )
            hook.remove()
        self._post_load_hooks_removed = True

    def finish_state_dict_load(self, *, succeeded: bool) -> None:
        if not self._post_load_hooks_removed:
            raise RuntimeError("FSDP2 weight arena post-load hooks were not removed.")
        for fsdp_param, _, _ in self.fsdp_param_placements:
            module = fsdp_param._module_info.module

            def _reset_sharded_param(*args, _fsdp_param=fsdp_param, **kwargs):
                _fsdp_param.reset_sharded_param()

            fsdp_param._post_load_hook_handle = module.register_load_state_dict_post_hook(_reset_sharded_param)
        self._post_load_hooks_removed = False
        if succeeded:
            self.assert_fsdp_bindings_unchanged()

    def assert_fsdp_bindings_unchanged(self) -> None:
        self.assert_virtual_addresses_unchanged()
        for fsdp_param, arena_index, offset_bytes in self.fsdp_param_placements:
            padded_data = fsdp_param._sharded_param_data
            expected_addr = self.arenas[arena_index].data_ptr() + offset_bytes
            if padded_data.numel() > 0 and padded_data.data_ptr() != expected_addr:
                raise RuntimeError(
                    f"FSDP2 parameter {getattr(fsdp_param, '_param_fqn', '<unknown>')} escaped its arena: "
                    f"expected data_ptr={expected_addr}, got {padded_data.data_ptr()}."
                )
            local_tensor = fsdp_param.sharded_param._local_tensor
            if local_tensor.untyped_storage().data_ptr() != self.arenas[arena_index].data_ptr():
                raise RuntimeError(
                    f"FSDP2 parameter {getattr(fsdp_param, '_param_fqn', '<unknown>')} local tensor "
                    "no longer shares its arena storage."
                )


def _synchronize_arena_devices(arenas: tuple[torch.Tensor, ...]) -> None:
    devices = {arena.device for arena in arenas if arena.device.type == "cuda"}
    for device in devices:
        torch.cuda.synchronize(device)


@torch.no_grad()
def snapshot_weight_arena_to_cpu(
    handle: WeightArenaHandle,
    *,
    pin_memory: bool,
) -> tuple[torch.Tensor, ...]:
    """Create a persistent CPU mirror in the arena's final storage layout."""
    handle.assert_virtual_addresses_unchanged()
    cached_arenas = tuple(torch.empty_like(arena, device="cpu", pin_memory=pin_memory) for arena in handle.arenas)
    for cached, arena in zip(cached_arenas, handle.arenas, strict=True):
        cached.copy_(arena, non_blocking=pin_memory and arena.device.type == "cuda")
    _synchronize_arena_devices(handle.arenas)
    return cached_arenas


@torch.no_grad()
def restore_weight_arena_from_cpu(
    handle: WeightArenaHandle,
    cached_arenas: tuple[torch.Tensor, ...],
) -> int:
    """Restore a CPU arena mirror directly into the final model storages."""
    handle.assert_virtual_addresses_unchanged()
    if len(cached_arenas) != len(handle.arenas):
        raise RuntimeError(
            f"Weight arena cache count changed: expected {len(handle.arenas)}, got {len(cached_arenas)}."
        )

    total_bytes = 0
    for index, (arena, cached) in enumerate(zip(handle.arenas, cached_arenas, strict=True)):
        if cached.device.type != "cpu":
            raise RuntimeError(f"Weight arena cache {index} must be on CPU, got {cached.device}.")
        if cached.shape != arena.shape or cached.dtype != arena.dtype:
            raise RuntimeError(
                f"Weight arena cache {index} changed layout: expected shape={arena.shape} dtype={arena.dtype}, "
                f"got shape={cached.shape} dtype={cached.dtype}."
            )
        arena.copy_(cached, non_blocking=cached.is_pinned() and arena.device.type == "cuda")
        total_bytes += cached.numel() * cached.element_size()
    _synchronize_arena_devices(handle.arenas)
    return total_bytes


@dataclass
class _TensorBinding:
    name: str
    tensor: torch.Tensor
    storage_offset_bytes: int
    size: torch.Size
    stride: tuple[int, ...]


@dataclass
class _StorageGroup:
    key: tuple[str, int, int]
    nbytes: int
    bindings: list[_TensorBinding]
    arena_index: int = -1
    arena_offset_bytes: int = -1


@dataclass
class _ChunkPlan:
    device: torch.device
    groups: list[_StorageGroup]
    nbytes: int = 0


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _validate_config(max_chunk_bytes: int, alignment_bytes: int) -> None:
    if alignment_bytes <= 0 or alignment_bytes & (alignment_bytes - 1):
        raise ValueError(f"alignment_bytes must be a positive power of two, got {alignment_bytes}.")
    if max_chunk_bytes <= 0:
        raise ValueError(f"max_chunk_bytes must be positive, got {max_chunk_bytes}.")
    if max_chunk_bytes % alignment_bytes != 0:
        raise ValueError(
            f"max_chunk_bytes ({max_chunk_bytes}) must be a multiple of alignment_bytes ({alignment_bytes})."
        )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor._local_tensor if isinstance(tensor, DTensor) else tensor


def _iter_named_parameters(module: nn.Module):
    try:
        return module.named_parameters(recurse=True, remove_duplicate=False)
    except TypeError:
        return module.named_parameters(recurse=True)


def _iter_named_buffers(module: nn.Module):
    try:
        return module.named_buffers(recurse=True, remove_duplicate=False)
    except TypeError:
        return module.named_buffers(recurse=True)


def _append_binding(
    bindings: list[tuple[str, torch.Tensor]],
    seen_tensor_ids: set[int],
    name: str,
    tensor: torch.Tensor,
) -> None:
    tensor = _local_tensor(tensor)
    if tensor.device.type == "meta" or tensor.untyped_storage().nbytes() == 0:
        return
    if id(tensor) in seen_tensor_ids:
        return
    seen_tensor_ids.add(id(tensor))
    bindings.append((name, tensor))


def _collect_module_bindings(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    bindings: list[tuple[str, torch.Tensor]] = []
    seen_tensor_ids: set[int] = set()
    for name, parameter in _iter_named_parameters(module):
        _append_binding(bindings, seen_tensor_ids, f"parameter:{name}", parameter)
    for name, buffer in _iter_named_buffers(module):
        _append_binding(bindings, seen_tensor_ids, f"buffer:{name}", buffer)
    return bindings


def _collect_module_direct_bindings(module: nn.Module) -> tuple[list[tuple[str, torch.Tensor]], list[torch.Tensor]]:
    """Collect real/meta tensors while retaining empty tensors for materialization."""
    bindings: list[tuple[str, torch.Tensor]] = []
    empty_tensors: list[torch.Tensor] = []
    seen_tensor_ids: set[int] = set()
    for kind, iterator in (
        ("parameter", _iter_named_parameters(module)),
        ("buffer", _iter_named_buffers(module)),
    ):
        for name, tensor in iterator:
            if isinstance(tensor, DTensor):
                raise RuntimeError(
                    "Module direct materialization does not support DTensor; use the FSDP2-specific materializer."
                )
            if id(tensor) in seen_tensor_ids:
                continue
            seen_tensor_ids.add(id(tensor))
            if tensor.untyped_storage().nbytes() == 0:
                empty_tensors.append(tensor)
            else:
                bindings.append((f"{kind}:{name}", tensor))
    return bindings, empty_tensors


def get_fsdp_param_groups(state: object) -> tuple[object, ...]:
    """Return parameter groups across the plural and singular FSDP2 state layouts."""
    if hasattr(state, "_fsdp_param_groups"):
        param_groups = tuple(state._fsdp_param_groups)
    elif hasattr(state, "_fsdp_param_group"):
        param_group = state._fsdp_param_group
        param_groups = () if param_group is None else (param_group,)
    else:
        raise RuntimeError("Unsupported FSDP2 state: missing _fsdp_param_group and _fsdp_param_groups.")
    for param_group in param_groups:
        if not hasattr(param_group, "fsdp_params"):
            raise RuntimeError("Unsupported FSDP2 parameter group: missing fsdp_params.")
    return param_groups


def prepare_module_for_direct_fsdp2_weight_arena(
    module: nn.Module,
    full_state: dict[str, torch.Tensor],
) -> dict[str, int]:
    """Dematerialize a pre-FSDP module while retaining its load-source state.

    FSDP2 constructs the load-source rank on CPU and the remaining ranks on
    meta. Both layouts are valid for a direct arena load: CPU tensors provide
    the state-dict payload, while meta tensors are placeholders on ranks that
    receive the broadcast. Only an already materialized accelerator state
    would retain the old allocation that direct materialization must avoid.
    """
    device_counts: dict[str, int] = {}
    unsupported: list[tuple[str, str]] = []
    for name, tensor in full_state.items():
        device_type = tensor.device.type
        device_counts[device_type] = device_counts.get(device_type, 0) + 1
        if device_type not in {"cpu", "meta"}:
            unsupported.append((name, str(tensor.device)))

    if unsupported:
        raise RuntimeError(
            "Actor direct weight arena requires the pre-FSDP full state on CPU "
            "for load-source ranks or meta for receiving ranks; "
            f"device_counts={dict(sorted(device_counts.items()))} "
            f"first_unsupported_tensors={unsupported[:8]}."
        )
    if len(device_counts) > 1:
        raise RuntimeError(
            "Actor direct weight arena requires a uniform pre-FSDP full state: "
            "all CPU on a load-source rank or all meta on a receiving rank; "
            f"device_counts={dict(sorted(device_counts.items()))}."
        )

    # CPU state_dict tensors retain their original storages after to_empty(),
    # so they remain available to set_model_state_dict(). An all-meta receiving
    # rank is already in the desired layout and must not be rematerialized.
    if device_counts.get("cpu", 0):
        module.to_empty(device="meta")

    non_meta = [
        (name, str(tensor.device)) for name, tensor in module.state_dict().items() if tensor.device.type != "meta"
    ]
    if non_meta:
        raise RuntimeError(
            "Actor direct weight arena failed to dematerialize the model before fully_shard: "
            f"device_counts={dict(sorted(device_counts.items()))} "
            f"first_tensors={non_meta[:8]}."
        )
    return dict(sorted(device_counts.items()))


def _collect_fsdp2_bindings_with_metadata(
    module: nn.Module,
) -> tuple[list[tuple[str, torch.Tensor]], dict[str, tuple[object, int]]]:
    try:
        from torch.distributed._composable.fsdp import FSDPModule
    except ImportError as exc:
        raise RuntimeError("FSDP2 weight arena requires torch.distributed._composable.fsdp.FSDPModule.") from exc

    bindings: list[tuple[str, torch.Tensor]] = []
    fsdp_param_specs: dict[str, tuple[object, int]] = {}
    seen_tensor_ids: set[int] = set()
    seen_fsdp_param_ids: set[int] = set()
    fsdp_param_count = 0

    for submodule in module.modules():
        if not isinstance(submodule, FSDPModule):
            continue
        state = submodule._get_fsdp_state()
        for param_group in get_fsdp_param_groups(state):
            for fsdp_param in param_group.fsdp_params:
                if id(fsdp_param) in seen_fsdp_param_ids:
                    continue
                seen_fsdp_param_ids.add(id(fsdp_param))
                fsdp_param_count += 1
                param_fqn = _fsdp_param_name(fsdp_param, fsdp_param_count)
                if not hasattr(fsdp_param, "_sharded_param_data") or not hasattr(fsdp_param, "sharded_param"):
                    raise RuntimeError(f"Unsupported FSDP2 parameter {param_fqn}: missing sharded storage references.")
                sharded_state = getattr(fsdp_param, "sharded_state", None)
                if getattr(sharded_state, "name", None) != "SHARDED":
                    raise RuntimeError(
                        f"FSDP2 parameter {param_fqn} must be SHARDED before arena packing, got {sharded_state}."
                    )
                padded_name = f"fsdp_padded:{param_fqn}"
                binding_count_before = len(bindings)
                _append_binding(
                    bindings,
                    seen_tensor_ids,
                    padded_name,
                    fsdp_param._sharded_param_data,
                )
                if len(bindings) != binding_count_before:
                    fsdp_param_specs[padded_name] = (
                        fsdp_param,
                        fsdp_param._sharded_param_data.storage_offset()
                        * fsdp_param._sharded_param_data.element_size(),
                    )
                _append_binding(
                    bindings,
                    seen_tensor_ids,
                    f"fsdp_local:{param_fqn}",
                    fsdp_param.sharded_param,
                )

    if fsdp_param_count == 0:
        raise RuntimeError("Actor weight arena was enabled, but no FSDP2 parameters were found.")

    for name, tensor in _collect_module_bindings(module):
        _append_binding(bindings, seen_tensor_ids, name, tensor)
    return bindings, fsdp_param_specs


def _fsdp_param_name(fsdp_param: object, ordinal: int) -> str:
    """Return the FSDP2 FQN or an ordinal name before FSDP2 assigns one."""
    param_fqn = getattr(fsdp_param, "_param_fqn", None)
    return str(param_fqn) if param_fqn else f"fsdp_param_{ordinal}"


def _collect_fsdp2_bindings(module: nn.Module) -> list[tuple[str, torch.Tensor]]:
    bindings, _ = _collect_fsdp2_bindings_with_metadata(module)
    return bindings


def _validate_binding(name: str, tensor: torch.Tensor, require_cuda: bool) -> None:
    if tensor.layout != torch.strided:
        raise RuntimeError(f"Weight arena only supports strided tensors, but {name} has layout {tensor.layout}.")
    if require_cuda and tensor.device.type != "cuda":
        raise RuntimeError(f"Weight arena requires CUDA tensors, but {name} is on {tensor.device}.")
    if any(stride < 0 for stride in tensor.stride()):
        raise RuntimeError(f"Weight arena does not support negative strides: {name} has stride {tensor.stride()}.")

    storage_nbytes = tensor.untyped_storage().nbytes()
    max_element_offset = tensor.storage_offset()
    for size, stride in zip(tensor.size(), tensor.stride()):
        if size > 0:
            max_element_offset += (size - 1) * stride
    required_bytes = (
        max_element_offset * tensor.element_size()
        if tensor.numel() == 0
        else (max_element_offset + 1) * tensor.element_size()
    )
    if required_bytes > storage_nbytes:
        raise RuntimeError(f"Tensor {name} spans {required_bytes} bytes, beyond its {storage_nbytes}-byte storage.")


def _group_bindings(
    named_tensors: list[tuple[str, torch.Tensor]],
    alignment_bytes: int,
    require_cuda: bool,
) -> list[_StorageGroup]:
    groups: list[_StorageGroup] = []
    by_key: dict[tuple[str, int, int], _StorageGroup] = {}
    for name, tensor in named_tensors:
        _validate_binding(name, tensor, require_cuda)
        if alignment_bytes % tensor.element_size() != 0:
            raise RuntimeError(
                f"alignment_bytes={alignment_bytes} is not divisible by {name}'s element size {tensor.element_size()}."
            )
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        group = by_key.get(key)
        if group is None:
            group = _StorageGroup(key=key, nbytes=storage.nbytes(), bindings=[])
            by_key[key] = group
            groups.append(group)
        group.bindings.append(
            _TensorBinding(
                name=name,
                tensor=tensor,
                storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
                size=tensor.size(),
                stride=tuple(tensor.stride()),
            )
        )
    if not groups:
        raise RuntimeError("Weight arena has no non-empty tensor storage to pack.")
    return groups


def _plan_chunks(
    groups: list[_StorageGroup],
    max_chunk_bytes: int,
    alignment_bytes: int,
    *,
    device_override: torch.device | None = None,
) -> list[_ChunkPlan]:
    chunks: list[_ChunkPlan] = []
    current: _ChunkPlan | None = None
    for group in groups:
        device = device_override or group.bindings[0].tensor.device
        aligned_offset = _align_up(current.nbytes, alignment_bytes) if current is not None else 0
        needs_new_chunk = (
            current is None
            or current.device != device
            or (current.groups and aligned_offset + group.nbytes > max_chunk_bytes)
        )
        if needs_new_chunk:
            current = _ChunkPlan(device=device, groups=[])
            chunks.append(current)
            aligned_offset = 0
        group.arena_index = len(chunks) - 1
        group.arena_offset_bytes = aligned_offset
        current.groups.append(group)
        current.nbytes = aligned_offset + group.nbytes
    return chunks


def _group_direct_bindings(
    named_tensors: list[tuple[str, torch.Tensor]],
    alignment_bytes: int,
    *,
    allowed_cuda_device: torch.device | None = None,
) -> list[_StorageGroup]:
    groups: list[_StorageGroup] = []
    by_key: dict[tuple[str, int, int], _StorageGroup] = {}
    for name, tensor in named_tensors:
        _validate_binding(name, tensor, require_cuda=False)
        if tensor.device.type == "cuda" and tensor.device != allowed_cuda_device:
            raise RuntimeError(
                f"Direct weight arena requires meta or CPU source tensors, but {name} is already on {tensor.device}."
            )
        if alignment_bytes % tensor.element_size() != 0:
            raise RuntimeError(
                f"alignment_bytes={alignment_bytes} is not divisible by {name}'s element size {tensor.element_size()}."
            )
        storage = tensor.untyped_storage()
        storage_identity = getattr(storage, "_cdata", None)
        if storage_identity is None:
            raise RuntimeError(f"Direct weight arena cannot identify the storage backing {name}.")
        key = (str(tensor.device), int(storage_identity), storage.nbytes())
        group = by_key.get(key)
        if group is None:
            group = _StorageGroup(key=key, nbytes=storage.nbytes(), bindings=[])
            by_key[key] = group
            groups.append(group)
        group.bindings.append(
            _TensorBinding(
                name=name,
                tensor=tensor,
                storage_offset_bytes=tensor.storage_offset() * tensor.element_size(),
                size=tensor.size(),
                stride=tuple(tensor.stride()),
            )
        )
    return groups


def _arena_backed_replacement(
    tensor: torch.Tensor,
    arena_storage: torch.UntypedStorage,
    offset_bytes: int,
    size: torch.Size,
    stride: tuple[int, ...],
) -> torch.Tensor:
    element_size = tensor.element_size()
    if offset_bytes % element_size != 0:
        raise RuntimeError(
            f"Arena offset {offset_bytes} for dtype={tensor.dtype} is not aligned to {element_size} bytes."
        )
    view = torch.empty(0, dtype=tensor.dtype, device=arena_storage.device).set_(
        arena_storage,
        storage_offset=offset_bytes // element_size,
        size=size,
        stride=stride,
    )
    replacement = torch.Tensor._make_subclass(type(tensor), view, tensor.requires_grad)
    replacement.__dict__.update(tensor.__dict__)
    return replacement


@torch.no_grad()
def materialize_module_weights_in_arena(
    module: nn.Module,
    *,
    device: torch.device | str,
    max_chunk_bytes: int,
    alignment_bytes: int = 256,
) -> WeightArenaHandle:
    """Materialize meta/CPU parameters and buffers directly into arena allocations.

    Existing Tensor objects are preserved with ``swap_tensors``. CPU storages are
    copied directly into their final arena ranges; meta storages are left
    uninitialized for the caller's normal weight loader to populate.
    """
    _validate_config(max_chunk_bytes, alignment_bytes)
    target_device = torch.device(device)
    if target_device.type == "meta":
        raise ValueError("Direct weight arena target device cannot be meta.")

    named_tensors, empty_tensors = _collect_module_direct_bindings(module)
    groups = _group_direct_bindings(named_tensors, alignment_bytes)
    chunks = _plan_chunks(
        groups,
        max_chunk_bytes,
        alignment_bytes,
        device_override=target_device,
    )
    started_at = time.perf_counter()
    arenas: list[torch.Tensor] = []
    placements: list[WeightArenaPlacement] = []

    for arena_index, chunk in enumerate(chunks):
        arena = torch.empty(chunk.nbytes, dtype=torch.uint8, device=target_device)
        arena_storage = arena.untyped_storage()
        arena_base_addr = arena_storage.data_ptr()
        for group in chunk.groups:
            source_tensor = group.bindings[0].tensor
            if source_tensor.device.type == "cpu":
                source_storage = source_tensor.untyped_storage()
                source_bytes = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
                    source_storage,
                    storage_offset=0,
                    size=(source_storage.nbytes(),),
                    stride=(1,),
                )
                arena.narrow(0, group.arena_offset_bytes, group.nbytes).copy_(source_bytes)
                del source_bytes

            for binding in group.bindings:
                new_offset_bytes = group.arena_offset_bytes + binding.storage_offset_bytes
                replacement = _arena_backed_replacement(
                    binding.tensor,
                    arena_storage,
                    new_offset_bytes,
                    binding.size,
                    binding.stride,
                )
                torch.utils.swap_tensors(binding.tensor, replacement)
                expected_addr = arena_base_addr + new_offset_bytes
                if binding.tensor.numel() > 0 and binding.tensor.data_ptr() != expected_addr:
                    raise RuntimeError(
                        f"Direct arena materialization failed for {binding.name}: expected data_ptr={expected_addr}, "
                        f"got {binding.tensor.data_ptr()}."
                    )
            placements.append(
                WeightArenaPlacement(
                    arena_index=arena_index,
                    offset_bytes=group.arena_offset_bytes,
                    nbytes=group.nbytes,
                    binding_names=tuple(binding.name for binding in group.bindings),
                )
            )
        arenas.append(arena)

    for tensor in empty_tensors:
        replacement = torch.Tensor._make_subclass(
            type(tensor),
            torch.empty_strided(tensor.size(), tensor.stride(), dtype=tensor.dtype, device=target_device),
            tensor.requires_grad,
        )
        replacement.__dict__.update(tensor.__dict__)
        torch.utils.swap_tensors(tensor, replacement)

    arena_tuple = tuple(arenas)
    _synchronize_arena_devices(arena_tuple)
    stats = WeightArenaStats(
        arena_count=len(arenas),
        unique_storage_count=len(groups),
        tensor_binding_count=sum(len(group.bindings) for group in groups) + len(empty_tensors),
        total_storage_bytes=sum(group.nbytes for group in groups),
        total_arena_bytes=sum(arena.numel() for arena in arenas),
        largest_storage_bytes=max((group.nbytes for group in groups), default=0),
        pack_seconds=time.perf_counter() - started_at,
    )
    return WeightArenaHandle(
        arenas=arena_tuple,
        placements=tuple(placements),
        stats=stats,
        base_addresses=tuple(arena.data_ptr() for arena in arena_tuple),
    )


def _collect_unique_fsdp2_params(module: nn.Module) -> list[object]:
    try:
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        from torch.distributed._composable.fsdp import FSDPModule

    fsdp_params: list[object] = []
    seen_ids: set[int] = set()
    for submodule in module.modules():
        if not isinstance(submodule, FSDPModule):
            continue
        for param_group in get_fsdp_param_groups(submodule._get_fsdp_state()):
            for fsdp_param in param_group.fsdp_params:
                if id(fsdp_param) in seen_ids:
                    continue
                seen_ids.add(id(fsdp_param))
                if not hasattr(fsdp_param, "padded_sharded_param_size"):
                    raise RuntimeError("Unsupported FSDP2 parameter: missing padded_sharded_param_size.")
                fsdp_params.append(fsdp_param)
    if not fsdp_params:
        raise RuntimeError("Actor direct weight arena was enabled, but no FSDP2 parameters were found.")
    return fsdp_params


@torch.no_grad()
def materialize_fsdp2_model_weights_in_arena(
    module: nn.Module,
    *,
    device: torch.device | str,
    max_chunk_bytes: int,
    alignment_bytes: int = 256,
) -> FSDP2WeightArenaHandle:
    """Materialize FSDP2 shards and module buffers directly into arenas.

    Parameter shards must still be CPU/meta so direct mode never retains an old
    CUDA weight allocation. FSDP may independently move small module buffers to
    the target CUDA device while sharding; preserve those values while rebinding
    them into the arena. Non-persistent buffers are restored again after the
    initial NIXL pull because they are not part of the transferred state dict.
    """
    _validate_config(max_chunk_bytes, alignment_bytes)
    target_device = torch.device(device)
    if target_device.type != "cuda":
        raise ValueError(f"FSDP2 direct weight arena requires a CUDA target, got {target_device}.")

    fsdp_params = _collect_unique_fsdp2_params(module)
    fsdp_param_by_tensor_id: dict[int, object] = {}
    named_tensors: list[tuple[str, torch.Tensor]] = []
    for fsdp_param in fsdp_params:
        padded_data = fsdp_param._sharded_param_data
        if padded_data.device.type not in ("cpu", "meta"):
            raise RuntimeError(
                f"FSDP2 direct weight arena expected CPU/meta shards before materialization, but "
                f"{getattr(fsdp_param, '_param_fqn', '<unknown>')} is on {padded_data.device}."
            )
        fsdp_param_by_tensor_id[id(padded_data)] = fsdp_param
        named_tensors.append((f"fsdp_padded:{getattr(fsdp_param, '_param_fqn', '<unknown>')}", padded_data))

    seen_buffer_ids: set[int] = set()
    empty_buffers: list[torch.Tensor] = []
    for name, buffer in _iter_named_buffers(module):
        if id(buffer) in seen_buffer_ids:
            continue
        seen_buffer_ids.add(id(buffer))
        if isinstance(buffer, DTensor):
            raise RuntimeError(f"FSDP2 direct weight arena does not support DTensor buffer {name}.")
        if buffer.device.type not in ("cpu", "meta") and buffer.device != target_device:
            raise RuntimeError(
                f"FSDP2 direct weight arena expected CPU/meta or target-device buffer {name}, "
                f"got {buffer.device} for target {target_device}."
            )
        if buffer.untyped_storage().nbytes() == 0:
            empty_buffers.append(buffer)
        else:
            named_tensors.append((f"buffer:{name}", buffer))

    groups = _group_direct_bindings(
        named_tensors,
        alignment_bytes,
        allowed_cuda_device=target_device,
    )
    chunks = _plan_chunks(
        groups,
        max_chunk_bytes,
        alignment_bytes,
        device_override=target_device,
    )
    started_at = time.perf_counter()
    arenas: list[torch.Tensor] = []
    placements: list[WeightArenaPlacement] = []
    fsdp_placements: list[tuple[object, int, int]] = []

    for arena_index, chunk in enumerate(chunks):
        arena = torch.empty(chunk.nbytes, dtype=torch.uint8, device=target_device)
        arena_storage = arena.untyped_storage()
        for group in chunk.groups:
            source_tensor = group.bindings[0].tensor
            fsdp_param = fsdp_param_by_tensor_id.get(id(source_tensor))
            if fsdp_param is not None:
                if len(group.bindings) != 1:
                    raise RuntimeError("FSDP2 padded shard unexpectedly aliases another direct arena binding.")
                binding = group.bindings[0]
                offset_bytes = group.arena_offset_bytes + binding.storage_offset_bytes
                padded_flat = torch.empty(0, dtype=source_tensor.dtype, device=target_device).set_(
                    arena_storage,
                    storage_offset=offset_bytes // source_tensor.element_size(),
                    size=source_tensor.size(),
                    stride=source_tensor.stride(),
                )
                padded_flat.zero_()
                padded_tensor = padded_flat.view(fsdp_param.padded_sharded_param_size)
                shard_dim = fsdp_param.fsdp_placement.dim
                local_length = fsdp_param.sharded_size[shard_dim]
                local_tensor = padded_tensor.narrow(shard_dim, 0, local_length)
                new_param = nn.Parameter(
                    fsdp_param.to_sharded_dtensor(local_tensor),
                    requires_grad=fsdp_param.sharded_param.requires_grad,
                )
                new_param.__dict__.update(fsdp_param.sharded_param.__dict__)
                old_param = fsdp_param.sharded_param
                torch.utils.swap_tensors(old_param, new_param)
                fsdp_param._setattr_on_modules(old_param)
                fsdp_param.sharded_param = old_param
                fsdp_param._sharded_param_data = padded_flat
                fsdp_param._sharding_spec = old_param._spec
                fsdp_placements.append((fsdp_param, arena_index, offset_bytes))
            else:
                if source_tensor.device.type != "meta":
                    source_storage = source_tensor.untyped_storage()
                    source_bytes = torch.empty(0, dtype=torch.uint8, device=source_tensor.device).set_(
                        source_storage,
                        storage_offset=0,
                        size=(source_storage.nbytes(),),
                        stride=(1,),
                    )
                    arena.narrow(0, group.arena_offset_bytes, group.nbytes).copy_(source_bytes)
                    del source_bytes
                for binding in group.bindings:
                    replacement = _arena_backed_replacement(
                        binding.tensor,
                        arena_storage,
                        group.arena_offset_bytes + binding.storage_offset_bytes,
                        binding.size,
                        binding.stride,
                    )
                    torch.utils.swap_tensors(binding.tensor, replacement)
            placements.append(
                WeightArenaPlacement(
                    arena_index=arena_index,
                    offset_bytes=group.arena_offset_bytes,
                    nbytes=group.nbytes,
                    binding_names=tuple(binding.name for binding in group.bindings),
                )
            )
        arenas.append(arena)

    for buffer in empty_buffers:
        replacement = torch.Tensor._make_subclass(
            type(buffer),
            torch.empty_strided(buffer.size(), buffer.stride(), dtype=buffer.dtype, device=target_device),
            buffer.requires_grad,
        )
        replacement.__dict__.update(buffer.__dict__)
        torch.utils.swap_tensors(buffer, replacement)

    arena_tuple = tuple(arenas)
    _synchronize_arena_devices(arena_tuple)
    handle = FSDP2WeightArenaHandle(
        arenas=arena_tuple,
        placements=tuple(placements),
        stats=WeightArenaStats(
            arena_count=len(arenas),
            unique_storage_count=len(groups),
            tensor_binding_count=sum(len(group.bindings) for group in groups) + len(empty_buffers),
            total_storage_bytes=sum(group.nbytes for group in groups),
            total_arena_bytes=sum(arena.numel() for arena in arenas),
            largest_storage_bytes=max((group.nbytes for group in groups), default=0),
            pack_seconds=time.perf_counter() - started_at,
        ),
        base_addresses=tuple(arena.data_ptr() for arena in arena_tuple),
        fsdp_param_placements=tuple(fsdp_placements),
    )
    handle.assert_fsdp_bindings_unchanged()
    return handle


def _storage_as_bytes(tensor: torch.Tensor, expected_key: tuple[str, int, int]) -> torch.Tensor:
    storage = tensor.untyped_storage()
    actual_key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
    if actual_key != expected_key:
        raise RuntimeError(
            "A source storage changed while the weight arena was being packed: "
            f"expected {expected_key}, got {actual_key}."
        )
    return torch.empty(0, dtype=torch.uint8, device=tensor.device).set_(
        storage,
        storage_offset=0,
        size=(storage.nbytes(),),
        stride=(1,),
    )


def _make_fsdp2_handle(
    handle: WeightArenaHandle,
    fsdp_param_specs: dict[str, tuple[object, int]],
) -> FSDP2WeightArenaHandle:
    placement_by_binding: dict[str, WeightArenaPlacement] = {}
    for placement in handle.placements:
        for binding_name in placement.binding_names:
            if binding_name in fsdp_param_specs:
                if binding_name in placement_by_binding:
                    raise RuntimeError(f"FSDP2 binding {binding_name} appeared in multiple arena placements.")
                placement_by_binding[binding_name] = placement

    missing_bindings = sorted(set(fsdp_param_specs) - set(placement_by_binding))
    if missing_bindings:
        raise RuntimeError(
            "FSDP2 arena packing lost padded shard placements: "
            f"count={len(missing_bindings)} first_bindings={missing_bindings[:8]}."
        )

    fsdp_param_placements = tuple(
        (
            fsdp_param,
            placement_by_binding[binding_name].arena_index,
            placement_by_binding[binding_name].offset_bytes + storage_offset_bytes,
        )
        for binding_name, (fsdp_param, storage_offset_bytes) in fsdp_param_specs.items()
    )
    return FSDP2WeightArenaHandle(
        arenas=handle.arenas,
        placements=handle.placements,
        stats=handle.stats,
        base_addresses=handle.base_addresses,
        fsdp_param_placements=fsdp_param_placements,
    )


@torch.no_grad()
def _pack_named_tensors(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    max_chunk_bytes: int,
    alignment_bytes: int,
    require_cuda: bool,
    fsdp_param_specs: dict[str, tuple[object, int]] | None = None,
) -> WeightArenaHandle:
    _validate_config(max_chunk_bytes, alignment_bytes)
    groups = _group_bindings(named_tensors, alignment_bytes, require_cuda)
    chunks = _plan_chunks(groups, max_chunk_bytes, alignment_bytes)
    started_at = time.perf_counter()
    arenas: list[torch.Tensor] = []
    placements: list[WeightArenaPlacement] = []

    for arena_index, chunk in enumerate(chunks):
        arena = torch.empty(chunk.nbytes, dtype=torch.uint8, device=chunk.device)
        arena_storage = arena.untyped_storage()
        arena_base_addr = arena_storage.data_ptr()
        for group in chunk.groups:
            source_bytes = _storage_as_bytes(group.bindings[0].tensor, group.key)
            arena.narrow(0, group.arena_offset_bytes, group.nbytes).copy_(source_bytes)
            for binding in group.bindings:
                element_size = binding.tensor.element_size()
                new_offset_bytes = group.arena_offset_bytes + binding.storage_offset_bytes
                if new_offset_bytes % element_size != 0:
                    raise RuntimeError(
                        f"Arena offset {new_offset_bytes} for {binding.name} is not aligned to {element_size} bytes."
                    )
                binding.tensor.set_(
                    arena_storage,
                    storage_offset=new_offset_bytes // element_size,
                    size=binding.size,
                    stride=binding.stride,
                )
                expected_addr = arena_base_addr + new_offset_bytes
                if binding.tensor.numel() > 0 and binding.tensor.data_ptr() != expected_addr:
                    raise RuntimeError(
                        f"Weight arena rebind failed for {binding.name}: expected data_ptr={expected_addr}, "
                        f"got {binding.tensor.data_ptr()}."
                    )
            placements.append(
                WeightArenaPlacement(
                    arena_index=arena_index,
                    offset_bytes=group.arena_offset_bytes,
                    nbytes=group.nbytes,
                    binding_names=tuple(binding.name for binding in group.bindings),
                )
            )
            del source_bytes
        if chunk.device.type == "cuda":
            torch.cuda.synchronize(chunk.device)
        arenas.append(arena)

    stats = WeightArenaStats(
        arena_count=len(arenas),
        unique_storage_count=len(groups),
        tensor_binding_count=sum(len(group.bindings) for group in groups),
        total_storage_bytes=sum(group.nbytes for group in groups),
        total_arena_bytes=sum(arena.numel() for arena in arenas),
        largest_storage_bytes=max(group.nbytes for group in groups),
        pack_seconds=time.perf_counter() - started_at,
    )
    arena_tuple = tuple(arenas)
    handle = WeightArenaHandle(
        arenas=arena_tuple,
        placements=tuple(placements),
        stats=stats,
        base_addresses=tuple(arena.data_ptr() for arena in arena_tuple),
    )
    if fsdp_param_specs is not None:
        fsdp2_handle = _make_fsdp2_handle(handle, fsdp_param_specs)
        fsdp2_handle.assert_fsdp_bindings_unchanged()
        return fsdp2_handle
    return handle


def _weight_tms_pool_key(mem_pools: dict, tms_tag: str, enable_cpu_backup: bool) -> tuple:
    """Return the TMS pool that owns model weights.

    TMS keeps separate pools for the same tag depending on whether CPU backup
    is enabled. The caller must select the setting used by the worker's model
    allocation context; a same-tag pool with the opposite setting belongs to a
    different allocation class and must remain untouched.
    """
    pool_key = (tms_tag, bool(enable_cpu_backup))
    if pool_key not in mem_pools:
        matching_keys = [key for key in mem_pools if isinstance(key, tuple) and len(key) == 2 and key[0] == tms_tag]
        raise RuntimeError(
            f"Weight arena expected model-weight TMS pool {pool_key!r}; available pools={matching_keys!r}."
        )
    return pool_key


def _install_packed_tms_pool(mem_pools: dict, pool_key: tuple, packed_pool: object) -> None:
    """Replace only the model-weight pool, leaving the optimizer pool intact."""
    mem_pools[pool_key] = packed_pool


def _pack_named_tensors_in_fresh_tms_pool(
    named_tensors: list[tuple[str, torch.Tensor]],
    *,
    max_chunk_bytes: int,
    alignment_bytes: int,
    require_cuda: bool,
    tms_tag: str,
    enable_cpu_backup: bool,
    fsdp_param_specs: dict[str, tuple[object, int]] | None = None,
) -> WeightArenaHandle:
    """Pack into a new TMS pool, then release the source pool's cached blocks."""
    started_at = time.perf_counter()
    try:
        from torch_memory_saver import torch_memory_saver
    except ImportError as exc:
        raise RuntimeError("Weight arena TMS pool replacement requires torch_memory_saver.") from exc

    torch_memory_saver._ensure_initialized()
    impl = torch_memory_saver._impl
    required_fields = ("_mem_pools", "_hook_util", "_with_region_config")
    missing_fields = tuple(field for field in required_fields if not hasattr(impl, field))
    if missing_fields:
        raise RuntimeError(
            f"Unsupported torch_memory_saver internals for weight arena pool replacement: missing {missing_fields}."
        )

    source_pool_key = _weight_tms_pool_key(impl._mem_pools, tms_tag, enable_cpu_backup)
    source_pool = impl._mem_pools[source_pool_key]
    source_pool_use_count = source_pool.use_count()
    if source_pool_use_count != 1:
        raise RuntimeError(
            "Weight arena must replace the TMS pool after its allocation region has exited. "
            f"tag={tms_tag!r} pool_key={source_pool_key!r} pool_use_count={source_pool_use_count}."
        )

    # Complete all validation before releasing the source pool object. Every
    # model storage must come from that pool; otherwise the TMS allocation
    # lifetime assumptions below do not hold.
    _validate_config(max_chunk_bytes, alignment_bytes)
    source_groups = _group_bindings(named_tensors, alignment_bytes, require_cuda)
    model_source_allocations = {
        (
            group.bindings[0].tensor.device.index,
            group.bindings[0].tensor.untyped_storage().data_ptr(),
        )
        for group in source_groups
    }
    source_snapshot = source_pool.snapshot()
    source_reserved_bytes = sum(int(segment.get("total_size", 0)) for segment in source_snapshot)
    active_source_blocks = [
        (int(segment["device"]), block)
        for segment in source_snapshot
        for block in segment.get("blocks", ())
        if block.get("state") == "active_allocated"
    ]
    active_source_allocations = {(device_index, int(block["address"])) for device_index, block in active_source_blocks}
    missing_model_allocations = model_source_allocations - active_source_allocations
    if missing_model_allocations:
        missing_preview = sorted(missing_model_allocations)[:8]
        raise RuntimeError(
            "Weight arena found model storages outside the source TMS pool: "
            f"tag={tms_tag!r} missing_count={len(missing_model_allocations)} "
            f"first_device_addresses={missing_preview}."
        )
    residual_source_blocks = [
        block
        for device_index, block in active_source_blocks
        if (device_index, int(block["address"])) not in model_source_allocations
    ]
    residual_source_bytes = sum(int(block.get("size", 0)) for block in residual_source_blocks)

    arena_pool = torch.cuda.MemPool(allocator=impl._hook_util.get_allocator())
    with torch.cuda.use_mem_pool(arena_pool):
        with impl._with_region_config(tag=tms_tag, enable_cpu_backup=enable_cpu_backup):
            handle = _pack_named_tensors(
                named_tensors,
                max_chunk_bytes=max_chunk_bytes,
                alignment_bytes=alignment_bytes,
                require_cuda=require_cuda,
                fsdp_param_specs=fsdp_param_specs,
            )

    torch.cuda.synchronize()
    gc.collect()
    # Swapping the pool after every binding has moved releases the old pool's
    # inactive cache. Live non-model allocations keep their TMS metadata and
    # remain pauseable after the MemPool object is destroyed.
    _install_packed_tms_pool(impl._mem_pools, source_pool_key, arena_pool)
    del source_pool
    gc.collect()
    torch.cuda.synchronize()
    handle.stats = replace(
        handle.stats,
        pack_seconds=time.perf_counter() - started_at,
        tms_source_reserved_bytes=source_reserved_bytes,
        tms_source_active_bytes=residual_source_bytes,
        tms_source_active_allocations=len(residual_source_blocks),
    )
    return handle


def pack_module_weights(
    module: nn.Module,
    *,
    max_chunk_bytes: int,
    alignment_bytes: int = 256,
    require_cuda: bool = True,
) -> WeightArenaHandle:
    """Pack registered parameters and buffers without replacing their Tensor objects."""
    return _pack_named_tensors(
        _collect_module_bindings(module),
        max_chunk_bytes=max_chunk_bytes,
        alignment_bytes=alignment_bytes,
        require_cuda=require_cuda,
    )


def pack_fsdp2_model_weights(
    module: nn.Module,
    *,
    max_chunk_bytes: int,
    alignment_bytes: int = 256,
) -> FSDP2WeightArenaHandle:
    """Pack FSDP2 padded shards, local DTensor views, parameters, and buffers."""
    named_tensors, fsdp_param_specs = _collect_fsdp2_bindings_with_metadata(module)
    return _pack_named_tensors(
        named_tensors,
        max_chunk_bytes=max_chunk_bytes,
        alignment_bytes=alignment_bytes,
        require_cuda=True,
        fsdp_param_specs=fsdp_param_specs,
    )


def pack_module_weights_in_fresh_tms_pool(
    module: nn.Module,
    *,
    max_chunk_bytes: int,
    alignment_bytes: int = 256,
    tms_tag: str = "weights",
    enable_cpu_backup: bool = False,
) -> WeightArenaHandle:
    """Pack module weights and replace the completed source TMS pool."""
    return _pack_named_tensors_in_fresh_tms_pool(
        _collect_module_bindings(module),
        max_chunk_bytes=max_chunk_bytes,
        alignment_bytes=alignment_bytes,
        require_cuda=True,
        tms_tag=tms_tag,
        enable_cpu_backup=enable_cpu_backup,
    )


def pack_fsdp2_model_weights_in_fresh_tms_pool(
    module: nn.Module,
    *,
    max_chunk_bytes: int,
    alignment_bytes: int = 256,
    tms_tag: str = "default",
    enable_cpu_backup: bool = False,
) -> FSDP2WeightArenaHandle:
    """Pack FSDP2 shards and replace the completed source TMS pool."""
    named_tensors, fsdp_param_specs = _collect_fsdp2_bindings_with_metadata(module)
    return _pack_named_tensors_in_fresh_tms_pool(
        named_tensors,
        max_chunk_bytes=max_chunk_bytes,
        alignment_bytes=alignment_bytes,
        require_cuda=True,
        tms_tag=tms_tag,
        enable_cpu_backup=enable_cpu_backup,
        fsdp_param_specs=fsdp_param_specs,
    )
