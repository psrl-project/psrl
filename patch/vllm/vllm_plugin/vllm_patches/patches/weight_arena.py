import logging
import os
from contextlib import contextmanager
from typing import Any

import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.model_executor.model_loader.base_loader import log_model_inspection
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_ORIGINAL_LOAD_MODEL = GPUModelRunner.load_model
DIRECT_ARENA_DUMMY_LOAD_FORMAT = "psrl_arena_dummy"
_DIRECT_LOADER_REGISTERED = False


def _role_materialization(arena_config: dict[str, Any], role: str) -> str:
    key = "reward_materialization" if role == "reward" else "rollout_materialization"
    return str(arena_config.get(key, "direct"))


def _max_chunk_bytes(arena_config: dict[str, Any]) -> int:
    from psrl.utils.weight_arena import gb_to_bytes

    return gb_to_bytes(arena_config["max_chunk_gb"], field_name="max_chunk_gb")


def _assert_direct_model_supported(vllm_config: Any, model_config: Any) -> None:
    if vllm_config.quant_config is not None or model_config.quantization is not None:
        raise RuntimeError("Direct weight arena currently supports unquantized models only.")
    if vllm_config.lora_config is not None:
        raise RuntimeError("Direct weight arena does not support LoRA.")


@contextmanager
def _materialize_ep_metadata_on_cpu(enabled: bool):
    """Keep deterministic EP maps real while the large model is built on meta."""
    if not enabled:
        yield
        return

    from vllm.model_executor.layers.fused_moe import expert_map_manager

    original = expert_map_manager.determine_expert_map

    def determine_expert_map_on_cpu(*args: Any, **kwargs: Any):
        with torch.device("cpu"):
            return original(*args, **kwargs)

    expert_map_manager.determine_expert_map = determine_expert_map_on_cpu
    try:
        yield
    finally:
        expert_map_manager.determine_expert_map = original


def _assert_direct_moe_modules_supported(
    model: torch.nn.Module,
    model_config: Any,
    parallel_config: Any,
) -> None:
    if not model_config.is_moe:
        return

    from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
        UnquantizedMoeBackend,
    )
    from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
        UnquantizedFusedMoEMethod,
    )

    fused_moe_count = 0
    expected_ep = bool(parallel_config.enable_expert_parallel)
    for module in model.modules():
        if not isinstance(module, FusedMoE):
            continue
        fused_moe_count += 1
        quant_method = module.quant_method
        if not isinstance(quant_method, UnquantizedFusedMoEMethod):
            raise RuntimeError(
                f"Direct MoE weight arena only supports UnquantizedFusedMoEMethod, got {type(quant_method).__name__}."
            )
        if quant_method.unquantized_backend != UnquantizedMoeBackend.TRITON:
            raise RuntimeError(
                "Direct MoE weight arena requires the storage-preserving Triton backend, "
                f"got {quant_method.unquantized_backend.value}."
            )
        if bool(module.use_ep) != expected_ep:
            raise RuntimeError(
                "Direct MoE weight arena found inconsistent expert parallel state: "
                f"parallel_config.enable_expert_parallel={expected_ep} module.use_ep={module.use_ep}."
            )
        if module.use_ep and (module.ep_size <= 1 or module._expert_map is None):
            raise RuntimeError(
                "Direct MoE EP weight arena requires a non-trivial EP group and a materialized expert map."
            )
    if fused_moe_count == 0:
        raise RuntimeError("vLLM marked the model as MoE but no FusedMoE modules were found.")


@torch.no_grad()
def _initialize_direct_runtime_buffers(model: torch.nn.Module) -> tuple[str, ...]:
    """Initialize model-created non-persistent buffers without changing storage.

    Direct mode constructs the model on meta, so computed runtime buffers do not
    contain their constructor values after arena materialization. They are also
    absent from state_dict and NIXL transfers. Recompute supported buffers and
    copy into their existing arena views so CUDA graph and arena addresses stay
    stable.
    """
    initialized: list[str] = []
    unsupported: list[str] = []
    for module_prefix, module in model.named_modules():
        for buffer_name in sorted(module._non_persistent_buffers_set):
            buffer = module._buffers.get(buffer_name)
            if buffer is None:
                continue
            full_name = f"{module_prefix}.{buffer_name}" if module_prefix else buffer_name
            if buffer_name != "cos_sin_cache" or not hasattr(module, "_compute_cos_sin_cache"):
                unsupported.append(f"{full_name} ({type(module).__name__})")
                continue

            expected = module._compute_cos_sin_cache()
            if expected.shape != buffer.shape:
                raise RuntimeError(
                    f"Direct weight arena runtime buffer shape changed for {full_name}: "
                    f"expected {tuple(buffer.shape)}, got {tuple(expected.shape)}."
                )
            buffer.copy_(expected.to(device=buffer.device, dtype=buffer.dtype))
            initialized.append(full_name)

    if unsupported:
        raise RuntimeError(
            "Direct weight arena found non-persistent buffers without an initializer: "
            f"count={len(unsupported)} first_buffers={unsupported[:8]}."
        )
    return tuple(initialized)


class DirectWeightArenaDummyLoader(DummyModelLoader):
    """Construct supported vLLM weights on meta and materialize them in arenas."""

    def load_model(self, vllm_config, model_config, prefix: str = ""):
        additional_config = vllm_config.additional_config or {}
        arena_config = additional_config.get("psrl_nixl_weight_arena", {})
        role = additional_config.get("psrl_role", "rollout")
        enabled_key = "reward_enabled" if role == "reward" else "rollout_enabled"
        if role not in ("rollout", "reward") or not arena_config.get(enabled_key, False):
            raise RuntimeError(f"The direct arena loader is not enabled for role={role!r}.")
        if _role_materialization(arena_config, role) != "direct":
            raise RuntimeError(f"The direct arena loader requires {role}_materialization=direct.")
        _assert_direct_model_supported(vllm_config, model_config)

        target_device = torch.device(
            vllm_config.device_config.device
            if vllm_config.load_config.device is None
            else vllm_config.load_config.device
        )
        with set_default_torch_dtype(model_config.dtype):
            with _materialize_ep_metadata_on_cpu(
                model_config.is_moe and vllm_config.parallel_config.enable_expert_parallel
            ):
                with torch.device("meta"):
                    model = initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
            log_model_inspection(model)
            _assert_direct_moe_modules_supported(
                model,
                model_config,
                vllm_config.parallel_config,
            )

            from psrl.utils.weight_arena import materialize_module_weights_in_arena

            handle = materialize_module_weights_in_arena(
                model,
                device=target_device,
                max_chunk_bytes=_max_chunk_bytes(arena_config),
                alignment_bytes=int(arena_config["alignment_bytes"]),
            )
            initialize_dummy_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)
            initialized_runtime_buffers = _initialize_direct_runtime_buffers(model)
            handle.assert_module_weights_in_arena(model)
            model._psrl_weight_arena_handle = handle
            model._psrl_weight_arena_runtime_buffers = initialized_runtime_buffers
            psrl_logger.warning(
                "[NIXL_WEIGHT_ARENA] role=%s stage=initialize_runtime_buffers count=%d",
                role,
                len(initialized_runtime_buffers),
            )
            if model_config.is_moe:
                model_kind = "moe_ep" if vllm_config.parallel_config.enable_expert_parallel else "moe_tp"
            else:
                model_kind = "dense"
            model._psrl_weight_arena_model_kind = model_kind
        return model.eval()


def register_direct_weight_arena_loader() -> None:
    global _DIRECT_LOADER_REGISTERED
    if _DIRECT_LOADER_REGISTERED:
        return
    from vllm.model_executor.model_loader import register_model_loader

    register_model_loader(DIRECT_ARENA_DUMMY_LOAD_FORMAT)(DirectWeightArenaDummyLoader)
    _DIRECT_LOADER_REGISTERED = True


def _get_model_cudagraph_wrappers(model_runner: GPUModelRunner) -> list[CUDAGraphWrapper]:
    model = getattr(model_runner, "model", None)
    wrappers: list[CUDAGraphWrapper] = []
    if isinstance(model, CUDAGraphWrapper):
        wrappers.append(model)
    nested_wrapper = getattr(model, "cudagraph_wrapper", None)
    if isinstance(nested_wrapper, CUDAGraphWrapper) and nested_wrapper not in wrappers:
        wrappers.append(nested_wrapper)
    return wrappers


def _assert_no_cudagraph_has_been_captured(model_runner: GPUModelRunner) -> None:
    wrappers = _get_model_cudagraph_wrappers(model_runner)
    for wrapper in list(getattr(CUDAGraphWrapper, "_all_instances", ())):
        if wrapper not in wrappers:
            wrappers.append(wrapper)
    for wrapper in wrappers:
        if wrapper.concrete_cudagraph_entries:
            raise RuntimeError("Rollout weight arena must be packed before the first CUDA graph capture.")


def _assert_supported_compilation_mode(model_runner: GPUModelRunner) -> None:
    compilation_config = model_runner.compilation_config
    mode = getattr(compilation_config, "mode", None)
    level = getattr(compilation_config, "level", None)
    mode_name = getattr(mode, "name", None)
    if mode_name == "STOCK_TORCH_COMPILE" or level == 1:
        raise RuntimeError(
            "Rollout weight arena does not support stock torch.compile because vLLM compiles inside "
            "load_model() before the arena hook runs."
        )


def _assert_weight_offload_disabled(model_runner: GPUModelRunner) -> None:
    # vLLM <= 0.17 stored UVA offload in cache_config. Keep rejecting it when
    # present, but do not require that legacy field on the 0.18 config layout.
    cache_config = getattr(model_runner.vllm_config, "cache_config", None)
    legacy_cpu_offload_gb = getattr(cache_config, "cpu_offload_gb", 0)
    if legacy_cpu_offload_gb > 0:
        raise RuntimeError("Rollout weight arena does not support vLLM weight offloading.")

    offload_config = getattr(model_runner.vllm_config, "offload_config", None)
    if offload_config is None:
        raise RuntimeError("Unsupported vLLM config: missing offload_config.")
    uva_config = getattr(offload_config, "uva", None)
    prefetch_config = getattr(offload_config, "prefetch", None)
    if uva_config is None or not hasattr(uva_config, "cpu_offload_gb"):
        raise RuntimeError("Unsupported vLLM config: missing offload_config.uva.cpu_offload_gb.")
    if prefetch_config is None or not hasattr(prefetch_config, "offload_group_size"):
        raise RuntimeError("Unsupported vLLM config: missing offload_config.prefetch.offload_group_size.")
    uva_offload_gb = uva_config.cpu_offload_gb
    offload_group_size = prefetch_config.offload_group_size
    if uva_offload_gb > 0 or offload_group_size > 0:
        raise RuntimeError("Rollout weight arena does not support vLLM weight offloading.")


def _pack_loaded_model(
    model_runner: GPUModelRunner,
    arena_config: dict[str, Any],
    *,
    replace_tms_pool: bool,
) -> None:
    _assert_no_cudagraph_has_been_captured(model_runner)
    if hasattr(model_runner, "_psrl_weight_arena_handle"):
        raise RuntimeError("Rollout weight arena cannot repack a model after initialization.")

    from psrl.utils.weight_arena import (
        pack_module_weights,
        pack_module_weights_in_fresh_tms_pool,
    )

    model = model_runner.get_model()
    packer = pack_module_weights_in_fresh_tms_pool if replace_tms_pool else pack_module_weights
    pack_kwargs = {
        "max_chunk_bytes": _max_chunk_bytes(arena_config),
        "alignment_bytes": int(arena_config["alignment_bytes"]),
    }
    if replace_tms_pool:
        additional_config = model_runner.vllm_config.additional_config or {}
        pack_kwargs["enable_cpu_backup"] = bool(additional_config.get("enable_weights_cpu_backup", False))
    model_runner._psrl_weight_arena_handle = packer(model, **pack_kwargs)
    stats = model_runner._psrl_weight_arena_handle.stats
    role = (model_runner.vllm_config.additional_config or {}).get("psrl_role", "rollout")
    psrl_logger.warning(
        "[NIXL_WEIGHT_ARENA] role=%s rank=%d materialization=repack arenas=%d unique_storages=%d "
        "tensor_bindings=%d storage_bytes=%d arena_bytes=%d largest_storage_bytes=%d pack_s=%.6f "
        "tms_source_reserved_bytes=%d tms_source_active_bytes=%d tms_source_active_allocations=%d "
        "tms_pool_replaced=%s cudagraph_captured_before_pack=false base_addresses=%s",
        role,
        getattr(model_runner.parallel_config, "rank", -1),
        stats.arena_count,
        stats.unique_storage_count,
        stats.tensor_binding_count,
        stats.total_storage_bytes,
        stats.total_arena_bytes,
        stats.largest_storage_bytes,
        stats.pack_seconds,
        stats.tms_source_reserved_bytes,
        stats.tms_source_active_bytes,
        stats.tms_source_active_allocations,
        replace_tms_pool,
        model_runner._psrl_weight_arena_handle.base_addresses,
    )


def _adopt_direct_weight_arena(model_runner: GPUModelRunner) -> None:
    _assert_no_cudagraph_has_been_captured(model_runner)
    model = model_runner.get_model()
    handle = getattr(model, "_psrl_weight_arena_handle", None)
    if handle is None:
        raise RuntimeError("Direct weight arena loader did not attach an arena handle.")
    handle.assert_module_weights_in_arena(model)
    model_runner._psrl_weight_arena_handle = handle
    stats = handle.stats
    role = (model_runner.vllm_config.additional_config or {}).get("psrl_role", "rollout")
    model_kind = getattr(model, "_psrl_weight_arena_model_kind", "unknown")
    psrl_logger.warning(
        "[NIXL_WEIGHT_ARENA] role=%s rank=%d model_kind=%s materialization=direct "
        "arenas=%d unique_storages=%d "
        "tensor_bindings=%d storage_bytes=%d arena_bytes=%d largest_storage_bytes=%d pack_s=%.6f "
        "tms_pool_replaced=false cudagraph_captured_before_pack=false base_addresses=%s",
        role,
        getattr(model_runner.parallel_config, "rank", -1),
        model_kind,
        stats.arena_count,
        stats.unique_storage_count,
        stats.tensor_binding_count,
        stats.total_storage_bytes,
        stats.total_arena_bytes,
        stats.largest_storage_bytes,
        stats.pack_seconds,
        handle.base_addresses,
    )


def finalize_pending_weight_arena(model_runner: GPUModelRunner) -> None:
    """Pack after Worker.load_model exits the original TMS weights pool."""
    arena_config = getattr(model_runner, "_psrl_weight_arena_pending_config", None)
    if arena_config is None:
        return
    delattr(model_runner, "_psrl_weight_arena_pending_config")
    _pack_loaded_model(model_runner, arena_config, replace_tms_pool=True)


def _weight_arena_enabled(additional_config: dict[str, Any] | None) -> bool:
    additional_config = additional_config or {}
    arena_config = additional_config.get("psrl_nixl_weight_arena", {})
    role = additional_config.get("psrl_role", "rollout")
    enabled_key = "reward_enabled" if role == "reward" else "rollout_enabled"
    return bool(arena_config.get(enabled_key, False))


@min_vllm_version("0.18.1")
class WeightArenaModelRunnerPatch(vLLMPatch[GPUModelRunner]):
    """Pack model storages after load and before vLLM warmup/CUDA graph capture."""

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        additional_config = self.vllm_config.additional_config or {}
        arena_config = additional_config.get("psrl_nixl_weight_arena", {})
        enabled = _weight_arena_enabled(additional_config)
        if enabled:
            if hasattr(self, "_psrl_weight_arena_handle"):
                raise RuntimeError("Rollout weight arena cannot repack a model after initialization.")
            _assert_no_cudagraph_has_been_captured(self)
            _assert_supported_compilation_mode(self)
            _assert_weight_offload_disabled(self)

        _ORIGINAL_LOAD_MODEL(self, *args, **kwargs)
        if not enabled:
            return

        role = additional_config.get("psrl_role", "rollout")
        materialization = _role_materialization(arena_config, role)
        if materialization == "direct":
            _adopt_direct_weight_arena(self)
            return
        if materialization != "repack":
            raise RuntimeError(
                f"psrl.nixl.weight_arena.{role}_materialization must be 'direct' or 'repack', got {materialization!r}."
            )

        if os.environ.get("PSRL_VLLM_PATCHES", "").startswith("TMS"):
            self._psrl_weight_arena_pending_config = arena_config
            return
        _pack_loaded_model(self, arena_config, replace_tms_pool=False)
