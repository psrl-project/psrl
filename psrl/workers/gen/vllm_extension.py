import gc
import logging
import os
import socket
import time
from contextlib import nullcontext
from copy import copy, deepcopy
from types import MethodType

import ray
import torch
from omegaconf import DictConfig
from torch.distributed.tensor import DTensor
from verl.utils.device import get_device_id

# from vllm.config import get_current_vllm_config
# from vllm.platforms import current_platform
from verl.utils.fs import copy_to_local
from verl.utils.memory_utils import aggressive_empty_cache
from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader
from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.config import set_current_vllm_config
from vllm.distributed.kv_transfer import get_kv_transfer_group
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config
from vllm_patches.interfaces import supports_weight_layout

from psrl.utils.common.nixl_names import NIXL_META_SERVER_NAME
from psrl.utils.common.worker_naming import gen_client_name
from psrl.utils.converter import create_parameter_mapping
from psrl.utils.converter.param_sync import ParamSyncPlan
from psrl.utils.converter.vllm_converter import convert_vllm_inplace
from psrl.utils.nixl import (
    NIXLClientType,
    NIXLStorageClient,
    fingerprint_log_record,
    report_precision_sensitive_parameters,
)

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def monkey_patch_compute_logits(model, vocab_size: int) -> None:
    """Prevent sampling logits outside the tokenizer vocabulary."""
    original_compute_logits = model.compute_logits

    def compute_logits(self, *args, **kwargs) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


def _iter_rank_local_checkpoint_weights(model_loader, model_config, model):
    """Yield CPU checkpoint tensors after enabling the loader's rank filter."""
    initialize_ep_filter = getattr(model_loader, "_init_ep_weight_filter", None)
    if initialize_ep_filter is not None:
        initialize_ep_filter(model_config)
    for name, tensor in model_loader.get_all_weights(model_config, model):
        if isinstance(tensor, DTensor):
            tensor = tensor.full_tensor()
        if tensor.device.type != "cpu":
            raise RuntimeError(
                f"Node-shared reward checkpoint loader must yield CPU tensors, but {name!r} is on {tensor.device}."
            )
        yield name, tensor


def _load_model_weights_with_vllm_config(model, weights, vllm_config):
    """Load checkpoint tensors and convert them to the inference kernel layout."""
    with set_current_vllm_config(vllm_config):
        loaded_params = model.load_weights(weights=weights)
        if loaded_params is None:
            raise RuntimeError("Reward checkpoint loader returned no load result.")
        # Checkpoint loaders write canonical tensors even when dummy initialization packed the kernel weights.
        process_weights_after_loading(model, vllm_config.model_config, vllm_config.device_config.device)
        return loaded_params


"""
class NIXLWorker(Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # We do the device initialization here before building the NIXL client 
        self.device = torch.device(f"cuda:{self.local_rank}")
        current_platform.set_device(self.device)
        current_platform.check_if_supports_dtype(self.model_config.dtype)
        
        vllm_config = get_current_vllm_config()
        assert vllm_config.additional_config is not None, "additional_config must be provided when using NIXL"
        assert vllm_config.additional_config.get("nixl_config") is not None, \
            "nixl_config must be provided when using NIXL"
        assert vllm_config.additional_config.get("nixl_interface") is not None, \
            "nixl_interface must be provided when using NIXL"
        assert vllm_config.additional_config.get("instance_id") is not None, \
            "instance_id must be provided when using NIXL"
        self.nixl_config = vllm_config.additional_config.get("nixl_config")
        self.nixl_interface = vllm_config.additional_config.get("nixl_interface")
        self.instance_id = vllm_config.additional_config.get("instance_id")
        
        assert hasattr(self, "init_nixl_client"), "init_nixl_client must be provided when using NIXL"
        self.init_nixl_client(self.nixl_config, self.nixl_interface, self.instance_id)
"""


class vLLMWorkerExtension:
    def monkey_patch_model(self, vocab_size: int) -> None:
        """Apply model patches required before serving requests."""
        model = self.model_runner.model
        monkey_patch_compute_logits(model, vocab_size)
        patch_vllm_moe_model_weight_loader(model)

    def _maybe_tms_weights_region(self):
        """Tag temporary GPU allocations as weights when the TMS vLLM patch is active."""
        if not os.environ.get("PSRL_VLLM_PATCHES", "").startswith("TMS"):
            return nullcontext()
        try:
            from torch_memory_saver import torch_memory_saver
        except ImportError:
            return nullcontext()
        additional_config = self.model_runner.vllm_config.additional_config or {}
        enable_cpu_backup = bool(additional_config.get("enable_weights_cpu_backup", False))
        return torch_memory_saver.region(tag="weights", enable_cpu_backup=enable_cpu_backup)

    def preload_weights_to_cpu_cache(self, weights_path: str | None = None, load_format: str | None = None) -> int:
        """
        Prepare checkpoint-backed CPU weights for reward-model wake up.

        Node-shared mode keeps lazy safetensors mappings so ranks share physical
        page-cache backing. Process-local mode retains the eager CPU clone.
        """
        try:
            model = self.model_runner.model
            if isinstance(model, CUDAGraphWrapper):
                model = model.unwrap()

            model_config = copy(self.model_config)
            model_id = str(weights_path if weights_path is not None else model_config.model)
            if weights_path is not None:
                model_config.model = copy_to_local(weights_path)

            load_config = deepcopy(self.model_runner.load_config)
            if load_format is not None:
                # Reward models are initialized with dummy weights on GPU, but
                # the CPU cache must come from the real checkpoint.
                load_config.load_format = "auto" if str(load_format).startswith("dummy") else load_format

            arena_config = (self.model_runner.vllm_config.additional_config or {}).get("psrl_nixl_weight_arena", {})
            if arena_config.get("reward_enabled", False):
                from psrl.utils.node_shared_weight_cache import validate_reward_cache_config

                cache_mode = validate_reward_cache_config(arena_config)
            else:
                cache_mode = "process_local"
            if cache_mode == "node_shared":
                if not hasattr(self.model_runner, "_psrl_weight_arena_handle"):
                    raise RuntimeError("Node-shared reward CPU cache requires an initialized reward weight arena.")
                from psrl.utils.node_shared_weight_cache import fingerprint_safetensors_checkpoint

                checkpoint_fingerprint = fingerprint_safetensors_checkpoint(model_config.model)
                checkpoint_load_format = str(load_config.load_format).lower()
                if checkpoint_load_format not in ("auto", "safetensors"):
                    raise NotImplementedError(
                        "Node-shared reward checkpoint loading requires load_format "
                        f"'auto' or 'safetensors', got {checkpoint_load_format!r}."
                    )
                # All ranks map the same read-only checkpoint pages. The model's
                # rank-local vLLM loaders then slice/filter on CPU before copying
                # only their TP/EP shard into the final GPU arena.
                load_config.load_format = "safetensors"
                load_config.safetensors_load_strategy = "lazy"
                load_config.model_loader_extra_config = {}
                model_loader = get_model_loader(load_config)
                if not hasattr(model_loader, "get_all_weights"):
                    raise NotImplementedError(
                        f"Node-shared CPU weight cache does not support load format `{load_config.load_format}`."
                    )
                self._psrl_streaming_weight_source = {
                    "model_config": model_config,
                    "load_config": load_config,
                    "model_id": model_id,
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                }
                self._psrl_reward_weight_cache_state = {
                    "ready": True,
                    "source": "node_shared_checkpoint",
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                }
                psrl_logger.info("Prepared safetensors streaming source for node-shared reward cache.")
                return 0

            model_loader = get_model_loader(load_config)
            if not hasattr(model_loader, "get_all_weights"):
                raise NotImplementedError(
                    f"CPU weight cache does not support load format `{load_config.load_format}`."
                )

            cache: list[tuple[str, torch.Tensor]] = []
            for name, tensor in model_loader.get_all_weights(model_config, model):
                if isinstance(tensor, DTensor):
                    tensor = tensor.full_tensor()
                cache.append((name, tensor.detach().cpu().clone()))

            self._psrl_cpu_weight_cache = cache
            self._psrl_reward_weight_cache_state = {
                "ready": True,
                "source": "checkpoint",
                "checkpoint_tensor_count": len(cache),
            }
            psrl_logger.info("Preloaded %d reward model tensors to CPU cache.", len(cache))
            return len(cache)
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.preload_weights_to_cpu_cache: {e}") from e

    def _publish_node_shared_reward_arena(
        self,
        arena_handle,
        arena_config: dict,
        model_id: str,
        checkpoint_fingerprint: str,
        populate_handle=None,
    ):
        """Publish or map this worker's topology-specific reward arena."""
        from vllm.distributed.parallel_state import get_ep_group, get_pp_group, get_tp_group

        from psrl.utils.node_shared_weight_cache import (
            WeightCacheTopology,
            publish_or_map_weight_arena,
        )
        from psrl.utils.weight_arena import gb_to_bytes

        tp_group = get_tp_group()
        pp_group = get_pp_group()
        parallel_config = self.model_runner.vllm_config.parallel_config
        if parallel_config.enable_expert_parallel:
            ep_group = get_ep_group()
            ep_size = ep_group.world_size
            ep_rank = ep_group.rank_in_group
        else:
            ep_size = 1
            ep_rank = 0
        topology = WeightCacheTopology(
            tp_size=tp_group.world_size,
            tp_rank=tp_group.rank_in_group,
            pp_size=pp_group.world_size,
            pp_rank=pp_group.rank_in_group,
            ep_size=ep_size,
            ep_rank=ep_rank,
        )
        runtime_context = ray.get_runtime_context()
        job_id_value = runtime_context.get_job_id()
        job_id = job_id_value.hex() if callable(getattr(job_id_value, "hex", None)) else str(job_id_value)
        shared_cache = publish_or_map_weight_arena(
            arena_handle,
            cache_root=arena_config.get("reward_node_cache_dir", "/dev/shm/psrl-rm-weight-cache"),
            job_id=job_id,
            node_id=runtime_context.get_node_id(),
            model_id=model_id,
            checkpoint_fingerprint=checkpoint_fingerprint,
            model_dtype=str(self.model_config.dtype),
            quantization=(
                None
                if getattr(self.model_config, "quantization", None) is None
                else str(self.model_config.quantization)
            ),
            topology=topology,
            timeout_s=float(arena_config.get("reward_node_cache_wait_timeout_s", 1800)),
            populate_handle=populate_handle,
            backend=str(arena_config.get("reward_node_cache_backend", "auto")),
            shm_reserve_bytes=gb_to_bytes(
                arena_config.get("reward_node_cache_shm_reserve_gb", 256),
                field_name="reward_node_cache_shm_reserve_gb",
                allow_zero=True,
            ),
            memfd_reserve_bytes=gb_to_bytes(
                arena_config.get("reward_node_cache_memfd_reserve_gb", 256),
                field_name="reward_node_cache_memfd_reserve_gb",
                allow_zero=True,
            ),
        )
        diagnostics = shared_cache.diagnostics()
        psrl_logger.warning(
            "[RM_NODE_SHARED_WEIGHT_CACHE] backend=%s node_id=%s cache_key=%s role=%s "
            "tp_rank=%d pp_rank=%d ep_rank=%d mapped_bytes=%d unique_cache_bytes=%d "
            "inode=%d private_bytes=%s pss_bytes=%s.",
            diagnostics["backend"],
            diagnostics["node_id"],
            diagnostics["cache_key"],
            diagnostics["role"],
            topology.tp_rank,
            topology.pp_rank,
            topology.ep_rank,
            diagnostics["mapped_bytes"],
            diagnostics["unique_cache_bytes"],
            diagnostics["inode"],
            diagnostics["private_bytes"],
            diagnostics["pss_bytes"],
        )
        return shared_cache

    def close_node_shared_weight_cache(self) -> None:
        """Close this worker's mmap views and release its job namespace lease."""
        shared_cache = getattr(self, "_psrl_node_shared_arena_cache", None)
        if shared_cache is None:
            return
        if hasattr(self, "_psrl_weight_arena_cpu_cache"):
            del self._psrl_weight_arena_cpu_cache
        gc.collect()
        shared_cache.close()
        del self._psrl_node_shared_arena_cache

    def load_weights_from_cpu_cache(self, blocking: bool = True):
        """
        Load reward-model weights from the worker-local CPU cache to GPU.
        """
        total_start = time.perf_counter()
        gpu_copy_elapsed_s = 0.0
        empty_cache_elapsed_s = 0.0
        arena_snapshot_elapsed_s = 0.0
        cache_release_elapsed_s = 0.0
        arena_restore_bytes = 0
        cache_source = "checkpoint"
        cache_transitioned = False
        model_validation_elapsed_s = 0.0
        arena_handle = getattr(self.model_runner, "_psrl_weight_arena_handle", None)
        arena_cache = getattr(self, "_psrl_weight_arena_cpu_cache", None)
        try:
            model = self.model_runner.get_model()
            cache_state = getattr(self, "_psrl_reward_weight_cache_state", None)
            if arena_handle is not None:
                validation_start = time.perf_counter()
                arena_handle.assert_module_weights_in_arena(model)
                model_validation_elapsed_s += time.perf_counter() - validation_start

            if arena_cache is not None:
                if arena_handle is None:
                    raise RuntimeError("Reward arena CPU cache exists without a weight arena handle.")
                if (
                    cache_state is None
                    or not cache_state.get("ready")
                    or cache_state.get("source") not in ("arena", "node_shared_arena")
                ):
                    raise RuntimeError(f"Reward arena CPU cache has invalid state: {cache_state!r}.")
                from psrl.utils.weight_arena import restore_weight_arena_from_cpu

                cache_source = cache_state["source"]
                torch.cuda.synchronize()
                gpu_copy_start = time.perf_counter()
                arena_restore_bytes = restore_weight_arena_from_cpu(arena_handle, arena_cache)
                gpu_copy_elapsed_s = time.perf_counter() - gpu_copy_start
                shared_cache = getattr(self, "_psrl_node_shared_arena_cache", None)
                if shared_cache is not None:
                    shared_cache.drop_resident_pages()
                validation_start = time.perf_counter()
                arena_handle.assert_module_weights_in_arena(model)
                model_validation_elapsed_s += time.perf_counter() - validation_start
                loaded_params = {"__psrl_weight_arena_restore__"}
            else:
                streaming_source = getattr(self, "_psrl_streaming_weight_source", None)
                process_local_source = hasattr(self, "_psrl_cpu_weight_cache")
                if streaming_source is None and not process_local_source:
                    raise RuntimeError("Reward CPU weight source is not initialized.")
                expected_source = "node_shared_checkpoint" if streaming_source is not None else "checkpoint"
                if cache_state is None or not cache_state.get("ready") or cache_state.get("source") != expected_source:
                    raise RuntimeError(f"Reward checkpoint CPU cache has invalid state: {cache_state!r}.")
                if arena_handle is not None and not blocking:
                    raise RuntimeError("Reward arena snapshot requires blocking=True.")

                current_device = None if streaming_source is not None else torch.cuda.current_device()

                def cached_weights_generator():
                    if streaming_source is not None:
                        model_loader = get_model_loader(streaming_source["load_config"])
                        yield from _iter_rank_local_checkpoint_weights(
                            model_loader,
                            streaming_source["model_config"],
                            model,
                        )
                        return
                    for name, tensor in self._psrl_cpu_weight_cache:
                        if isinstance(tensor, DTensor):
                            tensor = tensor.full_tensor()
                        yield (name, tensor.to(current_device, non_blocking=True))

                arena_config = (self.model_runner.vllm_config.additional_config or {}).get(
                    "psrl_nixl_weight_arena", {}
                )
                if arena_config.get("reward_enabled", False):
                    from psrl.utils.node_shared_weight_cache import validate_reward_cache_config

                    cache_mode = validate_reward_cache_config(arena_config)
                else:
                    cache_mode = "process_local"
                node_shared_first_load = (
                    arena_handle is not None and cache_mode == "node_shared" and streaming_source is not None
                )
                if node_shared_first_load:
                    builder_timings: dict[str, float] = {}

                    def populate_shared_arena() -> None:
                        nonlocal loaded_params
                        torch.cuda.synchronize()
                        checkpoint_load_start = time.perf_counter()
                        with self._maybe_tms_weights_region():
                            loaded_params = _load_model_weights_with_vllm_config(
                                model,
                                cached_weights_generator(),
                                self.model_runner.vllm_config,
                            )
                        if loaded_params is None:
                            raise RuntimeError("Reward checkpoint loader returned no load result.")
                        torch.cuda.synchronize()
                        builder_timings["checkpoint_load_s"] = time.perf_counter() - checkpoint_load_start
                        validation_start = time.perf_counter()
                        arena_handle.assert_module_weights_in_arena(model)
                        builder_timings["validation_s"] = time.perf_counter() - validation_start
                        empty_cache_start = time.perf_counter()
                        aggressive_empty_cache(force_sync=True)
                        builder_timings["empty_cache_s"] = time.perf_counter() - empty_cache_start

                    snapshot_start = time.perf_counter()
                    shared_cache = self._publish_node_shared_reward_arena(
                        arena_handle,
                        arena_config,
                        streaming_source["model_id"],
                        streaming_source["checkpoint_fingerprint"],
                        populate_handle=populate_shared_arena,
                    )
                    arena_snapshot_elapsed_s = time.perf_counter() - snapshot_start
                    self._psrl_node_shared_arena_cache = shared_cache
                    self._psrl_weight_arena_cpu_cache = shared_cache.arenas
                    if shared_cache.builder:
                        cache_source = "node_shared_checkpoint_builder"
                        gpu_copy_elapsed_s = builder_timings.get("checkpoint_load_s", 0.0)
                        empty_cache_elapsed_s = builder_timings.get("empty_cache_s", 0.0)
                        model_validation_elapsed_s += builder_timings.get("validation_s", 0.0)
                    else:
                        from psrl.utils.weight_arena import restore_weight_arena_from_cpu

                        cache_source = "node_shared_arena_consumer"
                        torch.cuda.synchronize()
                        gpu_copy_start = time.perf_counter()
                        arena_restore_bytes = restore_weight_arena_from_cpu(
                            arena_handle,
                            shared_cache.arenas,
                        )
                        torch.cuda.synchronize()
                        gpu_copy_elapsed_s = time.perf_counter() - gpu_copy_start
                        shared_cache.drop_resident_pages()
                        loaded_params = {"__psrl_weight_arena_restore__"}
                        validation_start = time.perf_counter()
                        arena_handle.assert_module_weights_in_arena(model)
                        model_validation_elapsed_s += time.perf_counter() - validation_start
                    arena_bytes = sum(
                        tensor.numel() * tensor.element_size() for tensor in self._psrl_weight_arena_cpu_cache
                    )
                    self._psrl_reward_weight_cache_state = {
                        "ready": True,
                        "source": "node_shared_arena",
                        "arena_count": len(self._psrl_weight_arena_cpu_cache),
                        "arena_bytes": arena_bytes,
                        "pin_memory": False,
                        **shared_cache.diagnostics(),
                    }
                    cache_transitioned = True
                    release_start = time.perf_counter()
                    del self._psrl_streaming_weight_source
                    gc.collect()
                    cache_release_elapsed_s = time.perf_counter() - release_start
                else:
                    torch.cuda.synchronize()
                    gpu_copy_start = time.perf_counter()
                    with self._maybe_tms_weights_region():
                        loaded_params = _load_model_weights_with_vllm_config(
                            model,
                            cached_weights_generator(),
                            self.model_runner.vllm_config,
                        )
                    if loaded_params is None:
                        raise RuntimeError("Reward checkpoint loader returned no load result.")
                    if blocking:
                        torch.cuda.synchronize()
                        gpu_copy_elapsed_s = time.perf_counter() - gpu_copy_start
                        if arena_handle is not None:
                            validation_start = time.perf_counter()
                            arena_handle.assert_module_weights_in_arena(model)
                            model_validation_elapsed_s += time.perf_counter() - validation_start
                        # Release checkpoint-layout CPU-to-GPU staging tensors before
                        # taking the persistent arena snapshot.
                        empty_cache_start = time.perf_counter()
                        aggressive_empty_cache(force_sync=True)
                        empty_cache_elapsed_s = time.perf_counter() - empty_cache_start

                if arena_handle is not None and not node_shared_first_load:
                    pin_memory = bool(arena_config.get("reward_cpu_cache_pin_memory", False))
                    snapshot_start = time.perf_counter()
                    if cache_mode == "node_shared":
                        if streaming_source is None:
                            raise RuntimeError(
                                "Node-shared reward arena publication requires a streaming checkpoint source."
                            )
                        shared_cache = self._publish_node_shared_reward_arena(
                            arena_handle,
                            arena_config,
                            streaming_source["model_id"],
                            streaming_source["checkpoint_fingerprint"],
                        )
                        self._psrl_node_shared_arena_cache = shared_cache
                        self._psrl_weight_arena_cpu_cache = shared_cache.arenas
                    else:
                        from psrl.utils.weight_arena import snapshot_weight_arena_to_cpu

                        self._psrl_weight_arena_cpu_cache = snapshot_weight_arena_to_cpu(
                            arena_handle,
                            pin_memory=pin_memory,
                        )
                    arena_snapshot_elapsed_s = time.perf_counter() - snapshot_start
                    arena_bytes = sum(
                        tensor.numel() * tensor.element_size() for tensor in self._psrl_weight_arena_cpu_cache
                    )
                    self._psrl_reward_weight_cache_state = {
                        "ready": True,
                        "source": "node_shared_arena" if cache_mode == "node_shared" else "arena",
                        "arena_count": len(self._psrl_weight_arena_cpu_cache),
                        "arena_bytes": arena_bytes,
                        "pin_memory": pin_memory,
                    }
                    if cache_mode == "node_shared":
                        self._psrl_reward_weight_cache_state.update(shared_cache.diagnostics())
                    cache_transitioned = True
                    release_start = time.perf_counter()
                    if streaming_source is not None:
                        del self._psrl_streaming_weight_source
                    else:
                        del self._psrl_cpu_weight_cache
                    gc.collect()
                    cache_release_elapsed_s = time.perf_counter() - release_start
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.load_weights_from_cpu_cache: {e}") from e
        elapsed_s = time.perf_counter() - total_start
        arena_count = len(arena_handle.arenas) if arena_handle is not None else 0
        arena_bytes = sum(arena.numel() * arena.element_size() for arena in arena_handle.arenas) if arena_handle else 0
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["load_weights_from_cpu_cache"] = {
            "cache_source": cache_source,
            "gpu_copy_elapsed_s": gpu_copy_elapsed_s,
            "empty_cache_elapsed_s": empty_cache_elapsed_s,
            "arena_snapshot_elapsed_s": arena_snapshot_elapsed_s,
            "cache_release_elapsed_s": cache_release_elapsed_s,
            "arena_restore_bytes": arena_restore_bytes,
            "arena_count": arena_count,
            "arena_bytes": arena_bytes,
            "cache_transitioned": cache_transitioned,
            "model_validation_elapsed_s": model_validation_elapsed_s,
            "elapsed_s": elapsed_s,
        }
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker operation=wake "
            "stage=load_weights_from_cpu_cache host=%s rank=%s tp_rank=%s pid=%s "
            "cache_source=%s gpu_copy_elapsed_s=%.6f empty_cache_elapsed_s=%.6f "
            "arena_snapshot_elapsed_s=%.6f cache_release_elapsed_s=%.6f "
            "arena_restore_bytes=%d arena_count=%d arena_bytes=%d cache_transitioned=%s "
            "model_validation_elapsed_s=%.6f elapsed_s=%.6f",
            socket.gethostname(),
            self.get_instance_local_rank(),
            self.get_instance_local_tp_rank(),
            os.getpid(),
            cache_source,
            gpu_copy_elapsed_s,
            empty_cache_elapsed_s,
            arena_snapshot_elapsed_s,
            cache_release_elapsed_s,
            arena_restore_bytes,
            arena_count,
            arena_bytes,
            cache_transitioned,
            model_validation_elapsed_s,
            elapsed_s,
        )
        return True

    def load_weights(self, weights, blocking: bool = True):
        """
        Load weights into the vLLM model runner.

        This method rebuilds the weights using the provided function and arguments stemming from `reduce_tensor` calls,
        transfers them to the current CUDA device, and loads them into the vLLM model runner.
        If the weight is a DTensor, it converts it to a full tensor before loading.
        If `blocking` is True, it ensures that all operations are completed before returning.
        If an error occurs during the process, it logs the error and returns None.

        Args:
            weights (List[tuple]): A list of tuples where each tuple contains:
                - name (str): The name of the weight.
                - handle (tuple): A tuple containing the function and its arguments to rebuild the weight.
            blocking (bool): If True, will block until all operations are completed.

        Returns:
            loaded_params: The loaded parameters from the model runner.

        Raises:
            Exception: If there is an error during the loading process.
        """

        def rebuild_weights_generator():
            current_device = torch.cuda.current_device()
            for name, handle in weights:
                func, args = handle
                list_args = list(args)
                # CPU bundle: (type(tensor), storage, metadata)
                if len(list_args) == 3:
                    tensor = func(*list_args)
                    tensor = tensor.to(current_device, non_blocking=True)
                    if isinstance(tensor, DTensor):
                        tensor = tensor.full_tensor()
                else:
                    list_args[6] = get_device_id()
                    tensor = func(*list_args)
                    if isinstance(tensor, DTensor):
                        tensor = tensor.full_tensor()
                yield (name, tensor)

        rebuild_weights = rebuild_weights_generator()
        torch.cuda.synchronize()
        loaded_params = self.model_runner.model.load_weights(weights=rebuild_weights)
        if blocking:
            # Ensure all operations are completed before returning
            torch.cuda.synchronize()
            with self._maybe_tms_weights_region():
                loaded_params = self.model_runner.model.load_weights(weights=rebuild_weights)
            if blocking:
                # Ensure all operations are completed before returning
                torch.cuda.synchronize()
        return loaded_params

    def cuda_synchronize(self):
        """Synchronize the CUDA device."""
        try:
            torch.cuda.synchronize()
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.cuda_synchronize: {e}") from e
        return None

    def get_gpu_memory_snapshot(self) -> dict[str, int | str]:
        """Return process-local and device-wide CUDA memory from this vLLM worker."""
        try:
            device = torch.cuda.current_device()
            torch.cuda.synchronize(device)
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            properties = torch.cuda.get_device_properties(device)
            return {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "node_id": ray.get_runtime_context().get_node_id(),
                "rank": self.get_instance_local_rank(),
                "tp_rank": self.get_instance_local_tp_rank(),
                "device": device,
                "device_name": properties.name,
                "device_uuid": str(getattr(properties, "uuid", "unknown")),
                "torch_allocated_bytes": torch.cuda.memory_allocated(device),
                "torch_reserved_bytes": torch.cuda.memory_reserved(device),
                "device_used_bytes": total_bytes - free_bytes,
                "device_free_bytes": free_bytes,
                "device_total_bytes": total_bytes,
            }
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.get_gpu_memory_snapshot: {e}") from e

    def get_last_tms_sleep_wake_timing(self, operation: str) -> dict:
        """Return the last TMS timing captured inside this GPU worker."""
        if operation not in ("sleep", "wake"):
            raise ValueError(f"Unsupported TMS timing operation: {operation}")
        return {
            "hostname": socket.gethostname(),
            "rank": self.get_instance_local_rank(),
            "tp_rank": self.get_instance_local_tp_rank(),
            "pid": os.getpid(),
            "timing": getattr(self, f"_psrl_last_{operation}_timing", None),
        }

    def get_last_worker_stage_timing(self, stage: str) -> dict:
        """Return the latest timing for a worker-extension stage."""
        timings = getattr(self, "_psrl_worker_stage_timings", {})
        return {
            "hostname": socket.gethostname(),
            "rank": self.get_instance_local_rank(),
            "tp_rank": self.get_instance_local_tp_rank(),
            "pid": os.getpid(),
            "timing": timings.get(stage),
        }

    def get_weight_arena_info(self) -> dict:
        """Return JSON-safe arena and latest NIXL registration state."""
        handle = getattr(self.model_runner, "_psrl_weight_arena_handle", None)
        arena_cache = getattr(self, "_psrl_weight_arena_cpu_cache", None)
        cache_state = getattr(self, "_psrl_reward_weight_cache_state", None)
        cache_info = None
        if arena_cache is not None:
            cache_info = {
                "arena_count": len(arena_cache),
                "total_bytes": sum(tensor.numel() * tensor.element_size() for tensor in arena_cache),
                "pin_memory": all(tensor.is_pinned() for tensor in arena_cache),
            }
            shared_cache = getattr(self, "_psrl_node_shared_arena_cache", None)
            if shared_cache is not None:
                cache_info.update(shared_cache.diagnostics())
        registration_timing = None
        nixl_client = getattr(self, "nixl_storage_client", None)
        if nixl_client is not None:
            registration_timing = getattr(nixl_client, "_last_reregister_timing", None)
        if handle is None:
            return {
                "enabled": False,
                "hostname": socket.gethostname(),
                "node_id": ray.get_runtime_context().get_node_id(),
                "rank": self.get_instance_local_rank(),
                "tp_rank": self.get_instance_local_tp_rank(),
                "pid": os.getpid(),
                "stats": None,
                "base_addresses": [],
                "reward_cpu_cache": cache_info,
                "reward_cpu_cache_state": cache_state,
                "nixl_registration": registration_timing,
            }
        handle.assert_virtual_addresses_unchanged()
        return {
            "enabled": True,
            "hostname": socket.gethostname(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "rank": self.get_instance_local_rank(),
            "tp_rank": self.get_instance_local_tp_rank(),
            "pid": os.getpid(),
            "stats": handle.stats.to_dict(),
            "base_addresses": list(handle.base_addresses),
            "reward_cpu_cache": cache_info,
            "reward_cpu_cache_state": cache_state,
            "nixl_registration": registration_timing,
        }

    def get_worker_stage_timing(self, stage: str) -> dict:
        """Return the latest extension timing with arena metadata."""
        return self.get_last_worker_stage_timing(stage)

    def patch_vllm_moe_model_weight_loader(self) -> None:
        """Patch the vLLM model weight loader for MoE models."""
        try:
            vllm_model = self.model_runner.model
            if isinstance(vllm_model, CUDAGraphWrapper):
                vllm_model = vllm_model.unwrap()
            patch_vllm_moe_model_weight_loader(vllm_model)
        except Exception as e:
            raise ValueError(f"Error in vLLMWorkerExtension.patch_vllm_moe_model_weight_loader: {e}") from e
        return None

    # ----------------------------- NIXL Related -----------------------------
    # Because the model is on another process since vllm V1, we must call the nixl methods via rpc
    def get_instance_local_rank(self):
        from vllm.distributed.parallel_state import get_world_group

        return get_world_group().rank

    def get_instance_local_tp_rank(self):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()

    def get_instance_local_ep_rank(self):
        from vllm.distributed.parallel_state import get_ep_group

        try:
            return get_ep_group().rank_in_group
        except AssertionError:
            # Non-MoE models do not initialize an EP group.
            return 0

    def init_nixl_client(
        self,
        nixl_config: DictConfig,
        replica_idx: int,
        logging_path: str | None = None,
    ):
        # NIXL attributes
        self.unified_state_dict = None
        self.unified_sharding_dict = None
        self.local_sharding_dict = None
        self.param_sync_plan = ParamSyncPlan()
        self.psrl_instance_id = int(replica_idx)
        # Initialize the NIXL client
        self.nixl_storage_client = NIXLStorageClient(
            client_name=gen_client_name(replica_idx, self.get_instance_local_rank()),
            server_name=NIXL_META_SERVER_NAME,
            use_gpu=True,
            client_type=NIXLClientType.PULL_SIDE,
            nixl_config=nixl_config,
            replica_idx=replica_idx,
            worker_index=self.get_instance_local_rank(),
            logging_path=logging_path,
        )
        psrl_logger.info(f"NIXL client initialized on port {self.nixl_storage_client.client_port}.")

    def nixl_convert_params(self, model_config: DictConfig):
        """Convert the model parameters to unified format.

        Args:
            config (DictConfig): Configuration object containing training settings.
        """
        vllm_model = self.model_runner.model
        if isinstance(vllm_model, CUDAGraphWrapper):
            vllm_model = vllm_model.unwrap()
        param_mapping = (
            None if supports_weight_layout(vllm_model) else create_parameter_mapping(type(vllm_model), model_config)
        )
        self.unified_state_dict, self.local_sharding_dict = convert_vllm_inplace(
            vllm_model,
            tp_rank=self.get_instance_local_tp_rank(),
            ep_rank=self.get_instance_local_ep_rank(),
            parameter_mapping=param_mapping,
        )
        fp32_patterns = () if param_mapping is None else param_mapping.get_external_fp32_param_patterns()
        fp32_keys = [
            key
            for key, tensor in self.unified_state_dict.items()
            if tensor.dtype == torch.float32 and any(pattern in key for pattern in fp32_patterns)
        ]
        if fp32_keys:
            psrl_logger.info(
                "vLLM precision invariant validated: fp32_parameters=%d patterns=%s",
                len(fp32_keys),
                fp32_patterns,
            )
        if self.param_sync_plan.actions:
            psrl_logger.info(
                "vLLM canonical parameter sync plan created: actions=%d keys=%s",
                len(self.param_sync_plan.actions),
                [action.key for action in self.param_sync_plan.actions],
            )

    def nixl_protocol(self, model_config: DictConfig, mode: str = "full"):
        """Run the NIXL server protocol.

        Args:
            model_config (DictConfig): Configuration object containing training settings.
            mode (str): Mode of registration, either 'meta' or 'full'.
                'meta' mode converts to meta tensors and skip registering their memory.
                'full' mode converts to full tensors.

            NOTE: ps storage may init with meta tensors, the register step would be different.
        """
        # Register the state dict and sharding dict to the NIXL client
        meta_only = mode == "meta"
        if self.unified_state_dict is None or self.local_sharding_dict is None:
            self.nixl_convert_params(model_config)
        psrl_logger.info("nixl client protocol step 1: connect_to_server")
        self.nixl_storage_client.connect_to_server()
        psrl_logger.info("nixl client protocol step 2: send_local_sharding")
        self.nixl_storage_client.send_local_sharding(self.local_sharding_dict)
        psrl_logger.info("nixl client protocol step 3: wait_for_server_sharding")
        unified_sharding_dict = self.nixl_storage_client.wait_for_server_sharding()
        # psrl_logger.info(f"unified_sharding_dict: {unified_sharding_dict}")
        psrl_logger.info("nixl client protocol step 4: register_local_tensors")
        self.nixl_storage_client.register_local_tensors(
            self.unified_state_dict, unified_sharding_dict, meta_only=meta_only
        )
        psrl_logger.info("nixl client protocol step 5: send_local_info")
        self.nixl_storage_client.send_local_info()
        psrl_logger.info("nixl client protocol step 6: wait_for_server_info")
        self.nixl_storage_client.wait_for_server_info()
        psrl_logger.info("nixl client protocol step 7: send_local_temp_mapping")
        self.nixl_storage_client.send_local_temp_mapping()
        psrl_logger.info("nixl client protocol step 8: wait_for_server_temp_mappings")
        self.nixl_storage_client.wait_for_server_temp_mappings()
        psrl_logger.info("nixl client protocol done.")
        self.unified_sharding_dict = unified_sharding_dict

    def nixl_register_after_wake_up(self):
        """Register the model parameters to NIXL after wake up from sleep.

        After sleep/wake_up, the physical memory backing the model weights
        has changed while virtual addresses remain the same. This method performs
        local re-registration:
        1. Reset nixl agent (clears UCX rcache)
        2. Re-registers memory with new physical pages (generates new rkeys)
        """
        total_start = time.perf_counter()
        sync_start = time.perf_counter()
        torch.cuda.synchronize()
        sync_elapsed_s = time.perf_counter() - sync_start
        arena_handle = getattr(self.model_runner, "_psrl_weight_arena_handle", None)
        arena_addresses = arena_handle.assert_virtual_addresses_unchanged() if arena_handle is not None else None
        # Reset nixl agent and reregister to handle physical memory changes
        register_start = time.perf_counter()
        self.nixl_storage_client.register_local_tensors(self.unified_state_dict, self.unified_sharding_dict)
        register_elapsed_s = time.perf_counter() - register_start
        elapsed_s = time.perf_counter() - total_start
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["nixl_register_after_wake_up"] = {
            "cuda_sync_elapsed_s": sync_elapsed_s,
            "register_elapsed_s": register_elapsed_s,
            "elapsed_s": elapsed_s,
            "arena_virtual_addresses": arena_addresses,
        }
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker operation=wake "
            "stage=nixl_register_after_wake_up host=%s rank=%s tp_rank=%s pid=%s "
            "cuda_sync_elapsed_s=%.6f register_elapsed_s=%.6f elapsed_s=%.6f "
            "arena_address_check=%s arena_count=%d",
            socket.gethostname(),
            self.get_instance_local_rank(),
            self.get_instance_local_tp_rank(),
            os.getpid(),
            sync_elapsed_s,
            register_elapsed_s,
            elapsed_s,
            "passed" if arena_addresses is not None else "disabled",
            len(arena_addresses or ()),
        )

    def nixl_deregister(self):
        """Deregister the model parameters from NIXL."""
        start = time.perf_counter()
        self.nixl_storage_client.deregister_local_tensors()
        elapsed_s = time.perf_counter() - start
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["nixl_deregister"] = {"elapsed_s": elapsed_s}
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker operation=sleep "
            "stage=nixl_deregister host=%s rank=%s tp_rank=%s pid=%s elapsed_s=%.6f",
            socket.gethostname(),
            self.get_instance_local_rank(),
            self.get_instance_local_tp_rank(),
            os.getpid(),
            elapsed_s,
        )

    def nixl_send_local_info_to(self, dst_agent_names: str | list[str]):
        """
        Send local NIXL info to the specified destination agents.
        """
        if isinstance(dst_agent_names, str):
            dst_agent_names = [dst_agent_names]
        self.nixl_storage_client.send_local_info_to(dst_agent_names)

    def nixl_wait_for_update_infos(self, info_num: int):
        """Wait for infos of updated clients for global synchronization.

        Args:
            info_num (int): Number of infos to wait for.
        """
        self.nixl_storage_client.wait_for_update_infos(info_num)

    def nixl_log_shard_info(self, label: str = "", max_elements: int = 8):
        """Debug log local NIXL shard info on this vLLM worker."""
        self.nixl_storage_client.log_shard_info(label=label, max_elements=max_elements)

    def _capture_weight_fingerprint(
        self,
        *,
        stage: str,
        model_version: int,
        options: dict | None,
    ) -> dict | None:
        if options is None:
            return None
        fingerprint = self.nixl_storage_client.get_weight_fingerprint(
            mode=options["mode"],
            sample_count=options["sample_count"],
            chunk_bytes=options["chunk_bytes"],
        )
        record = fingerprint_log_record(
            fingerprint,
            flow="transfer_chain",
            stage=stage,
            role="rollout",
            model_version=model_version,
            rank=self.get_instance_local_rank(),
            instance_id=self.psrl_instance_id,
            tp_rank=self.get_instance_local_tp_rank(),
            client_name=self.nixl_storage_client.client_name,
            include_tensor_digests=options["include_tensor_digests"],
        )
        return record

    def nixl_pull_model_core(
        self,
        ps_nixl_agent_names,
        ps_nixl_gen_storage_client_names,
        model_version: int = -1,
        fingerprint_options: dict | None = None,
    ):
        """Pull the model parameters from PS workers via NIXL.

        Args:
            ps_nixl_agent_names (list[str]): List of PS NIXL agent names
            ps_nixl_train_storage_client_names (list[str]): List of PS NIXL train storage client names
        """
        if not hasattr(self, "pull_times"):
            self.pull_times = 0
        self.pull_times += 1
        total_start = time.perf_counter()
        wait_operations = []
        issue_reads_start = time.perf_counter()
        for key in self.unified_state_dict:
            for target_agent_name, target_client_name in zip(ps_nixl_agent_names, ps_nixl_gen_storage_client_names):
                shards_to_transfer = self.nixl_storage_client.client_read(
                    target_agent_name,
                    target_client_name,
                    key,
                    f"gen_pull_{self.pull_times}",
                )
                if len(shards_to_transfer) > 0:
                    wait_operations.append((key, target_client_name, shards_to_transfer))
        issue_reads_elapsed_s = time.perf_counter() - issue_reads_start
        # Generation cannot be overlapped with the NIXL pull, so we need to wait for all operations to complete
        wait_start = time.perf_counter()
        for key, target_client_name, shards_to_transfer in wait_operations:
            self.nixl_storage_client.wait(
                key,
                f"gen_pull_{self.pull_times}",
                "READ",
                target_client=target_client_name,
            )
            # self.nixl_storage_client.wait(key, "gen_pull", "READ", target_client=target_client_name)
        wait_elapsed_s = time.perf_counter() - wait_start
        finish_start = time.perf_counter()
        self.nixl_storage_client.merge_and_finish_cached_xfer()
        self.cuda_synchronize()
        raw_fingerprint = self._capture_weight_fingerprint(
            stage="rollout_after_raw_pull",
            model_version=model_version,
            options=fingerprint_options,
        )
        sync_plan_start = time.perf_counter()
        self.param_sync_plan.after_pull(self.unified_state_dict)
        self.cuda_synchronize()
        final_fingerprint = self._capture_weight_fingerprint(
            stage="rollout_after_param_sync",
            model_version=model_version,
            options=fingerprint_options,
        )
        sync_plan_elapsed_s = time.perf_counter() - sync_plan_start
        report_precision_sensitive_parameters(
            self.unified_state_dict,
            logger=psrl_logger,
            role="rollout",
            rank=self.get_instance_local_rank(),
            pull_times=self.pull_times,
            tp_rank=self.get_instance_local_tp_rank(),
        )
        # self.nixl_log_shard_info(label=f"AFTER_GEN_PULL_{self.pull_times}")
        self.nixl_storage_client.clear_intermediate_cached_data()
        finish_elapsed_s = time.perf_counter() - finish_start
        elapsed_s = time.perf_counter() - total_start
        self._psrl_worker_stage_timings = getattr(self, "_psrl_worker_stage_timings", {})
        self._psrl_worker_stage_timings["nixl_pull_model_core"] = {
            "issue_reads_elapsed_s": issue_reads_elapsed_s,
            "wait_elapsed_s": wait_elapsed_s,
            "sync_plan_elapsed_s": sync_plan_elapsed_s,
            "finish_elapsed_s": finish_elapsed_s,
            "elapsed_s": elapsed_s,
        }
        psrl_logger.info(
            f"{self.nixl_storage_client}: NIXL pull model core done ({self.pull_times} times). time: {elapsed_s}s"
        )
        # vLLM V1 runs this extension in an EngineCore process, which does not
        # inherit the GenWorker's file logger. Return records to the caller so
        # the outer Ray actor can persist them in its component log.
        return {
            "model_version": model_version,
            "raw_fingerprint": raw_fingerprint,
            "final_fingerprint": final_fingerprint,
        }

    def get_total_kv_cache_tokens(self) -> int:
        """Total KV-cache token capacity of this instance, post-allocation.

        Uses vLLM's own `max_concurrency * max_model_len` formula (the same one
        behind the "GPU KV cache size: N tokens" startup log line), which is
        exact for hybrid/sliding-window layouts and already accounts for
        DCP/PCP sharding, unlike a naive `num_gpu_blocks * block_size`.
        """
        assert hasattr(self, "vllm_config"), "vllm_config must be set"
        kv_cache_config = getattr(self.model_runner, "kv_cache_config", None)
        if kv_cache_config is None or not kv_cache_config.kv_cache_groups:
            return 0
        max_concurrency = get_max_concurrency_for_kv_cache_config(self.vllm_config, kv_cache_config)
        return int(max_concurrency * self.vllm_config.model_config.max_model_len)

    # --- LMCache KV cache management methods ---
    # Invoked via `collective_rpc` from `KVCacheManager`.

    def _get_lmcache_engine(self):
        """
        Return the `LMCacheEngine` from the active KV transfer group.

        vLLM initializes the transfer group when `kv_transfer_config` is set.

        Returns:
            LMCacheEngine: The `lmcache_engine` from the active KV transfer group.
        """
        connector = get_kv_transfer_group()
        assert connector is not None, (
            "KV transfer group is None. Ensure kv_transfer_config is set during vLLM engine initialization."
        )
        assert hasattr(connector, "_lmcache_engine"), (
            f"Connector lacks _lmcache_engine: type={type(connector).__name__!r}. Expected LMCacheConnectorV1."
        )
        engine = connector._lmcache_engine.lmcache_engine
        assert engine is not None, "LMCacheEngine lmcache_engine is None."
        return engine

    def _get_lmcache_chunk_keys(self, tokens: list[int]) -> list:
        """
        Convert a token sequence to a list of `CacheEngineKey` objects.

        Uses the token database attached to the `LMCacheEngine`.

        Args:
            tokens (list[int]): Full token sequence.

        Returns:
            list: `CacheEngineKey` objects for each chunk.
        """
        engine = self._get_lmcache_engine()
        triples = engine.token_database.process_tokens(tokens)
        # `process_tokens` returns `(start, end, key)` triples. Extract the keys.
        return [key for _, _, key in triples]

    def _get_lmcache_total_bytes(self) -> int:
        """
        Return the total byte capacity of the LMCache local CPU backend allocator.

        Returns:
            int: Total bytes, or 0 if the allocator does not expose `total_size`.
        """
        engine = self._get_lmcache_engine()
        backend = engine.storage_manager.local_cpu_backend
        allocator = backend.memory_allocator
        # `MixedMemoryAllocator` / `PagedCpuGpuMemoryAllocator` expose `total_size`.
        return int(getattr(allocator, "total_size", 0))

    def lmcache_pin_backend(self, tokens: list[int]) -> int:
        """
        Pin the cached backend chunks for `tokens` to prevent LRU eviction.

        A held chunk has `ref_count == 2` and cannot be evicted. Exhausting all
        evictable chunks can block retrievals and reject new stores.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of chunks newly pinned (`ref_count` incremented).
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_memory_objs"):
            # Map each PSRL-pinned cache key to its memory object.
            self._psrl_pinned_memory_objs: dict = {}

        engine = self._get_lmcache_engine()
        keys = self._get_lmcache_chunk_keys(tokens)
        backend = engine.storage_manager.local_cpu_backend

        pinned = 0
        for key in keys:
            if key in self._psrl_pinned_memory_objs:
                # Repeated pins must not increment `ref_count`.
                continue
            # `get_blocking` raises `ref_count` to make the object ineligible for eviction.
            memory_obj = backend.get_blocking(key)
            if memory_obj is None:
                # Chunk not present in backend (prefix may be shorter than key list).
                continue
            self._psrl_pinned_memory_objs[key] = memory_obj
            pinned += 1

        # High pin pressure can reject stores and block retrievals.
        hot_cache_size = len(backend.hot_cache)
        if hot_cache_size > 0:
            psrl_pinned_total = len(self._psrl_pinned_memory_objs)
            usage_pct = psrl_pinned_total / hot_cache_size
            if usage_pct > 0.8:
                psrl_logger.warning(
                    f"[LMCache] Backend pin pressure: PSRL holds {psrl_pinned_total}/"
                    f"{hot_cache_size} hot_cache chunks pinned ({usage_pct:.0%}). "
                    "New store() calls may fail and retrieve() calls may spin "
                    "until a PSRL pin is released via lmcache_unpin_backend."
                )
        return pinned

    def lmcache_unpin_backend(self, tokens: list[int]) -> int:
        """
        Unpin the cached backend chunks for `tokens`, allowing LRU eviction.

        Only chunks tracked in `_psrl_pinned_memory_objs` are released, preventing
        interference with active request references.

        Args:
            tokens (list[int]): Full token sequence for the trajectory.

        Returns:
            int: Number of chunks whose `ref_count` was decremented.
        """
        assert tokens, "tokens must be a non-empty list."
        if not hasattr(self, "_psrl_pinned_memory_objs"):
            self._psrl_pinned_memory_objs: dict = {}

        keys = self._get_lmcache_chunk_keys(tokens)
        freed = 0
        for key in keys:
            memory_obj = self._psrl_pinned_memory_objs.pop(key, None)
            if memory_obj is None:
                continue
            # The hot cache and PSRL each hold one reference.
            assert memory_obj.get_ref_count() > 1, (
                f"Invalid backend reference count before unpin: ref_count={memory_obj.get_ref_count()}. "
                "Expected more than one reference. "
                "Possible double-unpin or external ref_count corruption."
            )
            memory_obj.ref_count_down()
            freed += 1
        return freed

    def lmcache_clear_all_from_backend(self) -> None:
        """
        Remove all cached KV chunks from the LMCache CPU backend.

        Weight updates must clear stale KV entries before subsequent requests.
        """
        engine = self._get_lmcache_engine()
        engine.clear()
