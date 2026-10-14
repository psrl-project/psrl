import gc
import logging
import os
import socket
import time
from contextlib import AbstractContextManager, nullcontext

import torch
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
from vllm.tracing import instrument
from vllm.utils.mem_utils import format_gib
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.worker.gpu_worker import Worker

from vllm_patches.core import min_vllm_version, vLLMPatch

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

_ORIGINAL_WORKER_LOAD_MODEL = Worker.load_model


def _should_restore_sleep_buffers(tags: list[str] | None) -> bool:
    return tags is None or "kv_cache" in tags


def _release_inactive_cuda_cache() -> dict[str, int]:
    """Release inactive blocks owned by this vLLM GPU worker process."""
    reserved_before = torch.cuda.memory_reserved()
    allocated_before = torch.cuda.memory_allocated()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    reserved_after = torch.cuda.memory_reserved()
    allocated_after = torch.cuda.memory_allocated()
    return {
        "reserved_before": reserved_before,
        "reserved_after": reserved_after,
        "reserved_freed": reserved_before - reserved_after,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
    }


@min_vllm_version("0.22.0")
class TMSWorkerPatch(vLLMPatch[Worker]):
    """
    Replace cuMemAllocator with torch_memory_saver
    for better memory management.

    Compatible with vLLM 0.22.0+
    """

    def load_model(self) -> None:
        _ORIGINAL_WORKER_LOAD_MODEL(self)
        from vllm_patches.patches.weight_arena import finalize_pending_weight_arena

        finalize_pending_weight_arena(self.model_runner)

    def log_tms_timing(self, operation: str, stage: str, elapsed_s: float, tag: str | None = None) -> None:
        timing = getattr(self, "_psrl_current_tms_timing", None)
        if timing is not None and timing.get("operation") == operation:
            timing["stages"].append(
                {
                    "stage": stage,
                    "tag": tag or "none",
                    "elapsed_s": elapsed_s,
                }
            )
            if stage == "total":
                setattr(self, f"_psrl_last_{operation}_timing", timing)
        psrl_logger.warning(
            "[VLLM_SLEEP_WAKE_TIMING] scope=tp_worker operation=%s stage=%s tag=%s "
            "host=%s rank=%s local_rank=%s pid=%s elapsed_s=%.6f",
            operation,
            stage,
            tag or "none",
            socket.gethostname(),
            self.rank,
            self.local_rank,
            os.getpid(),
            elapsed_s,
        )

    def sleep(self, level: int = 1) -> None:
        """Put the worker into sleep mode to free up GPU memory."""
        from torch_memory_saver import torch_memory_saver

        total_start = time.perf_counter()
        self._psrl_current_tms_timing = {"operation": "sleep", "stages": []}
        free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

        # Save the buffers before level 2 sleep
        if level == 2:
            stage_start = time.perf_counter()
            model = self.model_runner.model
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
            self.log_tms_timing("sleep", "save_buffers", time.perf_counter() - stage_start)

        if level == 1:
            raise NotImplementedError(
                "Level 1 sleep is not implemented for TMS because we always need to save kv cache."
            )
        else:
            stage_start = time.perf_counter()
            torch_memory_saver.pause("weights")
            self.log_tms_timing("sleep", "pause", time.perf_counter() - stage_start, tag="weights")
            stage_start = time.perf_counter()
            torch_memory_saver.pause("kv_cache")
            self.log_tms_timing("sleep", "pause", time.perf_counter() - stage_start, tag="kv_cache")
            if os.environ.get("PSRL_VLLM_PATCHES", "") == "TMS:GRAPH":
                stage_start = time.perf_counter()
                torch_memory_saver.pause("graph")
                self.log_tms_timing("sleep", "pause", time.perf_counter() - stage_start, tag="graph")

        # TMS only releases allocations in its tagged pools. Inference
        # temporaries use the worker's default caching allocator, so clear its
        # inactive blocks in this process before another colocated role wakes.
        stage_start = time.perf_counter()
        cache_stats = _release_inactive_cuda_cache()
        self.log_tms_timing(
            "sleep",
            "empty_cache",
            time.perf_counter() - stage_start,
            tag="default_allocator",
        )

        free_bytes_after_sleep, total = torch.cuda.mem_get_info()
        freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
        used_bytes = total - free_bytes_after_sleep
        assert freed_bytes >= 0, "Memory usage increased after sleeping."
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        gib = 1024**3
        psrl_logger.warning(
            "[VLLM_SLEEP_MEMORY] stage=worker_after_sleep host=%s rank=%s local_rank=%s "
            "pid=%s device=cuda:%s gpu_uuid=%s gpu_name=%s freed=%.2f GB "
            "cache_reserved_before=%.2f GB cache_reserved_after=%.2f GB cache_reserved_freed=%.2f GB "
            "cache_allocated_before=%.2f GB cache_allocated_after=%.2f GB "
            "torch_allocated=%.2f GB torch_reserved=%.2f GB "
            "device_used=%.2f GB device_free=%.2f GB device_total=%.2f GB",
            socket.gethostname(),
            self.rank,
            self.local_rank,
            os.getpid(),
            device,
            str(getattr(properties, "uuid", "unknown")),
            properties.name,
            freed_bytes / gib,
            cache_stats["reserved_before"] / gib,
            cache_stats["reserved_after"] / gib,
            cache_stats["reserved_freed"] / gib,
            cache_stats["allocated_before"] / gib,
            cache_stats["allocated_after"] / gib,
            torch.cuda.memory_allocated(device) / gib,
            torch.cuda.memory_reserved(device) / gib,
            used_bytes / gib,
            free_bytes_after_sleep / gib,
            total / gib,
        )
        self.log_tms_timing("sleep", "total", time.perf_counter() - total_start)
        # psrl_logger.info(
        #     "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
        #     format_gib(freed_bytes),
        #     format_gib(used_bytes),
        # )

    def wake_up(self, tags: list[str] | None = None) -> None:
        """Wake up from sleep mode and restore memory."""
        from torch_memory_saver import torch_memory_saver

        total_start = time.perf_counter()
        self._psrl_current_tms_timing = {"operation": "wake", "stages": []}
        free_bytes_before_wake_up = torch.cuda.mem_get_info()[0]

        for tag in tags:
            stage_start = time.perf_counter()
            torch_memory_saver.resume(tag)
            self.log_tms_timing("wake", "resume", time.perf_counter() - stage_start, tag=tag)

        # Restore the buffers after level 2 sleep
        if _should_restore_sleep_buffers(tags) and len(self._sleep_saved_buffers):
            stage_start = time.perf_counter()
            model = self.model_runner.model
            for name, buffer in model.named_buffers():
                if name in self._sleep_saved_buffers:
                    buffer.data.copy_(self._sleep_saved_buffers[name].data)
            torch.cuda.synchronize()
            self._sleep_saved_buffers = {}
            self.log_tms_timing("wake", "restore_buffers", time.perf_counter() - stage_start)

        if tags is None or "kv_cache" in tags:
            self.model_runner.post_kv_cache_wake_up()

        free_bytes_after_wake_up, total = torch.cuda.mem_get_info()
        increased_bytes = free_bytes_before_wake_up - free_bytes_after_wake_up
        used_bytes = total - free_bytes_after_wake_up
        assert increased_bytes >= 0, "Memory usage increased after waking up."
        psrl_logger.info(
            "Wake up mode increased %.2f GiB memory, %.2f GiB memory is still in use.",
            format_gib(increased_bytes),
            format_gib(used_bytes),
        )
        self.log_tms_timing("wake", "total", time.perf_counter() - total_start)

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
        """Get the memory pool context manager if sleep mode is enabled."""
        if not self.vllm_config.model_config.enable_sleep_mode:
            return nullcontext()

        from torch_memory_saver import torch_memory_saver

        enable_weights_cpu_backup = self.vllm_config.additional_config.get("enable_weights_cpu_backup", False)
        return torch_memory_saver.region(
            tag=tag,
            enable_cpu_backup=enable_weights_cpu_backup,
        )

    @instrument(span_name="Allocate KV cache")
    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        """Allocate GPU KV cache with the specified kv_cache_config."""

        # Update local config with adjusted num blocks after profiling,
        # so that it's available to the warmup stage.
        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks

        # Init kv cache connector here, because it requires
        # `kv_cache_config`.
        # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
        # because `initialize_kv_cache` will inject kv cache groups not
        # related to kv cache connector (e.g. kv cache sharing layers).
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        if self.vllm_config.model_config.enable_sleep_mode:
            from torch_memory_saver import torch_memory_saver

            with torch_memory_saver.region(tag="kv_cache"):
                self.model_runner.initialize_kv_cache(kv_cache_config)
        else:
            self.model_runner.initialize_kv_cache(kv_cache_config)

        if self.model_config.enable_return_routed_experts:
            self.model_runner.init_routed_experts_capturer()

        # Build KV-zero metadata outside the CuMem pool so the bookkeeping
        # GPU tensors (seg_addrs, block-id buffers) use the standard PyTorch
        # allocator and are not discarded during sleep/wake cycles.
        if kv_cache_config.needs_kv_cache_zeroing and hasattr(self.model_runner, "_init_kv_zero_meta"):
            self.model_runner._init_kv_zero_meta()
