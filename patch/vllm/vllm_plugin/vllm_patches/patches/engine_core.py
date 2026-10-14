import asyncio
import logging
import os
from datetime import datetime, timezone

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))


def apply_engine_core_patches() -> None:
    """
    Expose scheduler observations and memory controls through vLLM utilities.
    """
    _patch_engine_core_psrl_methods()
    _patch_engine_client_runtime_snapshots()


def _patch_engine_core_psrl_methods() -> None:
    """
    Keep memory restoration separate from scheduler resumption.
    """
    from vllm.v1.engine.core import EngineCore

    _SENTINEL = "_psrl_engine_core_methods_patched"

    if getattr(EngineCore, _SENTINEL, False):
        psrl_logger.debug("EngineCore PSRL utility methods are already patched.")
        return

    def psrl_get_gpu_cache_info(
        self,
        tokens: list,
    ) -> dict:
        return self.scheduler.psrl_get_gpu_cache_info(tokens)

    def psrl_pin_gpu(
        self,
        tokens: list,
    ) -> int:
        return self.scheduler.psrl_pin_gpu(tokens)

    def psrl_unpin_gpu(
        self,
        tokens: list,
    ) -> int:
        return self.scheduler.psrl_unpin_gpu(tokens)

    def psrl_get_runtime_snapshot(
        self,
    ) -> dict:
        stats = self.scheduler.psrl_get_runtime_stats()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scheduler_stats": stats,
            "is_sleeping": self.psrl_is_memory_sleeping(),
        }

    def psrl_is_memory_sleeping(
        self,
    ) -> bool:
        """
        Report executor sleep independently of scheduler pause state.
        """
        return self.model_executor.is_sleeping

    def psrl_wake_up_memory(
        self,
        tags: list[str],
    ) -> None:
        if not self.is_scheduler_paused():
            raise RuntimeError("Memory restoration requires a paused scheduler.")
        self.model_executor.wake_up(tags)

    EngineCore.psrl_get_gpu_cache_info = psrl_get_gpu_cache_info
    EngineCore.psrl_pin_gpu = psrl_pin_gpu
    EngineCore.psrl_unpin_gpu = psrl_unpin_gpu
    EngineCore.psrl_get_runtime_snapshot = psrl_get_runtime_snapshot
    EngineCore.psrl_is_memory_sleeping = psrl_is_memory_sleeping
    EngineCore.psrl_wake_up_memory = psrl_wake_up_memory
    setattr(EngineCore, _SENTINEL, True)

    psrl_logger.info("Patched EngineCore PSRL utility methods.")


def _patch_engine_client_runtime_snapshots() -> None:
    """
    Preserve every managed DP engine's utility response.
    """
    from vllm.v1.engine.core_client import AsyncMPClient

    async def psrl_get_runtime_snapshots_async(
        self,
    ) -> dict[int, dict]:
        engines = tuple(self.core_engines)
        snapshots = await asyncio.gather(*(
            self._call_utility_async("psrl_get_runtime_snapshot", engine=engine)
            for engine in engines
        ))
        return {
            int.from_bytes(engine, "little"): snapshot
            for engine, snapshot in zip(engines, snapshots, strict=True)
        }

    AsyncMPClient.psrl_get_runtime_snapshots_async = psrl_get_runtime_snapshots_async
