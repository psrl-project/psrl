"""
TMS / LD_PRELOAD helpers shared by rollout (`configure_worker`) and trainer (`RayWorkerGroup.worker_env`).

Aligned with main_psrl agentic_rl: train workers use `TMS_INIT_ENABLE=1` and `PSRL_TMS_ENABLE`
when `tms.range` is train/all. rollout gen_worker uses `TMS_INIT_ENABLE=0` (vLLM subprocess path).
"""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from omegaconf import DictConfig, OmegaConf


def prepend_python_library_path(ld_library_path: str | None = None) -> str:
    """Prepend the active Python environment's native library directory."""
    python_library_path = os.path.join(sys.prefix, "lib")
    current_value = os.environ.get("LD_LIBRARY_PATH", "") if ld_library_path is None else ld_library_path
    paths = [python_library_path, *current_value.split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(path for path in paths if path))


def build_ray_train_worker_tms_env(psrl_config: DictConfig | dict[str, Any] | None) -> dict[str, str] | None:
    """
    Environment variables for verl `RayWorkerGroup(worker_env=...)`, matching main_psrl ray_trainer.

    Returns None when TMS preload is not required for training workers.
    """
    if psrl_config is None:
        return None

    tms_range = OmegaConf.select(psrl_config, "tms.range", default=None)
    enable_nixl = bool(OmegaConf.select(psrl_config, "tms.enable_nixl", default=False))
    if tms_range not in ("train", "all") and not enable_nixl:
        return None

    torch_memory_saver = importlib.import_module("torch_memory_saver")
    dynlib_path = os.path.join(
        os.path.dirname(os.path.dirname(torch_memory_saver.__file__)),
        "torch_memory_saver_hook_mode_preload.abi3.so",
    )
    if not os.path.exists(dynlib_path):
        raise FileNotFoundError(f"LD_PRELOAD so file {dynlib_path} does not exist.")

    return {
        "LD_PRELOAD": dynlib_path,
        "LD_LIBRARY_PATH": prepend_python_library_path(),
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        "PSRL_TMS_ENABLE": "1" if tms_range in ("train", "all") else "",
    }
