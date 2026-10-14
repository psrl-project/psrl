"""
Runtime configuration for the AIRS-Bench recipe.

Mirrors the structured-config pattern used by `examples/mem_agent/config.py`, so
values can be supplied from the launch script and merged onto typed defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

DEFAULT_IMAGE = "psrl/airs-bench-agent:latest"
DEFAULT_DATA_ROOT = "/apdcephfs_zwfy10/share_303541817/lhy/airs_bench_data"
DEFAULT_MLGYM_REPO = "/apdcephfs_zwfy10/share_303541817/lhy/science_infra/MLGym"
_THIS_DIR = Path(__file__).resolve().parent
DEFAULT_AGENT_CONFIG = str(_THIS_DIR / "agent_config.yaml")


@dataclass
class AirsBenchRuntimeConfig:
    """
    Per-episode settings owned by the AIRS-Bench recipe.

    Attributes:
        image (str): Sandbox container image.
        sandbox_cpus (float): Docker CPU limit for one sandbox.
        sandbox_memory (str): Docker memory limit for one sandbox.
        data_root (str): Root holding airs_prepared and airs_configs.
        mlgym_repo (str): Path to the read-only MLGym checkout.
        agent_config_path (str): MLGym agent config. Must pin DefaultHistoryProcessor.
        max_observation_chars (int): Head and tail budget per observation.
        episode_timeout_s (float): Wall-clock cap for one whole episode.
        per_action_timeout_s (float): Cap for one agent action. Covers MLGym's
            3600 second training_timeout.
        startup_timeout_s (float): Cap for sandbox shell readiness.
    """

    image: str = DEFAULT_IMAGE
    sandbox_cpus: float = 8.0
    sandbox_memory: str = "32g"
    data_root: str = DEFAULT_DATA_ROOT
    mlgym_repo: str = DEFAULT_MLGYM_REPO
    agent_config_path: str = DEFAULT_AGENT_CONFIG
    max_observation_chars: int = 8000
    episode_timeout_s: float = 10800.0
    per_action_timeout_s: float = 3600.0
    startup_timeout_s: float = 600.0


def build_runtime_config(value: DictConfig | dict[str, Any] | None) -> AirsBenchRuntimeConfig:
    """
    Merge YAML values onto the structured AIRS-Bench schema.

    Args:
        value (DictConfig | dict[str, Any] | None): Overrides from configuration.

    Returns:
        AirsBenchRuntimeConfig: Fully populated runtime config.
    """
    raw = OmegaConf.create(value or {})
    merged = OmegaConf.merge(OmegaConf.structured(AirsBenchRuntimeConfig), raw)
    config: AirsBenchRuntimeConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]
    return config
