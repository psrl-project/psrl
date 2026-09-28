"""
SciBuddy batch-rollout runtime configuration.

Mirrors examples/sciaccel_rl/config.py but trims keys that sciaccel added for
its own task-specific needs (GPU overrides, hint infrastructure) so the defaults
stay as narrow as this CPU-only task family requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from omegaconf import OmegaConf


@dataclass
class HarborConfig:
    """
    Harbor Job execution settings for the SciBuddy family.
    """

    jobs_dir: str = "/tmp/scibuddy_jobs"
    agent_name: str = "terminus-2"
    # Byte cap on one terminal observation. Harbor hardcodes 10000.
    # A lower value requires the subclass pointed at by agent_name.
    max_observation_bytes: int = 8000
    # Raise the task container memory above the task.toml declaration. 0 keeps
    # the value the task declares, which is fine for these CPU analysis tasks.
    memory_mb_override: int = 0
    # Harbor containers are invisible to Ray scheduling. Teardown is asynchronous,
    # so an episode's containers outlive the episode slot.
    max_concurrent_episodes: int = 4


@dataclass
class SciBuddyRuntimeConfig:
    """
    Top-level runtime config for the SciBuddy PSRL integration.
    """

    harbor: HarborConfig = field(default_factory=HarborConfig)
    # Harbor enforces this externally as the agent budget. Do not set below the
    # task timeout. PSRL wait_for is only a longer backstop.
    task_timeout_sec: float = 28800.0
    # Verifier budget. Must be >= the task's [verifier] timeout_sec so the outer
    # guard never pre-empts grading.
    verifier_timeout_sec: float = 1800.0


def build_runtime_config(yaml_kwargs: dict[str, Any]) -> SciBuddyRuntimeConfig:
    """
    Build config by merging YAML kwargs onto the structured schema.
    """
    raw = OmegaConf.to_container(OmegaConf.create(yaml_kwargs), resolve=True)
    if not isinstance(raw, dict):
        raw = {}
    raw.pop("name", None)
    raw.pop("_target_", None)
    schema = OmegaConf.structured(SciBuddyRuntimeConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create(raw))
    return OmegaConf.to_object(merged)
