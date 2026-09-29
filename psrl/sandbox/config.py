from dataclasses import dataclass, field
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from psrl.sandbox.capacity import SandboxCapacityConfig
from psrl.sandbox.core import SandboxBackend
from psrl.sandbox.manager import SandboxManager


@dataclass
class SandboxManagerConfig:
    """Worker-local sandbox backend registry."""

    default_backend: str = "docker"
    backends: dict[str, Any] = field(default_factory=dict)
    capacity: SandboxCapacityConfig = field(default_factory=SandboxCapacityConfig)


def build_sandbox_manager(
    config: DictConfig | SandboxManagerConfig,
    capacity_coordinator=None,
    owner_id: str | None = None,
) -> SandboxManager:
    """Instantiate the configured backends and bind the worker's node capacity."""
    backend_configs = config.backends or {}
    if isinstance(backend_configs, DictConfig):
        backend_configs = OmegaConf.to_container(backend_configs, resolve=True)
    backends: dict[str, SandboxBackend] = {}
    for name, backend_config in backend_configs.items():
        if isinstance(backend_config, SandboxBackend):
            raise TypeError(
                f"Sandbox backend {name!r} is already instantiated; "
                "sandbox configuration must remain declarative until worker initialization."
            )
        backend = hydra.utils.instantiate(backend_config)
        if not isinstance(backend, SandboxBackend):
            raise TypeError(f"Configured sandbox backend {name!r} is not a SandboxBackend.")
        backends[name] = backend
    capacity = None
    if capacity_coordinator is not None:
        capacity = config.capacity
        if isinstance(config.capacity, DictConfig):
            capacity = OmegaConf.to_container(capacity, resolve=True)
        capacity = capacity if isinstance(capacity, SandboxCapacityConfig) else SandboxCapacityConfig(**capacity)
        if not owner_id:
            raise ValueError("Sandbox capacity coordinator requires a non-empty owner_id.")
    return SandboxManager(
        backends,
        config.default_backend,
        capacity_coordinator=capacity_coordinator,
        capacity_owner_id=owner_id,
        capacity_heartbeat_interval_s=capacity.heartbeat_interval_s if capacity is not None else None,
    )
