"""Shared evaluation infrastructure: model serving for offline eval harnesses.

This package holds what every eval needs regardless of benchmark. It is imported
by `examples/mini_swe/eval` and `examples/sciaccel_rl/eval`, which previously
reached into each other for it.

`psrl.eval.serve` is the CLI entry point and the only Hydra-aware module. The
modules below it take plain dataclasses, so their logic is exercisable without
composing a config or holding a GPU.
"""

from psrl.eval.vllm_fleet import FleetResult, FleetSpec, launch_fleet, partition_gpus, read_endpoints
from psrl.eval.vllm_server import Endpoint, ServerHandle, ServerSpec, build_command, launch, wait_ready

__all__ = [
    "Endpoint",
    "FleetResult",
    "FleetSpec",
    "ServerHandle",
    "ServerSpec",
    "build_command",
    "launch",
    "launch_fleet",
    "partition_gpus",
    "read_endpoints",
    "wait_ready",
]
