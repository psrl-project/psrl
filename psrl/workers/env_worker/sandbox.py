"""Define sandbox data and translate it into Docker arguments."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

psrl_logger = logging.getLogger(__file__)
psrl_logger.setLevel(os.getenv("PSRL_LOGGING_LEVEL", "WARN"))

# NOTE(claude): The `--gpus` flag requires unavailable `nvidia-container-runtime`,
# so expose character devices and bind-mount the driver libraries.
NVIDIA_CONTROL_DEVICES: tuple[str, ...] = (
    "/dev/nvidiactl",
    "/dev/nvidia-uvm",
    "/dev/nvidia-uvm-tools",
)

DEFAULT_DRIVER_LIB_GLOBS: tuple[str, ...] = (
    "/usr/lib64/libnvidia-ml.so.1",
    "/usr/lib64/libcuda.so.1",
)

DEFAULT_NVIDIA_SMI_PATH = "/usr/bin/nvidia-smi"


@dataclass(frozen=True)
class SandboxSpec:
    """
    Declarative description of one sandbox container.

    Attributes:
        image (str): Container image reference.
        cpus (float | None): CPU limit passed to docker --cpus. None means no limit.
        memory (str | None): Memory limit passed to docker --memory, for example 32g.
        gpus (int): Number of GPUs required. 0 means a CPU-only sandbox.
        mounts (tuple): Bind mounts as (host_path, container_path, mode) triples.
        env (dict[str, str]): Environment variables to set inside the container.
        network (str): Docker network mode.
        labels (dict[str, str]): Docker labels, used for reaper-based cleanup.
        startup_timeout_s (float): Maximum seconds to wait for the shell to become ready.
        idle_timeout_s (float): Seconds of inactivity after which the watchdog reaps
            the sandbox. Guards against agent loops that die without calling destroy.
    """

    image: str
    cpus: float | None = None
    memory: str | None = None
    gpus: int = 0
    mounts: tuple[tuple[str, str, str], ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    network: str = "host"
    labels: dict[str, str] = field(default_factory=dict)
    startup_timeout_s: float = 300.0
    idle_timeout_s: float = 7200.0


@dataclass
class ExecResult:
    """
    Outcome of one command executed inside a sandbox shell.

    Attributes:
        stdout (str): Combined stdout and stderr with the sentinel line removed.
        exit_code (int | None): Shell exit status, or None when the sentinel did not
            carry a parseable integer, which happens when a command fails badly.
        timed_out (bool): Whether the read deadline elapsed before the sentinel arrived.
        duration_s (float): Wall-clock seconds spent executing.
    """

    stdout: str
    exit_code: int | None
    timed_out: bool
    duration_s: float


def build_gpu_argv(
    gpu_device_indices: tuple[int, ...],
    driver_libs: tuple[str, ...] = DEFAULT_DRIVER_LIB_GLOBS,
    nvidia_smi_path: str = DEFAULT_NVIDIA_SMI_PATH,
) -> list[str]:
    """
    Build the device and bind-mount arguments that give a container GPU access.

    Args:
        gpu_device_indices (tuple[int, ...]): Physical GPU indices to expose.
        driver_libs (tuple[str, ...]): Host driver library paths to bind-mount.
            Paths that do not exist on this host are skipped.
        nvidia_smi_path (str): Host path to nvidia-smi. Skipped when absent.

    Returns:
        list[str]: Docker arguments, empty when no indices were requested.
    """
    if not gpu_device_indices:
        return []

    argv: list[str] = []
    for index in gpu_device_indices:
        argv.extend(["--device", f"/dev/nvidia{index}"])
    for control_device in NVIDIA_CONTROL_DEVICES:
        if os.path.exists(control_device):
            argv.extend(["--device", control_device])
        else:
            psrl_logger.warning(f"Control device={control_device!r} is absent, skipping passthrough.")

    for lib_path in driver_libs:
        if os.path.exists(lib_path):
            argv.extend(["-v", f"{lib_path}:{lib_path}:ro"])
    if os.path.exists(nvidia_smi_path):
        argv.extend(["-v", f"{nvidia_smi_path}:{nvidia_smi_path}:ro"])

    return argv


def build_docker_run_argv(
    spec: SandboxSpec,
    container_name: str,
    gpu_device_indices: tuple[int, ...] = (),
    driver_libs: tuple[str, ...] = DEFAULT_DRIVER_LIB_GLOBS,
    nvidia_smi_path: str = DEFAULT_NVIDIA_SMI_PATH,
) -> list[str]:
    """
    Translate a `SandboxSpec` into a full docker run argument vector.

    The container is started with an interactive login shell on stdin so that
    shell state persists across `exec` calls, which MLGym depends on.

    Args:
        spec (SandboxSpec): Sandbox description.
        container_name (str): Docker container name, used for cleanup.
        gpu_device_indices (tuple[int, ...]): Physical GPU indices assigned by the worker.
            Length must equal `spec.gpus`.
        driver_libs (tuple[str, ...]): Host driver library paths to bind-mount.
        nvidia_smi_path (str): Host path to nvidia-smi.

    Returns:
        list[str]: Argument vector suitable for `subprocess.Popen`.

    Raises:
        ValueError: If the number of assigned devices does not match `spec.gpus`.
    """
    if len(gpu_device_indices) != spec.gpus:
        raise ValueError(
            f"Assigned GPU device count {len(gpu_device_indices)} does not match requested spec.gpus {spec.gpus}."
        )

    argv: list[str] = ["docker", "run", "-i", "--rm", "--name", container_name]

    if spec.network:
        argv.extend(["--network", spec.network])
    if spec.cpus is not None:
        argv.append(f"--cpus={spec.cpus}")
    if spec.memory is not None:
        argv.append(f"--memory={spec.memory}")

    for host_path, container_path, mode in spec.mounts:
        argv.extend(["-v", f"{host_path}:{container_path}:{mode}"])
    # Sort so that argv construction is deterministic and diffable in logs.
    for key in sorted(spec.env):
        argv.extend(["-e", f"{key}={spec.env[key]}"])
    for key in sorted(spec.labels):
        argv.extend(["--label", f"{key}={spec.labels[key]}"])

    argv.extend(build_gpu_argv(gpu_device_indices, driver_libs, nvidia_smi_path))
    argv.extend([spec.image, "/bin/bash", "-l"])
    return argv
