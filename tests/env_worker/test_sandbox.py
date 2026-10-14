import os
from unittest.mock import patch

import pytest
from psrl.workers.env_worker.sandbox import (
    NVIDIA_CONTROL_DEVICES,
    SandboxSpec,
    build_docker_run_argv,
)


@pytest.mark.cpu_test
def test_cpu_only_argv_has_interactive_shell_and_no_gpu_flags():
    spec = SandboxSpec(image="mlgym:latest")
    argv = build_docker_run_argv(spec, container_name="psrl-env-abc123")

    assert argv[:4] == ["docker", "run", "-i", "--rm"], f"Unexpected argv prefix: {argv[:4]!r}."
    assert argv[-3:] == ["mlgym:latest", "/bin/bash", "-l"], f"Unexpected argv suffix: {argv[-3:]!r}."
    assert "--name" in argv, "Container name must always be set for cleanup."
    assert argv[argv.index("--name") + 1] == "psrl-env-abc123"
    assert "--gpus" not in argv, "The --gpus flag is unusable on this cluster."
    assert "--device" not in argv, "A CPU-only sandbox must not request devices."


@pytest.mark.cpu_test
def test_resource_limits_and_mounts_are_translated():
    spec = SandboxSpec(
        image="mlgym:latest",
        cpus=8.0,
        memory="32g",
        mounts=(("/host/data", "/data", "ro"),),
        env={"HTTP_PROXY": "http://proxy:3128"},
        labels={"psrl.env_sandbox_id": "s1"},
        network="host",
    )
    argv = build_docker_run_argv(spec, container_name="c1")

    assert "--cpus=8.0" in argv
    assert "--memory=32g" in argv
    assert "-v" in argv
    assert "/host/data:/data:ro" in argv
    assert "HTTP_PROXY=http://proxy:3128" in argv
    assert "psrl.env_sandbox_id=s1" in argv
    assert argv[argv.index("--network") + 1] == "host"


@pytest.mark.cpu_test
def test_gpu_sandbox_uses_device_passthrough_not_gpus_flag():
    spec = SandboxSpec(image="mlgym:latest", gpus=2)
    with patch.object(os.path, "exists", return_value=True):
        argv = build_docker_run_argv(spec, container_name="c1", gpu_device_indices=(3, 5))

    assert "--gpus" not in argv, "Must use device passthrough, not the broken --gpus flag."
    assert "/dev/nvidia3" in argv, "Requested physical device must be passed through."
    assert "/dev/nvidia5" in argv
    assert "/dev/nvidia0" not in argv, "Must not leak devices that were not assigned."
    for control_device in NVIDIA_CONTROL_DEVICES:
        assert control_device in argv, f"Missing required control device={control_device!r}."


@pytest.mark.cpu_test
def test_gpu_sandbox_bind_mounts_driver_libraries():
    spec = SandboxSpec(image="mlgym:latest", gpus=1)
    argv = build_docker_run_argv(
        spec,
        container_name="c1",
        gpu_device_indices=(0,),
        driver_libs=("/usr/lib64/libnvidia-ml.so.1",),
        nvidia_smi_path="/usr/bin/nvidia-smi",
    )

    joined = " ".join(argv)
    assert "/usr/lib64/libnvidia-ml.so.1:" in joined, "Driver library must be bind-mounted read-only."
    assert joined.count(":ro") >= 2, "Driver mounts must be read-only."
    assert "/usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro" in joined


@pytest.mark.cpu_test
def test_gpu_count_must_match_assigned_devices():
    spec = SandboxSpec(image="mlgym:latest", gpus=2)
    with pytest.raises(ValueError, match="does not match"):
        build_docker_run_argv(spec, container_name="c1", gpu_device_indices=(0,))


@pytest.mark.cpu_test
def test_argv_is_deterministic():
    spec = SandboxSpec(
        image="mlgym:latest",
        env={"B": "2", "A": "1"},
        labels={"z": "1", "a": "2"},
    )
    first = build_docker_run_argv(spec, container_name="c1")
    second = build_docker_run_argv(spec, container_name="c1")

    assert first == second, "Argv construction must be deterministic for reproducibility."
