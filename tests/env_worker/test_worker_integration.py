"""
Integration tests that start real containers.

Skipped automatically when Docker is unavailable or no local image exists.
"""

import asyncio
import shutil
import subprocess

import pytest
from psrl.workers.env_worker.sandbox import SandboxSpec
from psrl.workers.env_worker.worker import EnvWorker


def _first_local_image() -> str | None:
    if shutil.which("docker") is None:
        return None
    completed = subprocess.run(
        ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    for line in completed.stdout.splitlines():
        if line.strip() and "<none>" not in line:
            return line.strip()
    return None


IMAGE = _first_local_image()
requires_docker = pytest.mark.skipif(IMAGE is None, reason="Docker or a local image is unavailable.")


@requires_docker
def test_shell_state_persists_across_exec_calls():
    """State set by one exec must be visible to the next, which docker exec cannot do."""

    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=0, exec_default_timeout_s=60.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=IMAGE, network="none"))
        try:
            await worker.exec(sandbox_id, "cd /tmp", timeout_s=30.0)
            await worker.exec(sandbox_id, "export PSRL_MARKER=persisted", timeout_s=30.0)
            result = await worker.exec(sandbox_id, "pwd && echo $PSRL_MARKER", timeout_s=30.0)

            assert "/tmp" in result.stdout, f"cwd did not persist: {result.stdout!r}."
            assert "persisted" in result.stdout, f"Exported variable did not persist: {result.stdout!r}."
            assert result.exit_code == 0
        finally:
            await worker.destroy_sandbox(sandbox_id)

    asyncio.run(scenario())


@requires_docker
def test_nonzero_exit_code_is_reported():
    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=0, exec_default_timeout_s=60.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=IMAGE, network="none"))
        try:
            # NOTE(claude): Using a subshell `(exit N)` keeps the persistent shell alive.
            # A bare `exit N` would terminate the shell process, ending the container.
            result = await worker.exec(sandbox_id, "(exit 42)", timeout_s=30.0)
            assert result.exit_code == 42, f"Expected exit code 42, got {result.exit_code!r}."
        finally:
            await worker.destroy_sandbox(sandbox_id)

    asyncio.run(scenario())


@requires_docker
def test_file_round_trip():
    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=0, exec_default_timeout_s=60.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=IMAGE, network="none"))
        try:
            payload = b"col_a,col_b\n1,2\n"
            await worker.write_file(sandbox_id, "/tmp/submission.csv", payload)
            result = await worker.exec(sandbox_id, "cat /tmp/submission.csv", timeout_s=30.0)

            assert "col_a,col_b" in result.stdout, f"File content missing: {result.stdout!r}."
        finally:
            await worker.destroy_sandbox(sandbox_id)

    asyncio.run(scenario())


@requires_docker
def test_read_file_returns_raw_bytes_not_tar():
    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=0, exec_default_timeout_s=60.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=IMAGE, network="none"))
        try:
            payload = b"col_a,col_b\n1,2\n"
            await worker.write_file(sandbox_id, "/tmp/submission.csv", payload)
            result = await worker.read_file(sandbox_id, "/tmp/submission.csv")

            assert result == payload, f"read_file must return raw file bytes, got {result[:80]!r}."
        finally:
            await worker.destroy_sandbox(sandbox_id)

    asyncio.run(scenario())


@requires_docker
def test_timed_out_command_does_not_leak_into_the_next_observation():
    """
    Regression test for a verified bug. A command that times out keeps running and
    later emits its output plus a sentinel. Without a drain before each exec, that
    text lands in the next command's observation and its stale exit code is reported
    as the next command's. MLGym routinely hits the 3600 second limit, so this path
    is common.
    """

    async def scenario() -> None:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=0, exec_default_timeout_s=60.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=IMAGE, network="none"))
        try:
            timed_out = await worker.exec(sandbox_id, "sleep 3; echo LATE_LEAK", timeout_s=1.0)
            assert timed_out.timed_out, "The slow command should have timed out."

            await asyncio.sleep(4.0)  # let the abandoned command finish and emit

            following = await worker.exec(sandbox_id, "echo SECOND", timeout_s=30.0)
            assert "LATE_LEAK" not in following.stdout, (
                f"Stale output leaked into the next observation: {following.stdout!r}."
            )
            assert "SECOND" in following.stdout
            assert following.exit_code == 0, f"A stale sentinel corrupted the exit code: {following.exit_code!r}."
        finally:
            await worker.destroy_sandbox(sandbox_id)

    asyncio.run(scenario())


@requires_docker
def test_destroy_removes_the_container():
    async def scenario() -> str:
        worker = EnvWorker(worker_id=0, cpu_slots=1, gpu_slots=0, exec_default_timeout_s=60.0)
        sandbox_id = await worker.create_sandbox(SandboxSpec(image=IMAGE, network="none"))
        await worker.destroy_sandbox(sandbox_id)
        return f"psrl-env-{sandbox_id}"

    container_name = asyncio.run(scenario())
    completed = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name={container_name}"],
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "", "Container must be removed after destroy_sandbox."
