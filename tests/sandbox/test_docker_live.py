"""Opt-in conformance test against a real Docker Engine."""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest
from psrl.sandbox import PauseMode, SandboxManager, SandboxSource, SandboxSpec, SandboxStatus
from psrl.sandbox.backends.docker import DockerBackend

pytestmark = pytest.mark.skipif(
    os.getenv("PSRL_RUN_DOCKER_INTEGRATION") != "1",
    reason="set PSRL_RUN_DOCKER_INTEGRATION=1 to exercise a real Docker daemon",
)


@pytest.mark.asyncio
async def test_live_docker_lifecycle_metrics_and_idempotency() -> None:
    backend = DockerBackend(
        docker_host=os.getenv("DOCKER_HOST"),
        auto_pull=os.getenv("PSRL_DOCKER_AUTO_PULL", "1") == "1",
        security={
            "require_rootless": os.getenv("PSRL_REQUIRE_ROOTLESS", "0") == "1",
            "pids_limit": 128,
            "cap_drop": ("ALL",),
            "no_new_privileges": True,
        },
    )
    manager = SandboxManager({"docker": backend}, "docker")
    spec = SandboxSpec(
        SandboxSource.image(os.getenv("PSRL_DOCKER_TEST_IMAGE", "python:3.11-slim")),
        idempotency_key=f"docker-live-{uuid.uuid4().hex}",
        idle_timeout_s=120,
    )

    try:
        first, second = await asyncio.gather(manager.acquire(spec), manager.acquire(spec))
        assert first is second
        assert (await first.session.exec("printf psrl")).stdout == "psrl"
        await first.session.write_bytes("/tmp/binary", b"\x00psrl\xff")
        assert await first.session.read_bytes("/tmp/binary") == b"\x00psrl\xff"
        usage = await first.session.stats()
        assert usage.memory_bytes > 0
        await first.session.pause(PauseMode.FREEZE)
        assert await first.session.status() == SandboxStatus.PAUSED
        await first.session.resume()
        assert await first.session.status() == SandboxStatus.RUNNING

        await first.release()
        await first.release()
        metrics = manager.metrics_snapshot()["docker"]
        assert metrics.operations["create"].count == 1
        assert metrics.operations["terminate"].count == 1
        assert metrics.peak_memory_bytes >= usage.memory_bytes
        assert metrics.active_sessions == 0
    finally:
        await manager.shutdown()
