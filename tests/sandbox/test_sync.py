from __future__ import annotations

import asyncio
from collections.abc import Mapping

import pytest
from psrl.sandbox import (
    ExecResult,
    SandboxBackend,
    SandboxCapabilities,
    SandboxRef,
    SandboxSession,
    SandboxSource,
    SandboxSpec,
    SandboxStatus,
)
from psrl.sandbox.manager import SandboxManager


class FakeSession(SandboxSession):
    def __init__(self) -> None:
        self.terminated = False
        self.files: dict[str, bytes] = {}

    @property
    def ref(self) -> SandboxRef:
        return SandboxRef("fake", "sync-session")

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        env: Mapping[str, str] | None = None,
        timeout_s: float | None = None,
    ) -> ExecResult:
        await asyncio.sleep(0)
        return ExecResult(0, command, cwd or "")

    async def read_bytes(self, path: str) -> bytes:
        return self.files[path]

    async def write_bytes(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def status(self) -> SandboxStatus:
        return SandboxStatus.TERMINATED if self.terminated else SandboxStatus.RUNNING

    async def terminate(self) -> None:
        self.terminated = True


class FakeBackend(SandboxBackend):
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    @property
    def name(self) -> str:
        return "fake"

    @property
    def capabilities(self) -> SandboxCapabilities:
        return SandboxCapabilities()

    async def create(self, spec: SandboxSpec) -> SandboxSession:
        session = FakeSession()
        self.sessions.append(session)
        return session

    async def connect(self, sandbox_id: str) -> SandboxSession:
        return FakeSession()


@pytest.mark.asyncio
async def test_sync_manager_runs_on_owner_loop_and_closes_sessions() -> None:
    backend = FakeBackend()
    manager = SandboxManager({"fake": backend}, "fake")
    sync_manager = manager.sync()

    session = await asyncio.to_thread(sync_manager.create, SandboxSpec(source=SandboxSource.image("image")))
    result = await asyncio.to_thread(session.exec, "echo ok", cwd="/workspace")
    await asyncio.to_thread(session.write_bytes, "/tmp/data", b"payload")
    data = await asyncio.to_thread(session.read_bytes, "/tmp/data")
    await asyncio.to_thread(session.close)

    assert result == ExecResult(0, "echo ok", "/workspace")
    assert data == b"payload"
    assert backend.sessions[0].terminated
    assert not manager._leases


@pytest.mark.asyncio
async def test_sync_manager_reclaims_sessions_left_by_failed_harness() -> None:
    backend = FakeBackend()
    manager = SandboxManager({"fake": backend}, "fake")
    sync_manager = manager.sync()

    await asyncio.to_thread(sync_manager.create, SandboxSpec(source=SandboxSource.image("image")))
    await sync_manager.aclose()

    assert backend.sessions[0].terminated
    assert not manager._leases


@pytest.mark.asyncio
async def test_sync_manager_releases_session_when_close_wins_create_race() -> None:
    started = asyncio.Event()
    finish_create = asyncio.Event()

    class DelayedBackend(FakeBackend):
        async def create(self, spec: SandboxSpec) -> SandboxSession:
            started.set()
            await finish_create.wait()
            return await super().create(spec)

    backend = DelayedBackend()
    manager = SandboxManager({"fake": backend}, "fake")
    sync_manager = manager.sync()
    create_task = asyncio.create_task(
        asyncio.to_thread(sync_manager.create, SandboxSpec(source=SandboxSource.image("image")))
    )
    await started.wait()

    await sync_manager.aclose()
    finish_create.set()

    with pytest.raises(RuntimeError, match="closed while creating"):
        await create_task
    assert backend.sessions[0].terminated
    assert not manager._leases
