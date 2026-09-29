"""Regression coverage for Docker preparation and failure cleanup."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from psrl.sandbox import SandboxManager, SandboxSource, SandboxSpec, SnapshotKind, SnapshotRef
from psrl.sandbox.backends.docker import DockerBackend
from psrl.sandbox.backends.docker_engine import DockerEngineClient, DockerEngineError, DockerExecOutputLimitError
from psrl.sandbox.utils import docker_utils

from tests.sandbox.test_docker_backend import FakeDockerEngine

pytestmark = pytest.mark.cpu_test


class StreamResponse:
    """
    Model a Docker response stream without requiring a daemon or socket.
    """

    status = 200
    headers = {}

    def __init__(self, chunks):
        self.chunks = chunks
        self.content = self
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    def __aiter__(self):
        return self.iter_chunked(65536)

    async def iter_chunked(self, size):
        for chunk in self.chunks:
            yield chunk


@pytest.mark.parametrize("field", ["error", "errorDetail"])
async def test_pull_rejects_errors_in_successful_http_stream(monkeypatch, field) -> None:
    error = b'{"error":"denied"}\n' if field == "error" else b'{"errorDetail":{"message":"denied"}}\n'
    response = StreamResponse([b'{"status":"Pulling"}\n', error])
    engine = DockerEngineClient()
    monkeypatch.setattr(engine, "_get_session", AsyncMock(return_value=SimpleNamespace(post=lambda *a, **k: response)))

    with pytest.raises(DockerEngineError, match="denied"):
        await engine.pull_image("private/image")

    assert response.closed, "Failed image pulls must release the connection."


async def test_exec_output_budget_closes_the_response(monkeypatch) -> None:
    response = StreamResponse([b"1234", b"5678"])
    engine = DockerEngineClient()
    monkeypatch.setattr(
        engine, "_get_session", AsyncMock(return_value=SimpleNamespace(request=lambda *a, **k: response))
    )

    with pytest.raises(DockerExecOutputLimitError):
        await engine._request("POST", "/exec/id/start", expected=(200,), max_response_bytes=5)

    assert response.closed, "Oversized output must release the connection."


async def test_prepare_shares_pull_and_survives_one_cancelled_waiter(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    manager = SandboxManager({"docker": backend}, "docker")
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def pull(reference, auth):
        engine.pulled.append(reference)
        entered.set()
        await finish.wait()

    monkeypatch.setattr(engine, "pull_image", pull)
    spec = SandboxSpec(SandboxSource.image("image"))
    first = asyncio.create_task(manager.prepare(spec))
    second = asyncio.create_task(manager.prepare(spec))
    await entered.wait()
    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    finish.set()
    await second
    await manager.prepare(spec)

    assert engine.pulled == ["image"], "Concurrent waiters must share one image download."
    assert engine.config is None, "Preparation must not allocate a container."
    assert not manager._leases, "Preparation must not acquire container capacity."
    await manager.shutdown()


async def test_prepare_limits_distinct_image_downloads_and_drains_on_shutdown(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine, image_pull_concurrency=2)
    entered = asyncio.Event()
    active = 0
    peak = 0

    async def pull(reference, auth):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        try:
            await asyncio.Future()
        finally:
            active -= 1

    monkeypatch.setattr(engine, "pull_image", pull)
    tasks = [asyncio.create_task(backend.prepare(SandboxSpec(SandboxSource.image(str(i))))) for i in range(10)]
    await entered.wait()
    await backend.shutdown()
    await asyncio.gather(*tasks, return_exceptions=True)

    assert peak == 2, "Image pulls must respect the configured concurrency bound."
    assert active == 0 and not backend._image_tasks, "Shutdown must drain pending preparation."
    assert engine.closed, "Shutdown must close the transport."


@pytest.mark.parametrize("error", [aiohttp.ClientPayloadError("broken"), DockerExecOutputLimitError("large")])
async def test_exec_stream_failure_terminates_the_container(monkeypatch, error) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    monkeypatch.setattr(engine, "exec", AsyncMock(side_effect=error))

    with pytest.raises(type(error)):
        await session.exec("background-work")

    assert engine.removes == 1, "Disconnected exec processes must not outlive their sandbox."


async def test_exec_deadline_includes_engine_setup(monkeypatch) -> None:
    engine = FakeDockerEngine()
    session = await DockerBackend(engine=engine).create(SandboxSpec(SandboxSource.image("image")))

    async def blocked(*args, **kwargs):
        await asyncio.Future()

    monkeypatch.setattr(engine, "exec", blocked)
    with pytest.raises(TimeoutError):
        await session.exec("blocked", timeout_s=0.01)

    assert engine.removes == 1, "An expired command must release its container."


async def test_snapshot_references_are_immutable(monkeypatch) -> None:
    engine = FakeDockerEngine()
    monkeypatch.setattr(engine, "commit_container", AsyncMock(return_value="sha256:image"), raising=False)
    session = await DockerBackend(engine=engine).create(SandboxSpec(SandboxSource.image("image")))

    first = await session.snapshot(SnapshotKind.FILESYSTEM)
    second = await session.snapshot(SnapshotKind.FILESYSTEM)

    assert first.snapshot_id != second.snapshot_id, "Later snapshots must not retarget earlier references."


def test_gc_does_not_treat_unreadable_lease_as_dead_owner(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(docker_utils, "_list_sandbox_containers", lambda command, **kwargs: [("container", "owner")])

    def unreadable(*args, **kwargs):
        raise PermissionError("unreadable lease")

    monkeypatch.setattr(docker_utils, "owner_heartbeat_age_s", unreadable)
    removed, remaining = docker_utils.sweep_stale_sandboxes(str(tmp_path), 120)

    assert removed == [] and remaining == 1, "Unreadable leases must not authorize container deletion."


@pytest.mark.parametrize("inspection", [{"Running": True, "ExitCode": 0}, {"Running": False, "ExitCode": None}, {}])
async def test_exec_requires_a_final_exit_status(monkeypatch, inspection) -> None:
    engine = DockerEngineClient()
    monkeypatch.setattr(
        engine,
        "_request",
        AsyncMock(side_effect=[({}, b'{"Id":"exec"}', 201), ({}, b"", 200), ({}, json.dumps(inspection), 200)]),
    )

    with pytest.raises(DockerEngineError, match="final exit status"):
        await engine.exec("container", ["work"])


def test_gc_listing_is_scoped_to_its_lease_namespace(monkeypatch) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(docker_utils.subprocess, "run", run)
    docker_utils._list_sandbox_containers(("docker",), lease_store="namespace")

    assert "label=psrl.lease_store=namespace" in calls[0], "GC must not inspect other lease namespaces."


async def test_snapshot_deletion_accepts_the_same_reference_as_restore(monkeypatch) -> None:
    engine = FakeDockerEngine()
    remove = AsyncMock()
    monkeypatch.setattr(engine, "remove_image", remove, raising=False)
    backend = DockerBackend(engine=engine)

    await backend.delete_snapshot(SnapshotRef("docker", "image-id", SnapshotKind.FILESYSTEM))

    remove.assert_awaited_once_with("image-id")


async def test_failed_lifecycle_cleanup_still_closes_engine(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)

    def fail():
        raise OSError("lease store unavailable")

    monkeypatch.setattr(backend.lifecycle, "close", fail)
    with pytest.raises(OSError):
        await backend.shutdown()

    assert engine.closed, "A cleanup failure must not leak the connection pool."


async def test_start_failure_is_not_replaced_by_cleanup_failure(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    monkeypatch.setattr(engine, "start_container", AsyncMock(side_effect=ValueError("start failed")))
    monkeypatch.setattr(engine, "remove_container", AsyncMock(side_effect=OSError("cleanup failed")))

    with pytest.raises(ValueError, match="start failed"):
        await backend.create(SandboxSpec(SandboxSource.image("image")))


async def test_closed_engine_cannot_reopen_its_connection_pool() -> None:
    engine = DockerEngineClient()
    await engine.close()

    with pytest.raises(RuntimeError, match="client is closed"):
        await engine.info()
