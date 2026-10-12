from __future__ import annotations

from types import SimpleNamespace

import pytest
from psrl.sandbox import PauseMode, ResourceSpec, SandboxFeature, SandboxSource, SandboxSpec, SnapshotKind
from psrl.sandbox.backends.e2b import (
    AgentEnvBackend,
    AgentEnvClientFactory,
    AgentEnvStateDriver,
    CubeSandboxClientFactory,
    CubeSandboxStateDriver,
    E2BBackend,
)


class FakeCommands:
    async def run(self, command: str, **kwargs):
        if command == "fail":
            error = RuntimeError("failed")
            error.exit_code = 3
            error.stdout = "partial"
            error.stderr = "failure"
            raise error
        return SimpleNamespace(exit_code=7, stdout=command, stderr=kwargs.get("cwd", ""))


class FakeFiles:
    def __init__(self) -> None:
        self.data = {}

    async def read(self, path: str, format: str):
        return bytearray(self.data[path])

    async def write(self, path: str, data: bytes):
        self.data[path] = data


class FakeClient:
    def __init__(self, sandbox_id: str) -> None:
        self.sandbox_id = sandbox_id
        self.commands = FakeCommands()
        self.files = FakeFiles()
        self.running = True
        self.paused = False

    async def is_running(self):
        return self.running

    async def kill(self):
        self.running = False

    async def pause(self, keep_memory: bool):
        assert keep_memory
        self.paused = True

    async def create_snapshot(self):
        return SimpleNamespace(snapshot_id="snapshot-1", names=["snapshot-name:v1"])

    async def fork(self, count: int):
        assert count == 1
        return [FakeClient("fork-1")]


class FakeFactory:
    def __init__(self) -> None:
        self.client = FakeClient("sandbox-1")
        self.deleted_snapshots: list[str] = []

    async def create(self, spec: SandboxSpec):
        return self.client

    async def connect(self, sandbox_id: str):
        self.client.paused = False
        return self.client

    async def close(self):
        return None

    async def delete_snapshot(self, snapshot_id: str):
        self.deleted_snapshots.append(snapshot_id)


class FakeControl:
    def __init__(self) -> None:
        self.api_key = "test-key"
        self.api_url = "http://provider.test"
        self.requests: list[tuple[str, str, dict]] = []
        self.deleted: list[str] = []

    async def request(self, method: str, path: str, *, expected: tuple[int, ...], payload=None):
        self.requests.append((method, path, dict(payload or {})))
        if path.endswith("/fork"):
            return [{"sandbox": {"sandboxID": "fork-1"}}]
        if path.endswith("/snapshots"):
            return {"snapshotID": "snapshot-1", "names": ["baseline"]}
        return {"sandboxID": "created-1"}

    async def delete_idempotent(self, path: str):
        self.deleted.append(path)

    async def close(self):
        return None


class FakeAgentEnvFactory(AgentEnvClientFactory):
    async def connect(self, sandbox_id: str):
        return FakeClient(sandbox_id)


class FakeCubeFactory(CubeSandboxClientFactory):
    async def connect(self, sandbox_id: str):
        return FakeClient(sandbox_id)


@pytest.mark.asyncio
async def test_e2b_compatible_data_plane_and_hibernate() -> None:
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))

    result = await session.exec("echo ok", cwd="/workspace")
    await session.write_bytes("/tmp/value", b"value")
    await session.pause(PauseMode.HIBERNATE)

    assert result.exit_code == 7
    assert result.stderr == "/workspace"
    assert await session.read_bytes("/tmp/value") == b"value"
    assert factory.client.paused
    assert backend.capabilities.supports(SandboxFeature.HIBERNATE)

    failed = await session.exec("fail")
    assert failed.exit_code == 3
    assert failed.stdout == "partial"


@pytest.mark.asyncio
async def test_e2b_native_state_capabilities() -> None:
    factory = FakeFactory()
    backend = E2BBackend(client_factory=factory)
    session = await backend.create(SandboxSpec(SandboxSource.template("base")))

    snapshot = await session.snapshot(SnapshotKind.FULL_STATE)
    child = await session.fork()
    restored = await backend.restore(snapshot)
    await backend.delete_snapshot(snapshot)

    assert snapshot.snapshot_id == "snapshot-1"
    assert child.ref.sandbox_id == "fork-1"
    assert restored.ref.sandbox_id == "sandbox-1"
    assert backend.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
    assert backend.capabilities.supports(SandboxFeature.NATIVE_FORK)
    assert factory.deleted_snapshots == ["snapshot-1"]


@pytest.mark.asyncio
async def test_generic_e2b_rejects_image_and_runtime_resource_overrides() -> None:
    # A custom factory owns its own mapping contract, so validate the official
    # factory directly rather than pretending all injected clients are E2B.
    from psrl.sandbox.backends.e2b import E2BSDKClientFactory

    sdk_factory = E2BSDKClientFactory()
    with pytest.raises(RuntimeError, match="template source"):
        await sdk_factory.create(SandboxSpec(SandboxSource.image("image")))
    with pytest.raises(RuntimeError, match="fixed by the template"):
        await sdk_factory.create(
            SandboxSpec(
                SandboxSource.template("template"),
                resources=ResourceSpec(memory_mb=1024),
            )
        )


def test_agentenv_does_not_overclaim_optional_e2b_state_apis() -> None:
    backend = AgentEnvBackend(client_factory=FakeFactory())

    assert backend.capabilities.supports(SandboxFeature.HIBERNATE)
    assert not backend.capabilities.supports(SandboxFeature.NATIVE_FORK)


@pytest.mark.asyncio
async def test_agentenv_factory_maps_cold_image_resources_timeout_and_auth_metadata() -> None:
    control = FakeControl()
    factory = FakeAgentEnvFactory(control=control)

    client = await factory.create(
        SandboxSpec(
            SandboxSource.image("registry/image:tag"),
            resources=ResourceSpec(cpu_count=4, memory_mb=8192, disk_mb=16384),
            env={"A": "B"},
            metadata={"task": "1"},
            idle_timeout_s=12.1,
            idempotency_key="rollout-1",
        )
    )

    assert client.sandbox_id == "created-1"
    _, path, payload = control.requests[0]
    assert path == "/sandboxes-cold"
    assert payload["image"] == "registry/image:tag"
    assert payload["cpuCount"] == 4
    assert payload["memoryMB"] == 8192
    assert payload["diskSizeMB"] == 16384
    assert payload["timeout"] == 13
    assert payload["envVars"] == {"A": "B"}
    assert payload["metadata"]["psrl.idempotency_key"] == "rollout-1"


@pytest.mark.asyncio
async def test_cube_factory_maps_template_resources_and_rejects_images() -> None:
    control = FakeControl()
    factory = FakeCubeFactory(control=control)
    spec = SandboxSpec(
        SandboxSource.template("template-1"),
        resources=ResourceSpec(cpu_count=2, memory_mb=4096, disk_mb=8192),
    )

    await factory.create(spec)

    _, path, payload = control.requests[0]
    assert path == "/sandboxes"
    assert payload["templateID"] == "template-1"
    assert payload["cpuCount"] == 2
    assert payload["memoryMB"] == 4096
    assert payload["diskSizeMB"] == 8192
    with pytest.raises(RuntimeError, match="built template"):
        await factory.create(SandboxSpec(SandboxSource.image("image")))


@pytest.mark.asyncio
async def test_provider_specific_state_drivers_advertise_only_real_semantics() -> None:
    control = FakeControl()
    factory = FakeFactory()
    agentenv = E2BBackend(
        name="agentenv",
        client_factory=factory,
        state_driver=AgentEnvStateDriver(control),
    )
    cube = E2BBackend(
        name="cubesandbox",
        client_factory=factory,
        state_driver=CubeSandboxStateDriver(control),
    )
    session = await agentenv.create(SandboxSpec(SandboxSource.template("base")))

    snapshot = await session.snapshot(SnapshotKind.FULL_STATE)
    child = await session.fork()
    await agentenv.delete_snapshot(snapshot)

    assert child.ref.sandbox_id == "fork-1"
    assert agentenv.capabilities.supports(SandboxFeature.NATIVE_FORK)
    assert cube.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
    assert not cube.capabilities.supports(SandboxFeature.NATIVE_FORK)
    assert control.deleted == ["/templates/snapshot-1"]
