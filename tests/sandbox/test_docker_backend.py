from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

import pytest
from psrl.sandbox import MountSpec, PauseMode, SandboxFeature, SandboxSource, SandboxSpec
from psrl.sandbox.backends.docker import DockerBackend, DockerSession
from psrl.sandbox.backends.docker_engine import DockerEngineError
from psrl.sandbox.utils import docker_utils


class FakeDockerEngine:
    def __init__(self) -> None:
        self.config: dict[str, Any] | None = None
        self.name = ""
        self.container_id = "container-id"
        self.state = "running"
        self.removes = 0
        self.exec_timeout = False
        self.files: dict[str, bytes] = {}
        self.closed = False
        self.pulled: list[str] = []
        self.pull_auth = None

    async def info(self):
        return {"SecurityOptions": ["name=rootless"]}

    async def pull_image(self, reference: str, auth=None):
        self.pulled.append(reference)
        self.pull_auth = auth

    async def image_exists(self, reference: str) -> bool:
        return reference in self.pulled

    async def create_container(self, name: str, config):
        self.name = name
        self.config = dict(config)
        return self.container_id

    async def start_container(self, container_id: str):
        self.state = "running"

    async def inspect_container(self, container_id: str):
        if self.state == "removed":
            return None
        return {
            "Id": self.container_id,
            "State": {"Status": self.state},
            "Config": {"Labels": (self.config or {}).get("Labels", {})},
        }

    async def remove_container(self, container_id: str):
        self.removes += 1
        self.state = "removed"

    async def pause_container(self, container_id: str):
        self.state = "paused"

    async def unpause_container(self, container_id: str):
        self.state = "running"

    async def exec(self, container_id: str, command: list[str], **kwargs):
        if self.exec_timeout:
            raise TimeoutError("timed out")
        return 0, command[-1].encode(), b""

    async def read_file(self, container_id: str, path: str):
        return self.files[path]

    async def write_file(self, container_id: str, path: str, data: bytes):
        self.files[path] = data

    async def stats(self, container_id: str):
        return {
            "memory_stats": {"usage": 1024, "max_usage": 4096},
            "cpu_stats": {"cpu_usage": {"total_usage": 123}},
        }

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_docker_create_maps_spec_and_secure_defaults() -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(
        engine=engine,
        security={"require_rootless": True},
        policy_profiles={
            "rl": {
                "network_mode": "none",
                "extra_hosts": ["host.docker.internal:host-gateway"],
            }
        },
    )
    session = await backend.create(
        SandboxSpec(
            SandboxSource.image("python:3.11"),
            workdir="/workspace",
            env={"A": "B"},
            metadata={"task": "1"},
            mounts=(MountSpec("/tmp/source", "/workspace/source", read_only=True),),
            policy_profile="rl",
            idempotency_key="rollout-1",
        )
    )

    assert session.ref.sandbox_id == "container-id"
    assert engine.config is not None
    assert engine.config["Image"] == "python:3.11"
    host_config = engine.config["HostConfig"]
    assert host_config["Init"]
    assert host_config["AutoRemove"]
    assert host_config["CapDrop"] == ["ALL"]
    assert "no-new-privileges" in host_config["SecurityOpt"]
    assert host_config["PidsLimit"] == 4096
    assert host_config["NetworkMode"] == "none"
    assert host_config["Mounts"][0]["ReadOnly"]
    assert engine.config["Labels"]["psrl.idempotency_key"] == "rollout-1"
    assert backend.capabilities.supports(SandboxFeature.FREEZE)
    assert not backend.capabilities.supports(SandboxFeature.FULL_STATE_SNAPSHOT)
    assert backend.capabilities.supports(SandboxFeature.RESTORE)
    assert not backend.capabilities.supports(SandboxFeature.NATIVE_FORK)


@pytest.mark.asyncio
async def test_docker_bridge_exposes_host_gateway_and_rewrites_loopback_proxy() -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(
        engine=engine,
        policy_profiles={
            "mini_swe": {
                "network_mode": "bridge",
                "host_gateway_alias": "host.docker.internal",
                "rewrite_loopback_proxies": True,
            }
        },
    )

    await backend.create(
        SandboxSpec(
            SandboxSource.image("python:3.11"),
            env={
                "HTTP_PROXY": "http://user:token@127.0.0.1:7890",
                "HTTPS_PROXY": "http://proxy.example.com:8443",
                "NO_PROXY": "localhost,127.0.0.1",
            },
            policy_profile="mini_swe",
        )
    )

    assert engine.config is not None
    host_config = engine.config["HostConfig"]
    assert host_config["NetworkMode"] == "bridge"
    assert host_config["ExtraHosts"] == ["host.docker.internal:host-gateway"]
    environment = dict(item.split("=", 1) for item in engine.config["Env"])
    assert environment["HTTP_PROXY"] == "http://user:token@host.docker.internal:7890"
    assert environment["HTTPS_PROXY"] == "http://proxy.example.com:8443"
    assert environment["NO_PROXY"] == "localhost,127.0.0.1,host.docker.internal"
    assert environment["no_proxy"] == environment["NO_PROXY"]


@pytest.mark.asyncio
async def test_docker_policy_can_enable_suid_without_granting_extra_capabilities() -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(
        engine=engine,
        policy_profiles={"root_task": {"no_new_privileges": False}},
    )

    await backend.create(
        SandboxSpec(
            SandboxSource.image("python:3.11"),
            policy_profile="root_task",
        )
    )

    assert engine.config is not None
    assert "no-new-privileges" not in engine.config["HostConfig"]["SecurityOpt"]
    assert engine.config["HostConfig"]["CapDrop"] == ["ALL"]


@pytest.mark.asyncio
async def test_docker_create_recovers_idempotent_conflict(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    spec = SandboxSpec(SandboxSource.image("image"), idempotency_key="same-request")
    expected_config = backend._build_container_config(spec, backend._resolve_policy(spec))
    engine.config = expected_config

    async def conflict(name: str, config):
        raise DockerEngineError(409, "name already exists")

    monkeypatch.setattr(engine, "create_container", conflict)
    session = await backend.create(spec)

    assert session.ref.sandbox_id == "container-id"


@pytest.mark.asyncio
async def test_docker_idempotent_retry_waits_for_concurrent_start(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    spec = SandboxSpec(SandboxSource.image("image"), idempotency_key="same-request")
    engine.config = backend._build_container_config(spec, backend._resolve_policy(spec))
    engine.state = "created"

    async def conflict(name: str, config):
        raise DockerEngineError(409, "name already exists")

    async def finish_start() -> None:
        await asyncio.sleep(0.01)
        engine.state = "running"

    monkeypatch.setattr(engine, "create_container", conflict)
    start_task = asyncio.create_task(finish_start())
    session = await backend.create(spec)
    await start_task

    assert session.ref.sandbox_id == "container-id"
    assert engine.removes == 0


@pytest.mark.asyncio
async def test_docker_auto_pull_forwards_registry_auth(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(
        engine=engine,
        registry_auth={"username": "robot", "password": "token"},
    )
    original_create = engine.create_container
    attempts = 0

    async def missing_then_create(name: str, config):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DockerEngineError(404, "image not found")
        return await original_create(name, config)

    monkeypatch.setattr(engine, "create_container", missing_then_create)
    session = await backend.create(SandboxSpec(SandboxSource.image("private/image:tag")))
    await session.terminate()

    assert engine.pulled == ["private/image:tag"]
    assert engine.pull_auth == {"username": "robot", "password": "token"}


@pytest.mark.asyncio
async def test_docker_auto_pull_can_join_an_idempotent_concurrent_create(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    spec = SandboxSpec(SandboxSource.image("image"), idempotency_key="same-request")
    engine.config = backend._build_container_config(spec, backend._resolve_policy(spec))
    attempts = 0

    async def missing_then_conflict(name: str, config):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise DockerEngineError(404, "image not found")
        raise DockerEngineError(409, "name already exists")

    monkeypatch.setattr(engine, "create_container", missing_then_conflict)

    session = await backend.create(spec)

    assert session.ref.sandbox_id == "container-id"
    assert engine.pulled == ["image"]


@pytest.mark.asyncio
async def test_docker_connect_resumes_frozen_container() -> None:
    engine = FakeDockerEngine()
    engine.state = "paused"
    backend = DockerBackend(engine=engine)

    session = await backend.connect("container-id")

    assert isinstance(session, DockerSession)
    assert engine.state == "running"


@pytest.mark.asyncio
async def test_docker_files_stats_metrics_and_idempotent_terminate() -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    payload = b"\x00\xffbinary\n"

    await session.write_bytes("/tmp/value", payload)
    usage = await session.stats()
    await session.terminate()
    await session.terminate()

    assert await engine.read_file("container-id", "/tmp/value") == payload
    assert usage.memory_bytes == 1024
    assert usage.peak_memory_bytes == 4096
    assert engine.removes == 1
    metrics = backend.metrics_snapshot()
    assert metrics.operations["create"].count == 1
    assert metrics.operations["terminate"].count == 1
    assert "mean_seconds" in metrics.as_dict()["operations"]["create"]
    assert metrics.peak_memory_bytes == 4096
    assert metrics.active_sessions == 0


@pytest.mark.asyncio
async def test_docker_rejects_hibernation() -> None:
    backend = DockerBackend(engine=FakeDockerEngine())
    session = DockerSession(backend, "container-id")

    with pytest.raises(Exception, match="hibernation"):
        await session.pause(PauseMode.HIBERNATE)


@pytest.mark.asyncio
async def test_docker_command_timeout_terminates_disposable_sandbox() -> None:
    engine = FakeDockerEngine()
    engine.exec_timeout = True
    backend = DockerBackend(engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))

    with pytest.raises(TimeoutError):
        await session.exec("sleep 100", timeout_s=0.01)

    assert engine.removes == 1


@pytest.mark.asyncio
async def test_docker_cancelled_exec_terminates_disposable_sandbox(monkeypatch) -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)
    session = await backend.create(SandboxSpec(SandboxSource.image("image")))

    async def cancelled_exec(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(engine, "exec", cancelled_exec)
    with pytest.raises(asyncio.CancelledError):
        await session.exec("sleep 100")

    assert engine.removes == 1


@pytest.mark.asyncio
async def test_docker_lifetime_timeout_reclaims_container() -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine)

    await backend.create(
        SandboxSpec(SandboxSource.image("image"), idle_timeout_s=0.01),
    )
    await asyncio.sleep(0.03)

    assert engine.removes == 1
    assert backend.metrics_snapshot().operations["lifetime_timeout"].count == 1


@pytest.mark.asyncio
async def test_docker_owner_lease_is_shared_and_removed_on_shutdown(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("PSRL_ACTOR_ID", "w0-host-999999-abcdef12")
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker_lifecycle.spawn_node_gc",
        lambda *args, **kwargs: SimpleNamespace(poll=lambda: None),
    )
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker_lifecycle.force_remove_containers_by_label",
        lambda *args, **kwargs: [],
    )
    engine = FakeDockerEngine()
    backend = DockerBackend(
        engine=engine,
        lifecycle={
            "heartbeat_dir": str(tmp_path),
            "lease_ttl_s": 10.0,
            "heartbeat_interval_s": 0.01,
        },
    )

    session = await backend.create(SandboxSpec(SandboxSource.image("image")))
    heartbeat = docker_utils.owner_heartbeat_path(str(tmp_path), backend.owner_id)
    assert os.path.exists(heartbeat)
    first_mtime = os.stat(heartbeat).st_mtime

    await asyncio.sleep(0.05)
    assert os.stat(heartbeat).st_mtime > first_mtime

    await session.terminate()
    assert os.path.exists(heartbeat)
    await backend.shutdown()
    assert not os.path.exists(heartbeat)


@pytest.mark.asyncio
async def test_docker_passes_opaque_cgroup_parent_to_engine() -> None:
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine, cgroup_parent="system.slice/psrl.slice")

    await backend.create(SandboxSpec(SandboxSource.image("image")))
    assert engine.config is not None
    assert engine.config["HostConfig"]["CgroupParent"] == "system.slice/psrl.slice"


@pytest.mark.asyncio
async def test_docker_disk_admission_blocks_when_free_space_is_low(monkeypatch) -> None:
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.shutil.disk_usage",
        lambda path: SimpleNamespace(total=0, used=0, free=1024 * 1024),  # 1 MiB free
    )
    engine = FakeDockerEngine()
    backend = DockerBackend(
        engine=engine,
        disk_admission={"path": "/dockerdata", "min_free_mb": 1024, "wait_timeout_s": 0},
    )

    with pytest.raises(RuntimeError, match="admission threshold"):
        await backend.create(SandboxSpec(SandboxSource.image("image")))
    assert engine.removes == 0  # nothing was created


@pytest.mark.asyncio
async def test_docker_disk_admission_allows_ample_free_space(monkeypatch) -> None:
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker.shutil.disk_usage",
        lambda path: SimpleNamespace(total=0, used=0, free=4096 * 1024 * 1024),
    )
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine, disk_admission={"path": "/dockerdata", "min_free_mb": 1024})

    session = await backend.create(SandboxSpec(SandboxSource.image("image")))

    assert session.ref.sandbox_id == "container-id"


@pytest.mark.asyncio
async def test_docker_disk_admission_fails_when_path_cannot_be_inspected(monkeypatch) -> None:
    def fail_disk_usage(path: str):
        raise OSError("not mounted")

    monkeypatch.setattr("psrl.sandbox.backends.docker.shutil.disk_usage", fail_disk_usage)
    backend = DockerBackend(
        engine=FakeDockerEngine(),
        disk_admission={"path": "/dockerdata", "min_free_mb": 1024},
    )

    with pytest.raises(RuntimeError, match="Could not inspect Docker data path"):
        await backend.create(SandboxSpec(SandboxSource.image("image")))


@pytest.mark.asyncio
async def test_docker_without_owner_skips_lease_and_node_gc(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("PSRL_ACTOR_ID", raising=False)
    spawned: list[str] = []
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker_lifecycle.spawn_node_gc",
        lambda *args, **kwargs: spawned.append("gc"),
    )
    engine = FakeDockerEngine()
    backend = DockerBackend(engine=engine, lifecycle={"heartbeat_dir": str(tmp_path)})

    session = await backend.create(SandboxSpec(SandboxSource.image("image")))

    assert spawned == []  # an owner-less backend never starts a collector
    assert list(tmp_path.iterdir()) == []  # and never writes a lease file
    await session.terminate()
