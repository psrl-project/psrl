from __future__ import annotations

from types import SimpleNamespace

import pytest
from psrl.sandbox.backends.docker_lifecycle import DockerLifecycle, DockerLifecycleConfig
from psrl.sandbox.capacity import SandboxCapacityConfig


def test_lifecycle_restarts_an_idle_collector(monkeypatch, tmp_path) -> None:
    spawned = []
    removed = []

    def spawn(*args, **kwargs):
        process = SimpleNamespace(poll=lambda: 0)
        spawned.append(process)
        return process

    monkeypatch.setattr("psrl.sandbox.backends.docker_lifecycle.spawn_node_gc", spawn)
    monkeypatch.setattr(
        "psrl.sandbox.backends.docker_lifecycle.force_remove_containers_by_label",
        lambda *args, **kwargs: removed.append((args, kwargs)),
    )
    lifecycle = DockerLifecycle(
        "worker-1",
        DockerLifecycleConfig(
            heartbeat_dir=str(tmp_path),
            heartbeat_interval_s=60,
            lease_ttl_s=120,
        ),
    )

    lifecycle.start()
    lifecycle.start()
    lifecycle.close()

    assert len(spawned) == 2
    assert len(removed) == 1


def test_lifecycle_config_attaches_explicit_docker_host() -> None:
    config = DockerLifecycleConfig.from_value(None, docker_host="tcp://docker.example:2376")

    assert config.docker_command == ("docker", "--host", "tcp://docker.example:2376")


def test_default_crash_recovery_precedes_capacity_expiry() -> None:
    lifecycle = DockerLifecycleConfig()
    capacity = SandboxCapacityConfig()

    assert lifecycle.lease_ttl_s + lifecycle.gc_interval_s < capacity.lease_ttl_s


def test_lifecycle_fails_when_crash_recovery_cannot_start(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("psrl.sandbox.backends.docker_lifecycle.spawn_node_gc", lambda *args, **kwargs: None)
    lifecycle = DockerLifecycle(
        "worker-1",
        DockerLifecycleConfig(heartbeat_dir=str(tmp_path)),
    )

    with pytest.raises(RuntimeError, match="Could not start Docker sandbox crash recovery"):
        lifecycle.start()

    lifecycle.close()
