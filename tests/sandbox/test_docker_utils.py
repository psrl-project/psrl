from __future__ import annotations

import os
import subprocess
import time
from types import SimpleNamespace

import pytest
from psrl.sandbox.utils import docker_utils


def test_force_remove_batches_all_matching_containers(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[1] == "ps":
            return SimpleNamespace(returncode=0, stdout=b"first\nsecond\n", stderr=b"")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    removed = docker_utils.force_remove_containers_by_label("psrl.actor_id", "actor")

    assert removed == ["first", "second"]
    assert calls[1] == ["docker", "rm", "-f", "-v", "first", "second"]


def _write_heartbeat(heartbeat_dir: str, owner_id: str, age_s: float) -> None:
    docker_utils.write_owner_heartbeat(heartbeat_dir, owner_id)
    when = time.time() - age_s
    os.utime(docker_utils.owner_heartbeat_path(heartbeat_dir, owner_id), (when, when))


def _install_docker_list(monkeypatch, containers: list[tuple[str, str]]) -> list[list[str]]:
    """Fake ``docker ps``/``docker rm -f`` and return the recorded argv calls."""
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(list(args))
        if args[1] == "ps":
            stdout = "".join(f"{container_id}\t{actor_id}\n" for container_id, actor_id in containers).encode()
            return SimpleNamespace(returncode=0, stdout=stdout, stderr=b"")
        return SimpleNamespace(returncode=0, stdout=("\n".join(args[3:]) + "\n").encode(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_sweep_reaps_only_containers_with_expired_owner_leases(monkeypatch, tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    _write_heartbeat(heartbeat_dir, "stale-owner", age_s=10_000)
    _write_heartbeat(heartbeat_dir, "fresh-owner", age_s=1)
    containers = [
        ("stale-sandbox", "stale-owner"),
        ("fresh-sandbox", "fresh-owner"),
    ]
    calls = _install_docker_list(monkeypatch, containers)

    reaped, _ = docker_utils.sweep_stale_sandboxes(heartbeat_dir, ttl_s=900)

    assert reaped == ["stale-sandbox"]
    assert ["docker", "rm", "-f", "-v", "stale-sandbox"] in calls
    assert os.path.exists(docker_utils.owner_heartbeat_path(heartbeat_dir, "fresh-owner"))


def test_sweep_reaps_missing_owner_heartbeat(monkeypatch, tmp_path) -> None:
    heartbeat_dir = str(tmp_path / "hb")
    calls = _install_docker_list(monkeypatch, [("orphan", "dead-actor")])

    reaped, _ = docker_utils.sweep_stale_sandboxes(heartbeat_dir, ttl_s=900)

    assert reaped == ["orphan"]
    assert ["docker", "rm", "-f", "-v", "orphan"] in calls


def test_gc_lock_is_process_scoped_and_reusable(tmp_path) -> None:
    lock_path = str(tmp_path / "gc.lock")
    first = docker_utils._acquire_gc_lock(lock_path)
    assert first is not None
    assert docker_utils._acquire_gc_lock(lock_path) is None
    docker_utils._release_gc_lock(first)
    second = docker_utils._acquire_gc_lock(lock_path)
    assert second is not None
    docker_utils._release_gc_lock(second)


def test_run_gc_loop_exits_when_node_is_idle(monkeypatch, tmp_path) -> None:
    sweeps: list[int] = []
    lock_handle = object()
    monkeypatch.setattr(docker_utils, "_acquire_gc_lock", lambda lock_path: lock_handle)
    monkeypatch.setattr(docker_utils, "_release_gc_lock", lambda handle: sweeps.append(1))
    monkeypatch.setattr(docker_utils, "sweep_stale_sandboxes", lambda *a, **k: ([], 0))

    assert docker_utils.run_gc_loop(str(tmp_path), ttl_s=900, interval_s=1, idle_exit_cycles=1) == 0
    assert sweeps == [1]  # the lock is always released


def test_spawn_node_gc_validates_interval_and_ttl(tmp_path) -> None:
    with pytest.raises(ValueError):
        docker_utils.spawn_node_gc(str(tmp_path), ttl_s=900, interval_s=0)
    with pytest.raises(ValueError):
        docker_utils.spawn_node_gc(str(tmp_path), ttl_s=0, interval_s=60)


def test_failed_docker_query_is_not_reported_as_idle(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(docker_utils, "_list_sandbox_containers", lambda command, **kwargs: None)

    reaped, remaining = docker_utils.sweep_stale_sandboxes(str(tmp_path), ttl_s=900)

    assert reaped == []
    assert remaining is None
