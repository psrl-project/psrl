import fcntl
import json
import os
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from psrl.utils import node_shared_weight_cache as cache_module
from psrl.utils.node_shared_weight_cache import (
    WeightCacheTopology,
    fingerprint_safetensors_checkpoint,
    publish_or_map_weight_arena,
    validate_node_cache_directory,
    validate_reward_cache_config,
)
from psrl.utils.weight_arena import pack_module_weights
from torch import nn

pytestmark = pytest.mark.cpu_test


def _arena_handle():
    module = nn.Module()
    module.register_buffer("weights", torch.arange(64, dtype=torch.uint8))
    return pack_module_weights(
        module,
        max_chunk_bytes=32,
        alignment_bytes=16,
        require_cuda=False,
    )


@pytest.fixture(autouse=True)
def _memory_backed_cache_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cache_module, "_filesystem_for_path", lambda path: ("tmpfs", "/dev/shm"))


def _topology(rank: int) -> WeightCacheTopology:
    return WeightCacheTopology(
        tp_size=4,
        tp_rank=rank,
        pp_size=1,
        pp_rank=0,
        ep_size=1,
        ep_rank=0,
    )


def _publish(
    cache_root: Path,
    rank: int = 0,
    fingerprint: str = "model",
    populate_handle=None,
    backend: str = "shm",
):
    return publish_or_map_weight_arena(
        _arena_handle(),
        cache_root=cache_root,
        job_id="job-1",
        node_id="node-1",
        model_id="/models/reward",
        checkpoint_fingerprint=fingerprint,
        model_dtype="torch.bfloat16",
        quantization=None,
        topology=_topology(rank),
        timeout_s=2,
        populate_handle=populate_handle,
        backend=backend,
        shm_reserve_bytes=0,
        memfd_reserve_bytes=0,
    )


def test_two_tp4_instances_share_four_cache_inodes(tmp_path: Path) -> None:
    caches = [_publish(tmp_path, rank) for _instance in range(2) for rank in range(4)]
    try:
        inodes_by_rank = {
            rank: {cache.inode for cache in caches if cache.topology.tp_rank == rank} for rank in range(4)
        }
        assert all(len(inodes) == 1 for inodes in inodes_by_rank.values())
        assert len({cache.inode for cache in caches}) == 4
        assert sum(cache.builder for cache in caches) == 4
        assert max(cache.unique_cache_bytes for cache in caches) == 4 * 64
    finally:
        for cache in caches:
            cache.close()
    assert not (tmp_path / "job-1").exists()


def test_first_lease_cleans_stale_job_namespace(tmp_path: Path) -> None:
    stale = tmp_path / "job-1" / "stale.arena"
    stale.parent.mkdir()
    stale.write_bytes(b"stale")
    cache = _publish(tmp_path)
    try:
        assert not stale.exists()
        assert cache.unique_cache_bytes == 64
    finally:
        cache.close()


def test_concurrent_publish_has_one_builder(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)
    caches = []
    errors = []
    populate_calls = 0
    state_lock = threading.Lock()

    def populate() -> None:
        nonlocal populate_calls
        with state_lock:
            populate_calls += 1

    def publish() -> None:
        try:
            barrier.wait()
            caches.append(_publish(tmp_path, populate_handle=populate))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=publish) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert not errors
        assert len(caches) == 2
        assert caches[0].inode == caches[1].inode
        assert sum(cache.builder for cache in caches) == 1
        assert populate_calls == 1
    finally:
        for cache in caches:
            cache.close()


def test_only_builder_populates_checkpoint_handle(tmp_path: Path) -> None:
    calls = []

    def populate() -> None:
        calls.append("checkpoint")

    first = _publish(tmp_path, populate_handle=populate)
    second = _publish(tmp_path, populate_handle=populate)
    try:
        assert first.builder
        assert not second.builder
        assert calls == ["checkpoint"]
        assert first.inode == second.inode
    finally:
        first.close()
        second.close()


def test_checkpoint_population_is_parallel_across_topology_ranks(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)
    state_lock = threading.Lock()
    both_active = threading.Event()
    release_population = threading.Event()
    active = 0
    max_active = 0
    caches = []
    errors = []

    def publish(rank: int) -> None:
        def populate() -> None:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
                if active == 2:
                    both_active.set()
            release_population.wait(1)
            with state_lock:
                active -= 1

        try:
            barrier.wait()
            caches.append(_publish(tmp_path, rank=rank, populate_handle=populate))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=publish, args=(rank,)) for rank in range(2)]
    for thread in threads:
        thread.start()
    parallel_population = both_active.wait(1)
    release_population.set()
    for thread in threads:
        thread.join()
    try:
        assert not errors
        assert parallel_population
        assert max_active == 2
        assert len(caches) == 2
    finally:
        for cache in caches:
            cache.close()


def test_restore_drops_private_mapping_without_evicting_shared_memory(tmp_path: Path) -> None:
    cache = _publish(tmp_path)
    try:
        assert cache.drop_resident_pages()
        assert cache.data_path.read_bytes() == bytes(range(64))
    finally:
        cache.close()


def test_incomplete_manifest_is_rebuilt(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    manifest_path = first.data_path.with_suffix(".json")
    manifest_path.write_text("{")
    second = _publish(tmp_path)
    try:
        assert second.builder
        assert json.loads(manifest_path.read_text())["total_bytes"] == 64
    finally:
        first.close()
        second.close()


def test_wrong_manifest_fingerprint_fails_fast(tmp_path: Path) -> None:
    first = _publish(tmp_path)
    manifest_path = first.data_path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["key"]["checkpoint_fingerprint"] = "other-model"
    manifest_path.write_text(json.dumps(manifest))
    try:
        with pytest.raises(RuntimeError, match="fingerprint does not match"):
            _publish(tmp_path)
    finally:
        first.close()


def test_lock_timeout_is_reported(tmp_path: Path) -> None:
    lock_path = tmp_path / "held.lock"
    first_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    second_fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(first_fd, fcntl.LOCK_EX)
        with pytest.raises(TimeoutError, match="Timed out"):
            cache_module._lock_with_timeout(second_fd, fcntl.LOCK_EX, 0.01, lock_path)
    finally:
        os.close(second_fd)
        os.close(first_fd)


def test_shared_memory_space_check_fails_before_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cache_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1, used=0, free=1),
    )
    with pytest.raises(RuntimeError, match="Insufficient shared memory"):
        _publish(tmp_path, backend="shm")


def test_auto_falls_back_to_memfd_when_shm_is_insufficient(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cache_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1, used=0, free=1),
    )
    monkeypatch.setattr(cache_module, "_mem_available_bytes", lambda: 1024**3)
    first = _publish(tmp_path, backend="auto")
    second = _publish(tmp_path, backend="auto")
    try:
        assert first.backend == "memfd"
        assert second.backend == "memfd"
        assert first.builder
        assert not second.builder
        assert first.inode == second.inode
        assert first.data_path is None
        assert second.data_path is None
        assert not list(tmp_path.rglob("*.arena"))
        assert bytes(first.arenas[0].tolist()) == bytes(range(64))
    finally:
        first.close()
        second.close()


def test_auto_uses_shm_without_starting_memfd_broker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cache_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1024**3, used=0, free=1024**3),
    )
    monkeypatch.setattr(
        cache_module,
        "_ensure_broker",
        lambda *args, **kwargs: pytest.fail("shm selection must not start a memfd broker"),
    )
    cache = _publish(tmp_path, backend="auto")
    try:
        assert cache.backend == "shm"
        assert cache.data_path is not None
    finally:
        cache.close()


def test_auto_recovers_from_publish_enospc_without_repopulating_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cache_module.shutil,
        "disk_usage",
        lambda path: SimpleNamespace(total=1024**3, used=0, free=1024**3),
    )
    monkeypatch.setattr(cache_module, "_mem_available_bytes", lambda: 1024**3)
    real_publish = cache_module._publish_arena
    publish_calls = 0
    populate_calls = 0

    def fail_first_shm_publish(*args, **kwargs):
        nonlocal publish_calls
        publish_calls += 1
        if publish_calls == 1:
            raise OSError(cache_module.errno.ENOSPC, "simulated tmpfs exhaustion")
        return real_publish(*args, **kwargs)

    def populate() -> None:
        nonlocal populate_calls
        populate_calls += 1

    monkeypatch.setattr(cache_module, "_publish_arena", fail_first_shm_publish)
    first = _publish(tmp_path, backend="auto", populate_handle=populate)
    second = _publish(tmp_path, backend="auto", populate_handle=populate)
    try:
        assert first.backend == second.backend == "memfd"
        assert first.inode == second.inode
        assert publish_calls == 1
        assert populate_calls == 1
    finally:
        first.close()
        second.close()


def test_memfd_broker_transfers_same_descriptor_to_subprocess() -> None:
    from psrl.utils.node_shared_memfd_broker import process_start_time, request

    token = f"test-{uuid.uuid4().hex}"
    cache_module._ensure_broker(token, 10)
    fd = os.memfd_create(
        "psrl-memfd-test",
        getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002),
    )
    os.write(fd, b"shared-arena")
    cache_module._seal_memfd(fd)
    identity = {
        "client_id": uuid.uuid4().hex,
        "pid": os.getpid(),
        "start_time": process_start_time(),
    }
    manifest = {"total_bytes": len(b"shared-arena")}
    response, returned_fd = request(
        token,
        {"command": "register", "key": "rank-0", "manifest": manifest, **identity},
        send_fd=fd,
    )
    assert response["found"]
    assert returned_fd is not None
    expected_inode = os.fstat(returned_fd).st_ino
    os.close(returned_fd)
    os.close(fd)
    code = """
import json, os, uuid
from psrl.utils.node_shared_memfd_broker import process_start_time, request
token, expected = os.environ['TOKEN'], int(os.environ['EXPECTED_INODE'])
identity = {'client_id': uuid.uuid4().hex, 'pid': os.getpid(), 'start_time': process_start_time()}
response, fd = request(token, {'command': 'get', 'key': 'rank-0', **identity})
try:
    print(json.dumps({'inode': os.fstat(fd).st_ino, 'data': os.pread(fd, 12, 0).decode()}))
finally:
    request(token, {'command': 'release', 'key': 'rank-0', **identity})
    os.close(fd)
"""
    child = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TOKEN": token, "EXPECTED_INODE": str(expected_inode)},
    )
    payload = json.loads(child.stdout)
    assert payload == {"inode": expected_inode, "data": "shared-arena"}
    request(token, {"command": "release", "key": "rank-0", **identity})


def test_only_memory_filesystems_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module, "_filesystem_for_path", lambda path: ("tmpfs", "/dev/shm"))
    assert validate_node_cache_directory(tmp_path) == tmp_path.resolve()
    for fs_type in ("ext4", "nfs"):
        monkeypatch.setattr(cache_module, "_filesystem_for_path", lambda path, value=fs_type: (value, "/mnt"))
        with pytest.raises(RuntimeError, match="must use node-local shared memory"):
            validate_node_cache_directory(tmp_path)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"reward_enabled": False}, "reward_enabled=true"),
        ({"reward_cpu_cache_pin_memory": True}, "pin_memory=false"),
        ({"reward_node_cache_wait_timeout_s": 0}, "must be positive"),
        ({"reward_cpu_cache_mode": "invalid"}, "must be 'node_shared'"),
        ({"reward_node_cache_backend": "disk"}, "must be 'auto', 'shm', or 'memfd'"),
        ({"reward_node_cache_shm_reserve_gb": -1}, "must be a non-negative GB value"),
    ],
)
def test_node_shared_config_fails_fast(
    tmp_path: Path,
    override: dict,
    error: str,
) -> None:
    config = {
        "reward_enabled": True,
        "reward_cpu_cache_mode": "node_shared",
        "reward_cpu_cache_pin_memory": False,
        "reward_node_cache_dir": str(tmp_path),
        "reward_node_cache_wait_timeout_s": 1,
        **override,
    }
    with pytest.raises((RuntimeError, ValueError), match=error):
        validate_reward_cache_config(config)


def test_safetensors_fingerprint_tracks_checkpoint_metadata(tmp_path: Path) -> None:
    shard = tmp_path / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"first")
    first = fingerprint_safetensors_checkpoint(tmp_path)
    shard.write_bytes(b"second-version")
    second = fingerprint_safetensors_checkpoint(tmp_path)
    assert first != second


def test_non_safetensors_checkpoint_fails_fast(tmp_path: Path) -> None:
    (tmp_path / "pytorch_model.bin").write_bytes(b"weights")
    with pytest.raises(RuntimeError, match="requires a safetensors checkpoint"):
        fingerprint_safetensors_checkpoint(tmp_path)
